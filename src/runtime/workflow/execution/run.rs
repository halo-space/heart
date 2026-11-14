//! Workflow execution loop and continuation handling.
//!
//! This module owns the mutable run loop and waiting continuations. The parent
//! execution module keeps admission, observation and state commit helpers.

use super::*;

impl Engine {
    pub(super) async fn run_inner(
        &mut self,
        initial_messages: Messages,
        next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: crate::Cancellation,
    ) -> Result<Messages, Error> {
        if let Some(existing) = &self.state.initial {
            if existing != &initial_messages {
                return Err(Error::new(
                    "CONFLICT",
                    "workflow initial Messages cannot change after activation",
                ));
            }
        } else {
            self.state.initial = Some(initial_messages);
        }

        loop {
            if self.state.status == graph::Status::Succeeded {
                return Ok(self.end_messages());
            }
            if self.state.status == graph::Status::Failed {
                return Err(self
                    .first_failure()
                    .unwrap_or_else(|| Error::new("FAILED", "workflow execution failed")));
            }
            if self.state.status == graph::Status::Cancelled || cancellation.is_cancelled() {
                self.cancel_active();
                self.state.status = graph::Status::Cancelled;
                return Err(Error::new("CANCELLED", "workflow run cancelled"));
            }
            if let Some((node_id, exec_id, _wait_id, response)) = self.state.resolved_wait() {
                let target = self.target_for(&node_id, exec_id)?;
                let input = match &target {
                    Target::Agent(..) => {
                        let execution = &self.state.nodes[&node_id].executions[&exec_id];
                        let mut input = self.agent_input_for(&node_id, &execution.from)?;
                        let reply = messages_to_agent_input(&response);
                        input.payload = reply.payload;
                        // Reply metadata may supply extra context, but conflicting
                        // ownership still reaches the Agent's canonical check.
                        input.metadata.extend(reply.metadata);
                        serde_json::to_value(input)
                            .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?
                    }
                    Target::Graph { .. } => self.input_for(
                        &node_id,
                        &self.state.nodes[&node_id].executions[&exec_id].from,
                    )?,
                    Target::Node(_) => serde_json::to_value(response)
                        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?,
                };
                let retry = self.definition().nodes[&node_id].retry.clone();
                let start_idx = self.state.nodes[&node_id].executions[&exec_id]
                    .attempts
                    .last()
                    .map(|attempt| attempt.idx)
                    .ok_or_else(|| Error::new("INVALID_STATE", "execution has no attempt"))?;
                let base = self.state.nodes[&node_id].executions[&exec_id]
                    .attempts
                    .last()
                    .expect("validated Attempt")
                    .clone();
                let view = self.view_for(&node_id, exec_id, input.clone(), Some(start_idx))?;
                let invocation = events::with_source(
                    self.state.chat_id,
                    &node_id,
                    exec_id,
                    invoke_with_retry(
                        target,
                        input,
                        view,
                        cancellation.clone(),
                        retry,
                        next_exec_id.clone(),
                        self.state.chat_id,
                        start_idx,
                        committed_tool_facts(&self.state.nodes[&node_id].executions[&exec_id]),
                        None,
                    ),
                );
                let invocation =
                    self.continue_invocation(&node_id, invocation, &cancellation, &base);
                let invocation = Box::pin(invocation).await?;
                let invocation_graph = invocation.graph;
                self.replace_resumed_attempts(&node_id, exec_id, invocation.attempts, &base)?;
                if let Some(graph) = invocation_graph {
                    self.attach_graph(&node_id, exec_id, graph)?;
                }
                match invocation.outcome {
                    Outcome::Complete(value) => {
                        let messages = self.output_messages(&node_id, value)?;
                        let to = self.outgoing(&node_id, &messages)?;
                        self.state
                            .complete_execution(&node_id, exec_id, messages, to)?;
                    }
                    Outcome::Wait(wait_id) => {
                        self.state.wait_execution(&node_id, exec_id, wait_id)?;
                    }
                    Outcome::Waiting => self.wait_call(&node_id, exec_id)?,
                    Outcome::Cancelled => {
                        self.state.cancel_execution(&node_id, exec_id)?;
                    }
                    Outcome::Failed(error) => {
                        self.state.fail_execution(&node_id, exec_id, error)?;
                    }
                }
                self.state.recompute_usage();
                self.observe_commit(&node_id, exec_id).await?;
                // Resume has the same Barrier semantics as an ordinary
                // invocation. Other active/waiting siblings still block it.
                self.state.advance_step();
                continue;
            }
            if Box::pin(self.resume_waiting(&next_exec_id, &cancellation)).await? {
                // Reenter each saved invocation once per run. A method that
                // waits again must yield its scene, not spin on the same wait.
                if !self.state.advance_step() {
                    self.state.status = graph::Status::Paused;
                    return Err(Error::new(
                        "WAITING",
                        "workflow batch did not reach a barrier",
                    ));
                }
                continue;
            }
            let ready = self.ready_nodes();
            if ready.is_empty() {
                if !self.state.waits.is_empty() || !self.state.pauses().is_empty() {
                    self.state.status = graph::Status::Paused;
                    return Err(Error::new("WAITING", "workflow is waiting for input"));
                }
                if self.reached_end() {
                    self.state.status = graph::Status::Succeeded;
                    return Ok(self.end_messages());
                }
                self.state.status = graph::Status::Failed;
                return Err(self.first_failure().unwrap_or_else(|| {
                    Error::new(
                        "INVALID_STATE",
                        "workflow has no ready node and has not reached END",
                    )
                }));
            }

            let mut specs = Vec::with_capacity(ready.len());
            let mut seen = BTreeSet::new();
            for node_id in ready {
                let exec_id = next_exec_id();
                if !seen.insert(exec_id) {
                    return Err(Error::new("CONFLICT", "exec_id must be globally unique"));
                }
                let from = self.source_executions(&node_id)?;
                let input = self.input_for(&node_id, &from)?;
                let target = if let Some(workflow) = self.workflow.graph_node(&node_id) {
                    Target::Graph {
                        workflow: Arc::new(workflow.clone()),
                        state: None,
                        history: self.inherited_graph(&node_id),
                    }
                } else if let Some(agent) = self.workflow.agent(&node_id) {
                    Target::Agent(agent, None)
                } else {
                    Target::Node(self.workflow.executor(&node_id).ok_or_else(|| {
                        Error::new("UNSUPPORTED", "workflow node has no executor")
                    })?)
                };
                let retry = self.definition().nodes[&node_id].retry.clone();
                specs.push((node_id, exec_id, from, input, target, retry));
            }

            let definition = self.definition().clone();
            self.state.start_batch(
                &definition,
                specs
                    .iter()
                    .map(|(node_id, exec_id, from, _, _, _)| {
                        (node_id.clone(), *exec_id, from.clone())
                    })
                    .collect(),
            )?;
            let specs = specs
                .into_iter()
                .map(|(node_id, exec_id, from, input, target, retry)| {
                    let view = self.view_for(&node_id, exec_id, input.clone(), Some(0))?;
                    Ok((node_id, exec_id, from, input, target, retry, view))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            for (node_id, exec_id, _, _, target, _, _) in &specs {
                if !matches!(target, Target::Graph { .. }) {
                    self.state
                        .node_mut(node_id)?
                        .execution_mut(*exec_id)?
                        .status = node::status::execution::Status::Pending;
                }
            }
            // Capture allocated child executions before polling their work,
            // including pending leaves. This is not a persistence trigger.
            Box::pin(self.publish(None, false)).await?;

            let task_cancellation = cancellation.clone();
            let task_next_exec_id = Arc::clone(&next_exec_id);
            let chat_id = self.state.chat_id;
            let relay = events::Relay::default();
            let slots = admission::slots();
            let changed = Arc::new(futures_util::task::AtomicWaker::new());
            let interaction = events::interaction();
            let mut active = Vec::new();
            let mut results = specs
                .into_iter()
                .map(|(node_id, exec_id, from, input, target, retry, view)| {
                    let cancellation = task_cancellation.clone();
                    let next_exec_id = Arc::clone(&task_next_exec_id);
                    let relay = relay.clone();
                    // Graphs are containers, not admitted work. Their leaves
                    // use this same outer semaphore, including after restore.
                    let slot = (!matches!(&target, Target::Graph { .. })).then(|| {
                        crate::runtime::agent::reply::Slot::waiting(slots.clone(), changed.clone())
                    });
                    let run = interaction.as_ref().map(|run| match &slot {
                        Some(slot) => run.branch(slot.clone()),
                        None => run.child(),
                    });
                    active.push((node_id.clone(), exec_id, run.clone(), slot.clone()));
                    async move {
                        let owner = run.as_ref().map(|run| run.owner());
                        let entering = match &run {
                            Some(run) => run.continue_execution(&cancellation).await,
                            None => match &slot {
                                Some(slot) => slot.acquire(&cancellation).await,
                                None => Ok(()),
                            },
                        };
                        if let Err(error) = entering {
                            return (
                                node_id,
                                exec_id,
                                from,
                                Invocation {
                                    attempts: vec![node::Attempt {
                                        status: node::status::attempt::Status::Cancelled,
                                        ..Default::default()
                                    }],
                                    outcome: Outcome::Failed(error),
                                    graph: None,
                                },
                            );
                        }
                        let result = events::with_optional_interaction(
                            run,
                            events::with_relay(
                                relay,
                                events::with_source(
                                    chat_id,
                                    &node_id,
                                    exec_id,
                                    invoke_with_retry(
                                        target,
                                        input,
                                        view,
                                        cancellation,
                                        retry,
                                        next_exec_id,
                                        chat_id,
                                        0,
                                        node::Attempt::default(),
                                        None,
                                    ),
                                ),
                            ),
                        )
                        .await;
                        drop(owner);
                        if let Some(slot) = slot {
                            slot.release();
                        }
                        (node_id, exec_id, from, result)
                    }
                })
                .collect::<futures_util::stream::FuturesUnordered<_>>();

            enum ProgressOrResult<T> {
                Facts(Box<events::Progress>),
                Finished(Box<T>),
            }
            loop {
                let next = std::future::poll_fn(|cx| {
                    changed.register(cx.waker());
                    match results.poll_next_unpin(cx) {
                        std::task::Poll::Ready(Some(result)) => {
                            return std::task::Poll::Ready(Some(ProgressOrResult::Finished(
                                Box::new(result),
                            )));
                        }
                        std::task::Poll::Ready(None) => return std::task::Poll::Ready(None),
                        std::task::Poll::Pending => {}
                    }
                    if let std::task::Poll::Ready(progress) = relay.poll(cx) {
                        return std::task::Poll::Ready(Some(ProgressOrResult::Facts(Box::new(
                            progress,
                        ))));
                    }
                    for (node_id, exec_id, run, _) in &active {
                        if self.state.nodes[node_id].executions[exec_id]
                            .attempts
                            .last()
                            .is_some_and(|attempt| {
                                attempt.status == node::status::attempt::Status::Waiting
                            })
                            && let Some(run) = run
                            && run.children_blocked()
                        {
                            run.suspend();
                        }
                    }
                    if active
                        .iter()
                        .all(|(_, _, _, slot)| slot.as_ref().is_none_or(|slot| !slot.held()))
                        && let Some(run) = &interaction
                    {
                        run.suspend();
                    }
                    std::task::Poll::Pending
                })
                .await;
                let Some(next) = next else { break };
                let (node_id, exec_id, _from, result) = match next {
                    ProgressOrResult::Facts(progress) => {
                        let events::Progress {
                            node_id,
                            exec_id,
                            attempt,
                            event,
                            entered,
                            ack,
                        } = *progress;
                        let result = Box::pin(self.accept_progress(
                            &node_id,
                            exec_id,
                            attempt,
                            event,
                            entered,
                            &cancellation,
                            None,
                        ))
                        .await;
                        let _ = ack.send(result.clone());
                        result?;
                        continue;
                    }
                    ProgressOrResult::Finished(result) => *result,
                };
                let invocation_graph = result.graph;
                self.replace_attempts(&node_id, exec_id, result.attempts)?;
                if let Some(graph) = invocation_graph {
                    self.attach_graph(&node_id, exec_id, graph)?;
                }
                match result.outcome {
                    Outcome::Complete(value) => {
                        let messages = self.output_messages(&node_id, value)?;
                        let to = self.outgoing(&node_id, &messages)?;
                        self.state
                            .complete_execution(&node_id, exec_id, messages, to)?;
                    }
                    Outcome::Wait(wait_id) => {
                        self.state.wait_execution(&node_id, exec_id, wait_id)?;
                    }
                    Outcome::Waiting => self.wait_call(&node_id, exec_id)?,
                    Outcome::Cancelled => {
                        self.state.cancel_execution(&node_id, exec_id)?;
                    }
                    Outcome::Failed(error) => {
                        self.state.fail_execution(&node_id, exec_id, error)?;
                    }
                }
                self.state.recompute_usage();
                self.observe_commit(&node_id, exec_id).await?;
            }
            if self.state.waits.is_empty() && self.state.pauses().is_empty() {
                self.state.status = graph::Status::Running;
            }
            self.state.recompute_usage();
            if cancellation.is_cancelled() {
                self.cancel_active();
                self.state.status = graph::Status::Cancelled;
                return Err(Error::new("CANCELLED", "workflow run cancelled"));
            }
            if self.state.nodes.values().any(|node| {
                node.executions.values().any(|execution| {
                    execution.idx == self.state.step
                        && execution.status == node::status::execution::Status::Cancelled
                })
            }) {
                self.state.status = graph::Status::Cancelled;
                return Err(Error::new("CANCELLED", "workflow execution cancelled"));
            }
            if !self.state.advance_step() {
                self.state.status = graph::Status::Paused;
                return Err(Error::new(
                    "WAITING",
                    "workflow batch did not reach a barrier",
                ));
            }
        }
    }

    async fn resume_waiting(
        &mut self,
        next_exec_id: &Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: &crate::Cancellation,
    ) -> Result<bool, Error> {
        let mut recovery_reply = self.recovery_reply.take();
        if let Some(reply) = &recovery_reply {
            let current = reply
                .path
                .first()
                .ok_or_else(|| Error::new("INVALID_STATE", "saved reply path is empty"))?;
            let execution = self
                .state
                .nodes
                .get(&current.node_id)
                .and_then(|node| node.executions.get(&current.exec_id))
                .filter(|execution| {
                    execution.idx == self.state.step
                        && execution.waits.is_empty()
                        && matches!(
                            execution.status,
                            node::status::execution::Status::Running
                                | node::status::execution::Status::Waiting
                        )
                })
                .filter(|execution| {
                    execution
                        .attempts
                        .last()
                        .is_some_and(|attempt| attempt.idx == current.idx)
                })
                .ok_or_else(|| Error::new("INVALID_STATE", "saved reply target is not active"))?;
            let _ = execution;
        }
        let targets = self
            .state
            .nodes
            .iter()
            .flat_map(|(node_id, node)| {
                node.executions
                    .values()
                    .filter(|execution| {
                        execution.idx == self.state.step
                            && execution.waits.is_empty()
                            && matches!(
                                execution.status,
                                node::status::execution::Status::Pending
                                    | node::status::execution::Status::Running
                                    | node::status::execution::Status::Waiting
                            )
                            && execution.attempts.last().is_some_and(|attempt| {
                                attempt.status == node::status::attempt::Status::Waiting
                            })
                    })
                    .map(move |execution| (node_id.clone(), execution.exec_id))
            })
            .collect::<Vec<_>>();
        if targets.is_empty() {
            return Ok(false);
        }
        let mut specs = Vec::new();
        let mut bases = std::collections::BTreeMap::new();
        for (node_id, exec_id) in targets {
            let execution = self.state.nodes[&node_id].executions[&exec_id].clone();
            let previous_idx = execution
                .attempts
                .last()
                .map(|attempt| attempt.idx)
                .ok_or_else(|| Error::new("INVALID_STATE", "waiting execution has no attempt"))?;
            let retry = self.definition().nodes[&node_id].retry.clone();
            let target = self.target_for(&node_id, exec_id)?;
            let mut input = self.input_for(&node_id, &execution.from)?;
            let mut nested_reply = None;
            if recovery_reply.as_ref().is_some_and(|reply| {
                reply
                    .path
                    .first()
                    .is_some_and(|target| target.node_id == node_id && target.exec_id == exec_id)
            }) {
                let mut reply = recovery_reply.take().expect("matched recovery reply");
                let target = reply.path.remove(0);
                if target.node_id != node_id || target.exec_id != exec_id {
                    return Err(Error::new("INVALID_STATE", "saved reply target changed"));
                }
                match self.target_for(&node_id, exec_id)? {
                    Target::Agent(_, _) if reply.path.is_empty() => {
                        let mut agent_input: Input = serde_json::from_value(input)
                            .map_err(|error| Error::new("INVALID_STATE", error.to_string()))?;
                        agent_input.payload = reply.input.payload;
                        agent_input.metadata.extend(reply.input.metadata);
                        input = serde_json::to_value(agent_input)
                            .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
                    }
                    Target::Graph { .. } if !reply.path.is_empty() => {
                        nested_reply = Some(reply);
                    }
                    _ => {
                        return Err(Error::new(
                            "INVALID_STATE",
                            "saved reply path does not match its Workflow Node",
                        ));
                    }
                }
            }
            let base = execution
                .attempts
                .last()
                .expect("validated current Attempt")
                .clone();
            let view = self.view_for(&node_id, exec_id, input.clone(), Some(previous_idx))?;
            bases.insert((node_id.clone(), exec_id), base);
            specs.push((
                node_id,
                exec_id,
                target,
                input,
                retry,
                previous_idx,
                committed_tool_facts(&execution),
                nested_reply,
                view,
            ));
        }
        if recovery_reply.is_some() {
            return Err(Error::new(
                "INVALID_STATE",
                "saved reply has no active continuation",
            ));
        }
        // Rebuild all current consumers concurrently under the same admission
        // limit. Awaiting one Agent to completion first can strand a FIFO head
        // whose owner is another saved branch.
        let relay = events::Relay::default();
        let slots = admission::slots();
        let changed = Arc::new(futures_util::task::AtomicWaker::new());
        let interaction = events::interaction();
        let chat_id = self.state.chat_id;
        let mut active = Vec::new();
        let mut results = specs
            .into_iter()
            .map(
                |(node_id, exec_id, target, input, retry, idx, messages, reply, view)| {
                    let cancellation = cancellation.clone();
                    let next_exec_id = Arc::clone(next_exec_id);
                    let relay = relay.clone();
                    let slot = (!matches!(&target, Target::Graph { .. })).then(|| {
                        crate::runtime::agent::reply::Slot::waiting(slots.clone(), changed.clone())
                    });
                    let run = interaction.as_ref().map(|run| match &slot {
                        Some(slot) => run.branch(slot.clone()),
                        None => run.child(),
                    });
                    active.push((node_id.clone(), exec_id, run.clone(), slot.clone()));
                    async move {
                        let owner = run.as_ref().map(|run| run.owner());
                        let entering = match &run {
                            Some(run) => run.continue_execution(&cancellation).await,
                            None => match &slot {
                                Some(slot) => slot.acquire(&cancellation).await,
                                None => Ok(()),
                            },
                        };
                        let invocation = match entering {
                            Ok(()) => {
                                events::with_optional_interaction(
                                    run,
                                    events::with_relay(
                                        relay,
                                        events::with_source(
                                            chat_id,
                                            &node_id,
                                            exec_id,
                                            invoke_with_retry(
                                                target,
                                                input,
                                                view,
                                                cancellation,
                                                retry,
                                                next_exec_id,
                                                chat_id,
                                                idx,
                                                messages,
                                                reply,
                                            ),
                                        ),
                                    ),
                                )
                                .await
                            }
                            Err(error) => Invocation {
                                attempts: Vec::new(),
                                outcome: if error.code == "CANCELLED" {
                                    Outcome::Cancelled
                                } else {
                                    Outcome::Failed(error)
                                },
                                graph: None,
                            },
                        };
                        drop(owner);
                        if let Some(slot) = slot {
                            slot.release();
                        }
                        (node_id, exec_id, invocation)
                    }
                },
            )
            .collect::<futures_util::stream::FuturesUnordered<_>>();
        enum Next<T> {
            Facts(Box<events::Progress>),
            Finished(Box<T>),
        }
        loop {
            let next = std::future::poll_fn(|cx| {
                changed.register(cx.waker());
                match results.poll_next_unpin(cx) {
                    std::task::Poll::Ready(Some(result)) => {
                        return std::task::Poll::Ready(Some(Next::Finished(Box::new(result))));
                    }
                    std::task::Poll::Ready(None) => return std::task::Poll::Ready(None),
                    std::task::Poll::Pending => {}
                }
                if let std::task::Poll::Ready(facts) = relay.poll(cx) {
                    return std::task::Poll::Ready(Some(Next::Facts(Box::new(facts))));
                }
                for (node_id, exec_id, run, _) in &active {
                    if self.state.nodes[node_id].executions[exec_id]
                        .attempts
                        .last()
                        .is_some_and(|attempt| {
                            attempt.status == node::status::attempt::Status::Waiting
                        })
                        && let Some(run) = run
                        && run.children_blocked()
                    {
                        run.suspend();
                    }
                }
                if active
                    .iter()
                    .all(|(_, _, _, slot)| slot.as_ref().is_none_or(|slot| !slot.held()))
                    && let Some(run) = &interaction
                {
                    run.suspend();
                }
                std::task::Poll::Pending
            })
            .await;
            let Some(next) = next else { break };
            let (node_id, exec_id, invocation) = match next {
                Next::Facts(progress) => {
                    let events::Progress {
                        node_id,
                        exec_id,
                        attempt,
                        event,
                        entered,
                        ack,
                    } = *progress;
                    let result = Box::pin(self.accept_progress(
                        &node_id,
                        exec_id,
                        attempt,
                        event,
                        entered,
                        cancellation,
                        bases.get(&(node_id.clone(), exec_id)),
                    ))
                    .await;
                    let _ = ack.send(result.clone());
                    result?;
                    continue;
                }
                Next::Finished(result) => *result,
            };
            let base = bases
                .get(&(node_id.clone(), exec_id))
                .expect("saved continuation baseline");
            // Admission can be cancelled before invoking any component. Keep
            // the entire original scene; there is no replacement Attempt.
            if invocation.attempts.is_empty() {
                match invocation.outcome {
                    Outcome::Cancelled => self.state.cancel_execution(&node_id, exec_id)?,
                    Outcome::Failed(error) => {
                        self.state.fail_execution(&node_id, exec_id, error)?
                    }
                    _ => return Err(Error::new("INVALID_STATE", "continuation has no Attempt")),
                }
                self.state.recompute_usage();
                self.observe_commit(&node_id, exec_id).await?;
                continue;
            }
            let invocation_graph = invocation.graph;
            self.replace_resumed_attempts(&node_id, exec_id, invocation.attempts, base)?;
            if let Some(graph) = invocation_graph {
                self.attach_graph(&node_id, exec_id, graph)?;
            }
            match invocation.outcome {
                Outcome::Complete(value) => {
                    let messages = self.output_messages(&node_id, value)?;
                    let to = self.outgoing(&node_id, &messages)?;
                    self.state
                        .complete_execution(&node_id, exec_id, messages, to)?;
                }
                Outcome::Wait(wait_id) => self.state.wait_execution(&node_id, exec_id, wait_id)?,
                Outcome::Waiting => self.wait_call(&node_id, exec_id)?,
                Outcome::Cancelled => self.state.cancel_execution(&node_id, exec_id)?,
                Outcome::Failed(error) => self.state.fail_execution(&node_id, exec_id, error)?,
            }
            self.state.recompute_usage();
            self.observe_commit(&node_id, exec_id).await?;
        }
        Ok(true)
    }

    /// Restored invocations need the same backpressured native handoff as
    /// fresh work. This is private execution plumbing, not a second run API.
    async fn continue_invocation<F: Future<Output = Invocation>>(
        &mut self,
        node_id: &str,
        invocation: F,
        cancellation: &crate::Cancellation,
        base: &node::Attempt,
    ) -> Result<Invocation, Error> {
        let relay = events::Relay::default();
        let changed = Arc::new(futures_util::task::AtomicWaker::new());
        let slot = self.workflow.graph_node(node_id).is_none().then(|| {
            crate::runtime::agent::reply::Slot::waiting(admission::slots(), changed.clone())
        });
        let run = events::interaction().map(|run| match &slot {
            Some(slot) => run.branch(slot.clone()),
            None => run.child(),
        });
        let _owner = run.as_ref().map(|run| run.owner());
        match &run {
            Some(run) => run.continue_execution(cancellation).await?,
            None => {
                if let Some(slot) = &slot {
                    slot.acquire(cancellation).await?;
                }
            }
        }
        let mut invocation = Box::pin(events::with_optional_interaction(
            run.clone(),
            events::with_relay(relay.clone(), invocation),
        ));
        loop {
            enum Next {
                Facts(Box<events::Progress>),
                Finished(Box<Invocation>),
            }
            let next = std::future::poll_fn(|cx| {
                changed.register(cx.waker());
                if let std::task::Poll::Ready(result) = invocation.as_mut().poll(cx) {
                    return std::task::Poll::Ready(Next::Finished(Box::new(result)));
                }
                if let std::task::Poll::Ready(facts) = relay.poll(cx) {
                    return std::task::Poll::Ready(Next::Facts(Box::new(facts)));
                }
                if self.state.nodes[node_id]
                    .executions
                    .values()
                    .any(|execution| {
                        execution.attempts.last().is_some_and(|attempt| {
                            attempt.status == node::status::attempt::Status::Waiting
                        })
                    })
                    && let Some(run) = &run
                    && run.children_blocked()
                {
                    run.suspend();
                }
                std::task::Poll::Pending
            })
            .await;
            match next {
                Next::Finished(result) => {
                    if let Some(slot) = &slot {
                        slot.release();
                    }
                    return Ok(*result);
                }
                Next::Facts(facts) => {
                    let events::Progress {
                        node_id,
                        exec_id,
                        attempt,
                        event,
                        entered,
                        ack,
                    } = *facts;
                    let result = Box::pin(self.accept_progress(
                        &node_id,
                        exec_id,
                        attempt,
                        event,
                        entered,
                        cancellation,
                        Some(base),
                    ))
                    .await;
                    let _ = ack.send(result.clone());
                    result?;
                }
            }
        }
    }
}
