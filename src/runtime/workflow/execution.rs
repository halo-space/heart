//! Workflow execution engine and nested graph invocation.
//!
//! This module owns mutable execution, retry, reply, checkpoint and subgraph
//! progression. The parent module exposes only the Runtime facade and graph
//! contracts.

use super::*;

pub(super) struct Engine {
    pub(super) workflow: CompiledWorkflow,
    pub(super) state: State,
    pub(super) max_concurrency: usize,
    pub(super) checkpoint: Option<CheckpointSave>,
    pub(super) memory: Option<History>,
    pub(super) history_run: Option<Run>,
    pub(super) history_closed: bool,
    // Derived from the parent's earlier Attempt.graph, not separate truth.
    // It supplies inherited scenes when a failed nested Graph Node is retried.
    pub(super) retry_source: Option<Box<graph::State>>,
    // Derived from the saved Agent Step and carried only until the existing
    // execution consumes it. This is transient routing, not saved Runtime state.
    pub(super) recovery_reply: Option<RecoveryReply>,
    // Saved replies are accepted only by a Runtime explicitly reconstructed
    // from Checkpoint, never by an abandoned live Runtime.
    pub(super) accepts_saved_replies: bool,
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    async fn accept_progress(
        &mut self,
        node_id: &str,
        exec_id: i64,
        mut attempt: node::Attempt,
        event: Option<crate::event::Event>,
        entered: bool,
        cancellation: &crate::Cancellation,
        base: Option<&node::Attempt>,
    ) -> Result<(), Error> {
        if entered {
            let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
            if !matches!(
                execution.status,
                node::ExecutionStatus::Pending
                    | node::ExecutionStatus::Running
                    | node::ExecutionStatus::Waiting
            ) {
                return Err(Error::new(
                    "INVALID_STATE",
                    "admission target is no longer active",
                ));
            }
            let current = execution
                .attempts
                .last_mut()
                .filter(|current| current.idx == attempt.idx)
                .ok_or_else(|| Error::new("INVALID_STATE", "admission target Attempt changed"))?;
            execution.status = node::ExecutionStatus::Running;
            current.status = node::AttemptStatus::Running;
            // In-memory handoff only: neither a new Event nor a Checkpoint.
            return Box::pin(self.publish(None, false)).await;
        }
        let delta = matches!(&event, Some(crate::event::Event::Delta { .. }));
        let question = event.as_ref().and_then(|event| match event {
            crate::event::Event::Complete { message, .. } => {
                message.metadata.get("reply_id").and_then(Value::as_i64)
            }
            _ => None,
        });
        let feedback = if delta {
            false
        } else {
            let previous = self
                .state
                .nodes
                .get(node_id)
                .and_then(|node| node.executions.get(&exec_id))
                .and_then(|execution| execution.attempts.last());
            let before = previous.map(attempt_feedback).unwrap_or_default();
            attempt_feedback(&attempt)
                .iter()
                .any(|(id, count)| *count > before.get(id).copied().unwrap_or(0))
        };
        if let Some(base) = base.filter(|_| attempt.agent.is_some()) {
            prepend_agent_observations(&mut attempt, base);
        }
        if let Some(id) = question
            && !attempt_question(&attempt, id)?
        {
            return Err(Error::new(
                "INVALID_STATE",
                "Workflow question has no native waiting Step",
            ));
        }
        if !delta {
            let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
            execution.status = if attempt_waiting(&attempt) {
                node::ExecutionStatus::Waiting
            } else {
                node::ExecutionStatus::Running
            };
            let idx = attempt.idx as usize;
            if idx > execution.attempts.len() {
                return Err(Error::new(
                    "INVALID_STATE",
                    "Workflow progress skips an Attempt",
                ));
            }
            if let Some(previous) = execution.attempts.get(idx) {
                attempt.created_time = previous.created_time;
            }
            if idx == execution.attempts.len() {
                execution.attempts.push(attempt);
            } else {
                execution.attempts[idx] = attempt;
            }
            self.state.recompute_usage();
            self.state.status = if self
                .state
                .nodes
                .values()
                .flat_map(|node| node.executions.values())
                .any(|execution| execution.attempts.last().is_some_and(attempt_waiting))
            {
                graph::Status::Paused
            } else {
                graph::Status::Running
            };
        }
        let chat_id = self.state.chat_id;
        self.state.bind_agent_chat_id(chat_id)?;
        if !delta {
            wait::update(&mut self.state, node_id, exec_id)?;
        }
        if !delta && let Some(run) = events::interaction() {
            let attempt = self.state.nodes[node_id].executions[&exec_id]
                .attempts
                .last()
                .expect("committed Workflow progress");
            commit_feedback(attempt, &run)?;
        }
        if question.is_some() || feedback {
            if let Some(checkpoint) = &self.checkpoint {
                checkpoint(
                    self.state.chat_id,
                    self.snapshot()?,
                    crate::Cancellation::new(),
                )
                .await?;
            }
            if cancellation.is_cancelled() {
                return Err(Error::new(
                    "CANCELLED",
                    "Workflow cancelled while committing feedback",
                ));
            }
        }
        Box::pin(self.publish(event, delta)).await
    }

    async fn publish(&self, event: Option<crate::event::Event>, delta: bool) -> Result<(), Error> {
        if events::has_parent() {
            let attempt = if delta {
                node::Attempt {
                    idx: events::attempt_index(),
                    ..Default::default()
                }
            } else {
                node::Attempt {
                    idx: events::attempt_index(),
                    status: if self.state.status == graph::Status::Paused {
                        node::AttemptStatus::Waiting
                    } else {
                        node::AttemptStatus::Running
                    },
                    usage: self.state.usage.clone(),
                    graph: Some(Box::new(self.state.clone())),
                    ..Default::default()
                }
            };
            events::progress(attempt, event).await
        } else {
            if let Some(event) = event {
                events::emit(event).await;
            }
            Ok(())
        }
    }
    pub fn new(chat_id: i64, workflow: CompiledWorkflow) -> Result<Self, Error> {
        if chat_id <= 0 {
            return Err(Error::new("INVALID_ARGUMENTS", "chat_id must be positive"));
        }
        let state = State::new(chat_id, workflow.definition());
        Ok(Self {
            workflow,
            state,
            max_concurrency: 100,
            checkpoint: None,
            memory: None,
            history_run: None,
            history_closed: false,
            retry_source: None,
            recovery_reply: None,
            accepts_saved_replies: false,
        })
    }

    pub fn from_snapshot(
        chat_id: i64,
        workflow: CompiledWorkflow,
        snapshot: Value,
    ) -> Result<Self, Error> {
        if chat_id <= 0 {
            return Err(Error::new("INVALID_ARGUMENTS", "chat_id must be positive"));
        }
        let state: State = serde_json::from_value(snapshot)
            .map_err(|error| Error::new("INVALID_CHECKPOINT", error.to_string()))?;
        state.check_identity(chat_id)?;
        wait::validate(&state)?;
        if state.definition_id != workflow.definition().definition_id
            || state.version != workflow.definition().version
        {
            return Err(Error::new(
                "CONFLICT",
                "checkpoint definition does not match compiled workflow",
            ));
        }
        let mut runtime = Self {
            workflow,
            state,
            max_concurrency: 100,
            checkpoint: None,
            memory: None,
            history_run: None,
            history_closed: false,
            retry_source: None,
            recovery_reply: None,
            accepts_saved_replies: true,
        };
        runtime.validate_tool_scenes()?;
        if runtime.state.mark_waiting() {
            runtime.state.status = graph::Status::Running;
        }
        runtime.state.recompute_usage();
        Ok(runtime)
    }

    pub fn definition(&self) -> &Definition {
        self.workflow.definition()
    }

    fn validate_tool_scenes(&self) -> Result<(), Error> {
        for (id, node) in &self.state.nodes {
            for execution in node.executions.values() {
                if let Some(child) = self.workflow.graph_node(id) {
                    for attempt in &execution.attempts {
                        if let Some(state) = &attempt.graph {
                            Self::from_snapshot(
                                self.state.chat_id,
                                child.clone(),
                                serde_json::to_value(state).map_err(|_| {
                                    Error::new("INVALID_STATE", "invalid child Tool scene")
                                })?,
                            )?;
                        }
                    }
                } else if execution
                    .attempts
                    .iter()
                    .any(|attempt| !attempt.steps.is_empty())
                    && self.workflow.agent(id).is_none()
                {
                    let executor = self
                        .workflow
                        .executor(id)
                        .ok_or_else(|| Error::new("INVALID_STATE", "ToolNode owner is missing"))?;
                    let input = self.input_for(id, &execution.from)?;
                    for attempt in &execution.attempts {
                        executor.validate_recovery(input.clone(), attempt)?;
                    }
                    executor.validate_recovery(input, &committed_tool_facts(execution))?;
                }
            }
        }
        Ok(())
    }

    pub fn ready_nodes(&self) -> Vec<String> {
        self.state.ready_nodes(self.definition())
    }

    pub fn start_batch(
        &mut self,
        executions: Vec<(String, i64, Vec<ExecutionRef>)>,
    ) -> Result<(), Error> {
        let definition = self.workflow.definition().clone();
        self.state.start_batch(&definition, executions)
    }

    pub fn complete(
        &mut self,
        node_id: &str,
        exec_id: i64,
        messages: Messages,
        to: Vec<String>,
    ) -> Result<(), Error> {
        self.state
            .complete_execution(node_id, exec_id, messages, to)?;
        self.state.recompute_usage();
        Ok(())
    }

    pub fn fail(&mut self, node_id: &str, exec_id: i64, error: Error) -> Result<(), Error> {
        self.state.fail_execution(node_id, exec_id, error)?;
        self.state.recompute_usage();
        Ok(())
    }

    pub fn cancel(&mut self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        self.state.cancel_execution(node_id, exec_id)?;
        self.state.recompute_usage();
        Ok(())
    }

    /// Submits the external response for an active wait. The next `run` call
    /// resumes the same execution and Attempt; it does not allocate a new
    /// `exec_id`.
    pub fn resume(&mut self, node_id: &str, response: Messages) -> Result<(), Error> {
        self.state.resume(node_id, response)
    }

    pub(super) fn reply_saved(
        &mut self,
        response: Messages,
        input: AgentInput,
        cancellation: &crate::Cancellation,
    ) -> Result<(), Error> {
        if !self.accepts_saved_replies {
            return Err(Error::new(
                "NOT_FOUND",
                "no restored Workflow Checkpoint is receiving this reply",
            ));
        }
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "reply submission cancelled"));
        }
        if input.metadata.get("chat_id").and_then(Value::as_i64) != Some(self.state.chat_id) {
            return Err(Error::new(
                "CONFLICT",
                "reply belongs to a different Workflow run",
            ));
        }
        if let Some(initial) = &self.state.initial {
            let saved = messages_to_agent_input(initial).metadata;
            for key in ["tenant_id", "user_id", "agent_id", "session_id"] {
                if saved.contains_key(key) && input.metadata.get(key) != saved.get(key) {
                    return Err(Error::new(
                        "CONFLICT",
                        "reply belongs to a different Workflow identity",
                    ));
                }
            }
        }
        if let Some(previous) = &self.recovery_reply {
            return if previous.response == response {
                Ok(())
            } else {
                Err(Error::new(
                    "CONFLICT",
                    "this question already received a different reply",
                ))
            };
        }
        let reply_id = input
            .metadata
            .get("reply_id")
            .and_then(Value::as_i64)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "reply requires reply_id"))?;
        if reply_duplicate(&self.state, &input)? {
            return Ok(());
        }
        let mut paths = self.state.reply_paths(reply_id)?;
        if paths.len() != 1 {
            return Err(Error::new(
                if paths.is_empty() {
                    "NOT_FOUND"
                } else {
                    "CONFLICT"
                },
                if paths.is_empty() {
                    "reply does not identify a saved waiting question"
                } else {
                    "saved question identity is ambiguous"
                },
            ));
        }
        let path = paths.pop().expect("one validated reply path");
        let scene = agent_scene_for_reply_path(&self.state, &path)?;
        if !crate::runtime::agent::reply::reply_duplicate(scene, &input)?
            && wait::ids(&self.state)?.first() != Some(&reply_id)
        {
            return Err(Error::new(
                "CONFLICT",
                "replies must be submitted in the saved Workflow question order",
            ));
        }
        let Some(_) = crate::runtime::agent::reply::prepare_restore(scene, &input)? else {
            return Err(Error::new(
                "NOT_FOUND",
                "reply does not identify a saved waiting question",
            ));
        };

        // Fully recorded Workflow waits already have their response slot in
        // the current State. Earlier waiting snapshots are routed transiently
        // to the existing active Attempt below.
        if self
            .state
            .pause(&path[0].node_id)?
            .is_some_and(|wait| wait.exec_id == path[0].exec_id)
        {
            return self.state.resume(&path[0].node_id, response);
        }
        self.recovery_reply = Some(RecoveryReply {
            response,
            input,
            path,
        });
        Ok(())
    }

    pub fn advance_step(&mut self) -> bool {
        self.state.advance_step()
    }

    /// Produces the exact graph-state JSON accepted by Checkpoint.
    pub fn snapshot(&self) -> Result<Value, Error> {
        serde_json::to_value(&self.state)
            .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))
    }

    pub(super) fn validate_run_input(&self, initial_messages: &Messages) -> Result<(), Error> {
        self.state.check_identity(self.state.chat_id)?;
        if self
            .state
            .initial
            .as_ref()
            .is_some_and(|initial| initial != initial_messages)
        {
            return Err(Error::new(
                "CONFLICT",
                "workflow initial Messages cannot change after activation",
            ));
        }
        if let Some(chat_id) = messages_to_agent_input(initial_messages)
            .metadata
            .get("chat_id")
            && chat_id.as_i64() != Some(self.state.chat_id)
        {
            return Err(Error::new(
                "CONFLICT",
                "input chat_id does not match this Workflow run",
            ));
        }
        Ok(())
    }

    // Private complete-returning execution engine. The only public run
    // contract is the observation stream; this is not a compatibility API.
    pub(super) async fn execute(
        &mut self,
        initial_messages: Messages,
        next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: crate::Cancellation,
    ) -> Result<Messages, Error> {
        admission::inherit(
            self.max_concurrency,
            Box::pin(self.execute_inner(initial_messages, next_exec_id, cancellation)),
        )
        .await
    }

    async fn execute_inner(
        &mut self,
        initial_messages: Messages,
        next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: crate::Cancellation,
    ) -> Result<Messages, Error> {
        self.validate_run_input(&initial_messages)?;
        // A terminal snapshot is already closed. Re-running it is a read of
        // the committed result and must not create another Chat record
        // or invoke any node again.
        if matches!(
            self.state.status,
            graph::Status::Succeeded | graph::Status::Failed | graph::Status::Cancelled
        ) {
            let result =
                Box::pin(self.run_inner(initial_messages, next_exec_id, cancellation)).await;
            if result.is_ok() {
                for event in terminal_observations(&self.state) {
                    events::emit(event).await;
                }
            } else if result
                .as_ref()
                .is_err_and(|error| error.code == "CANCELLED")
            {
                for event in cancelled_observations(&self.state) {
                    events::emit(event).await;
                }
            }
            return result;
        }
        self.begin_history(&initial_messages, &cancellation).await?;
        let result = Box::pin(self.run_inner(initial_messages, next_exec_id, cancellation)).await;
        if let Err(error) = &result
            && error.code != "WAITING"
        {
            if error.code == "CANCELLED" {
                self.cancel_active();
                self.state.status = graph::Status::Cancelled;
            } else if self.state.status != graph::Status::Failed {
                let active = self
                    .state
                    .nodes
                    .iter()
                    .flat_map(|(id, node)| {
                        node.executions
                            .iter()
                            .filter(|(_, execution)| {
                                matches!(
                                    execution.status,
                                    node::ExecutionStatus::Pending
                                        | node::ExecutionStatus::Running
                                        | node::ExecutionStatus::Waiting
                                )
                            })
                            .map(move |(exec_id, _)| (id.clone(), *exec_id))
                    })
                    .collect::<Vec<_>>();
                for (node_id, exec_id) in active {
                    self.state
                        .fail_execution(&node_id, exec_id, error.clone())?;
                }
                self.state.status = graph::Status::Failed;
            }
            self.state.recompute_usage();
        }
        if let Some(checkpoint) = self.checkpoint.clone() {
            let state = self.snapshot()?;
            checkpoint(self.state.chat_id, state, crate::Cancellation::new()).await?;
        }
        self.close_history(&result).await?;
        result
    }

    async fn begin_history(
        &mut self,
        initial_messages: &Messages,
        cancellation: &crate::Cancellation,
    ) -> Result<(), Error> {
        let Some(history) = self.memory.clone() else {
            return Ok(());
        };
        if self.history_run.is_some() {
            return Ok(());
        }
        let input = messages_to_agent_input(initial_messages);
        let run = if self.state.initial.is_some() {
            history
                .resume_for(input, crate::memory::short::Type::Workflow, cancellation)
                .await?
        } else {
            history
                .begin_for(input, crate::memory::short::Type::Workflow, cancellation)
                .await?
        };
        self.history_run = Some(run);
        Ok(())
    }

    async fn close_history(&mut self, result: &Result<Messages, Error>) -> Result<(), Error> {
        if self.history_closed {
            return Ok(());
        }
        let Some(run) = self.history_run.clone() else {
            return Ok(());
        };
        let status = match result {
            Err(error) if error.code == "WAITING" => return Ok(()),
            Ok(_) => crate::memory::short::chat::Status::Completed,
            Err(error) if error.code == "CANCELLED" => crate::memory::short::chat::Status::Stopped,
            Err(_) => crate::memory::short::chat::Status::Failed,
        };
        run.settle(self.state.usage.clone());
        let graph = serde_json::to_value(self.definition())
            .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))?;
        let state = self.snapshot()?;
        let message = result.as_ref().ok().and_then(|messages| {
            messages
                .as_slice()
                .iter()
                .rev()
                .find(|message| message.role == Role::Assistant)
                .cloned()
        });
        run.close_workflow(status, message, Some(graph), Some(state))
            .await?;
        self.history_closed = true;
        Ok(())
    }

    /// Executes the compiled graph until it reaches END, waiting, failure or
    /// cancellation. The caller's ID source supplies one id for each new
    /// logical activation; retries keep the original id.
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
                            .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))?
                    }
                    Target::Graph { .. } => self.input_for(
                        &node_id,
                        &self.state.nodes[&node_id].executions[&exec_id].from,
                    )?,
                    Target::Node(_) => serde_json::to_value(response)
                        .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))?,
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
                        .status = node::ExecutionStatus::Pending;
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
                                        status: node::AttemptStatus::Cancelled,
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
                            .is_some_and(|attempt| attempt.status == node::AttemptStatus::Waiting)
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
                        && execution.status == node::ExecutionStatus::Cancelled
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

    fn reached_end(&self) -> bool {
        self.state.nodes.values().any(|state| {
            state.executions.values().any(|execution| {
                execution.status == node::ExecutionStatus::Succeeded
                    && execution.to.iter().any(|target| target == graph::END)
            })
        })
    }

    fn end_messages(&self) -> Messages {
        let mut messages = Messages::default();
        for node_id in self.definition().nodes.keys() {
            if let Some(state) = self.state.nodes.get(node_id) {
                for execution in state.executions.values() {
                    if execution.status == node::ExecutionStatus::Succeeded
                        && execution.to.iter().any(|target| target == graph::END)
                    {
                        messages.0.extend(execution.messages.0.clone());
                    }
                }
            }
        }
        messages
    }

    fn outgoing(&self, node_id: &str, messages: &Messages) -> Result<Vec<String>, Error> {
        let mut targets = outgoing(self.definition(), node_id);
        let allowed: BTreeSet<String> = self
            .definition()
            .graph
            .conditional_edges
            .iter()
            .filter(|relation| relation.from == node_id)
            .flat_map(|relation| relation.to.iter().cloned())
            .collect();
        if !allowed.is_empty() {
            let selected = self.workflow.route(node_id, messages)?;
            if selected.is_empty() {
                return Err(Error::new(
                    "INVALID_ROUTE",
                    "conditional route selected no target",
                ));
            }
            if selected.iter().any(|target| !allowed.contains(target)) {
                return Err(Error::new(
                    "INVALID_ROUTE",
                    "conditional route selected an undeclared target",
                ));
            }
            targets.retain(|target| !allowed.contains(target));
            targets.extend(selected);
        }
        let loop_targets: Vec<String> = self
            .definition()
            .graph
            .loop_definitions
            .iter()
            .filter(|relation| relation.from == node_id)
            .filter(|_| self.workflow.loop_enabled(node_id, messages))
            .map(|relation| relation.to.clone())
            .collect();
        targets.extend(loop_targets);
        targets.sort();
        targets.dedup();
        Ok(targets)
    }

    fn first_failure(&self) -> Option<Error> {
        self.state
            .nodes
            .values()
            .flat_map(|state| state.executions.values())
            .find_map(|execution| execution.error.clone())
    }

    fn cancel_active(&mut self) {
        let active: Vec<(String, i64)> = self
            .state
            .nodes
            .iter()
            .flat_map(|(node_id, state)| {
                state
                    .executions
                    .values()
                    .filter(|execution| {
                        matches!(
                            execution.status,
                            node::ExecutionStatus::Pending
                                | node::ExecutionStatus::Running
                                | node::ExecutionStatus::Waiting
                        )
                    })
                    .map(|execution| (node_id.clone(), execution.exec_id))
            })
            .collect();
        for (node_id, exec_id) in active {
            let _ = self.state.cancel_execution(&node_id, exec_id);
        }
    }

    fn source_executions(&self, node_id: &str) -> Result<Vec<ExecutionRef>, Error> {
        self.state
            .activation_sources(self.definition(), node_id)
            .ok_or_else(|| Error::new("INVALID_STATE", "node has no committed activation"))
    }

    fn input_for(&self, node_id: &str, from: &[ExecutionRef]) -> Result<Value, Error> {
        if self.is_failure_activation(node_id, from) {
            if self.definition().nodes[node_id].agent {
                return serde_json::to_value(AgentInput::default())
                    .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()));
            }
            if self.workflow.input_is_messages(node_id) {
                return serde_json::to_value(Messages::default())
                    .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()));
            }
            return Ok(Value::Object(serde_json::Map::new()));
        }
        if self.definition().nodes[node_id].agent {
            return serde_json::to_value(self.agent_input_for(node_id, from)?)
                .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()));
        }
        if !self.definition().nodes[node_id].attrs.input.is_empty() {
            return self.bound_values(node_id, from).map(Value::Object);
        }
        if from.is_empty() {
            if self.workflow.input_is_messages(node_id) {
                return serde_json::to_value(self.state.initial.as_ref().ok_or_else(|| {
                    Error::new("INVALID_STATE", "workflow initial Messages are missing")
                })?)
                .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()));
            }
            let values = self
                .state
                .initial
                .as_ref()
                .ok_or_else(|| {
                    Error::new("INVALID_STATE", "workflow initial Messages are missing")
                })?
                .as_slice()
                .last()
                .map(|message| message.values.clone())
                .unwrap_or_default();
            return Ok(Value::Object(values));
        }
        if self.workflow.input_is_messages(node_id) {
            let mut messages = Messages::default();
            for source in from {
                let execution = self.state.nodes[&source.node_id]
                    .executions
                    .get(&source.exec_id)
                    .ok_or_else(|| Error::new("NOT_FOUND", "source execution missing"))?;
                messages.0.extend(execution.messages.0.clone());
            }
            return serde_json::to_value(messages)
                .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()));
        }
        self.bound_values(node_id, from).map(Value::Object)
    }

    /// Snapshot only the current node's local execution facts for an
    /// invocation. This deliberately does not expose the owning `State` or
    /// graph mutators; the Node/Agent receives a clone through `View`.
    fn view_for(
        &self,
        node_id: &str,
        exec_id: i64,
        input: Value,
        attempt_idx: Option<u32>,
    ) -> Result<View, Error> {
        let execution = self
            .state
            .nodes
            .get(node_id)
            .and_then(|state| state.executions.get(&exec_id));
        let mut view = View {
            node_id: node_id.to_owned(),
            exec_id: Some(exec_id),
            attempt_idx,
            status: execution
                .map(|execution| {
                    serde_json::to_value(&execution.status)
                        .ok()
                        .and_then(|value| value.as_str().map(str::to_owned))
                        .unwrap_or_else(|| "pending".into())
                })
                .unwrap_or_else(|| "pending".into()),
            input,
            ..View::default()
        };
        if let Some(execution) = execution {
            view.result = (!execution.messages.is_empty())
                .then(|| serde_json::to_value(&execution.messages))
                .transpose()
                .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))?;
            view.attempts = execution
                .attempts
                .iter()
                .map(|attempt| {
                    serde_json::to_value(attempt)
                        .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))
                })
                .collect::<Result<_, _>>()?;
            view.error = execution
                .error
                .as_ref()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))?;
            view.waiting = execution
                .waits
                .values()
                .map(|wait| {
                    serde_json::to_value(wait)
                        .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))
                })
                .collect::<Result<_, _>>()?;
            view.usage = execution
                .usage
                .as_ref()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))?;
            if let Some(attempt) = execution.attempts.last() {
                if let Some(scene) = &attempt.agent {
                    view.children.extend(
                        scene
                            .get("nodes")
                            .and_then(Value::as_object)
                            .into_iter()
                            .flat_map(|nodes| {
                                nodes.iter().map(|(child_id, state)| {
                                    serde_json::json!({
                                        "node_id": child_id,
                                        "status": state.get("status"),
                                        "messages": state.get("messages"),
                                        "waits": state.get("waits"),
                                        "usage": state.get("usage"),
                                        "error": state.get("error"),
                                    })
                                })
                            }),
                    );
                }
                if let Some(graph) = &attempt.graph {
                    let nodes = graph
                        .nodes
                        .iter()
                        .map(|(child_id, state)| {
                            serde_json::json!({
                                "node_id": child_id,
                                "status": state.status,
                                "executions": state.executions.len(),
                            })
                        })
                        .collect::<Vec<_>>();
                    view.children.push(serde_json::json!({
                        "type": "workflow",
                        "status": graph.status,
                        "step": graph.step,
                        "nodes": nodes,
                    }));
                }
            }
            for source in &execution.from {
                if let Some(source_execution) = self
                    .state
                    .nodes
                    .get(&source.node_id)
                    .and_then(|state| state.executions.get(&source.exec_id))
                {
                    let key = format!("{}#{}", source.node_id, source.exec_id);
                    view.sources.insert(
                        key,
                        serde_json::json!({
                            "node_id": source.node_id,
                            "exec_id": source.exec_id,
                            "status": source_execution.status,
                            "messages": source_execution.messages,
                        }),
                    );
                }
            }
            let mut bindings = serde_json::Map::new();
            for (target, binding) in &self.definition().nodes[node_id].attrs.input {
                let source = execution
                    .from
                    .iter()
                    .find(|source| source.node_id == binding.from.node_id);
                let value = source
                    .and_then(|source| {
                        self.state
                            .nodes
                            .get(&source.node_id)
                            .and_then(|state| state.executions.get(&source.exec_id))
                    })
                    .and_then(|source| {
                        source
                            .messages
                            .as_slice()
                            .iter()
                            .rev()
                            .find_map(|message| message.values.get(&binding.from.field))
                    })
                    .or_else(|| {
                        (binding.from.node_id == graph::START)
                            .then(|| {
                                self.state
                                    .initial
                                    .as_ref()?
                                    .as_slice()
                                    .iter()
                                    .rev()
                                    .find_map(|message| message.values.get(&binding.from.field))
                            })
                            .flatten()
                    })
                    .cloned()
                    .unwrap_or(Value::Null);
                bindings.insert(
                    target.clone(),
                    serde_json::json!({
                        "node_id": binding.from.node_id,
                        "field": binding.from.field,
                        "value": value,
                    }),
                );
            }
            if !bindings.is_empty() {
                view.sources
                    .insert("attrs.input".into(), Value::Object(bindings));
            }
        }
        Ok(view)
    }

    fn agent_input_for(&self, node_id: &str, from: &[ExecutionRef]) -> Result<AgentInput, Error> {
        let attrs = &self.definition().nodes[node_id].attrs.input;
        if !attrs.is_empty() {
            let values = self.bound_values(node_id, from)?;
            return serde_json::from_value(Value::Object(values))
                .map_err(|error| Error::new("INVALID_ARGUMENTS", error.to_string()));
        }
        let messages = if from.is_empty() {
            self.state.initial.clone().unwrap_or_default()
        } else {
            let mut messages = Messages::default();
            for source in from {
                if let Some(execution) = self.state.nodes[&source.node_id]
                    .executions
                    .get(&source.exec_id)
                {
                    messages.0.extend(execution.messages.0.clone());
                }
            }
            messages
        };
        Ok(messages_to_agent_input(&messages))
    }

    // Only field adaptation is shared with Agent execution. Graph activation,
    // execution selection, state types and Pregel behavior remain separate.
    fn bound_values(
        &self,
        node_id: &str,
        from: &[ExecutionRef],
    ) -> Result<serde_json::Map<String, Value>, Error> {
        let bindings =
            self.definition().nodes[node_id]
                .attrs
                .input
                .iter()
                .map(|(field, binding)| {
                    (
                        field.as_str(),
                        binding.from.node_id.as_str(),
                        binding.from.field.as_str(),
                    )
                });
        crate::components::node::input::fields(bindings, |source_id| {
            if source_id == graph::START {
                return self.state.initial.as_ref().ok_or_else(|| {
                    Error::new("INVALID_STATE", "workflow initial Messages are missing")
                });
            }
            let source = from
                .iter()
                .find(|source| source.node_id == source_id)
                .ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", format!("missing source {source_id}"))
                })?;
            let execution = self
                .state
                .nodes
                .get(source_id)
                .and_then(|state| state.executions.get(&source.exec_id))
                .ok_or_else(|| Error::new("NOT_FOUND", "source execution missing"))?;
            if execution.status != node::ExecutionStatus::Succeeded {
                return Err(Error::new(
                    "INVALID_STATE",
                    "input source execution has not succeeded",
                ));
            }
            Ok(&execution.messages)
        })
    }

    fn is_failure_activation(&self, node_id: &str, from: &[ExecutionRef]) -> bool {
        self.definition()
            .graph
            .failure_relations
            .iter()
            .any(|relation| {
                relation.to.iter().any(|target| target == node_id)
                    && relation.from.len() == from.len()
                    && relation
                        .from
                        .iter()
                        .all(|source| from.iter().any(|item| &item.node_id == source))
                    && from.iter().any(|source| {
                        self.state
                            .nodes
                            .get(&source.node_id)
                            .and_then(|state| state.executions.get(&source.exec_id))
                            .is_some_and(|execution| {
                                execution.status == node::ExecutionStatus::Failed
                                    && (relation.codes.is_empty()
                                        || execution.error.as_ref().is_some_and(|error| {
                                            relation.codes.iter().any(|code| code == &error.code)
                                        }))
                            })
                    })
                    && (relation.on == "node_failed" || relation.on == "join_failed")
            })
    }

    fn output_messages(&self, node_id: &str, value: Value) -> Result<Messages, Error> {
        if self.workflow.output_is_messages(node_id) {
            serde_json::from_value(value)
                .map_err(|error| Error::new("INVALID_OUTPUT", error.to_string()))
        } else {
            Ok(Messages::new([crate::Message::function(value)]))
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
                            node::ExecutionStatus::Running | node::ExecutionStatus::Waiting
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
                                node::ExecutionStatus::Pending
                                    | node::ExecutionStatus::Running
                                    | node::ExecutionStatus::Waiting
                            )
                            && execution.attempts.last().is_some_and(|attempt| {
                                attempt.status == node::AttemptStatus::Waiting
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
                        let mut agent_input: AgentInput = serde_json::from_value(input)
                            .map_err(|error| Error::new("INVALID_STATE", error.to_string()))?;
                        agent_input.payload = reply.input.payload;
                        agent_input.metadata.extend(reply.input.metadata);
                        input = serde_json::to_value(agent_input).map_err(|error| {
                            Error::new("SERIALIZATION_ERROR", error.to_string())
                        })?;
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
                        .is_some_and(|attempt| attempt.status == node::AttemptStatus::Waiting)
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
                        execution
                            .attempts
                            .last()
                            .is_some_and(|attempt| attempt.status == node::AttemptStatus::Waiting)
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

    async fn observe_commit(&self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        let definition = &self.definition().nodes[node_id];
        if definition.agent || definition.workflow.is_some() {
            // Agent calls already forward their real stream, and Graph Node
            // containers must not duplicate the child's observations.
            // Their final committed status still belongs in the parent scene,
            // including while a sibling keeps this graph waiting.
            Box::pin(self.publish(None, false)).await?;
            return Ok(());
        }
        let execution = &self.state.nodes[node_id].executions[&exec_id];
        let messages = if execution.status == node::ExecutionStatus::Succeeded {
            execution
                .messages
                .as_slice()
                .iter()
                .map(events::project_message)
                .collect::<Vec<_>>()
        } else {
            let mut seen = BTreeSet::new();
            execution
                .attempts
                .iter()
                .flat_map(|attempt| attempt.messages.as_slice())
                .filter(|message| {
                    message
                        .metadata
                        .get("call_id")
                        .and_then(Value::as_str)
                        .is_some_and(|call_id| seen.insert(call_id.to_owned()))
                })
                .map(events::project_message)
                .collect::<Vec<_>>()
        };
        for message in messages {
            let mut event = crate::event::Event::Complete {
                message,
                usage: None,
                finish_reason: None,
            };
            crate::runtime::observation::associate(
                &mut event,
                Some(self.state.chat_id),
                Some((node_id, exec_id)),
            );
            // Before presenting a nested node's result, its owning Graph
            // Attempt must receive the already-committed native scene too.
            // This updates memory only, not another Checkpoint write.
            Box::pin(self.publish(Some(event), false)).await?;
        }
        Ok(())
    }

    fn target_for(&self, node_id: &str, exec_id: i64) -> Result<Target, Error> {
        if let Some(workflow) = self.workflow.graph_node(node_id) {
            let execution = self.state.nodes[node_id].executions.get(&exec_id);
            let state = execution
                .and_then(|execution| execution.attempts.last())
                .and_then(|attempt| attempt.graph.clone());
            let history = execution
                .and_then(|execution| {
                    execution
                        .attempts
                        .iter()
                        .rev()
                        .skip(1)
                        .find_map(|attempt| attempt.graph.clone())
                })
                .or_else(|| self.inherited_graph(node_id));
            return Ok(Target::Graph {
                workflow: Arc::new(workflow.clone()),
                state,
                history,
            });
        }
        if let Some(agent) = self.workflow.agent(node_id) {
            let restore = self.state.nodes[node_id]
                .executions
                .get(&exec_id)
                .and_then(|execution| execution.attempts.last())
                .and_then(|attempt| attempt.agent.clone());
            return Ok(Target::Agent(agent, restore));
        }
        Ok(Target::Node(self.workflow.executor(node_id).ok_or_else(
            || Error::new("UNSUPPORTED", "workflow node has no executor"),
        )?))
    }

    fn inherited_graph(&self, node_id: &str) -> Option<Box<graph::State>> {
        self.retry_source
            .as_ref()?
            .nodes
            .get(node_id)?
            .executions
            .values()
            .max_by_key(|execution| (execution.idx, execution.exec_id))?
            .attempts
            .last()?
            .graph
            .clone()
    }

    fn replace_attempts(
        &mut self,
        node_id: &str,
        exec_id: i64,
        attempts: Vec<node::Attempt>,
    ) -> Result<(), Error> {
        let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
        if attempts.is_empty() || attempts[0].idx != 0 {
            return Err(Error::new("INVALID_STATE", "invalid attempt history"));
        }
        execution.attempts = attempts;
        execution.usage = execution.attempts.iter().fold(None, |total, attempt| {
            crate::components::model::token::merge(total, attempt.usage.clone())
        });
        Ok(())
    }

    // A Tool lifecycle wait has no fabricated human question or response.
    fn wait_call(&mut self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        self.state.node_mut(node_id)?.execution_mut(exec_id)?.status =
            node::ExecutionStatus::Waiting;
        Ok(())
    }

    fn replace_resumed_attempts(
        &mut self,
        node_id: &str,
        exec_id: i64,
        mut attempts: Vec<node::Attempt>,
        base: &node::Attempt,
    ) -> Result<(), Error> {
        let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
        // Native progress may already have appended a retry. Replace only
        // the suffix that this continuation owns, using its original base,
        // not the now-advanced last Attempt. Earlier history stays immutable.
        let previous = base.idx as usize;
        if execution
            .attempts
            .get(previous)
            .is_none_or(|attempt| attempt.idx != base.idx)
            || attempts.is_empty()
            || attempts
                .iter()
                .enumerate()
                .any(|(offset, attempt)| attempt.idx as usize != previous + offset)
        {
            return Err(Error::new(
                "INVALID_STATE",
                "invalid resumed attempt history",
            ));
        }
        let original = base;
        attempts[0].created_time = original.created_time;
        if original.agent.is_some() {
            let cumulative = attempts[0]
                .agent
                .as_ref()
                .and_then(|scene| scene.get("usage"))
                .is_some_and(|usage| {
                    serde_json::from_value::<Option<crate::model::token::Usage>>(usage.clone())
                        .is_ok()
                });
            if !cumulative {
                // Early validation/preparation errors may produce no new
                // scene. Custom Agents can also report only segment usage.
                // Neither case erases consumption from the saved Attempt.
                attempts[0].usage = crate::components::model::token::merge(
                    original.usage.clone(),
                    attempts[0].usage.take(),
                );
            }
            if attempts[0].agent.is_none() {
                attempts[0].agent = original.agent.clone();
            }
            // The Agent scene is canonical for recovery. Keep already
            // observed process facts too, rather than replacing the current
            // Workflow Attempt with only the resumed stream segment.
            prepend_agent_observations(&mut attempts[0], original);
        }
        execution.attempts.truncate(previous);
        execution.attempts.extend(attempts);
        execution.usage = execution.attempts.iter().fold(None, |total, attempt| {
            crate::components::model::token::merge(total, attempt.usage.clone())
        });
        Ok(())
    }

    fn attach_graph(
        &mut self,
        node_id: &str,
        exec_id: i64,
        graph: Box<graph::State>,
    ) -> Result<(), Error> {
        let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
        let attempt = execution
            .attempts
            .last_mut()
            .ok_or_else(|| Error::new("INVALID_STATE", "execution has no attempt"))?;
        attempt.graph = Some(graph);
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct Invocation {
    pub(crate) attempts: Vec<node::Attempt>,
    pub(crate) outcome: Outcome,
    pub(crate) graph: Option<Box<graph::State>>,
}

fn attempt_waiting(attempt: &node::Attempt) -> bool {
    attempt
        .steps
        .iter()
        .any(|step| step.stage.is_some() && step.status == "waiting")
        || attempt
            .agent
            .as_ref()
            .and_then(|scene| scene["waits"].as_array())
            .is_some_and(|waits| !waits.is_empty())
        || attempt.graph.as_ref().is_some_and(|state| {
            !state.waits.is_empty()
                || !state.pauses().is_empty()
                || state
                    .nodes
                    .values()
                    .flat_map(|node| node.executions.values())
                    .any(|execution| execution.attempts.last().is_some_and(attempt_waiting))
        })
}

fn attempt_feedback(attempt: &node::Attempt) -> std::collections::BTreeMap<String, usize> {
    let mut progress = attempt
        .agent
        .as_ref()
        .map(crate::runtime::agent::wait::input_progress)
        .unwrap_or_default();
    if let Some(graph) = &attempt.graph {
        for execution in graph
            .nodes
            .values()
            .flat_map(|node| node.executions.values())
        {
            if execution.idx == graph.step
                && let Some(attempt) = execution.attempts.last()
            {
                progress.extend(attempt_feedback(attempt));
            }
        }
    }
    progress
}

fn commit_feedback(
    attempt: &node::Attempt,
    run: &crate::runtime::agent::reply::Run,
) -> Result<(), Error> {
    if let Some(scene) = &attempt.agent {
        run.commit_scene_feedback(scene)?;
    }
    if let Some(graph) = &attempt.graph {
        for execution in graph
            .nodes
            .values()
            .flat_map(|node| node.executions.values())
        {
            if execution.idx == graph.step
                && let Some(attempt) = execution.attempts.last()
            {
                commit_feedback(attempt, run)?;
            }
        }
    }
    Ok(())
}

fn prepend_agent_observations(attempt: &mut node::Attempt, base: &node::Attempt) {
    let mut messages = base.messages.clone();
    messages.0.append(&mut attempt.messages.0);
    attempt.messages = messages;
    let mut steps = base.steps.clone();
    for (index, step) in attempt.steps.iter_mut().enumerate() {
        step.id = format!("complete_{}", base.steps.len() + index + 1);
    }
    steps.append(&mut attempt.steps);
    attempt.steps = steps;
}

// A resolved contribution remains a fact even after its locator leaves the
// pending queue or its containing graph finishes. It is not a new submission.
/// Search the complete Workflow snapshot for a reply already accepted by any
/// embedded Agent scene. A duplicate is harmless and must not wake a second
/// waiting owner.
fn reply_duplicate(state: &graph::State, input: &AgentInput) -> Result<bool, Error> {
    let mut duplicate = false;
    for node in state.nodes.values() {
        for execution in node.executions.values() {
            if let Some(attempt) = execution.attempts.last() {
                if let Some(scene) = &attempt.agent {
                    duplicate |= crate::runtime::agent::reply::reply_duplicate(scene, input)?;
                }
                if let Some(child) = &attempt.graph {
                    duplicate |= reply_duplicate(child, input)?;
                }
            }
        }
    }
    Ok(duplicate)
}

fn agent_scene_for_reply_path<'a>(
    state: &'a graph::State,
    path: &[graph::ReplyExecution],
) -> Result<&'a Value, Error> {
    let Some((first, rest)) = path.split_first() else {
        return Err(Error::new("INVALID_STATE", "saved reply path is empty"));
    };
    let execution = state
        .nodes
        .get(&first.node_id)
        .and_then(|node| node.executions.get(&first.exec_id))
        .filter(|execution| execution.idx == state.step)
        .ok_or_else(|| Error::new("INVALID_STATE", "saved reply execution is stale"))?;
    let attempt = execution
        .attempts
        .last()
        .filter(|attempt| attempt.idx == first.idx)
        .ok_or_else(|| Error::new("INVALID_STATE", "saved reply Attempt is stale"))?;
    if rest.is_empty() {
        return attempt
            .agent
            .as_ref()
            .ok_or_else(|| Error::new("INVALID_STATE", "saved reply has no Agent scene"));
    }
    agent_scene_for_reply_path(
        attempt
            .graph
            .as_deref()
            .ok_or_else(|| Error::new("INVALID_STATE", "saved reply Graph scene is missing"))?,
        rest,
    )
}

pub(crate) fn attempt_question(attempt: &node::Attempt, id: i64) -> Result<bool, Error> {
    if let Some(scene) = &attempt.agent
        && scene["waits"]
            .as_array()
            .is_some_and(|waits| !waits.is_empty())
        && crate::runtime::agent::wait::question(scene, id)?
    {
        return Ok(true);
    }
    if let Some(state) = &attempt.graph {
        for execution in state
            .nodes
            .values()
            .flat_map(|node| node.executions.values())
        {
            if matches!(
                execution.status,
                node::ExecutionStatus::Running | node::ExecutionStatus::Waiting
            ) && let Some(attempt) = execution.attempts.last()
                && attempt_question(attempt, id)?
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[derive(Debug)]
pub(crate) enum Outcome {
    Complete(Value),
    Wait(String),
    Waiting,
    Failed(Error),
    Cancelled,
}

pub(crate) enum Target {
    Node(Arc<dyn graph::ErasedNode + 'static>),
    Agent(Arc<dyn graph::ErasedAgent + 'static>, Option<Value>),
    Graph {
        workflow: Arc<CompiledWorkflow>,
        state: Option<Box<graph::State>>,
        history: Option<Box<graph::State>>,
    },
}

// Reconstruct successful sub-call facts from the entire logical execution,
// not just its latest Attempt: an in-flight retry may have no messages yet.
// This is private ToolNode recovery input, never a partial normal Node output.
fn committed_tool_facts(execution: &node::Execution) -> node::Attempt {
    let mut committed = std::collections::BTreeMap::new();
    let mut steps = std::collections::BTreeMap::new();
    for step in execution.attempts.iter().flat_map(|attempt| &attempt.steps) {
        steps.insert(step.id.clone(), step.clone());
    }
    for message in execution
        .attempts
        .iter()
        .flat_map(|attempt| attempt.messages.as_slice())
    {
        if let Some(call_id) = message.metadata.get("call_id").and_then(Value::as_str) {
            committed.insert(call_id, message);
        }
    }
    node::Attempt {
        messages: Messages::new(committed.into_values().cloned()),
        steps: steps.into_values().collect(),
        ..Default::default()
    }
}

// Replay only actual saved cancellation Completes, never synthesize one from
// Delta or turn the cancelled execution into a successful downstream result.
fn terminal_observations(state: &graph::State) -> Vec<crate::event::Event> {
    let mut events = Vec::new();
    for (node_id, node) in &state.nodes {
        for execution in node.executions.values() {
            if execution.status != node::ExecutionStatus::Succeeded
                || !execution.to.iter().any(|to| to == graph::END)
            {
                continue;
            }
            if let Some(child) = execution
                .attempts
                .last()
                .and_then(|attempt| attempt.graph.as_ref())
            {
                events.extend(terminal_observations(child));
            } else {
                for message in execution.messages.as_slice() {
                    let mut event = crate::event::Event::Complete {
                        message: events::project_message(message),
                        usage: message.usage.clone(),
                        finish_reason: None,
                    };
                    crate::runtime::observation::associate(
                        &mut event,
                        Some(state.chat_id),
                        Some((node_id, execution.exec_id)),
                    );
                    events.push(event);
                }
            }
        }
    }
    events
}

fn cancelled_observations(state: &graph::State) -> Vec<crate::event::Event> {
    let mut events = Vec::new();
    for (node_id, node) in &state.nodes {
        for execution in node.executions.values() {
            for attempt in &execution.attempts {
                if let Some(child) = &attempt.graph {
                    events.extend(cancelled_observations(child));
                }
                for (step, message) in attempt.steps.iter().zip(attempt.messages.as_slice()) {
                    if let Some(call) = step.attempts.last()
                        && call.finish_reason.as_deref() == Some("cancelled")
                        && message.role == Role::Assistant
                    {
                        let mut event = crate::event::Event::Complete {
                            message: events::project_message(message),
                            usage: call.usage.clone(),
                            finish_reason: call.finish_reason.clone(),
                        };
                        crate::runtime::observation::associate(
                            &mut event,
                            Some(state.chat_id),
                            Some((node_id, execution.exec_id)),
                        );
                        events.push(event);
                    }
                }
            }
        }
    }
    events
}

// The final argument is the already committed ToolNode context restored from
// Checkpoint; keeping it alongside the retry inputs makes recovery explicit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn invoke_with_retry(
    target: Target,
    input: Value,
    view: View,
    cancellation: crate::Cancellation,
    retry: graph::RetryPolicy,
    next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
    chat_id: i64,
    attempt_start: u32,
    committed: node::Attempt,
    recovery_reply: Option<RecoveryReply>,
) -> Pin<Box<dyn Future<Output = Invocation> + Send + 'static>> {
    Box::pin(async move {
        let mut attempts: Vec<node::Attempt> = Vec::new();
        if !matches!(&target, Target::Graph { .. })
            && let Err(error) = events::entered(attempt_start).await
        {
            return Invocation {
                attempts: vec![node::Attempt {
                    idx: attempt_start,
                    status: node::AttemptStatus::Failed,
                    error: Some(error.clone()),
                    ..Default::default()
                }],
                outcome: Outcome::Failed(error),
                graph: None,
            };
        }
        // ToolNode-only committed call facts. They remain private to this one
        // logical execution and are restored from all saved Attempts on recovery;
        // ordinary Nodes keep the unchanged single-call path.
        let mut committed_tool_facts = committed;
        let mut graph_retry_state = match &target {
            Target::Graph { state, .. } => state.clone(),
            _ => None,
        };
        let mut graph_retry_history = match &target {
            Target::Graph { history, .. } => history.clone(),
            _ => None,
        };
        let retry_policy = retry.clone();
        let mut recovery_reply = recovery_reply;
        // A resumed wait continues the same Attempt. Its existing idx consumes
        // the original execution's retry budget; recovery never resets it.
        for offset in 0..=retry.max_retries.saturating_sub(attempt_start) {
            let idx = attempt_start.saturating_add(offset);
            let mut current_view = view.clone();
            current_view.attempt_idx = Some(idx);
            current_view.status = "running".into();
            if !attempts.is_empty() {
                current_view.attempts = attempts
                    .iter()
                    .filter_map(|attempt| serde_json::to_value(attempt).ok())
                    .collect();
            }
            if let Some(previous) = attempts.last() {
                current_view.error = previous
                    .error
                    .as_ref()
                    .and_then(|error| serde_json::to_value(error).ok());
            }
            if current_view.result.is_none() && !committed_tool_facts.messages.is_empty() {
                current_view.result = serde_json::to_value(&committed_tool_facts.messages).ok();
            }
            let mut attempt = node::Attempt {
                idx,
                status: node::AttemptStatus::Running,
                ..Default::default()
            };
            if cancellation.is_cancelled() {
                attempt.status = node::AttemptStatus::Cancelled;
                attempts.push(attempt);
                return Invocation {
                    attempts,
                    outcome: Outcome::Cancelled,
                    graph: None,
                };
            }
            let call_result = match &target {
                Target::Node(executor) => {
                    if let Some(future) = executor.call_with_recovery(
                        input.clone(),
                        cancellation.clone(),
                        node::Attempt {
                            idx,
                            ..committed_tool_facts.clone()
                        },
                    ) {
                        match events::with_attempt(idx, future).await {
                            Ok((Signal::Complete(value), steps)) => {
                                attempt.steps = steps;
                                if let Ok(messages) =
                                    serde_json::from_value::<Messages>(value.clone())
                                {
                                    attempt.messages = messages;
                                }
                                Ok(Signal::Complete(value))
                            }
                            Ok((Signal::Wait(wait), steps)) => {
                                attempt.steps = steps;
                                Ok(Signal::Wait(wait))
                            }
                            Err(failure) => {
                                committed_tool_facts = node::Attempt {
                                    messages: failure.completed.clone(),
                                    steps: failure.steps.clone(),
                                    ..Default::default()
                                };
                                attempt.messages = failure.completed;
                                attempt.steps = failure.steps;
                                if failure.error.code == "WAITING" {
                                    attempt.status = node::AttemptStatus::Waiting;
                                    attempts.push(attempt);
                                    return Invocation {
                                        attempts,
                                        outcome: Outcome::Waiting,
                                        graph: None,
                                    };
                                }
                                Err(failure.error)
                            }
                        }
                    } else {
                        executor
                            .call_with_view(
                                input.clone(),
                                cancellation.clone(),
                                current_view.clone(),
                            )
                            .await
                    }
                }
                Target::Agent(executor, restore) => {
                    let mut agent_input: AgentInput = match serde_json::from_value(input.clone()) {
                        Ok(input) => input,
                        Err(error) => {
                            attempt.status = node::AttemptStatus::Failed;
                            let error = Error::new("INVALID_ARGUMENTS", error.to_string());
                            attempt.error = Some(error.clone());
                            attempts.push(attempt);
                            return Invocation {
                                attempts,
                                outcome: Outcome::Failed(error),
                                graph: None,
                            };
                        }
                    };
                    if agent_input
                        .metadata
                        .get("chat_id")
                        .is_some_and(|id| id.as_i64() != Some(chat_id))
                    {
                        let error = Error::new(
                            "CONFLICT",
                            "Agent input chat_id does not match its Workflow",
                        );
                        attempt.status = node::AttemptStatus::Failed;
                        attempt.error = Some(error.clone());
                        attempts.push(attempt);
                        return Invocation {
                            attempts,
                            outcome: Outcome::Failed(error),
                            graph: None,
                        };
                    }
                    agent_input
                        .metadata
                        .insert("chat_id".into(), Value::from(chat_id));
                    if let Some(reply) = recovery_reply.take() {
                        if !reply.path.is_empty() {
                            let error = Error::new(
                                "INVALID_STATE",
                                "saved reply was not routed through its nested Graph",
                            );
                            attempt.status = node::AttemptStatus::Failed;
                            attempt.error = Some(error.clone());
                            attempts.push(attempt);
                            return Invocation {
                                attempts,
                                outcome: Outcome::Failed(error),
                                graph: None,
                            };
                        }
                        agent_input.payload = reply.input.payload;
                        agent_input.metadata.extend(reply.input.metadata);
                    }
                    let run = events::with_attempt(
                        idx,
                        executor.call(
                            agent_input,
                            cancellation.clone(),
                            restore.clone(),
                            current_view.clone(),
                        ),
                    )
                    .await;
                    attempt.steps = run.steps;
                    attempt.messages = run.messages;
                    attempt.usage = run.usage;
                    let agent_wait_id = run
                        .scene
                        .as_ref()
                        .and_then(|scene| scene.get("wait_id"))
                        .and_then(Value::as_str)
                        .filter(|wait_id| !wait_id.trim().is_empty())
                        .map(str::to_owned);
                    attempt.agent = run.scene.map(|mut scene| {
                        // An embedded Agent does not own a top-level checkpoint,
                        // so the Workflow supplies the single run identity.
                        scene["chat_id"] = Value::from(chat_id);
                        scene
                    });
                    match run.result {
                        Ok(messages) => serde_json::to_value(messages)
                            .map(Signal::Complete)
                            .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string())),
                        // Agent implementations use the shared Error contract for
                        // an external wait.  Translate it at the Workflow
                        // boundary into the same Signal used by ordinary Nodes;
                        // it must never consume retry budget or become a failure.
                        Err(error) if error.code == "WAITING" => {
                            let wait_id = agent_wait_id
                                .as_deref()
                                .or(error.details.as_deref())
                                .filter(|wait_id| !wait_id.trim().is_empty())
                                .unwrap_or("agent")
                                .to_owned();
                            Ok(Signal::Wait(crate::components::node::Wait { wait_id }))
                        }
                        Err(error) => Err(error),
                    }
                }
                Target::Graph { workflow, .. } => {
                    let nested_reply = recovery_reply.take();
                    match events::with_attempt(
                        idx,
                        run_subgraph(
                            Arc::clone(workflow),
                            input.clone(),
                            cancellation.clone(),
                            Arc::clone(&next_exec_id),
                            chat_id,
                            SubgraphScenes {
                                snapshot: graph_retry_state.clone(),
                                history: graph_retry_history.clone(),
                                reply: nested_reply,
                            },
                        ),
                    )
                    .await
                    {
                        Ok((value, state)) => {
                            attempt.status = node::AttemptStatus::Succeeded;
                            attempt.usage = state.usage.clone();
                            attempts.push(attempt);
                            return Invocation {
                                attempts,
                                outcome: Outcome::Complete(value),
                                graph: Some(Box::new(state)),
                            };
                        }
                        Err(failure) => {
                            if failure.error.code == "WAITING" {
                                let Some(state) = failure.state else {
                                    attempt.status = node::AttemptStatus::Failed;
                                    let error = Error::new(
                                        "INVALID_STATE",
                                        "waiting child workflow did not return its state",
                                    );
                                    attempt.error = Some(error.clone());
                                    attempts.push(attempt);
                                    return Invocation {
                                        attempts,
                                        outcome: Outcome::Failed(error),
                                        graph: None,
                                    };
                                };
                                let pauses = state.pauses();
                                if pauses.is_empty()
                                    && state
                                        .nodes
                                        .values()
                                        .flat_map(|node| node.executions.values())
                                        .any(|execution| {
                                            execution.attempts.last().is_some_and(attempt_waiting)
                                        })
                                {
                                    attempt.status = node::AttemptStatus::Waiting;
                                    attempt.graph = Some(Box::new(state));
                                    attempts.push(attempt);
                                    return Invocation {
                                        attempts,
                                        outcome: Outcome::Waiting,
                                        graph: None,
                                    };
                                }
                                let Some(wait_id) =
                                    (pauses.len() == 1).then(|| pauses[0].wait_id.clone())
                                else {
                                    attempt.status = node::AttemptStatus::Failed;
                                    let error = Error::new(
                                        "CONFLICT",
                                        "waiting child workflow must have exactly one active wait",
                                    );
                                    attempt.error = Some(error.clone());
                                    attempts.push(attempt);
                                    return Invocation {
                                        attempts,
                                        outcome: Outcome::Failed(error),
                                        graph: None,
                                    };
                                };
                                attempt.status = node::AttemptStatus::Waiting;
                                attempt.usage = state.usage.clone();
                                attempt.graph = Some(Box::new(state));
                                attempts.push(attempt);
                                return Invocation {
                                    attempts,
                                    outcome: Outcome::Wait(wait_id),
                                    graph: None,
                                };
                            }
                            if let Some(state) = failure.state {
                                if retry_policy.should_retry(idx, &failure.error) {
                                    // Only a real outer retry inherits successes.
                                    // Checkpoint continuation still uses the exact
                                    // original current-Attempt scene above.
                                    graph_retry_state =
                                        Some(Box::new(retry_graph_state(workflow, &state)));
                                    graph_retry_history = Some(Box::new(state.clone()));
                                }
                                attempt.usage = state.usage.clone();
                                attempt.graph = Some(Box::new(state));
                            }
                            Err(failure.error)
                        }
                    }
                }
            };
            match call_result {
                Ok(crate::node::Signal::Complete(value)) => {
                    attempt.status = node::AttemptStatus::Succeeded;
                    attempts.push(attempt);
                    return Invocation {
                        attempts,
                        outcome: Outcome::Complete(value),
                        graph: None,
                    };
                }
                Ok(crate::node::Signal::Wait(wait)) => {
                    attempt.status = node::AttemptStatus::Waiting;
                    attempts.push(attempt);
                    return Invocation {
                        attempts,
                        outcome: Outcome::Wait(wait.wait_id),
                        graph: None,
                    };
                }
                Err(error) => {
                    if error.code == "CANCELLED" {
                        attempt.status = node::AttemptStatus::Cancelled;
                        attempts.push(attempt);
                        return Invocation {
                            attempts,
                            outcome: Outcome::Cancelled,
                            graph: None,
                        };
                    }
                    attempt.status = node::AttemptStatus::Failed;
                    attempt.error = Some(error.clone());
                    if events::has_parent()
                        && let Err(owner_error) = events::progress(attempt.clone(), None).await
                    {
                        attempts.push(attempt);
                        return Invocation {
                            attempts,
                            outcome: Outcome::Failed(owner_error),
                            graph: None,
                        };
                    }
                    attempts.push(attempt);
                    if !retry_policy.should_retry(idx, &error) {
                        return Invocation {
                            attempts,
                            outcome: Outcome::Failed(error),
                            graph: None,
                        };
                    }
                    if let Err(error) =
                        crate::core::middleware::wait(&retry_policy, idx, &cancellation).await
                    {
                        // Waiting does not constitute a new actual Attempt. Keep
                        // the original failed Attempt and report run cancellation
                        // or a timer setup error separately.
                        return Invocation {
                            attempts,
                            outcome: if error.code == "CANCELLED" {
                                Outcome::Cancelled
                            } else {
                                Outcome::Failed(error)
                            },
                            graph: None,
                        };
                    }
                }
            }
        }
        unreachable!()
    })
}

pub(crate) fn messages_to_agent_input(messages: &Messages) -> AgentInput {
    let message = messages
        .as_slice()
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .or_else(|| messages.as_slice().last());
    let Some(message) = message else {
        return AgentInput::default();
    };
    let payload = message
        .content
        .iter()
        .map(|part| AgentPayload {
            r#type: part.r#type.clone(),
            data: serde_json::Value::Object(part.data.clone()),
            mime_type: part
                .data
                .get("mime_type")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
        .collect();
    AgentInput {
        metadata: message.metadata.clone(),
        payload,
    }
}

struct SubgraphFailure {
    error: Error,
    state: Option<graph::State>,
}

// Derive the next outer Attempt's child scene from existing graph facts. The
// failed Attempt keeps its full unmodified scene; only complete, unaffected
// successful child executions are inherited into this new scene.
fn retry_graph_state(workflow: &CompiledWorkflow, previous: &graph::State) -> graph::State {
    let definition = workflow.definition();
    let mut affected = previous
        .nodes
        .iter()
        .filter(|(_, state)| {
            state
                .executions
                .values()
                .any(|execution| execution.status == node::ExecutionStatus::Failed)
        })
        .map(|(node_id, _)| node_id.clone())
        .collect::<BTreeSet<_>>();
    let mut pending = affected.iter().cloned().collect::<Vec<_>>();
    while let Some(node_id) = pending.pop() {
        let mut downstream = outgoing(definition, &node_id);
        downstream.extend(
            definition
                .graph
                .conditional_edges
                .iter()
                .filter(|relation| relation.from == node_id)
                .flat_map(|relation| relation.to.iter().cloned()),
        );
        downstream.extend(
            definition
                .graph
                .loop_definitions
                .iter()
                .filter(|relation| relation.from == node_id)
                .map(|relation| relation.to.clone()),
        );
        downstream.extend(
            definition
                .graph
                .failure_relations
                .iter()
                .filter(|relation| relation.from.contains(&node_id))
                .flat_map(|relation| relation.to.iter().cloned()),
        );
        for to in downstream {
            if definition.nodes.contains_key(&to) && affected.insert(to.clone()) {
                pending.push(to);
            }
        }
    }
    let mut state = graph::State::new(previous.chat_id, definition);
    state.initial = previous.initial.clone();
    state.step = previous
        .nodes
        .iter()
        .filter(|(node_id, _)| affected.contains(*node_id))
        .flat_map(|(_, node)| node.executions.values().map(|execution| execution.idx))
        .min()
        .unwrap_or(previous.step);
    for (node_id, node) in &previous.nodes {
        if affected.contains(node_id) {
            continue;
        }
        let executions = node
            .executions
            .iter()
            .filter(|(_, execution)| execution.status == node::ExecutionStatus::Succeeded)
            .map(|(exec_id, execution)| (*exec_id, execution.clone()))
            .collect::<std::collections::BTreeMap<_, _>>();
        if !executions.is_empty() {
            state.nodes.insert(
                node_id.clone(),
                node::State {
                    status: node::Status::Succeeded,
                    executions,
                },
            );
        }
    }
    state.recompute_usage();
    state
}

type SubgraphFuture = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = Result<(Value, graph::State), SubgraphFailure>>
            + Send
            + 'static,
    >,
>;

struct SubgraphScenes {
    snapshot: Option<Box<graph::State>>,
    history: Option<Box<graph::State>>,
    reply: Option<RecoveryReply>,
}

fn run_subgraph(
    workflow: Arc<CompiledWorkflow>,
    input: Value,
    cancellation: crate::Cancellation,
    next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
    chat_id: i64,
    scenes: SubgraphScenes,
) -> SubgraphFuture {
    Box::pin(async move {
        let SubgraphScenes {
            snapshot,
            history,
            reply,
        } = scenes;
        let messages: Messages =
            serde_json::from_value(input).map_err(|error| SubgraphFailure {
                error: Error::new("INVALID_ARGUMENTS", error.to_string()),
                state: None,
            })?;
        // A nested Graph Node can have a new execution while still reusing
        // its unchanged-input internal successes from the parent's history.
        let stable_input = snapshot
            .as_ref()
            .and_then(|scene| scene.initial.as_ref())
            .unwrap_or(&messages);
        let history = history.filter(|previous| {
            previous.chat_id == chat_id
                && previous.definition_id == workflow.definition().definition_id
                && previous.version == workflow.definition().version
                && previous.initial.as_ref() == Some(stable_input)
        });
        let snapshot = snapshot.or_else(|| {
            history
                .as_ref()
                .map(|previous| Box::new(retry_graph_state(&workflow, previous)))
        });
        let mut runtime = match snapshot {
            Some(snapshot) => Engine::from_snapshot(
                chat_id,
                (*workflow).clone(),
                serde_json::to_value(*snapshot).map_err(|error| SubgraphFailure {
                    error: Error::new("SERIALIZATION_ERROR", error.to_string()),
                    state: None,
                })?,
            )
            .map_err(|error| SubgraphFailure { error, state: None })?,
            None => Engine::new(chat_id, (*workflow).clone())
                .map_err(|error| SubgraphFailure { error, state: None })?,
        };
        runtime.retry_source = history;
        runtime.recovery_reply = reply;
        let run_input = runtime.state.initial.clone().unwrap_or(messages);
        let output = runtime
            .execute(run_input, next_exec_id, cancellation)
            .await
            .map_err(|error| SubgraphFailure {
                error,
                state: Some(runtime.state.clone()),
            })?;
        let state = runtime.state.clone();
        let value = serde_json::to_value(output).map_err(|error| SubgraphFailure {
            error: Error::new("SERIALIZATION_ERROR", error.to_string()),
            state: Some(state.clone()),
        })?;
        Ok((value, state))
    })
}

fn outgoing(definition: &Definition, node_id: &str) -> Vec<String> {
    let mut result: Vec<String> = definition
        .graph
        .ordinary_edges
        .iter()
        .filter(|edge| edge.from.node() == Some(node_id))
        .map(|edge| match &edge.to {
            Endpoint::Node(id) => id.clone(),
            Endpoint::End => graph::END.into(),
            Endpoint::Start => graph::START.into(),
        })
        .collect();
    result.extend(
        definition
            .graph
            .fan_out_definitions
            .iter()
            .filter(|relation| relation.from == node_id)
            .map(|relation| relation.to.clone()),
    );
    result.extend(
        definition
            .graph
            .join_edges
            .iter()
            .filter(|relation| relation.from.iter().any(|source| source == node_id))
            .map(|relation| relation.to.clone()),
    );
    result.sort();
    result.dedup();
    result
}
