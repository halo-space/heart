//! Private task dispatch for ReAct. The graph remains runtime::agent state;
//! tasks reuse the Agent's model and configured Tool/MCP/Skill loop rather than
//! looking up names.
use super::*;
use crate::components::model::content::part::object as part_data;
use crate::runtime::agent::{graph::Start, node::Status};

struct Task<M: Chat> {
    node_id: String,
    exec_id: i64,
    state: ReActState<M>,
    messages: Messages,
    previous_usage: Option<crate::model::token::Usage>,
    slot: crate::runtime::agent::reply::Slot,
}

fn task_messages(mut messages: Messages, input: Message) -> Messages {
    let mut instruction = Message::new(Role::System);
    instruction.content.push(Part {
        r#type: "text".into(),
        data: part_data(json!({"value": "Complete only the current planned task objective using the original query and its committed source results. Choose tools and arguments yourself when needed. Return this task's result, not the final answer for unrelated tasks."})),
    });
    if let Some(system) = messages
        .0
        .first_mut()
        .filter(|message| message.role == Role::System)
    {
        system.content.extend(instruction.content);
    } else {
        messages.0.insert(0, instruction);
    }
    messages.push(input);
    messages
}

pub(super) struct Active {
    run: Run,
    slot: crate::runtime::agent::reply::Slot,
}
impl Active {
    pub(super) fn slot_held(&self) -> bool {
        self.slot.held()
    }
}

pub(super) struct TaskProgress<M: Chat> {
    task: Box<Task<M>>,
    result: Option<Result<Event, Error>>,
}

async fn advance<M: Chat + 'static>(mut task: Box<Task<M>>) -> TaskProgress<M> {
    let (result, state) = next_react_progress(task.state)
        .await
        .expect("live plan task");
    task.state = state;
    TaskProgress { task, result }
}

pub(super) fn fill_ready<M: Chat + 'static>(state: &mut ReActState<M>) -> Result<(), Error> {
    if state.cancellation.is_cancelled() || state.dispatch_failed {
        return Ok(());
    }
    let graph = state
        .graph
        .as_mut()
        .ok_or_else(|| Error::new("INVALID_STATE", "plan graph is not initialized"))?;
    for (id, active) in &state.task_runs {
        let node = graph.state_mut(id)?;
        node.status = if active.slot.held() {
            Status::Running
        } else {
            Status::Waiting
        };
        let status = node.status.clone();
        node.current_attempt_mut()?.status = status;
    }
    let capacity = state.task_slots.available_permits();
    // Recreate waiting consumers in saved arrival order, not node-name order.
    // Other interrupted work follows without duplicating any execution.
    let mut ordered = Vec::new();
    for wait in &state.waits {
        if let Some(node) = wait.nodes.first()
            && state.resume_nodes.contains(&node.node_id)
            && !ordered.contains(&node.node_id)
        {
            ordered.push(node.node_id.clone());
        }
    }
    for node_id in &state.resume_nodes {
        if !ordered.contains(node_id) {
            ordered.push(node_id.clone());
        }
    }
    let resumed = ordered.into_iter().take(capacity).collect::<Vec<_>>();
    let ready = graph
        .ready_nodes()
        .into_iter()
        .take(capacity.saturating_sub(resumed.len()))
        .collect::<Vec<_>>();
    if resumed.is_empty() && ready.is_empty() {
        return Ok(());
    }
    let id_fn = state.id_fn;
    if !ready.is_empty() && id_fn.is_none() {
        return Err(Error::new(
            "INVALID_STATE",
            "plan execution requires an id function",
        ));
    }
    // Prepare and validate every source before claiming any node. Only the
    // existing Graph batch operation mutates execution identities/states.
    let selected = resumed
        .iter()
        .cloned()
        .map(|node_id| (node_id, true))
        .chain(ready.iter().cloned().map(|node_id| (node_id, false)))
        .collect::<Vec<_>>();
    let mut tasks = Vec::new();
    let mut starts = Vec::new();
    for (node_id, resumed) in &selected {
        let node = &graph.plan.nodes[node_id];
        let data = json!({
            "plan_node": node_id,
            "objective": node.objective,
            "sources": graph.source_messages(node_id)?,
        });
        let mut message = Message::new(Role::User);
        message.content.push(Part {
            r#type: "text".into(),
            data: part_data(json!({"value": data.to_string()})),
        });
        message.values = serde_json::from_value(data).expect("task object");
        let exec_id = if *resumed {
            graph
                .state(node_id)?
                .execution
                .as_ref()
                .map(|execution| execution.exec_id)
                .ok_or_else(|| Error::new("INVALID_STATE", "waiting Agent node has no execution"))?
        } else {
            id_fn.expect("validated above")()
        };
        if !*resumed {
            starts.push(Start {
                node_id: node_id.clone(),
                exec_id,
            });
        }
        let existing = if *resumed {
            graph
                .state(node_id)?
                .execution
                .as_ref()
                .and_then(|execution| execution.attempts.last())
                .map(|attempt| attempt.messages.clone())
                .unwrap_or_default()
        } else {
            Messages::default()
        };
        tasks.push((node_id.clone(), exec_id, message, existing, *resumed));
    }
    if !starts.is_empty() {
        graph.start_batch(state.max_concurrency, &starts)?;
    }
    for node_id in resumed {
        state.resume_nodes.remove(&node_id);
    }
    for (node_id, exec_id, input, existing, resumed) in tasks {
        let mut messages = if resumed && !existing.is_empty() {
            existing.clone()
        } else {
            task_messages(state.task_base.clone(), input)
        };
        if resumed
            && state.external_resume_nodes.remove(&node_id)
            && !graph
                .state(&node_id)?
                .execution
                .as_ref()
                .and_then(|execution| execution.attempts.last())
                .is_some_and(|attempt| tools::pending(&attempt.steps))
            && !graph
                .state(&node_id)?
                .execution
                .as_ref()
                .and_then(|execution| execution.attempts.last())
                .is_some_and(|attempt| {
                    crate::runtime::agent::model_steps::continuing(&attempt.steps)
                        || attempt
                            .completion
                            .as_ref()
                            .is_some_and(crate::runtime::agent::completion::State::pending)
                })
            && let Some(input) = state.resume_input.clone()
        {
            messages.push(crate::components::agent_input::user_message(input));
        }
        let mut metadata = state.metadata.clone();
        metadata.insert("node_id".into(), json!(node_id));
        metadata.insert("exec_id".into(), json!(exec_id));
        let previous_usage = if resumed {
            graph
                .state(&node_id)?
                .execution
                .as_ref()
                .and_then(|execution| execution.attempts.last())
                .and_then(|attempt| attempt.usage.clone())
        } else {
            None
        };
        let steps = if resumed {
            graph
                .state(&node_id)?
                .execution
                .as_ref()
                .and_then(|execution| execution.attempts.last())
                .map(|attempt| attempt.steps.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let child_nodes = if resumed {
            graph
                .state(&node_id)?
                .execution
                .as_ref()
                .and_then(|execution| execution.attempts.last())
                .map(|attempt| attempt.nodes.clone())
                .unwrap_or_default()
        } else {
            BTreeMap::new()
        };
        let output = if resumed {
            graph
                .state(&node_id)?
                .execution
                .as_ref()
                .map(|execution| execution.messages.clone())
                .unwrap_or_default()
        } else {
            Messages::default()
        };
        let slot = crate::runtime::agent::reply::Slot::new(
            state.task_slots.clone(),
            state.task_ready.clone(),
        )?;
        let interaction = state
            .interaction
            .as_ref()
            .map(|run| run.branch(slot.clone()));
        let waits = if resumed {
            match (&state.interaction, &interaction) {
                (Some(parent), Some(task)) => parent.restore_consumer(
                    &crate::runtime::agent::wait::Node {
                        node_id: node_id.clone(),
                        exec_id,
                        idx: graph.state_mut(&node_id)?.current_attempt_mut()?.idx,
                    },
                    task,
                ),
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let completion = if resumed {
            graph
                .state(&node_id)?
                .execution
                .as_ref()
                .and_then(|execution| execution.attempts.last())
                .and_then(|attempt| attempt.completion.clone())
        } else {
            None
        };
        let node_trace = state.trace.child_root("agent.node");
        node_trace.property("node_id", node_id.clone());
        node_trace.property("exec_id", exec_id.to_string());
        let mut task = Task {
            node_id,
            exec_id,
            messages: output,
            previous_usage,
            slot: slot.clone(),
            state: ReActState {
                completion,
                restored_output: None,
                replayed_completion: false,
                preparation: None,
                _interaction_owner: interaction
                    .as_ref()
                    .map(crate::runtime::agent::reply::Run::owner),
                interaction,
                waits,
                stream: state.stream,
                model: state.model.clone(),
                dispatch: state.dispatch.clone(),
                subagents: state.subagents.clone(),
                max_iters: state.max_iters,
                max_concurrency: state.max_concurrency,
                cancellation: state.cancellation.clone(),
                task_base: messages.clone(),
                messages,
                iteration: state.iteration.clone(),
                provider: None,
                query: state.query.clone(),
                graph: None,
                resume_nodes: BTreeSet::new(),
                external_resume_nodes: BTreeSet::new(),
                id_fn: state.id_fn,
                checkpoint: None,
                checkpoint_id: None,
                plan_execution: false,
                plan_finalizing: false,
                active_tasks: FuturesUnordered::new(),
                task_runs: BTreeMap::new(),
                task_slots: state.task_slots.clone(),
                task_ready: state.task_ready.clone(),
                task_error: None,
                dispatch_failed: false,
                is_plan_task: true,
                tool_slots: state.tool_slots.clone(),
                pending_plans: VecDeque::new(),
                planning: None,
                pending_replan: None,
                plan_result: None,
                planning_error: None,
                resume_input: None,
                pending_calls: VecDeque::new(),
                active_tools: FuturesUnordered::new(),
                tool_progress: VecDeque::new(),
                child_events: VecDeque::new(),
                child_relay: crate::runtime::agent::subagent::Relay::default(),
                pending_subagents: BTreeMap::new(),
                tool_results: BTreeMap::new(),
                evaluations: BTreeMap::new(),
                steps,
                child_nodes,
                child_nodes_dirty: false,
                feedback_dirty: false,
                phase_progress: false,
                root: crate::runtime::agent::subagent::Root::default(),
                partial: Message::new(Role::Assistant),
                done: false,
                hooks: state.hooks.clone(),
                access: state.access.clone(),
                evaluator: None, // The Agent Evaluator only judges the final answer.
                metadata,
                run: Run::default(), // No second Chat/history lifecycle per task.
                model_pending: false,
                model_settled: false,
                input_tokens: 0,
                request: None,
                trace: node_trace.clone(),
                model_span: None,
            },
        };
        if resumed {
            restore_output(&mut task.state)?;
            tools::restore(&mut task.state)?;
        }
        state.task_runs.insert(
            task.node_id.clone(),
            Active {
                run: task.state.run.clone(),
                slot,
            },
        );
        state.active_tasks.push(Box::pin(
            node_trace.in_span("agent.node.execute", advance(Box::new(task))),
        ));
    }
    Ok(())
}

pub(super) fn commit<M: Chat + 'static>(
    state: &mut ReActState<M>,
    progress: TaskProgress<M>,
) -> Result<(Option<Event>, bool), Error> {
    let TaskProgress {
        mut task,
        mut result,
    } = progress;
    if waiting_on_feedback(&task.state) {
        task.slot.release();
    }
    let mut question = false;
    if let Some(Ok(Event::Complete { message, .. })) = &result
        && let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64)
    {
        question = task.state.steps.iter().any(|step| {
            step.waits
                .get(&id)
                .is_some_and(|record| record.status == "waiting")
        });
        for node in task.state.child_nodes.values() {
            if let Some(scene) = node
                .execution
                .as_ref()
                .and_then(|execution| execution.attempts.last())
                .and_then(|attempt| attempt.agent.as_ref())
                .filter(|scene| matches!(scene["status"].as_str(), Some("waiting" | "running")))
            {
                question |= crate::runtime::agent::wait::question(scene, id)?;
            }
        }
        if !question {
            return Err(Error::new(
                "INVALID_STATE",
                "task question has no actual waiting Step",
            ));
        }
    }
    let coordinate = crate::runtime::agent::wait::Node {
        node_id: task.node_id.clone(),
        exec_id: task.exec_id,
        idx: state
            .graph
            .as_ref()
            .expect("executing plan graph")
            .state(&task.node_id)?
            .execution
            .as_ref()
            .and_then(|execution| execution.attempts.last())
            .ok_or_else(|| Error::new("INVALID_STATE", "plan task has no current Attempt"))?
            .idx,
    };
    let desired = task
        .state
        .waits
        .iter()
        .map(|wait| {
            let mut wait = wait.clone();
            wait.nodes.insert(0, coordinate.clone());
            wait
        })
        .collect::<Vec<_>>();
    // Preserve global arrival order, removing only this task's resolved
    // locators. Question/feedback facts remain in its actual Attempt.steps.
    state
        .waits
        .retain(|wait| wait.nodes.first() != Some(&coordinate) || desired.contains(wait));
    for wait in desired {
        if !state.waits.contains(&wait) {
            state.waits.push(wait);
        }
    }
    // A forwarded child Complete is observation, not this task's output or
    // model input. Only the separately committed delegation result belongs here.
    let child_observation = match &result {
        Some(Ok(Event::Delta { message } | Event::Complete { message, .. })) => {
            match (
                message.metadata.get("node_id").and_then(Value::as_str),
                message.metadata.get("exec_id").and_then(Value::as_i64),
            ) {
                (Some(node_id), Some(exec_id)) => {
                    node_id != task.node_id || exec_id != task.exec_id
                }
                _ => false,
            }
        }
        _ => false,
    };
    if !child_observation
        && !question
        && let Some(Ok(Event::Complete { message, .. })) = &result
    {
        let mut original = message.clone();
        crate::runtime::observation::clear_source(&mut original);
        task.messages.push(original);
    }
    if !task.state.done {
        let mut committed = task.state.messages.clone();
        if !child_observation
            && matches!(
                &result,
                Some(Ok(Event::Complete {
                    finish_reason: Some(reason),
                    ..
                })) if reason == "tool"
            )
            && let Some(Ok(Event::Complete { message, .. })) = &result
        {
            let mut original = message.clone();
            crate::runtime::observation::clear_source(&mut original);
            committed.push(original);
        }
        let graph = state.graph.as_mut().expect("executing plan graph");
        let node = graph.state_mut(&task.node_id)?;
        let status = if task.slot.held() {
            Status::Running
        } else {
            Status::Waiting
        };
        node.status = status.clone();
        node.messages = Some(task.messages.clone());
        if let Some(execution) = node.execution.as_mut() {
            execution.messages = task.messages.clone();
        }
        let attempt = node.current_attempt_mut()?;
        attempt.status = status;
        attempt.messages = committed;
        attempt.steps = task.state.steps.clone();
        attempt.completion = task.state.completion.clone();
        if task.state.child_nodes_dirty {
            attempt.nodes = task.state.child_nodes.clone();
            task.state.child_nodes_dirty = false;
        }
        if let Some(run) = &state.interaction {
            // The task and all nested child facts now belong to the outer
            // graph. Embedded owners defer FIFO advancement to their parent.
            run.commit_feedback(&attempt.steps);
            for node in attempt.nodes.values() {
                if let Some(scene) = node
                    .execution
                    .as_ref()
                    .and_then(|execution| execution.attempts.last())
                    .and_then(|attempt| attempt.agent.as_ref())
                {
                    run.commit_scene_feedback(scene)?;
                }
            }
        }
        node.set_attempt_result(
            if task.slot.held() {
                Status::Running
            } else {
                Status::Waiting
            },
            crate::model::token::merge(task.previous_usage.clone(), task.state.run.usage()),
            None,
        )?;
        if let Some(Ok(event)) = &mut result {
            crate::runtime::observation::associate(
                event,
                state.query.metadata.get("chat_id").and_then(Value::as_i64),
                Some((&task.node_id, task.exec_id)),
            );
        }
        let feedback = std::mem::take(&mut task.state.feedback_dirty);
        let phase = std::mem::take(&mut task.state.phase_progress);
        let save = question
            || (!phase
                && (result.is_none() || feedback)
                && (!state.waits.is_empty() || crate::runtime::agent::scope::active()));
        state.active_tasks.push(Box::pin(advance(task)));
        return Ok((result.and_then(Result::ok), save));
    }
    let result = result.expect("a finished task has a terminal result");
    let new_usage = task.state.run.usage();
    task.slot.release();
    state.task_runs.remove(&task.node_id);
    // Settle each task's actual model calls once, including failure/cancel.
    // Individual forwarded Events retain their per-call Usage unchanged.
    state.run.settle(new_usage.clone());
    let usage = crate::model::token::merge(task.previous_usage, new_usage);
    let graph = state.graph.as_mut().expect("executing plan graph");
    // Live approval may have marked the task waiting without ending its
    // invocation. Its terminal transition still belongs to that same Attempt.
    if graph.state(&task.node_id)?.status == Status::Waiting {
        graph.resume(&task.node_id)?;
    }
    let mut context = task.state.messages.clone();
    // An eval error may stop the task before the normal ordered drain. Keep
    // successful call facts already emitted in that incomplete batch too.
    context.0.extend(task.state.tool_results.values().cloned());
    let committed = match &result {
        Err(error) if error.code == "WAITING" => {
            graph.wait(&task.node_id)?;
            graph
                .state_mut(&task.node_id)?
                .set_attempt_result(Status::Waiting, usage, None)?;
            state.task_error.get_or_insert_with(|| error.clone());
            Ok((None, true)) // Finish independent siblings before returning waiting.
        }
        Err(error) => {
            graph.fail_with_usage(&task.node_id, Some(error.clone()), usage)?;
            state.task_error.get_or_insert_with(|| error.clone());
            Ok((None, true)) // Finish independent siblings before the terminal error.
        }
        Ok(Event::Complete {
            finish_reason,
            message,
            ..
        }) if finish_reason.as_deref() == Some("cancelled") => {
            graph.cancel(&task.node_id)?;
            graph.set_finish_reason(&task.node_id, finish_reason.clone())?;
            graph
                .state_mut(&task.node_id)?
                .set_attempt_result(Status::Cancelled, usage, None)?;
            state.partial = message.clone();
            crate::runtime::observation::clear_source(&mut state.partial);
            // Other in-flight tasks get a chance to emit and settle their own
            // partial cancellation result. The last one closes the outer run.
            state.done = state.active_tasks.is_empty();
            if !state.cancellation.is_cancelled() {
                state
                    .task_error
                    .get_or_insert_with(|| Error::new("CANCELLED", "planned task cancelled"));
                state.done = false;
            }
            Ok((result.ok(), true))
        }
        Ok(Event::Complete { finish_reason, .. }) => {
            graph.succeed(&task.node_id, task.messages, usage)?;
            graph.set_finish_reason(&task.node_id, finish_reason.clone())?;
            Ok((result.ok(), true))
        }
        Ok(Event::Delta { .. }) => {
            graph.succeed(&task.node_id, task.messages, usage)?;
            Ok((result.ok(), true))
        }
    };
    // Attempt messages retain the actual invocation context, including Tool
    // evaluation feedback used by this task. Node/execution messages remain
    // the result stream exposed to dependent tasks, not internal feedback.
    let attempt = graph.state_mut(&task.node_id)?.current_attempt_mut()?;
    attempt.messages = context;
    attempt.steps = task.state.steps;
    attempt.completion = task.state.completion;
    attempt.nodes = task.state.child_nodes;
    let mut committed = committed?;
    if let Some(event) = &mut committed.0 {
        crate::runtime::observation::associate(
            event,
            state.query.metadata.get("chat_id").and_then(Value::as_i64),
            Some((&task.node_id, task.exec_id)),
        );
    }
    Ok(committed)
}

/// A fatal parent error (including failed question persistence) ends active
/// task consumers too. Keep their committed facts, not actionable orphan waits.
pub(super) fn cancel_active<M: Chat>(state: &mut ReActState<M>) {
    for active in std::mem::take(&mut state.task_runs).into_values() {
        // These handles contain only work since this invocation resumed,
        // unlike Node.usage which also retains previously settled attempts.
        state.run.settle(active.run.usage());
        active.slot.release();
    }
    let Some(graph) = state.graph.as_mut() else {
        return;
    };
    for node in graph
        .nodes
        .values_mut()
        .filter(|node| matches!(node.status, Status::Running | Status::Waiting))
    {
        if let Some(attempt) = node
            .execution
            .as_mut()
            .and_then(|execution| execution.attempts.last_mut())
        {
            crate::runtime::agent::step::cancel_pending(&mut attempt.steps);
        }
        node.status = Status::Cancelled;
        // Active graph nodes have a current Attempt by construction.
        node.set_attempt_result(Status::Cancelled, node.usage.clone(), None)
            .expect("active graph node has a current Attempt");
    }
}

pub(super) fn waiting_on_feedback<M: Chat>(state: &ReActState<M>) -> bool {
    // Selection creates Steps for the whole Tool batch. Calls still queued
    // behind a suspended call are not executing and cannot retain admission.
    !state.waits.is_empty()
        && !state.steps.iter().any(|step| {
            step.status == "running"
                && !(step.kind == "agent"
                    && state
                        .interaction
                        .as_ref()
                        .is_some_and(|run| run.children_blocked()))
                && !state
                    .pending_calls
                    .iter()
                    .any(|(_, call)| step.call_id.as_deref() == Some(call.call_id.as_str()))
        })
}

pub(super) fn usage<M: Chat>(state: &ReActState<M>) -> Option<crate::model::token::Usage> {
    state
        .task_runs
        .values()
        .fold(state.run.usage(), |total, active| {
            crate::model::token::merge(total, active.run.usage())
        })
}

pub(super) fn finalize<M: Chat>(state: &mut ReActState<M>) -> Result<(), Error> {
    let graph = state.graph.as_ref().expect("executing plan graph");
    if !graph.is_complete() {
        return Err(Error::new(
            "INVALID_STATE",
            "plan has no executable nodes and is not complete",
        ));
    }
    let results = graph
        .nodes
        .iter()
        .map(|(id, node)| (id.clone(), node.messages.clone()))
        .collect::<BTreeMap<_, _>>();
    let data = json!({"plan_complete": true, "results": results});
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "text".into(),
        data: part_data(json!({"value": data.to_string()})),
    });
    message.values = serde_json::from_value(data).expect("results object");
    state.messages.push(message);
    state.plan_finalizing = true;
    Ok(())
}

#[cfg(test)]
mod tests;
