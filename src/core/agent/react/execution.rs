//! Private task dispatch for ReAct. The graph remains runtime::agent state;
//! tasks reuse the Agent's model/Toolkit loop rather than looking up names.
use super::*;
use crate::runtime::agent::{graph::Start, node::Status};

struct Task<M: Chat> {
    node_id: String,
    exec_id: i64,
    state: ReActState<M>,
    messages: Messages,
}

pub(super) struct TaskProgress<M: Chat> {
    task: Task<M>,
    result: Result<Event, Error>,
}

async fn advance<M: Chat + 'static>(mut task: Task<M>) -> TaskProgress<M> {
    let (result, state) = next_react_event(task.state).await.expect("live plan task");
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
    let ready = graph
        .ready_nodes()
        .into_iter()
        .take(
            state
                .max_concurrency
                .saturating_sub(state.active_tasks.len()),
        )
        .collect::<Vec<_>>();
    if ready.is_empty() {
        return Ok(());
    }
    let id_fn = state
        .id_fn
        .ok_or_else(|| Error::new("INVALID_STATE", "plan execution requires an id function"))?;
    // Prepare and validate every source before claiming any node. Only the
    // existing Graph batch operation mutates execution identities/states.
    let mut inputs = Vec::new();
    let mut starts = Vec::new();
    for node_id in ready {
        let node = &graph.plan.nodes[&node_id];
        let data = json!({
            "plan_node": node_id,
            "objective": node.objective,
            "sources": graph.source_messages(&node_id)?,
        });
        let mut message = Message::new(Role::User);
        message.content.push(Part {
            r#type: "text".into(),
            data: json!({"value": data.to_string()}),
        });
        message.values = serde_json::from_value(data).expect("task object");
        inputs.push(message);
        starts.push(Start {
            node_id,
            exec_id: id_fn(),
        });
    }
    graph.start_batch(state.max_concurrency, &starts)?;
    for (start, input) in starts.into_iter().zip(inputs) {
        let mut messages = state.task_base.clone();
        let mut instruction = Message::new(Role::System);
        instruction.content.push(Part {
            r#type: "text".into(),
            data: json!({"value": "Complete only the current planned task objective using the original query and its committed source results. Choose tools and arguments yourself when needed. Return this task's result, not the final answer for unrelated tasks."}),
        });
        messages.0.insert(0, instruction);
        messages.push(input);
        let mut metadata = state.metadata.clone();
        metadata.insert("node_id".into(), json!(start.node_id));
        metadata.insert("exec_id".into(), json!(start.exec_id));
        let task = Task {
            node_id: start.node_id,
            exec_id: start.exec_id,
            messages: Messages::default(),
            state: ReActState {
                model: state.model.clone(),
                toolkit: state.toolkit.clone(),
                max_iters: state.max_iters,
                max_concurrency: state.max_concurrency,
                cancellation: state.cancellation.clone(),
                task_base: state.task_base.clone(),
                messages,
                iteration: state.iteration.clone(),
                provider: None,
                query: state.query.clone(),
                graph: None,
                id_fn: None,
                checkpoint: None,
                checkpoint_id: None,
                plan_execution: false,
                plan_finalizing: false,
                active_tasks: FuturesUnordered::new(),
                task_error: None,
                dispatch_failed: false,
                is_plan_task: true,
                tool_slots: state.tool_slots.clone(),
                pending_plans: VecDeque::new(),
                planning: None,
                pending_replan: None,
                plan_result: None,
                planning_error: None,
                pending_calls: VecDeque::new(),
                active_tools: FuturesUnordered::new(),
                tool_results: BTreeMap::new(),
                evaluations: BTreeMap::new(),
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
            },
        };
        state.active_tasks.push(Box::pin(advance(task)));
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
    if let Ok(event) = &mut result {
        let message = match event {
            Event::Delta { message } | Event::Complete { message, .. } => message,
        };
        message
            .metadata
            .insert("node_id".into(), json!(task.node_id));
        message
            .metadata
            .insert("exec_id".into(), json!(task.exec_id));
        if matches!(event, Event::Complete { .. }) {
            let Event::Complete { message, .. } = event else {
                unreachable!()
            };
            task.messages.push(message.clone());
        }
    }
    if !task.state.done {
        state.active_tasks.push(Box::pin(advance(task)));
        return Ok((result.ok(), false));
    }
    let usage = task.state.run.usage();
    // Settle each task's actual model calls once, including failure/cancel.
    // Individual forwarded Events retain their per-call Usage unchanged.
    state.run.settle(usage.clone());
    let graph = state.graph.as_mut().expect("executing plan graph");
    match &result {
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
            graph
                .state_mut(&task.node_id)?
                .set_attempt_result(Status::Cancelled, usage, None)?;
            state.partial = message.clone();
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
        Ok(_) => {
            graph.succeed(&task.node_id, task.messages, usage)?;
            Ok((result.ok(), true))
        }
    }
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
        data: json!({"value": data.to_string()}),
    });
    message.values = serde_json::from_value(data).expect("results object");
    state.messages.push(message);
    state.plan_finalizing = true;
    Ok(())
}
