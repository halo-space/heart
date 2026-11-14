//! Workflow retry facts and one logical invocation.
//!
//! This module preserves committed call facts while retrying only unfinished work.

use super::subgraph::{SubgraphScenes, retry_graph_state, run_subgraph};
use super::*;

pub(super) fn committed_tool_facts(execution: &node::Execution) -> node::Attempt {
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
pub(super) fn terminal_observations(state: &graph::State) -> Vec<crate::event::Event> {
    let mut events = Vec::new();
    for (node_id, node) in &state.nodes {
        for execution in node.executions.values() {
            if execution.status != node::status::execution::Status::Succeeded
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

pub(super) fn cancelled_observations(state: &graph::State) -> Vec<crate::event::Event> {
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
#[doc(hidden)]
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
                    status: node::status::attempt::Status::Failed,
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
            current_view.status = crate::components::node::Status::Running;
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
                status: node::status::attempt::Status::Running,
                ..Default::default()
            };
            if cancellation.is_cancelled() {
                attempt.status = node::status::attempt::Status::Cancelled;
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
                                    attempt.status = node::status::attempt::Status::Waiting;
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
                    let mut agent_input: Input = match serde_json::from_value(input.clone()) {
                        Ok(input) => input,
                        Err(error) => {
                            attempt.status = node::status::attempt::Status::Failed;
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
                        attempt.status = node::status::attempt::Status::Failed;
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
                            attempt.status = node::status::attempt::Status::Failed;
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
                            .map_err(|error| Error::new("SERIALIZATION", error.to_string())),
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
                            attempt.status = node::status::attempt::Status::Succeeded;
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
                                    attempt.status = node::status::attempt::Status::Failed;
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
                                    attempt.status = node::status::attempt::Status::Waiting;
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
                                    attempt.status = node::status::attempt::Status::Failed;
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
                                attempt.status = node::status::attempt::Status::Waiting;
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
                    attempt.status = node::status::attempt::Status::Succeeded;
                    attempts.push(attempt);
                    return Invocation {
                        attempts,
                        outcome: Outcome::Complete(value),
                        graph: None,
                    };
                }
                Ok(crate::node::Signal::Wait(wait)) => {
                    attempt.status = node::status::attempt::Status::Waiting;
                    attempts.push(attempt);
                    return Invocation {
                        attempts,
                        outcome: Outcome::Wait(wait.wait_id),
                        graph: None,
                    };
                }
                Err(error) => {
                    if error.code == "CANCELLED" {
                        attempt.status = node::status::attempt::Status::Cancelled;
                        attempts.push(attempt);
                        return Invocation {
                            attempts,
                            outcome: Outcome::Cancelled,
                            graph: None,
                        };
                    }
                    attempt.status = node::status::attempt::Status::Failed;
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
