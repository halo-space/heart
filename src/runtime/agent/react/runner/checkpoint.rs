//! ReAct checkpoint serialization, restoration, and cancellation snapshots.

use super::*;

pub(crate) fn checkpoint_snapshot<M: Chat>(
    state: &ReActState<M>,
    result: &Result<Event, Error>,
) -> Value {
    let (graph, nodes, history) = state
        .graph
        .as_ref()
        .map(|graph| {
            (
                Some(graph::State {
                    plan: graph.plan.clone(),
                }),
                graph.nodes.clone(),
                graph.history.clone(),
            )
        })
        .unwrap_or((None, BTreeMap::new(), Vec::new()));
    let status = result_status(result);
    let mut snapshot = serde_json::to_value(AgentState {
        completion: state.completion.clone(),
        preparation: state.preparation.clone(),
        chat_id: state.query.metadata.get("chat_id").and_then(Value::as_i64),
        r#type: "agent".into(),
        status,
        stream: state.stream,
        usage: execution::usage(state),
        execution: state.root.snapshot(
            status,
            execution::usage(state),
            result.as_ref().err().cloned(),
        ),
        graph,
        nodes,
        history,
        messages: state.messages.clone(),
        steps: state.steps.clone(),
        waits: state.waits.clone(),
        error: result.as_ref().err().cloned(),
        final_message: match result {
            Ok(Event::Complete { message, .. }) => {
                Some(crate::runtime::observation::original_message(message))
            }
            _ => None,
        },
        metadata: state.query.metadata.clone().into_iter().collect(),
    })
    .unwrap_or_else(|_| json!({}));
    snapshot["query"] = serde_json::to_value(&state.query).unwrap_or(Value::Null);
    snapshot["task_base"] = serde_json::to_value(&state.task_base).unwrap_or(Value::Null);
    snapshot["iteration"] = json!(state.iteration.load(Ordering::Relaxed));
    snapshot["plan_execution"] = json!(state.plan_execution);
    snapshot["plan_finalizing"] = json!(state.plan_finalizing);
    snapshot["tool_results"] = serde_json::to_value(&state.tool_results).unwrap_or(Value::Null);
    snapshot["planning"] = serde_json::to_value(&state.planning).expect("planning control");
    crate::runtime::agent::checkpoint::record_result(&mut snapshot, result);
    snapshot
}

pub(crate) fn checkpoint_id(metadata: &Map<String, Value>) -> Result<i64, Error> {
    let id = metadata
        .get("chat_id")
        .and_then(Value::as_i64)
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "checkpoint requires chat_id"))?;
    if id <= 0 {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "checkpoint chat_id must be positive",
        ));
    }
    Ok(id)
}

pub(crate) fn save_checkpoint<M: Chat>(
    state: &ReActState<M>,
    snapshot: Value,
) -> impl Future<Output = Result<(), Error>> + 'static {
    // Own just the save handle and ID, not a borrow of the whole stream state.
    // Send event streams need not also be Sync.
    let save = state.checkpoint.clone();
    let id = state.checkpoint_id;
    async move {
        if crate::runtime::agent::scope::capture(snapshot.clone()) {
            Ok(())
        } else if let (Some(save), Some(id)) = (save, id) {
            save.save(id, snapshot, Cancellation::new()).await
        } else {
            Ok(())
        }
    }
}

pub(crate) fn restore_output<M: Chat>(state: &mut ReActState<M>) -> Result<(), Error> {
    crate::runtime::agent::model_steps::validate_owner(
        &state.steps,
        state.completion.as_ref(),
        state.hooks.len(),
    )?;
    state.restored_output =
        if let Some(step) = crate::runtime::agent::model_steps::pending(&state.steps) {
            Some(crate::runtime::agent::model_steps::event(step)?)
        } else {
            state
                .completion
                .as_ref()
                .filter(|completion| completion.pending())
                .map(|saved| Event::Complete {
                    message: saved.message.clone(),
                    usage: saved.message.usage.clone(),
                    finish_reason: saved.finish_reason.clone(),
                })
        };
    if let Some(Event::Complete { message, .. }) = &state.restored_output {
        state.partial = message.clone();
        state.model_settled = true;
        state.model_pending = true;
        state.replayed_completion = true;
        state.input_tokens = state
            .steps
            .iter()
            .rev()
            .find(|step| step.kind == "model")
            .and_then(|step| step.request.as_deref())
            .map(crate::runtime::agent::chat::estimate_input_tokens)
            .unwrap_or(0);
    }
    Ok(())
}

pub(crate) fn restore_state<M: Chat>(
    state: &mut ReActState<M>,
    snapshot: Value,
) -> Result<(), Error> {
    fn field<T: serde::de::DeserializeOwned>(snapshot: &Value, name: &str) -> Result<T, Error> {
        serde_json::from_value(snapshot[name].clone())
            .map_err(|_| invalid_checkpoint(&format!("invalid Agent checkpoint field: {name}")))
    }
    state.messages = field(&snapshot, "messages")?;
    state.task_base = field(&snapshot, "task_base")?;
    state.query = field(&snapshot, "query")?;
    state.iteration = Arc::new(AtomicUsize::new(field(&snapshot, "iteration")?));
    state.plan_execution = field(&snapshot, "plan_execution")?;
    state.plan_finalizing = field(&snapshot, "plan_finalizing")?;
    state.root = crate::runtime::agent::subagent::Root::restore(&snapshot)?;
    state.tool_results = field(&snapshot, "tool_results")?;
    state.steps = snapshot
        .get("steps")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| invalid_checkpoint("invalid Agent checkpoint steps"))?
        .unwrap_or_default();
    state.completion = snapshot
        .get("completion")
        .filter(|value| !value.is_null())
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| invalid_checkpoint("invalid Agent completion"))?;
    state.planning = snapshot
        .get("planning")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| invalid_checkpoint("invalid Model planning control"))?
        .flatten();
    restore_output(state)?;
    state.waits = snapshot
        .get("waits")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| invalid_checkpoint("invalid saved waiting locators"))?
        .unwrap_or_default();
    tools::restore(state)?;
    // A parent waiting on a child is resumed by replaying only the registered
    // child hand-off.  The parent model is not called again for that Call,
    // and a committed tool result always wins over a stale pending entry.
    let mut pending = state.root.resume_calls(&state.steps, &mut state.messages)?;
    for (index, call) in &pending {
        state.pending_subagents.insert(
            call.call_id.clone(),
            PendingSubAgent {
                index: *index,
                call: call.clone(),
            },
        );
    }
    pending.sort_by_key(|(index, _)| *index);
    state.pending_calls.extend(pending);
    state.run.settle(field(&snapshot, "usage")?);
    if let Some(graph) = snapshot.get("graph").filter(|graph| !graph.is_null()) {
        let mut graph: Graph = serde_json::from_value(json!({
            "plan": graph["plan"],
            "history": snapshot.get("history").cloned().unwrap_or_else(|| json!([])),
            "nodes": snapshot["nodes"],
        }))
        .map_err(|_| invalid_checkpoint("invalid Agent checkpoint graph"))?;
        graph.plan.validate_structure()?;
        if graph.plan.nodes.keys().ne(graph.nodes.keys()) {
            return Err(invalid_checkpoint(
                "checkpoint graph nodes do not match the plan",
            ));
        }
        use crate::runtime::agent::node::status::Status;
        let waiting_nodes = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.status, Status::Running | Status::Waiting))
            .map(|(node_id, _)| node_id.clone())
            .collect::<Vec<_>>();
        let external_nodes = graph
            .nodes
            .iter()
            .filter(|(_, node)| node.status == Status::Waiting)
            .map(|(node_id, _)| node_id.clone())
            .collect::<BTreeSet<_>>();
        if matches!(
            snapshot.get("status").and_then(Value::as_str),
            Some("waiting" | "paused")
        ) && external_nodes.is_empty()
            && state.pending_subagents.is_empty()
            && state.waits.is_empty()
            && state.restored_output.is_none()
            && crate::runtime::agent::model_steps::pending_input(&state.steps).is_none()
        {
            return Err(invalid_checkpoint(
                "waiting Agent checkpoint has no waiting node",
            ));
        }
        for node in graph.nodes.values_mut() {
            if node.status == Status::Running {
                node.status = Status::Waiting;
                if let Some(attempt) = node
                    .execution
                    .as_mut()
                    .and_then(|execution| execution.attempts.last_mut())
                {
                    attempt.status = Status::Waiting;
                }
            } else if node.status == Status::Succeeded {
                if node.messages.is_none()
                    || node
                        .execution
                        .as_ref()
                        .and_then(|execution| execution.attempts.last())
                        .is_none_or(|attempt| attempt.status != Status::Succeeded)
                {
                    return Err(invalid_checkpoint(
                        "successful Agent node has no committed result",
                    ));
                }
            } else if node.status == Status::Failed {
                state.task_error = node
                    .execution
                    .as_ref()
                    .and_then(|execution| execution.error.clone())
                    .or_else(|| Some(Error::new("EXECUTION_FAILED", "restored Agent node failed")));
            }
        }
        graph.activate_ready()?;
        for node_id in waiting_nodes {
            graph.resume(&node_id)?;
            state.resume_nodes.insert(node_id);
        }
        state.external_resume_nodes = external_nodes;
        state.graph = Some(graph);
    }
    // A waiting snapshot without a plan graph is a valid external wait.  The
    // next call with the same chat_id supplies the continuation payload; no
    // internal node or wait ID is required in the public input.
    Ok(())
}

pub(crate) fn is_waiting_snapshot(snapshot: &Value) -> bool {
    matches!(
        snapshot.get("status").and_then(Value::as_str),
        Some("running" | "waiting" | "paused")
    ) || snapshot
        .get("nodes")
        .and_then(Value::as_object)
        .is_some_and(|nodes| {
            nodes.values().any(|node| {
                matches!(
                    node.get("status").and_then(Value::as_str),
                    Some("running" | "waiting")
                )
            })
        })
}

pub(crate) fn is_external_wait_snapshot(snapshot: &Value) -> bool {
    if snapshot
        .get("completion")
        .and_then(|state| state.get("stage"))
        .and_then(Value::as_str)
        .is_some_and(|stage| stage != "committed")
        || snapshot
            .get("steps")
            .and_then(Value::as_array)
            .is_some_and(|steps| {
                steps.iter().any(|step| {
                    step["kind"] == "model"
                        && (matches!(step["stage"].as_str(), Some("before" | "after"))
                            || (step["stage"] == "call"
                                && step["attempts"]
                                    .as_array()
                                    .and_then(|attempts| attempts.last())
                                    .is_none_or(|attempt| attempt["status"] != "waiting")))
                })
            })
    {
        return false;
    }
    let waiting_child =
        crate::runtime::agent::subagent::Root::restore(snapshot).is_ok_and(|root| root.pending());
    let waiting_tool = snapshot
        .get("steps")
        .and_then(Value::as_array)
        .is_some_and(|steps| {
            steps.iter().any(|step| {
                step.get("kind").and_then(Value::as_str) == Some("tool")
                    && matches!(
                        step.get("stage").and_then(Value::as_str),
                        Some("before" | "call" | "after" | "eval" | "complete")
                    )
            })
        });
    !waiting_child
        && !waiting_tool
        && (snapshot.get("status").and_then(Value::as_str) == Some("waiting")
            || snapshot
                .get("nodes")
                .and_then(Value::as_object)
                .is_some_and(|nodes| {
                    nodes
                        .values()
                        .any(|node| node.get("status").and_then(Value::as_str) == Some("waiting"))
                }))
}

pub(crate) fn checkpoint_identity_matches(snapshot: &Value, metadata: &Map<String, Value>) -> bool {
    crate::runtime::agent::checkpoint::identity_matches(snapshot, metadata)
}

pub(crate) fn running_snapshot<M: Chat>(state: &ReActState<M>) -> Value {
    checkpoint_snapshot(
        state,
        &Ok(Event::Delta {
            message: state.partial.clone(),
        }),
    )
}

pub(crate) fn waiting_snapshot<M: Chat>(state: &ReActState<M>) -> Value {
    let mut snapshot = checkpoint_snapshot(
        state,
        &Ok(Event::Delta {
            message: state.partial.clone(),
        }),
    );
    snapshot["status"] = serde_json::to_value(crate::runtime::agent::status::Status::Waiting)
        .expect("serializable waiting status");
    if let Some(execution) = state.root.snapshot(
        crate::runtime::agent::status::Status::Waiting,
        execution::usage(state),
        None,
    ) {
        snapshot["execution"] =
            serde_json::to_value(execution).expect("serializable root execution");
    }
    snapshot
}

// Cancellation-only work must not enlarge the ordinary nested poll frame.
#[inline(never)]
pub(crate) fn cancel_children<M: Chat + 'static>(
    state: &mut ReActState<M>,
    cx: &mut std::task::Context<'_>,
) -> Result<Vec<Event>, Error> {
    let relay = state.child_relay.clone();
    let (mut completed, events) = subagent::cancel_ready(
        cx,
        &relay,
        |cx| state.active_tools.poll_next_unpin(cx),
        |observation| {
            if state.is_plan_task {
                state.child_nodes_dirty |= observation.has_facts();
                subagent::observe(&mut state.child_nodes, observation)?;
            } else {
                state.root.observe(observation)?;
            }
            state.run.settle_child(observation.usage.clone());
            Ok(())
        },
    )?;
    completed.extend(state.tool_progress.drain(..));
    for progress in completed {
        if let ToolProgress::SubAgent(_, call, result, _) = progress {
            let message = result
                .as_ref()
                .as_ref()
                .ok()
                .map(|progress| progress.message.clone());
            if state.is_plan_task {
                subagent::complete(
                    &mut state.child_nodes,
                    &mut state.steps,
                    &call,
                    &result,
                    message,
                )?;
            } else {
                state
                    .root
                    .complete(&mut state.steps, &call, &result, message)?;
            }
            state.run.settle_child(match &*result {
                Ok(progress) => progress.usage.clone(),
                Err(failure) => failure.usage.clone(),
            });
        }
    }
    if state.is_plan_task {
        subagent::cancel_pending(&mut state.child_nodes);
    } else {
        state.root.cancel_pending();
    }
    state.active_tools.clear();
    // No live question may still point into a child just made terminal.
    // Historical question records remain on their actual Tool Steps.
    state.waits.clear();
    Ok(events)
}
