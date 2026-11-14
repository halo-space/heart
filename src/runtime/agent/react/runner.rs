//! ReAct execution state machine.
//!
//! Builder and Agent remain in the parent module; this module owns model/tool
//! progression, planning, and child coordination. Checkpoint and message
//! projection helpers live in focused child modules.

use super::*;
use crate::runtime::agent::{chat, completion, lifecycle, subagent};

mod checkpoint;
mod message;
mod run;
mod state;

pub(crate) use checkpoint::{
    cancel_children, checkpoint_id, checkpoint_identity_matches, checkpoint_snapshot,
    is_external_wait_snapshot, is_waiting_snapshot, restore_output, restore_state,
    running_snapshot, save_checkpoint, waiting_snapshot,
};
pub(crate) use message::{
    ErrorDetails, collect_delta, decode_call, tool_error_message, tool_success_message,
};
pub(crate) use run::{next_react_event, next_react_progress, result_status};
pub(crate) use state::ReActState;

fn register_cancellation(
    waiter: &mut Option<crate::cancellation::Registration>,
    cancellation: &Cancellation,
    waker: &std::task::Waker,
) {
    if let Some(waiter) = waiter {
        waiter.update(waker);
    } else {
        *waiter = Some(cancellation.register(waker));
    }
}

// Distinguish model-selected control from business evaluation. Replanning is
// not a Tool invocation and must not invent a Call or acknowledge a fake one.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) enum Planning {
    Selected(usize, Call),
    Evaluation,
}

pub(crate) enum ToolProgress {
    Interaction(crate::runtime::agent::reply::Change),
    Observation(Box<crate::runtime::agent::subagent::Observation>),
    Called(
        usize,
        Call,
        Box<Result<(Call, Message), Failure>>,
        Option<crate::runtime::agent::reply::Slot>,
    ),
    Evaluated(String, Result<ToolEvaluation, Error>),
    SubAgent(
        usize,
        Call,
        Box<Result<SubAgentProgress, Box<SubAgentFailure>>>,
        Option<tokio::sync::OwnedSemaphorePermit>,
    ),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct PendingSubAgent {
    index: usize,
    call: Call,
}

// Private per-call dependencies, not a new component Context or Runtime state.
#[derive(Clone)]
pub(crate) struct ToolExecution {
    interaction: Option<crate::runtime::agent::reply::Handle>,
    id_fn: Option<fn() -> i64>,
    dispatch: std::sync::Arc<Dispatch>,
    hooks: Hooks,
    access: Access,
    metadata: Map<String, Value>,
    cancellation: Cancellation,
    trace: Root,
}

pub(crate) async fn prepare_tool(
    state: &ToolExecution,
    call: &Call,
    saved: &Step,
) -> Result<Call, Failure> {
    Box::pin(lifecycle::tool_prepare(
        &state.dispatch,
        &state.access,
        &state.hooks,
        &state.metadata,
        state.interaction.as_ref(),
        state.id_fn,
        saved,
        call,
        |effective| async move {
            if let Some(run) = &state.interaction {
                run.tool_stage(effective, None, &state.cancellation).await?;
            }
            Ok(())
        },
        |permission| async move {
            if let Some(run) = &state.interaction {
                let call = permission.call()?;
                run.permission_stage(call, permission, &state.cancellation)
                    .await?;
            }
            Ok(())
        },
        &state.cancellation,
    ))
    .await
}

pub(crate) async fn execute_prepared_tool(
    state: &ToolExecution,
    call: Call,
    saved: Step,
) -> Result<(Call, Message), Error> {
    lifecycle::tool_call(
        &state.dispatch,
        &saved,
        call,
        state.interaction.clone(),
        state.id_fn,
        |call, message| async move {
            if let Some(run) = &state.interaction {
                run.tool_stage(call, Some(message), &state.cancellation)
                    .await?;
            }
            Ok(())
        },
        &state.cancellation,
    )
    .await
}

pub(crate) fn decode_subagent_input(
    call: &Call,
    parent: &Map<String, Value>,
    operation: &str,
) -> Result<(String, Input), Error> {
    let name = call
        .arguments
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", format!("{operation} name is missing")))?
        .to_owned();
    let input =
        call.arguments.get("input").cloned().ok_or_else(|| {
            Error::new("INVALID_ARGUMENTS", format!("{operation} input is missing"))
        })?;
    let mut input: Input = serde_json::from_value(input).map_err(|_| {
        Error::new(
            "INVALID_ARGUMENTS",
            format!("{operation} input must be agent::Input"),
        )
    })?;
    if operation == "team_assign" && input.payload.is_empty() {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "team_assign input.payload must contain at least one item",
        ));
    }
    crate::runtime::agent::subagent::inherit_identity(&mut input, parent)?;
    let depth = parent.get("depth").and_then(Value::as_u64).unwrap_or(1);
    if depth >= 2 {
        return Err(Error::new("INVALID_ARGUMENTS", "SubAgent depth limit is 2"));
    }
    input
        .metadata
        .insert("depth".into(), Value::from(depth + 1));
    Ok((name, input))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_subagent(
    subagents: Arc<Vec<SubAgent>>,
    handoff_targets: Arc<Vec<String>>,
    call: Call,
    parent: Map<String, Value>,
    restore: Option<Value>,
    cancellation: Cancellation,
    id_fn: Option<fn() -> i64>,
    observer: Option<crate::runtime::agent::subagent::Observer>,
) -> Result<SubAgentProgress, Box<SubAgentFailure>> {
    let handoff = call.name == "agent_handoff";
    let team_mode = call.name == "team_assign";
    let operation = if handoff {
        "agent_handoff"
    } else if team_mode {
        "team_assign"
    } else {
        "agent_spawn"
    };
    let (name, input) =
        decode_subagent_input(&call, &parent, operation).map_err(SubAgentFailure::new)?;
    if handoff && !handoff_targets.iter().any(|target| target == &name) {
        return Err(SubAgentFailure::new(Error::new(
            "NOT_FOUND",
            format!("agent_handoff target is not allowed: {name}"),
        )));
    }
    let subagent = subagents
        .iter()
        .find(|subagent| subagent.has_name(&name))
        .ok_or_else(|| {
            SubAgentFailure::new(Error::new(
                "NOT_FOUND",
                format!("{operation} target is not registered"),
            ))
        })?;
    crate::runtime::agent::subagent::run(
        subagent,
        input,
        &call,
        restore,
        cancellation,
        id_fn,
        observer.map(|observer| observer.suppress_terminal(handoff)),
    )
    .await
}

pub(crate) fn subagent_error_message(call: &Call, error: &Error) -> Message {
    let error_value = json!({
        "code": error.code,
        "message": error.message,
        "details": error.details,
    });
    Message {
        role: Role::Tool,
        content: vec![Part {
            r#type: "text".into(),
            data: part_data(json!({"value": error_value.to_string()})),
        }],
        values: Map::from_iter([
            (String::from("error"), error_value),
            (String::from("call_id"), Value::String(call.call_id.clone())),
            (String::from("name"), Value::String(call.name.clone())),
        ]),
        metadata: Map::new(),
        usage: None,
    }
}
