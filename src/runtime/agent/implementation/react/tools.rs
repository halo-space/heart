use super::*;
pub(super) use crate::runtime::agent::tool_steps::{
    called, committed, evaluated, failed_message, pending, selected, stopped,
};
use crate::runtime::agent::tool_steps::{current, new_step};

pub(super) fn restore<M: Chat>(state: &mut ReActState<M>) -> Result<(), Error> {
    for step in &state.steps {
        if step.kind == "tool"
            && matches!(step.stage.as_deref(), Some("call" | "after" | "eval"))
            && let Some(call) = &step.call
        {
            state.dispatch.validate(call)?;
        }
    }
    let pending =
        crate::runtime::agent::tool_steps::resume_calls(&state.steps, &mut state.messages)?;
    state.pending_calls.extend(pending);
    if state.is_plan_task {
        let pending = crate::runtime::agent::implementation::subagent::resume_calls(
            &state.child_nodes,
            &state.steps,
            &mut state.messages,
        )?;
        state.pending_calls.extend(pending);
        state
            .pending_calls
            .make_contiguous()
            .sort_by_key(|(index, _)| *index);
    }
    Ok(())
}

pub(super) fn continue_step<M: Chat + 'static>(
    state: &mut ReActState<M>,
    index: usize,
    call: &Call,
) -> Result<bool, Error> {
    if current(&mut state.steps, &call.call_id).is_none() {
        new_step(&mut state.steps, call);
    }
    let step = current(&mut state.steps, &call.call_id).expect("selected Tool step");
    match step.stage.as_deref() {
        Some("before" | "call" | "after") => {
            step.status = if step.waits.values().any(|wait| wait.status == "waiting") {
                "waiting"
            } else {
                "running"
            }
            .into();
            step.error = None;
            Ok(false)
        }
        Some("eval" | "complete") => {
            if step.stage.as_deref() == Some("complete") && step.status == "failed" {
                let error = step
                    .error
                    .clone()
                    .ok_or_else(|| invalid_checkpoint("failed Tool step has no error"))?;
                state
                    .tool_results
                    .insert(index, tool_error_message(call, error));
                return Ok(true);
            }
            let business = step
                .message
                .clone()
                .ok_or_else(|| invalid_checkpoint("Tool step lost its completed result"))?;
            let effective = step
                .call
                .clone()
                .ok_or_else(|| invalid_checkpoint("Tool step lost its actual arguments"))?;
            state
                .tool_results
                .insert(index, tool_success_message(&effective, business.clone()));
            if step.stage.as_deref() == Some("complete") {
                if let Some(evaluation) = &step.eval {
                    state
                        .evaluations
                        .insert(call.call_id.clone(), evaluation.clone());
                }
            } else {
                step.status = "running".into();
                step.error = None;
                let dispatch = state.dispatch.clone();
                let cancellation = state.cancellation.clone();
                let slots = state.tool_slots.clone();
                state.active_tools.push(Box::pin(async move {
                    let result = cancellable(
                        async {
                            let _permit = slots
                                .acquire_owned()
                                .await
                                .map_err(|_| Error::new("CANCELLED", "tool capacity closed"))?;
                            dispatch
                                .evaluate(&effective, &business, &cancellation)
                                .await
                        },
                        &cancellation,
                    )
                    .await;
                    ToolProgress::Evaluated(effective.call_id, result)
                }));
            }
            Ok(true)
        }
        _ => Err(invalid_checkpoint("invalid active Tool lifecycle stage")),
    }
}
