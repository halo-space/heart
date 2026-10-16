use super::*;
pub(super) use crate::harness::tool_steps::{
    called, committed, evaluated, failed_message, pending, selected, stopped,
};
use crate::harness::tool_steps::{current, new_step};

pub(super) fn restore<M: Chat>(state: &mut ReActState<M>) -> Result<(), Error> {
    for step in &state.steps {
        if step.stage.as_deref() == Some("eval")
            && let Some(call) = &step.call
        {
            state.toolkit.validate(call)?;
        }
    }
    let pending = crate::harness::tool_steps::resume_calls(&state.steps, &mut state.messages)?;
    state.pending_calls.extend(pending);
    if state.is_plan_task {
        let pending = crate::harness::subagent::resume_calls(
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
        Some("call") => {
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
                let toolkit = state.toolkit.clone();
                let cancellation = state.cancellation.clone();
                let slots = state.tool_slots.clone();
                state.active_tools.push(Box::pin(async move {
                    let result = cancellable(
                        async {
                            let _permit = slots
                                .acquire_owned()
                                .await
                                .map_err(|_| Error::new("CANCELLED", "tool capacity closed"))?;
                            toolkit.evaluate(&effective, &business, &cancellation).await
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
