//! Tool lifecycle facts. Futures/permits are transient; saved Steps own the
//! effective Call, completed business result and completed eval feedback.
use crate::runtime::agent::checkpoint::invalid as invalid_checkpoint;
use crate::{
    Error, Message, Messages,
    message::Role,
    runtime::agent::node::Step,
    tool::{Call, Evaluation as ToolEvaluation},
};
use serde_json::Value;
use std::collections::BTreeSet;

pub(crate) fn active(step: &Step) -> bool {
    step.kind == "tool"
        && matches!(
            step.stage.as_deref(),
            Some("before" | "call" | "after" | "eval" | "complete")
        )
}

pub(crate) fn current<'a>(steps: &'a mut [Step], id: &str) -> Option<&'a mut Step> {
    steps
        .iter_mut()
        .rev()
        .find(|step| active(step) && step.call_id.as_deref() == Some(id))
}

pub(crate) fn pending(steps: &[Step]) -> bool {
    steps.iter().any(active)
}

pub(crate) fn new_step(steps: &mut Vec<Step>, call: &Call) {
    steps.push(Step {
        id: format!("tool_{}", steps.len()),
        kind: "tool".into(),
        status: "running".into(),
        call_id: Some(call.call_id.clone()),
        call: Some(call.clone()),
        stage: Some("before".into()),
        ..Step::default()
    });
}

pub(crate) fn selected(steps: &mut Vec<Step>, calls: &[Call]) -> Result<(), Error> {
    let mut ids = BTreeSet::new();
    if calls.iter().any(|call| !ids.insert(&call.call_id)) {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "model selected duplicate Call identities in one batch",
        ));
    }
    for call in calls.iter().filter(|call| {
        call.name != "plan" && call.name != "agent_spawn" && call.name != "team_assign"
    }) {
        new_step(steps, call);
    }
    Ok(())
}

pub(crate) fn prepared(steps: &mut [Step], call: &Call) {
    if let Some(step) = current(steps, &call.call_id) {
        step.call = Some(call.clone());
        step.stage = Some("call".into());
        step.status = "running".into();
        step.error = None;
    }
}

pub(crate) fn returned(steps: &mut [Step], call: &Call, message: &Message) {
    if let Some(step) = current(steps, &call.call_id) {
        step.call = Some(call.clone());
        step.message = Some(crate::runtime::observation::original_message(message));
        step.stage = Some("after".into());
        step.status = "running".into();
        step.error = None;
    }
}

pub(crate) fn called(steps: &mut [Step], call: &Call, message: &Message) {
    if let Some(step) = current(steps, &call.call_id) {
        step.call = Some(call.clone());
        step.message = Some(crate::runtime::observation::original_message(message));
        step.stage = Some("eval".into());
        step.status = "running".into();
        step.error = None;
    }
}

pub(crate) fn evaluated(steps: &mut [Step], id: &str, evaluation: &ToolEvaluation) {
    if let Some(step) = current(steps, id) {
        step.eval = Some(evaluation.clone());
        if let Some(feedback) = step.eval.as_mut().and_then(|eval| eval.feedback.as_mut()) {
            for message in &mut feedback.0 {
                crate::runtime::observation::clear_source(message);
            }
        }
        step.stage = Some("complete".into());
        step.status = "succeeded".into();
        step.error = None;
    }
}

pub(crate) fn stopped(steps: &mut [Step], id: &str, error: &Error) {
    if let Some(step) = current(steps, id) {
        step.status = match error.code.as_str() {
            "WAITING" => "waiting",
            "CANCELLED" => "cancelled",
            _ => "failed",
        }
        .into();
        step.error = (error.code != "WAITING").then(|| error.clone());
    }
}

pub(crate) fn failed_message(steps: &mut [Step], id: &str, message: &Message) {
    if let Some(step) = current(steps, id) {
        // after can fail after call has already succeeded. Keep that result;
        // Step.error remains the source of the direct model error context.
        if step.message.is_none() {
            step.message = Some(crate::runtime::observation::original_message(message));
        }
        step.stage = Some("complete".into());
    }
}

pub(crate) fn committed(steps: &mut [Step]) {
    for step in steps
        .iter_mut()
        .filter(|step| step.kind == "tool" && step.stage.as_deref() == Some("complete"))
    {
        step.stage = Some("committed".into());
    }
}

pub(crate) fn cancel_pending(steps: &mut [Step]) {
    for step in steps.iter_mut().filter(|step| {
        step.kind == "tool"
            && matches!(
                step.stage.as_deref(),
                Some("before" | "call" | "after" | "eval")
            )
            && matches!(step.status.as_str(), "running" | "waiting")
    }) {
        step.status = "cancelled".into();
    }
}

pub(crate) fn resume_calls(
    steps: &[Step],
    messages: &mut Messages,
) -> Result<Vec<(usize, Call)>, Error> {
    for step in steps {
        for wait in step.waits.values() {
            if wait.kind.trim().is_empty()
                || !matches!(wait.status.as_str(), "waiting" | "resolved")
            {
                return Err(invalid_checkpoint("invalid Agent Step waiting record"));
            }
        }
    }
    for step in steps
        .iter()
        .filter(|step| step.kind == "tool" && step.stage.is_some())
    {
        if !matches!(
            step.stage.as_deref(),
            Some("before" | "call" | "after" | "eval" | "complete" | "committed")
        ) {
            return Err(invalid_checkpoint("unknown Tool lifecycle stage"));
        }
    }
    // The current model Complete supplies Call order only. It is never
    // replayed, and execution stage/success are read exclusively from Steps.
    let batch = messages.0.iter().rposition(|message| {
        message.role == Role::Assistant
            && message
                .content
                .iter()
                .any(|part| part.r#type == "tool_call")
    });
    let mut pending = Vec::new();
    let mut ids = BTreeSet::new();
    for step in steps.iter().filter(|step| active(step)) {
        let call = step
            .call
            .as_ref()
            .ok_or_else(|| invalid_checkpoint("Tool step has no actual Call"))?;
        if step.call_id.as_deref() != Some(&call.call_id) || !ids.insert(call.call_id.clone()) {
            return Err(invalid_checkpoint(
                "invalid or duplicate active Tool step identity",
            ));
        }
        let message = batch
            .and_then(|index| messages.0.get(index))
            .ok_or_else(|| invalid_checkpoint("Tool steps have no selecting model Complete"))?;
        let indexes = message
            .content
            .iter()
            .filter(|part| part.r#type == "tool_call")
            .enumerate()
            .filter(|(_, part)| {
                part.data.get("call_id").and_then(Value::as_str) == Some(&call.call_id)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if indexes.len() != 1 {
            return Err(invalid_checkpoint(
                "Tool step does not match one selected Call",
            ));
        }
        let selected = message
            .content
            .iter()
            .filter(|part| part.r#type == "tool_call")
            .nth(indexes[0])
            .expect("validated selected Call index");
        if selected.data.get("name").and_then(Value::as_str) != Some(&call.name) {
            return Err(invalid_checkpoint(
                "Tool step changed the selected capability identity",
            ));
        }
        match step.stage.as_deref() {
            Some("before" | "call")
                if step.message.is_none()
                    && step.eval.is_none()
                    && matches!(step.status.as_str(), "running" | "waiting") => {}
            Some("after" | "eval")
                if step.message.is_some()
                    && step.eval.is_none()
                    && matches!(step.status.as_str(), "running" | "waiting") => {}
            Some("complete")
                if step.message.is_some()
                    && ((step.status == "succeeded" && step.eval.is_some())
                        || (step.status == "failed" && step.error.is_some())) => {}
            _ => return Err(invalid_checkpoint("inconsistent Tool lifecycle stage")),
        }
        pending.push((indexes[0], call.clone()));
    }
    if let Some(batch) = batch {
        let mut index = 0;
        messages.0.retain(|message| {
            let keep = index <= batch
                || !message
                    .metadata
                    .get("call_id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| ids.contains(id));
            index += 1;
            keep
        });
    }
    pending.sort_by_key(|(index, _)| *index);
    Ok(pending)
}
