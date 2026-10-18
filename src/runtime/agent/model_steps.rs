//! Actual Model calls retain their native input and output continuations.
use super::node::{Status, Step, StepAttempt};
use crate::{Error, Message, event::Event, model::token::Usage};

pub(crate) fn pending_input(steps: &[Step]) -> Option<&Step> {
    steps.iter().rev().find(|step| {
        step.kind == "model" && matches!(step.stage.as_deref(), Some("before" | "call"))
    })
}

/// Provider-origin ordinary WAITING accepts new external input. It is not an
/// unfinished input Hook; the next actual attempt must recheck that new input.
pub(crate) fn external_wait(steps: &[Step]) -> bool {
    pending_input(steps).is_some_and(|step| {
        step.stage.as_deref() == Some("call")
            && step
                .attempts
                .last()
                .is_some_and(|attempt| attempt.status == Status::Waiting)
    })
}

pub(crate) fn continuing(steps: &[Step]) -> bool {
    pending(steps).is_some() || (pending_input(steps).is_some() && !external_wait(steps))
}

pub(crate) fn input(steps: &mut Vec<Step>, request: crate::model::chat::Request) -> Step {
    if external_wait(steps) {
        let step = steps
            .iter_mut()
            .rev()
            .find(|step| step.kind == "model" && step.stage.as_deref() == Some("call"))
            .expect("actual waiting Model call");
        step.stage = Some("before".into());
        step.idx = Some(0);
        step.request = Some(Box::new(request));
        step.status = "running".into();
        step.error = None;
        return step.clone();
    }
    if let Some(saved) = pending_input(steps) {
        return saved.clone();
    }
    let step = Step {
        id: format!("model_{}", steps.len()),
        kind: "model".into(),
        status: "running".into(),
        stage: Some("before".into()),
        idx: Some(0),
        request: Some(Box::new(request)),
        ..Default::default()
    };
    steps.push(step.clone());
    step
}

/// Enter only the Provider call, not a successful input Hook. A later fresh
/// dispatch retains an earlier unsuccessful attempt rather than overwriting it.
pub(crate) fn calling(steps: &mut [Step]) {
    if let Some(step) = steps
        .iter_mut()
        .rev()
        .find(|step| step.kind == "model" && step.stage.as_deref() == Some("call"))
    {
        if let Some(previous) = step.attempts.last_mut()
            && previous.status == Status::Running
        {
            previous.status = Status::Waiting;
        }
        step.attempts.push(StepAttempt {
            status: Status::Running,
            ..Default::default()
        });
        step.status = "running".into();
        step.error = None;
    }
}

pub(crate) fn pending(steps: &[Step]) -> Option<&Step> {
    steps
        .iter()
        .rev()
        .find(|step| step.kind == "model" && step.stage.as_deref() == Some("after"))
}
pub(crate) fn returned(
    steps: &mut Vec<Step>,
    request: Option<Box<crate::model::chat::Request>>,
    message: Message,
    usage: Option<Usage>,
    finish_reason: Option<String>,
) {
    let previous = pending_input(steps).map(|step| step.id.clone());
    let mut step = Step {
        id: format!("model_{}", steps.len()),
        kind: "model".into(),
        status: "running".into(),
        stage: Some("after".into()),
        idx: Some(0),
        request,
        message: Some(message),
        attempts: vec![StepAttempt {
            status: Status::Succeeded,
            usage,
            finish_reason,
            ..Default::default()
        }],
        ..Default::default()
    };
    if let Some(id) = previous {
        let owner = steps
            .iter_mut()
            .find(|step| step.id == id)
            .expect("Model owner");
        step.id = owner.id.clone();
        let succeeded = step.attempts.pop().expect("successful model attempt");
        step.attempts = std::mem::take(&mut owner.attempts);
        if let Some(attempt) = step.attempts.last_mut()
            && attempt.status == Status::Running
        {
            *attempt = succeeded;
        } else {
            step.attempts.push(succeeded);
        }
        *owner = step;
    } else {
        steps.push(step);
    }
}
pub(crate) fn validate(steps: &[Step], count: usize) -> Result<(), Error> {
    let mut active = 0;
    let mut ids = std::collections::BTreeSet::new();
    if steps.iter().any(|step| {
        step.id.is_empty()
            || !ids.insert(&step.id)
            || (step.kind != "model" && step.attempts.iter().any(|a| a.message.is_some()))
    }) {
        return Err(invalid());
    }
    for step in steps.iter().filter(|step| step.kind == "model") {
        if step.attempts.iter().any(|attempt| {
            attempt.message.as_ref().is_some_and(|message| {
                !matches!(
                    attempt.status,
                    Status::Waiting | Status::Failed | Status::Cancelled
                ) || message.usage != attempt.usage
            })
        }) {
            return Err(invalid());
        }
        if matches!(step.stage.as_deref(), Some("before" | "call")) {
            if step.request.is_none()
                || step.message.is_some()
                || step.idx.is_none_or(|idx| idx > count)
                || (step.stage.as_deref() == Some("call") && step.idx != Some(count))
                || (step.stage.as_deref() == Some("before")
                    && step.attempts.iter().any(|attempt| {
                        !matches!(
                            attempt.status,
                            Status::Waiting | Status::Failed | Status::Cancelled
                        )
                    }))
                || step
                    .attempts
                    .iter()
                    .any(|attempt| attempt.status == Status::Succeeded)
                || !matches!(
                    step.status.as_str(),
                    "running" | "waiting" | "failed" | "cancelled"
                )
            {
                return Err(invalid());
            }
            step.request
                .as_ref()
                .expect("Model request")
                .validate()
                .map_err(|_| invalid())?;
            active += 1;
            continue;
        }
        let Some(attempt) = step.attempts.last() else {
            return Err(invalid());
        };
        if !matches!(step.stage.as_deref(), Some("after" | "committed"))
            || step.message.is_none()
            || step.request.is_none()
            || step.idx.is_none_or(|idx| idx > count)
            || attempt.status != Status::Succeeded
            || step
                .message
                .as_ref()
                .and_then(|message| message.usage.as_ref())
                != attempt.usage.as_ref()
            || (step.stage.as_deref() == Some("committed") && step.idx != Some(count))
            || (step.stage.as_deref() == Some("committed") && step.status != "succeeded")
            || (step.stage.as_deref() == Some("after")
                && !matches!(
                    step.status.as_str(),
                    "running" | "waiting" | "failed" | "cancelled"
                ))
        {
            return Err(invalid());
        }
        step.request
            .as_ref()
            .expect("validated request")
            .validate()
            .map_err(|_| invalid())?;
        if step.stage.as_deref() == Some("after") {
            active += 1;
        }
    }
    if active > 1 {
        return Err(invalid());
    }
    Ok(())
}
pub(crate) fn validate_owner(
    steps: &[Step],
    completion: Option<&super::completion::State>,
    count: usize,
) -> Result<(), Error> {
    for step in steps {
        if let Some(permission) = &step.permission {
            if step.kind != "tool" {
                return Err(Error::new(
                    "INVALID_STATE",
                    "permission facts have no Tool owner",
                ));
            }
            permission.validate(
                step.call
                    .as_ref()
                    .ok_or_else(|| Error::new("INVALID_STATE", "permission owner has no Call"))?,
                count,
            )?;
        }
    }
    validate(steps, count)?;
    if let Some(completion) = completion {
        completion.validate(count)?;
        if completion.pending() && (pending(steps).is_some() || pending_input(steps).is_some()) {
            return Err(invalid());
        }
        let last = steps
            .iter()
            .rev()
            .find(|step| step.kind == "model")
            .ok_or_else(invalid)?;
        if last.stage.as_deref() != Some("committed")
            || last
                .message
                .as_ref()
                .and_then(|message| message.usage.as_ref())
                != completion.message.usage.as_ref()
        {
            return Err(invalid());
        }
    }
    Ok(())
}
pub(crate) fn refine(steps: &mut [Step], usage: &Usage) {
    if let Some(step) = steps.iter_mut().rev().find(|step| step.kind == "model") {
        if let Some(attempt) = step.attempts.last_mut() {
            attempt.usage = Some(usage.clone());
        }
        if let Some(message) = &mut step.message {
            message.usage = Some(usage.clone());
        }
    }
}

/// Retain only observed unfinished output on its actual call Attempt, never
/// on Step.message (which remains the Provider Complete/after continuation).
pub(crate) fn unfinished(steps: &mut [Step], message: &Message, usage: &Usage) {
    if let Some(step) = steps
        .iter_mut()
        .rev()
        .find(|step| step.kind == "model" && step.stage.as_deref() == Some("call"))
        && let Some(attempt) = step.attempts.last_mut()
    {
        let mut message = message.clone();
        message.usage = Some(usage.clone());
        attempt.message = Some(Box::new(message));
        attempt.usage = Some(usage.clone());
    }
}
fn invalid() -> Error {
    Error::new("INVALID_STATE", "invalid Model lifecycle facts")
}
/// Validate this Agent and its own planned attempts before component work or
/// persistence. Nested child Agents validate with their own middleware config.
pub(crate) fn validate_scene(scene: &serde_json::Value, count: usize) -> Result<(), Error> {
    let steps: Vec<Step> = scene
        .get("steps")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| invalid())?
        .unwrap_or_default();
    let completion: Option<super::completion::State> = scene
        .get("completion")
        .filter(|value| !value.is_null())
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| invalid())?;
    validate_owner(&steps, completion.as_ref(), count)?;
    if let Some(nodes) = scene.get("nodes").and_then(serde_json::Value::as_object) {
        for node in nodes.values() {
            if let Some(attempt) = node
                .get("execution")
                .and_then(|execution| execution.get("attempts"))
                .and_then(serde_json::Value::as_array)
                .and_then(|attempts| attempts.last())
            {
                validate_scene(attempt, count)?;
            }
        }
    }
    Ok(())
}
pub(crate) fn event(step: &Step) -> Result<Event, Error> {
    let attempt = step.attempts.last().ok_or_else(invalid)?;
    Ok(Event::Complete {
        message: step.message.clone().ok_or_else(invalid)?,
        usage: attempt.usage.clone(),
        finish_reason: attempt.finish_reason.clone(),
    })
}
pub(crate) fn stopped(steps: &mut [Step], error: &Error) {
    if let Some(step) = steps.iter_mut().rev().find(|step| {
        step.kind == "model" && matches!(step.stage.as_deref(), Some("before" | "call" | "after"))
    }) {
        step.status = match error.code.as_str() {
            "WAITING" => "waiting",
            "CANCELLED" => "cancelled",
            _ => "failed",
        }
        .into();
        step.error = (error.code != "WAITING").then(|| error.clone());
        if step.stage.as_deref() == Some("call")
            && let Some(attempt) = step.attempts.last_mut()
        {
            attempt.status = match error.code.as_str() {
                "WAITING" => Status::Waiting,
                "CANCELLED" => Status::Cancelled,
                _ => Status::Failed,
            };
            attempt.error = step.error.clone();
        }
    }
}
