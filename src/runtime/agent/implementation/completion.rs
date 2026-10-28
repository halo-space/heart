//! Native owner binding to the existing Middleware/shared lifecycle driver.
use super::hooks::{Hooks, Stage};
use crate::runtime::agent::{completion::State, model_steps, node::Step};
use crate::{Cancellation, Error, Message, event::Event};

pub(super) async fn model(
    hooks: &Hooks,
    mut saved: Step,
    mut commit: impl FnMut(Step) -> Result<(), Error>,
    cancellation: &Cancellation,
) -> Result<Event, Error> {
    let message = saved
        .message
        .clone()
        .ok_or_else(|| Error::new("INVALID_STATE", "missing Model result"))?;
    let result = Box::pin(hooks.after_from(
        Stage::Model,
        message,
        saved.idx.unwrap_or(0),
        |idx, message| {
            saved.idx = Some(idx);
            saved.message = Some(message);
            commit(saved.clone())
        },
        cancellation,
    ))
    .await;
    match result {
        Ok(message) => {
            saved.message = Some(message);
            saved.stage = Some("committed".into());
            saved.status = "succeeded".into();
            saved.error = None;
            commit(saved.clone())?;
            model_steps::event(&saved)
        }
        Err(error) => {
            model_steps::stopped(std::slice::from_mut(&mut saved), &error);
            commit(saved)?;
            Err(error)
        }
    }
}

pub(super) async fn agent(
    hooks: &Hooks,
    mut saved: State,
    mut commit: impl FnMut(State) -> Result<(), Error>,
    cancellation: &Cancellation,
) -> Result<Event, Error> {
    if saved.stage == "after" {
        let message = Box::pin(hooks.after_from(
            Stage::Execute,
            saved.message.clone(),
            saved.idx,
            |idx, message| {
                saved.idx = idx;
                saved.message = message;
                commit(saved.clone())
            },
            cancellation,
        ))
        .await?;
        saved.message = message;
        saved.stage = "complete".into();
        commit(saved.clone())?;
    }
    Ok(Event::Complete {
        usage: saved.message.usage.clone(),
        message: saved.message,
        finish_reason: saved.finish_reason,
    })
}

pub(super) fn has_calls(message: &Message) -> bool {
    message
        .content
        .iter()
        .any(|part| part.r#type == "tool_call")
}

/// Cancellation observes the same acknowledged result as restoration, not an
/// unfinished middleware working copy or the stale pre-transform raw output.
pub(super) fn partial(completion: Option<&State>, steps: &[Step]) -> Option<Message> {
    completion
        .filter(|state| state.pending())
        .map(|state| state.message.clone())
        .or_else(|| model_steps::pending(steps).and_then(|step| step.message.clone()))
}
