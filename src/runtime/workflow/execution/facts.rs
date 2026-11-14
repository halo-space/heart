//! Recovery facts shared by the Workflow execution loop.
//!
//! These helpers only inspect or combine already committed execution facts.
//! They do not advance the graph and do not create a second state owner.

use super::{Error, Input, Messages, Payload, Role, Value, graph, node};
use std::collections::BTreeMap;

pub(super) fn attempt_waiting(attempt: &node::Attempt) -> bool {
    attempt
        .steps
        .iter()
        .any(|step| step.stage.is_some() && step.status == node::status::step::Status::Waiting)
        || attempt
            .agent
            .as_ref()
            .and_then(|scene| scene["waits"].as_array())
            .is_some_and(|waits| !waits.is_empty())
        || attempt.graph.as_ref().is_some_and(|state| {
            !state.waits.is_empty()
                || !state.pauses().is_empty()
                || state
                    .nodes
                    .values()
                    .flat_map(|node| node.executions.values())
                    .any(|execution| execution.attempts.last().is_some_and(attempt_waiting))
        })
}

pub(super) fn attempt_feedback(attempt: &node::Attempt) -> BTreeMap<String, usize> {
    let mut progress = attempt
        .agent
        .as_ref()
        .map(crate::runtime::agent::wait::input_progress)
        .unwrap_or_default();
    if let Some(graph) = &attempt.graph {
        for execution in graph
            .nodes
            .values()
            .flat_map(|node| node.executions.values())
        {
            if execution.idx == graph.step
                && let Some(attempt) = execution.attempts.last()
            {
                progress.extend(attempt_feedback(attempt));
            }
        }
    }
    progress
}

pub(super) fn commit_feedback(
    attempt: &node::Attempt,
    run: &crate::runtime::agent::reply::Handle,
) -> Result<(), Error> {
    if let Some(scene) = &attempt.agent {
        run.commit_scene_feedback(scene)?;
    }
    if let Some(graph) = &attempt.graph {
        for execution in graph
            .nodes
            .values()
            .flat_map(|node| node.executions.values())
        {
            if execution.idx == graph.step
                && let Some(attempt) = execution.attempts.last()
            {
                commit_feedback(attempt, run)?;
            }
        }
    }
    Ok(())
}

pub(super) fn prepend_agent_observations(attempt: &mut node::Attempt, base: &node::Attempt) {
    let mut messages = base.messages.clone();
    messages.0.append(&mut attempt.messages.0);
    attempt.messages = messages;
    let mut steps = base.steps.clone();
    for (index, step) in attempt.steps.iter_mut().enumerate() {
        step.id = format!("complete_{}", base.steps.len() + index + 1);
    }
    steps.append(&mut attempt.steps);
    attempt.steps = steps;
}

/// Search the complete Workflow snapshot for a reply already accepted by any
/// embedded Agent scene. A duplicate is harmless and must not wake a second
/// waiting owner.
pub(super) fn reply_duplicate(state: &graph::State, input: &Input) -> Result<bool, Error> {
    let mut duplicate = false;
    for node in state.nodes.values() {
        for execution in node.executions.values() {
            if let Some(attempt) = execution.attempts.last() {
                if let Some(scene) = &attempt.agent {
                    duplicate |= crate::runtime::agent::reply::reply_duplicate(scene, input)?;
                }
                if let Some(child) = &attempt.graph {
                    duplicate |= reply_duplicate(child, input)?;
                }
            }
        }
    }
    Ok(duplicate)
}

pub(super) fn agent_scene_for_reply_path<'a>(
    state: &'a graph::State,
    path: &[graph::ReplyExecution],
) -> Result<&'a Value, Error> {
    let Some((first, rest)) = path.split_first() else {
        return Err(Error::new("INVALID_STATE", "saved reply path is empty"));
    };
    let execution = state
        .nodes
        .get(&first.node_id)
        .and_then(|node| node.executions.get(&first.exec_id))
        .filter(|execution| execution.idx == state.step)
        .ok_or_else(|| Error::new("INVALID_STATE", "saved reply execution is stale"))?;
    let attempt = execution
        .attempts
        .last()
        .filter(|attempt| attempt.idx == first.idx)
        .ok_or_else(|| Error::new("INVALID_STATE", "saved reply Attempt is stale"))?;
    if rest.is_empty() {
        return attempt
            .agent
            .as_ref()
            .ok_or_else(|| Error::new("INVALID_STATE", "saved reply has no Agent scene"));
    }
    agent_scene_for_reply_path(
        attempt
            .graph
            .as_deref()
            .ok_or_else(|| Error::new("INVALID_STATE", "saved reply Graph scene is missing"))?,
        rest,
    )
}

pub(crate) fn attempt_question(attempt: &node::Attempt, id: i64) -> Result<bool, Error> {
    if let Some(scene) = &attempt.agent
        && scene["waits"]
            .as_array()
            .is_some_and(|waits| !waits.is_empty())
        && crate::runtime::agent::wait::question(scene, id)?
    {
        return Ok(true);
    }
    if let Some(state) = &attempt.graph {
        for execution in state
            .nodes
            .values()
            .flat_map(|node| node.executions.values())
        {
            if matches!(
                execution.status,
                node::status::execution::Status::Running | node::status::execution::Status::Waiting
            ) && let Some(attempt) = execution.attempts.last()
                && attempt_question(attempt, id)?
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Reconstruct successful sub-call facts from the entire logical execution,
/// not just its latest Attempt. An in-flight retry may have no messages yet.
pub(crate) fn messages_to_agent_input(messages: &Messages) -> Input {
    let message = messages
        .as_slice()
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .or_else(|| messages.as_slice().last());
    let Some(message) = message else {
        return Input::default();
    };
    let payload = message
        .content
        .iter()
        .map(|part| Payload {
            r#type: part.r#type.clone(),
            data: serde_json::Value::Object(part.data.clone()),
            mime_type: part
                .data
                .get("mime_type")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
        .collect();
    Input {
        metadata: message.metadata.clone(),
        payload,
    }
}
