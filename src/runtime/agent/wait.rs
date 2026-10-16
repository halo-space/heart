//! Ordered locators into the actual Agent scene, never a copy of a question.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Error, runtime::agent::node::Step};

/// Existing execution coordinates of one containing node, outermost first.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Node {
    pub node_id: String,
    pub exec_id: i64,
    /// Current Attempt index, not a scheduling round or Step index.
    pub idx: usize,
}

/// One unresolved human-feedback location in State.waits.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Wait {
    /// Empty for a root Step; no synthetic root execution is required.
    pub nodes: Vec<Node>,
    /// The actual Step.id, not its position in the steps array.
    pub step: String,
}

fn invalid(message: &str) -> Error {
    Error::new("INVALID_STATE", message)
}

fn active(value: &Value) -> Result<(), Error> {
    if matches!(value["status"].as_str(), Some("running" | "waiting")) {
        Ok(())
    } else {
        Err(invalid("waiting locator references an inactive execution"))
    }
}

fn current_attempt(execution: &Value) -> Result<&Value, Error> {
    let attempts = execution["attempts"]
        .as_array()
        .ok_or_else(|| invalid("waiting locator execution has no attempts"))?;
    let attempt = attempts
        .last()
        .ok_or_else(|| invalid("waiting locator execution has no current attempt"))?;
    if attempt["idx"].as_u64() != Some((attempts.len() - 1) as u64) {
        return Err(invalid("waiting locator current attempt index is invalid"));
    }
    active(attempt)?;
    Ok(attempt)
}

/// Resolve only along existing containment; never search history or siblings
/// for a matching exec/Step after the specified position has become stale.
fn scene_node<'a>(scene: &'a Value, id: &str) -> Result<Option<&'a Value>, Error> {
    check_scene(scene)?;
    let planned = scene.get("nodes").and_then(|nodes| nodes.get(id));
    let root = scene.get("execution").filter(|value| !value.is_null());
    let direct = root
        .and_then(|execution| execution["attempts"].as_array())
        .and_then(|attempts| attempts.last())
        .and_then(|attempt| attempt.get("nodes"))
        .and_then(|nodes| nodes.get(id));
    match (planned, direct) {
        (Some(_), Some(_)) => Err(invalid("waiting locator node ownership is ambiguous")),
        (None, Some(node)) => {
            current_attempt(root.expect("direct node has a root execution"))?;
            Ok(Some(node))
        }
        (planned, _) => Ok(planned),
    }
}

fn check_scene(scene: &Value) -> Result<(), Error> {
    if scene["type"] != "agent" {
        return Err(invalid("waiting locator child scene is not Agent state"));
    }
    if !matches!(
        scene["status"].as_str(),
        Some("running" | "waiting" | "paused")
    ) {
        return Err(invalid("waiting locator Agent scene is not active"));
    }
    Ok(())
}

pub(crate) fn scene_step<'a>(
    scene: &'a Value,
    nodes: &[Node],
    id: &str,
) -> Result<&'a Value, Error> {
    check_scene(scene)?;
    match nodes.split_first() {
        None => step(scene, id),
        Some((coordinate, tail)) => {
            let node = scene_node(scene, &coordinate.node_id)?
                .ok_or_else(|| invalid("waiting locator node does not exist"))?;
            node_step(node, coordinate, tail, id)
        }
    }
}

fn node_step<'a>(
    node: &'a Value,
    coordinate: &Node,
    tail: &[Node],
    id: &str,
) -> Result<&'a Value, Error> {
    active(node)?;
    let execution = &node["execution"];
    if execution["exec_id"].as_i64() != Some(coordinate.exec_id) {
        return Err(invalid("waiting locator execution identity is stale"));
    }
    let attempt = current_attempt(execution)?;
    if attempt["idx"].as_u64() != Some(coordinate.idx as u64) {
        return Err(invalid("waiting locator attempt is not current"));
    }
    let child = attempt.get("agent").filter(|value| !value.is_null());
    match tail.split_first() {
        None => match child {
            Some(scene) => scene_step(scene, &[], id),
            None => step(attempt, id),
        },
        Some((coordinate, rest)) => {
            let direct = attempt
                .get("nodes")
                .and_then(|nodes| nodes.get(&coordinate.node_id));
            let inside = child
                .map(|scene| scene_node(scene, &coordinate.node_id))
                .transpose()?
                .flatten();
            match (direct, inside) {
                (Some(_), Some(_)) => Err(invalid("waiting locator node ownership is ambiguous")),
                (Some(node), None) | (None, Some(node)) => node_step(node, coordinate, rest, id),
                (None, None) => Err(invalid("waiting locator child node does not exist")),
            }
        }
    }
}

fn step<'a>(owner: &'a Value, id: &str) -> Result<&'a Value, Error> {
    let mut matching = owner["steps"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|step| step["id"].as_str() == Some(id));
    let step = matching
        .next()
        .ok_or_else(|| invalid("waiting locator Step does not exist"))?;
    if matching.next().is_some() {
        return Err(invalid("waiting locator Step identity is ambiguous"));
    }
    Ok(step)
}

// Check coverage without rebuilding/reordering the FIFO list or traversing
// business JSON. Only actual current Step containers belong to this walk.
fn covered_steps(owner: &Value, covered: &[&Value]) -> Result<(), Error> {
    for step in owner["steps"].as_array().into_iter().flatten() {
        // A cancelled operation retains its unresolved question as history,
        // but it is no longer an active external feedback consumer.
        if step["status"] == "cancelled" {
            continue;
        }
        let unresolved = step["waits"]
            .as_object()
            .into_iter()
            .flat_map(|waits| waits.values())
            .any(|record| record["status"] == "waiting");
        if unresolved && !covered.iter().any(|saved| std::ptr::eq(*saved, step)) {
            return Err(invalid("Agent waiting question has no active locator"));
        }
    }
    Ok(())
}

fn covered_nodes(nodes: Option<&Value>, covered: &[&Value]) -> Result<(), Error> {
    for node in nodes
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|nodes| nodes.values())
    {
        if !matches!(node["status"].as_str(), Some("running" | "waiting")) {
            continue;
        }
        let attempt = current_attempt(&node["execution"])?;
        if let Some(scene) = attempt.get("agent").filter(|value| !value.is_null()) {
            covered_scene(scene, covered)?;
        } else {
            covered_steps(attempt, covered)?;
        }
        covered_nodes(attempt.get("nodes"), covered)?;
    }
    Ok(())
}

fn covered_scene(scene: &Value, covered: &[&Value]) -> Result<(), Error> {
    covered_steps(scene, covered)?;
    covered_nodes(scene.get("nodes"), covered)?;
    if let Some(attempt) = scene
        .get("execution")
        .and_then(|execution| execution["attempts"].as_array())
        .and_then(|attempts| attempts.last())
        .filter(|attempt| matches!(attempt["status"].as_str(), Some("running" | "waiting")))
    {
        covered_nodes(attempt.get("nodes"), covered)?;
    }
    Ok(())
}

/// Restore-time validation only. It neither consumes feedback nor grants
/// approval, and preserves the stored FIFO ordering without rebuilding it.
pub(crate) fn validate(scene: &Value) -> Result<bool, Error> {
    let waits: Vec<Wait> = scene
        .get("waits")
        .map(|waits| serde_json::from_value(waits.clone()))
        .transpose()
        .map_err(|_| invalid("invalid Agent waiting locator list"))?
        .unwrap_or_default();
    let has_waits = !waits.is_empty();
    let mut positions = BTreeSet::new();
    let mut replies = BTreeSet::new();
    let mut covered = Vec::new();
    for wait in waits {
        if wait.step.trim().is_empty()
            || wait.nodes.iter().any(|node| node.node_id.trim().is_empty())
        {
            return Err(invalid("Agent waiting locator identity must not be empty"));
        }
        if !positions.insert(wait.clone()) {
            return Err(invalid("duplicate Agent waiting locator"));
        }
        let facts = scene_step(scene, &wait.nodes, &wait.step)?;
        covered.push(facts);
        let step: Step = serde_json::from_value(facts.clone())
            .map_err(|_| invalid("waiting locator has invalid Step facts"))?;
        if step.status != "waiting" {
            return Err(invalid("waiting locator Step is not waiting"));
        }
        let mut unresolved = false;
        for (reply_id, record) in step.waits {
            if record.kind.trim().is_empty()
                || !matches!(record.status.as_str(), "waiting" | "resolved")
            {
                return Err(invalid("invalid Agent Step waiting record"));
            }
            if record.status == "waiting" {
                unresolved = true;
                if !replies.insert(reply_id) {
                    return Err(invalid("Agent waiting reply identity is ambiguous"));
                }
            }
        }
        if !unresolved {
            return Err(invalid("waiting locator has no unresolved question"));
        }
    }
    covered_scene(scene, &covered)?;
    Ok(has_waits)
}

/// A forwarded question must be backed by its own saved waiting facts.
pub(crate) fn question(scene: &Value, id: i64) -> Result<bool, Error> {
    if !validate(scene)? {
        return Ok(false);
    }
    let waits: Vec<Wait> = serde_json::from_value(scene["waits"].clone())
        .map_err(|_| invalid("invalid Agent waiting locator list"))?;
    for wait in waits {
        if scene_step(scene, &wait.nodes, &wait.step)?["waits"][id.to_string()]["status"]
            == "waiting"
        {
            return Ok(true);
        }
    }
    Ok(false)
}
