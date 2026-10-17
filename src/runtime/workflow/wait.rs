//! Ordered human-feedback locators into native Workflow/Agent scenes.
//! Ordinary Node pause records are not human questions and stay in Execution.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    Error,
    runtime::workflow::{graph::State, node},
};

/// Existing containing execution coordinates, outermost first.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Node {
    pub node_id: String,
    pub exec_id: i64,
    /// Current Attempt index, not the Workflow scheduling step.
    pub idx: usize,
}

/// Locator only: question, feedback and Call stay on the actual Agent Step.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Wait {
    pub nodes: Vec<Node>,
    pub step: String,
}

fn invalid(message: &str) -> Error {
    Error::new("INVALID_STATE", message)
}

/// Validate the native Agent scenes and collect membership, not processing
/// order. Only the existing ordered locator list defines cross-branch order.
fn membership(state: &State) -> Result<BTreeSet<Wait>, Error> {
    let mut result = BTreeSet::new();
    for (node_id, state_node) in &state.nodes {
        for execution in state_node.executions.values().filter(|execution| {
            execution.idx == state.step
                && matches!(
                    execution.status,
                    node::ExecutionStatus::Running | node::ExecutionStatus::Waiting
                )
        }) {
            let Some(attempt) = execution.attempts.last() else {
                continue;
            };
            if !matches!(
                attempt.status,
                node::AttemptStatus::Running | node::AttemptStatus::Waiting
            ) {
                continue;
            }
            if attempt.idx as usize != execution.attempts.len() - 1 {
                return Err(invalid("Workflow current Attempt index is invalid"));
            }
            if attempt.agent.is_some() && attempt.graph.is_some() {
                return Err(invalid(
                    "Workflow waiting execution has ambiguous native owners",
                ));
            }
            let owner = Node {
                node_id: node_id.clone(),
                exec_id: execution.exec_id,
                idx: attempt.idx as usize,
            };
            let local = if let Some(scene) = &attempt.agent {
                crate::runtime::agent::wait::validate(scene)?;
                let waits: Vec<crate::runtime::agent::wait::Wait> = serde_json::from_value(
                    scene
                        .get("waits")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!([])),
                )
                .map_err(|_| invalid("invalid native Agent wait locators"))?;
                waits
                    .into_iter()
                    .map(|wait| Wait {
                        nodes: wait
                            .nodes
                            .into_iter()
                            .map(|node| Node {
                                node_id: node.node_id,
                                exec_id: node.exec_id,
                                idx: node.idx,
                            })
                            .collect(),
                        step: wait.step,
                    })
                    .collect::<Vec<_>>()
            } else if let Some(child) = &attempt.graph {
                validate(child)?;
                child.waits.clone()
            } else {
                Vec::new()
            };
            for mut wait in local {
                wait.nodes.insert(0, owner.clone());
                if !result.insert(wait) {
                    return Err(invalid("duplicate Workflow wait locator"));
                }
            }
        }
    }
    Ok(result)
}

/// Resolve every level against the current execution, never old Attempts,
/// arbitrary business JSON, or a reply ID guessed from a node name.
fn step<'a>(state: &'a State, wait: &Wait) -> Result<&'a Value, Error> {
    let (head, rest) = wait
        .nodes
        .split_first()
        .ok_or_else(|| invalid("Workflow wait has no containing node"))?;
    let execution = state
        .nodes
        .get(&head.node_id)
        .and_then(|node| node.executions.get(&head.exec_id))
        .filter(|execution| {
            execution.exec_id == head.exec_id
                && execution.idx == state.step
                && matches!(
                    execution.status,
                    node::ExecutionStatus::Running | node::ExecutionStatus::Waiting
                )
        })
        .ok_or_else(|| invalid("Workflow wait references an inactive execution"))?;
    let attempt = execution
        .attempts
        .last()
        .filter(|attempt| {
            attempt.idx as usize == head.idx
                && head.idx.checked_add(1) == Some(execution.attempts.len())
                && matches!(
                    attempt.status,
                    node::AttemptStatus::Running | node::AttemptStatus::Waiting
                )
        })
        .ok_or_else(|| invalid("Workflow wait references a stale Attempt"))?;
    if let Some(scene) = &attempt.agent {
        let nodes = rest
            .iter()
            .map(|node| crate::runtime::agent::wait::Node {
                node_id: node.node_id.clone(),
                exec_id: node.exec_id,
                idx: node.idx,
            })
            .collect::<Vec<_>>();
        crate::runtime::agent::wait::scene_step(scene, &nodes, &wait.step)
    } else if let Some(child) = &attempt.graph {
        step(
            child,
            &Wait {
                nodes: rest.to_vec(),
                step: wait.step.clone(),
            },
        )
    } else {
        Err(invalid("Workflow human wait has no native Agent Step"))
    }
}

/// Ordered IDs are derived from the validated actual Step records, never
/// serialized as another source of truth alongside the locators.
pub(crate) fn ids(state: &State) -> Result<Vec<i64>, Error> {
    let expected = membership(state)?;
    let actual = state.waits.iter().cloned().collect::<BTreeSet<_>>();
    if actual.len() != state.waits.len() || actual != expected {
        return Err(invalid(
            "Workflow wait locators do not match native waiting Steps",
        ));
    }
    // A parent may interleave independent owners, but cannot reverse the
    // relative order already recorded by one child graph or Agent scene.
    let owners = state
        .waits
        .iter()
        .filter_map(|wait| wait.nodes.first().cloned())
        .collect::<BTreeSet<_>>();
    for owner in owners {
        let attempt = state
            .nodes
            .get(&owner.node_id)
            .and_then(|node| node.executions.get(&owner.exec_id))
            .and_then(|execution| execution.attempts.last())
            .ok_or_else(|| invalid("Workflow waiting owner is missing"))?;
        let local: Vec<Wait> = if let Some(scene) = &attempt.agent {
            serde_json::from_value(scene["waits"].clone())
                .map_err(|_| invalid("invalid native Agent waits"))?
        } else if let Some(child) = &attempt.graph {
            child.waits.clone()
        } else {
            Vec::new()
        };
        let expected = local
            .into_iter()
            .map(|mut wait| {
                wait.nodes.insert(0, owner.clone());
                wait
            })
            .collect::<Vec<_>>();
        let observed = state
            .waits
            .iter()
            .filter(|wait| wait.nodes.first() == Some(&owner))
            .cloned()
            .collect::<Vec<_>>();
        if observed != expected {
            return Err(invalid("Workflow and native child waiting orders disagree"));
        }
    }
    let mut seen = BTreeSet::new();
    state
        .waits
        .iter()
        .map(|wait| {
            let step: crate::runtime::agent::node::Step =
                serde_json::from_value(step(state, wait)?.clone())
                    .map_err(|_| invalid("invalid native waiting Step"))?;
            let records = step
                .waits
                .iter()
                .filter(|(_, record)| record.status == "waiting")
                .collect::<Vec<_>>();
            if records.len() != 1 {
                return Err(invalid("ambiguous Workflow waiting Step"));
            }
            let (&id, record) = records[0];
            record.validate()?;
            if record
                .question
                .metadata
                .get("chat_id")
                .and_then(Value::as_i64)
                != Some(state.chat_id)
                || record
                    .question
                    .metadata
                    .get("reply_id")
                    .and_then(Value::as_i64)
                    != Some(id)
                || !seen.insert(id)
            {
                return Err(invalid("invalid or duplicate Workflow question identity"));
            }
            Ok(id)
        })
        .collect()
}

pub(crate) fn validate(state: &State) -> Result<(), Error> {
    ids(state).map(|_| ())
}

/// Called only after an actual native progress handoff. Preserve the global
/// arrival order of existing entries; append newly observed entries in the
/// owning Agent/child graph's already recorded order, never map order.
pub(crate) fn update(state: &mut State, node_id: &str, exec_id: i64) -> Result<(), Error> {
    let expected = membership(state)?;
    state.waits.retain(|wait| expected.contains(wait));
    let execution = state
        .nodes
        .get(node_id)
        .and_then(|node| node.executions.get(&exec_id))
        .ok_or_else(|| invalid("Workflow progress has no execution"))?;
    let attempt = execution
        .attempts
        .last()
        .ok_or_else(|| invalid("Workflow progress has no Attempt"))?;
    let local: Vec<Wait> = if let Some(scene) = &attempt.agent {
        serde_json::from_value(
            scene
                .get("waits")
                .cloned()
                .unwrap_or_else(|| serde_json::json!([])),
        )
        .map_err(|_| invalid("invalid native Agent waits"))?
    } else if let Some(child) = &attempt.graph {
        child.waits.clone()
    } else {
        Vec::new()
    };
    let owner = Node {
        node_id: node_id.into(),
        exec_id,
        idx: attempt.idx as usize,
    };
    for mut wait in local {
        wait.nodes.insert(0, owner.clone());
        if expected.contains(&wait) && !state.waits.contains(&wait) {
            state.waits.push(wait);
        }
    }
    validate(state)
}
