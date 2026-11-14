//! Mutable, serializable Workflow execution State.
//!
//! This module stores node executions, waits, replies and aggregate usage. It
//! does not compile definitions or execute nodes.

use super::*;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    #[default]
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Paused,
}

pub use crate::runtime::workflow::wait::Wait;

/// Derived lookup for an ordinary Node pause. The actual record remains in
/// Execution.waits; this is never a parallel persisted index.
#[derive(Clone)]
#[doc(hidden)]
pub struct Pause {
    pub node_id: String,
    pub exec_id: i64,
    pub wait_id: String,
}

/// Serializable Workflow Runtime state for one `chat_id`.
///
/// Node execution data remains nested under `nodes[node_id]`; `waits` is a
/// ordered human-feedback locator list. Ordinary Node pause facts remain in
/// their owning Execution.waits. This type intentionally does not run a
/// scheduler or persist an external Checkpoint.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct State {
    pub chat_id: i64,
    #[serde(rename = "type")]
    pub r#type: String,
    pub definition_id: String,
    pub version: String,
    pub step: u64,
    pub status: Status,
    /// Recomputed aggregate of all model calls committed in this graph state.
    /// It is a query field; routing and retry decisions never depend on it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial: Option<Messages>,
    #[serde(default)]
    pub waits: Vec<Wait>,
    pub nodes: BTreeMap<String, crate::runtime::workflow::node::State>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct ReplyExecution {
    pub node_id: String,
    pub exec_id: i64,
    pub idx: u32,
}

impl State {
    pub fn new(chat_id: i64, definition: &Definition) -> Self {
        let nodes = definition
            .nodes
            .keys()
            .map(|node_id| {
                (
                    node_id.clone(),
                    crate::runtime::workflow::node::State::new(
                        crate::runtime::workflow::node::status::Status::Idle,
                    ),
                )
            })
            .collect();

        Self {
            chat_id,
            r#type: "workflow".into(),
            definition_id: definition.definition_id.clone(),
            version: definition.version.clone(),
            step: 0,
            status: Status::Running,
            usage: None,
            initial: None,
            waits: Vec::new(),
            nodes,
        }
    }

    /// Child graphs share the outer run identity. Check this before any
    /// restored execution is mutated or any component is invoked.
    pub(crate) fn check_identity(&self, chat_id: i64) -> Result<(), Error> {
        if self.chat_id != chat_id || self.r#type != "workflow" {
            return Err(Error::new(
                "CONFLICT",
                "checkpoint identity does not match this Workflow run",
            ));
        }
        for node in self.nodes.values() {
            for execution in node.executions.values() {
                for attempt in &execution.attempts {
                    if let Some(child) = &attempt.graph {
                        child.check_identity(chat_id)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Embedded Agent scenes share the Workflow's top-level run identity.
    /// Normalize that existing identity before publishing or checkpointing
    /// progress, including scenes nested in child Graph attempts.
    pub(crate) fn bind_agent_chat_id(&mut self, chat_id: i64) -> Result<(), Error> {
        for node in self.nodes.values_mut() {
            for execution in node.executions.values_mut() {
                for attempt in &mut execution.attempts {
                    if let Some(scene) = attempt.agent.as_mut()
                        && let Some(scene) = scene.as_object_mut()
                    {
                        if scene
                            .get("chat_id")
                            .and_then(Value::as_i64)
                            .is_some_and(|saved| saved != chat_id)
                        {
                            return Err(Error::new(
                                "CONFLICT",
                                "embedded Agent Checkpoint has a different chat_id",
                            ));
                        }
                        scene.insert("chat_id".into(), chat_id.into());
                    }
                    if let Some(child) = attempt.graph.as_deref_mut() {
                        child.bind_agent_chat_id(chat_id)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Marks unfinished Attempts from a persisted snapshot as waiting.
    /// A queued execution stays Pending until actual admission; normalization
    /// does not claim it ran or consume a retry.
    /// The logical execution remains active so Runtime can reuse its exec_id;
    /// only a later call may continue the same execution. Explicit external
    /// waits are intentionally left untouched.
    pub(crate) fn mark_waiting(&mut self) -> bool {
        let mut changed = false;
        for node in self.nodes.values_mut() {
            for execution in node.executions.values_mut() {
                if !matches!(
                    execution.status,
                    crate::runtime::workflow::node::status::execution::Status::Pending
                        | crate::runtime::workflow::node::status::execution::Status::Running
                ) {
                    continue;
                }
                if let Some(attempt) = execution.attempts.last_mut()
                    && attempt.status
                        == crate::runtime::workflow::node::status::attempt::Status::Running
                {
                    attempt.status =
                        crate::runtime::workflow::node::status::attempt::Status::Waiting;
                    changed = true;
                }
                if let Some(attempt) = execution.attempts.last_mut()
                    && let Some(child) = attempt.graph.as_mut()
                {
                    changed |= child.mark_waiting();
                }
            }
        }
        changed
    }

    /// Locate saved feedback only through active Workflow executions and
    /// their actual embedded Agent/Graph scenes.
    pub(crate) fn reply_paths(&self, reply_id: i64) -> Result<Vec<Vec<ReplyExecution>>, Error> {
        let mut paths = Vec::new();
        for (node_id, node) in &self.nodes {
            for execution in node.executions.values().filter(|execution| {
                execution.idx == self.step
                    && matches!(
                        execution.status,
                        crate::runtime::workflow::node::status::execution::Status::Running
                            | crate::runtime::workflow::node::status::execution::Status::Waiting
                    )
            }) {
                let Some(attempt) = execution.attempts.last() else {
                    continue;
                };
                if !matches!(
                    attempt.status,
                    crate::runtime::workflow::node::status::attempt::Status::Running
                        | crate::runtime::workflow::node::status::attempt::Status::Waiting
                ) {
                    continue;
                }
                let current = ReplyExecution {
                    node_id: node_id.clone(),
                    exec_id: execution.exec_id,
                    idx: attempt.idx,
                };
                if let Some(scene) = &attempt.agent
                    && crate::runtime::agent::wait::question(scene, reply_id)?
                {
                    paths.push(vec![current.clone()]);
                }
                if let Some(child) = &attempt.graph {
                    for mut path in child.reply_paths(reply_id)? {
                        path.insert(0, current.clone());
                        paths.push(path);
                    }
                }
            }
        }
        Ok(paths)
    }

    /// Rebuilds the graph-level usage from committed node Attempt details.
    /// This is deliberately idempotent and safe to call after restore.
    #[doc(hidden)]
    pub fn recompute_usage(&mut self) {
        let mut usage = None;
        for state in self.nodes.values_mut() {
            for execution in state.executions.values_mut() {
                let mut execution_usage = None;
                // Inherited child successes can appear in multiple outer
                // Attempt scenes. They are one settled call, not new usage.
                let mut child_calls = BTreeSet::new();
                for attempt in &mut execution.attempts {
                    // A Graph Node's Attempt is a query projection of the
                    // child's calls, not an extra model call of its own.
                    if let Some(child) = &mut attempt.graph {
                        child.recompute_usage();
                        attempt.usage = child.unseen_call_usage(&mut child_calls);
                    }
                    execution_usage = crate::components::model::token::merge(
                        execution_usage,
                        attempt.usage.clone(),
                    );
                }
                execution.usage = execution_usage;
                usage = crate::components::model::token::merge(usage, execution.usage.clone());
            }
        }
        self.usage = usage;
    }

    fn unseen_call_usage(&self, seen: &mut BTreeSet<(i64, u32)>) -> Option<Usage> {
        let mut usage = None;
        for node in self.nodes.values() {
            for execution in node.executions.values() {
                for attempt in &execution.attempts {
                    let settled = if let Some(child) = &attempt.graph {
                        child.unseen_call_usage(seen)
                    } else if seen.insert((execution.exec_id, attempt.idx)) {
                        attempt.usage.clone()
                    } else {
                        None
                    };
                    usage = crate::components::model::token::merge(usage, settled);
                }
            }
        }
        usage
    }

    pub fn node(&self, node_id: &str) -> Result<&crate::runtime::workflow::node::State, Error> {
        self.nodes
            .get(node_id)
            .ok_or_else(|| Error::new("NOT_FOUND", "workflow node state was not found"))
    }

    pub fn node_mut(
        &mut self,
        node_id: &str,
    ) -> Result<&mut crate::runtime::workflow::node::State, Error> {
        self.nodes
            .get_mut(node_id)
            .ok_or_else(|| Error::new("NOT_FOUND", "workflow node state was not found"))
    }

    pub(crate) fn pauses(&self) -> Vec<Pause> {
        self.nodes
            .iter()
            .flat_map(|(node_id, node)| {
                node.executions
                    .values()
                    .filter(move |execution| {
                        execution.idx == self.step
                            && execution.status
                                == crate::runtime::workflow::node::status::execution::Status::Waiting
                    })
                    .flat_map(move |execution| {
                        execution
                            .waits
                            .iter()
                            .filter(|(_, record)| {
                                record.status == crate::runtime::workflow::node::status::wait::Status::Waiting
                            })
                            .map(move |(wait_id, _)| Pause {
                                node_id: node_id.clone(),
                                exec_id: execution.exec_id,
                                wait_id: wait_id.clone(),
                            })
                    })
            })
            .collect()
    }

    #[doc(hidden)]
    pub fn pause(&self, node_id: &str) -> Result<Option<Pause>, Error> {
        let mut matches = self
            .nodes
            .get(node_id)
            .into_iter()
            .flat_map(|node| node.executions.values())
            .filter(|execution| {
                execution.idx == self.step
                    && execution.status
                        == crate::runtime::workflow::node::status::execution::Status::Waiting
            })
            .flat_map(|execution| {
                execution
                    .waits
                    .iter()
                    .filter(|(_, record)| {
                        record.status
                            == crate::runtime::workflow::node::status::wait::Status::Waiting
                    })
                    .map(move |(wait_id, _)| Pause {
                        node_id: node_id.into(),
                        exec_id: execution.exec_id,
                        wait_id: wait_id.clone(),
                    })
            });
        let first = matches.next();
        if matches.next().is_some() {
            return Err(Error::new(
                "INVALID_STATE",
                "workflow node has ambiguous active pauses",
            ));
        }
        Ok(first)
    }

    fn common_terminal_step(&self, sources: &[String]) -> Option<u64> {
        let mut common: Option<BTreeSet<u64>> = None;
        for source in sources {
            let steps = self
                .nodes
                .get(source)?
                .executions
                .values()
                .filter(|execution| {
                    execution.idx < self.step
                        && matches!(
                            execution.status,
                            crate::runtime::workflow::node::status::execution::Status::Succeeded
                                | crate::runtime::workflow::node::status::execution::Status::Failed
                                | crate::runtime::workflow::node::status::execution::Status::Cancelled
                        )
                })
                .map(|execution| execution.idx)
                .collect::<BTreeSet<_>>();
            common = Some(match common {
                Some(current) => current.intersection(&steps).copied().collect(),
                None => steps,
            });
        }
        common.and_then(|steps| steps.into_iter().max())
    }

    /// Applies the external response to the single active wait for a node.
    /// The response is kept in execution facts so a restored runtime can
    /// resume without a second mailbox or scheduler record.
    #[doc(hidden)]
    pub fn resume(&mut self, node_id: &str, response: Messages) -> Result<(), Error> {
        let wait = match self.pause(node_id)? {
            Some(wait) => wait,
            None => {
                let already_resolved = self.nodes.get(node_id).and_then(|node| {
                    node.executions.values().find_map(|execution| {
                        execution.waits.values().find_map(|wait| {
                            (wait.status
                                == crate::runtime::workflow::node::status::wait::Status::Resolved)
                                .then(|| wait.response.clone())
                        })
                    })
                });
                return match already_resolved {
                    Some(Some(previous)) if previous == response => Ok(()),
                    Some(Some(_)) => Err(Error::new(
                        "CONFLICT",
                        "wait was already resolved differently",
                    )),
                    _ => Err(Error::new("NOT_FOUND", "workflow node has no active wait")),
                };
            }
        };
        // Stage only this execution; child and parent validation must both
        // succeed before any live waiting state is changed.
        let mut execution = self
            .node(node_id)?
            .executions
            .get(&wait.exec_id)
            .ok_or_else(|| Error::new("NOT_FOUND", "execution not found"))?
            .clone();
        // Only the current Attempt owns the active wait. Historical Attempts
        // remain immutable execution facts and must never be resumed or have
        // their nested graph state changed.
        if let Some(attempt) = execution.attempts.last_mut()
            && let Some(child) = attempt.graph.as_mut()
        {
            let child_waits = child.pauses();
            let reply_id = response
                .as_slice()
                .iter()
                .rev()
                .find(|message| message.role == crate::components::message::Role::User)
                .and_then(|message| message.metadata.get("reply_id"))
                .and_then(serde_json::Value::as_i64);
            let child_node_id = if let Some(reply_id) = reply_id {
                let mut matching = Vec::new();
                for wait in &child_waits {
                    let Some(attempt) = child
                        .nodes
                        .get(&wait.node_id)
                        .and_then(|node| node.executions.get(&wait.exec_id))
                        .and_then(|execution| execution.attempts.last())
                    else {
                        continue;
                    };
                    if crate::runtime::workflow::attempt_question(attempt, reply_id)? {
                        matching.push(wait.node_id.clone());
                    }
                }
                if matching.len() > 1 {
                    return Err(Error::new(
                        "CONFLICT",
                        "reply identifies multiple nested Workflow waits",
                    ));
                }
                matching.pop().ok_or_else(|| {
                    Error::new(
                        "NOT_FOUND",
                        "reply does not identify a nested Workflow wait",
                    )
                })?
            } else if child_waits.len() == 1 {
                child_waits[0].node_id.clone()
            } else if child_waits.is_empty() {
                String::new()
            } else {
                return Err(Error::new(
                    "CONFLICT",
                    "nested workflow has multiple active waits",
                ));
            };
            if !child_node_id.is_empty() {
                child.resume(&child_node_id, response.clone())?;
            }
        }
        let record = execution
            .waits
            .get_mut(&wait.wait_id)
            .ok_or_else(|| Error::new("INVALID_STATE", "active wait is missing from execution"))?;
        if record.status == crate::runtime::workflow::node::status::wait::Status::Resolved {
            if record.response.as_ref() == Some(&response) {
                return Ok(());
            }
            return Err(Error::new(
                "CONFLICT",
                "wait was already resolved differently",
            ));
        }
        if record.status != crate::runtime::workflow::node::status::wait::Status::Waiting
            || execution.status
                != crate::runtime::workflow::node::status::execution::Status::Waiting
        {
            return Err(Error::new("INVALID_STATE", "execution is not waiting"));
        }
        record.status = crate::runtime::workflow::node::status::wait::Status::Resolved;
        record.resolved_time = Some(0);
        record.response = Some(response);
        execution.status = crate::runtime::workflow::node::status::execution::Status::Running;
        if let Some(attempt) = execution.attempts.last_mut() {
            attempt.status = crate::runtime::workflow::node::status::attempt::Status::Running;
        }
        let node = self.node_mut(node_id)?;
        node.executions.insert(wait.exec_id, execution);
        node.status = crate::runtime::workflow::node::status::Status::Active;
        self.status = Status::Running;
        Ok(())
    }

    /// Returns one resolved wait that must be resumed by the runtime.
    pub(crate) fn resolved_wait(&self) -> Option<(String, i64, String, Messages)> {
        self.nodes.iter().find_map(|(node_id, node)| {
            node.executions.values().find_map(|execution| {
                if execution.status
                    != crate::runtime::workflow::node::status::execution::Status::Running
                {
                    return None;
                }
                execution.waits.iter().find_map(|(wait_id, wait)| {
                    (wait.status == crate::runtime::workflow::node::status::wait::Status::Resolved)
                        .then(|| {
                            (
                                node_id.clone(),
                                execution.exec_id,
                                wait_id.clone(),
                                wait.response.clone().unwrap_or_default(),
                            )
                        })
                })
            })
        })
    }

    /// Returns nodes that can be activated in the current step. The actual
    /// source executions and committed `to` decisions, not aggregate node
    /// status, determine readiness.
    pub fn ready_nodes(&self, definition: &Definition) -> Vec<String> {
        if self.status != Status::Running {
            return Vec::new();
        }
        if self.nodes.values().any(|node| {
            node.executions.values().any(|execution| {
                execution.idx == self.step
                    && matches!(
                        execution.status,
                        crate::runtime::workflow::node::status::execution::Status::Pending
                            | crate::runtime::workflow::node::status::execution::Status::Running
                            | crate::runtime::workflow::node::status::execution::Status::Waiting
                    )
            })
        }) {
            return Vec::new();
        }
        definition
            .nodes
            .keys()
            .filter(|node_id| self.activation_sources(definition, node_id).is_some())
            .cloned()
            .collect()
    }

    /// The full committed source set for one activation. `None` means the
    /// target must not be scheduled in this step; `Some([])` is START.
    pub fn activation_sources(
        &self,
        definition: &Definition,
        node_id: &str,
    ) -> Option<Vec<crate::runtime::workflow::node::ExecutionRef>> {
        let target = self.nodes.get(node_id)?;
        if target
            .executions
            .values()
            .any(|execution| execution.idx == self.step)
        {
            return None;
        }
        if self.step == 0 {
            return definition
                .graph
                .ordinary_edges
                .iter()
                .any(|edge| matches!(edge.from, Endpoint::Start) && edge.to.node() == Some(node_id))
                .then(Vec::new);
        }

        // A join-failure relation fires only after every declared source has
        // reached a terminal execution in an earlier step and at least one
        // source failed. The handler receives the complete source execution
        // set so the failure remains auditable without synthesising a
        // business message.
        if let Some(relation) = definition.graph.failure_relations.iter().find(|relation| {
            relation.on == "join_failed" && relation.to.iter().any(|target| target == node_id)
        }) {
            let common_step = self.common_terminal_step(&relation.from)?;
            let mut refs = Vec::with_capacity(relation.from.len());
            let mut any_failed = false;
            for source in &relation.from {
                let execution = self.nodes.get(source).and_then(|state| {
                    state
                        .executions
                        .values()
                        .filter(|execution| {
                            execution.idx == common_step
                                && matches!(
                                    execution.status,
                                    crate::runtime::workflow::node::status::execution::Status::Succeeded
                                        | crate::runtime::workflow::node::status::execution::Status::Failed
                                        | crate::runtime::workflow::node::status::execution::Status::Cancelled
                                )
                        })
                        .max_by_key(|execution| (execution.idx, execution.exec_id))
                })?;
                any_failed |= execution.status
                    == crate::runtime::workflow::node::status::execution::Status::Failed
                    && (relation.codes.is_empty()
                        || execution.error.as_ref().is_some_and(|error| {
                            relation.codes.iter().any(|code| code == &error.code)
                        }));
                refs.push(crate::runtime::workflow::node::ExecutionRef {
                    node_id: source.clone(),
                    exec_id: execution.exec_id,
                });
            }
            if !any_failed {
                return None;
            }
            if target
                .executions
                .values()
                .any(|execution| execution.from == refs)
            {
                return None;
            }
            return Some(refs);
        }

        let join = definition
            .graph
            .join_edges
            .iter()
            .find(|join| join.to == node_id);
        let join_step = join.and_then(|join| self.common_terminal_step(&join.from));
        if join.is_some() && join_step.is_none() {
            return None;
        }
        let mut sources = Vec::new();
        if let Some(join) = join {
            sources.extend(join.from.iter().map(String::as_str));
        } else {
            for edge in &definition.graph.ordinary_edges {
                if edge.to.node() == Some(node_id)
                    && let Some(source) = edge.from.node()
                {
                    sources.push(source);
                }
            }
            for relation in &definition.graph.conditional_edges {
                if relation.to.iter().any(|target| target == node_id) {
                    sources.push(&relation.from);
                }
            }
            for relation in &definition.graph.fan_out_definitions {
                if relation.to == node_id {
                    sources.push(&relation.from);
                }
            }
            for relation in &definition.graph.loop_definitions {
                if relation.to == node_id {
                    sources.push(&relation.from);
                }
            }
            for relation in &definition.graph.failure_relations {
                if relation.on == "node_failed"
                    && relation.to.iter().any(|target| target == node_id)
                {
                    sources.extend(relation.from.iter().map(String::as_str));
                }
            }
        }
        let mut seen = BTreeSet::new();
        sources.retain(|source| seen.insert(*source));
        let mut from = Vec::new();
        for source in &sources {
            let matched = self.nodes.get(*source).and_then(|state| {
                state
                    .executions
                    .values()
                    .filter(|execution| {
                        let normal = execution.status
                            == crate::runtime::workflow::node::status::execution::Status::Succeeded
                            && execution.to.iter().any(|target| target == node_id);
                        let failure = execution.status
                            == crate::runtime::workflow::node::status::execution::Status::Failed
                            && execution.error.is_some()
                            && definition.graph.failure_relations.iter().any(|relation| {
                                relation.on == "node_failed"
                                    && relation.from.len() == 1
                                    && relation.from[0] == *source
                                    && relation.to.iter().any(|target| target == node_id)
                                    && (relation.codes.is_empty()
                                        || execution.error.as_ref().is_some_and(|error| {
                                            relation.codes.iter().any(|code| code == &error.code)
                                        }))
                            });
                        (normal || failure)
                            && (if join.is_some() {
                                Some(execution.idx) == join_step
                            } else {
                                execution.idx == self.step - 1
                            })
                    })
                    .max_by_key(|execution| (execution.idx, execution.exec_id))
            });
            if let Some(execution) = matched {
                from.push(crate::runtime::workflow::node::ExecutionRef {
                    node_id: (*source).to_owned(),
                    exec_id: execution.exec_id,
                });
            } else if join.is_some() {
                return None;
            }
        }
        if from.is_empty() {
            return None;
        }
        if join.is_some()
            && target
                .executions
                .values()
                .any(|execution| execution.from == from)
        {
            return None;
        }
        Some(from)
    }

    pub fn start_execution(
        &mut self,
        definition: &Definition,
        node_id: &str,
        exec_id: i64,
        from: Vec<crate::runtime::workflow::node::ExecutionRef>,
    ) -> Result<(), Error> {
        if self
            .nodes
            .values()
            .any(|state| state.executions.contains_key(&exec_id))
        {
            return Err(Error::new("CONFLICT", "exec_id already exists in workflow"));
        }
        self.start_execution_inner(definition, node_id, exec_id, from, true)
    }

    pub fn start_batch(
        &mut self,
        definition: &Definition,
        executions: Vec<(
            String,
            i64,
            Vec<crate::runtime::workflow::node::ExecutionRef>,
        )>,
    ) -> Result<(), Error> {
        let mut ids = BTreeSet::new();
        if executions.iter().any(|(_, exec_id, _)| {
            !ids.insert(*exec_id)
                || self
                    .nodes
                    .values()
                    .any(|state| state.executions.contains_key(exec_id))
        }) {
            return Err(Error::new("CONFLICT", "exec_id already exists in workflow"));
        }
        let ready: BTreeSet<String> = self.ready_nodes(definition).into_iter().collect();
        let mut seen = BTreeSet::new();
        if executions
            .iter()
            .any(|(node_id, _, _)| !ready.contains(node_id) || !seen.insert(node_id.clone()))
        {
            return Err(Error::new(
                "INVALID_STATE",
                "every batch execution must target a distinct ready node",
            ));
        }
        let mut candidate = self.clone();
        for (node_id, exec_id, from) in executions {
            candidate.start_execution_inner(definition, &node_id, exec_id, from, false)?;
        }
        *self = candidate;
        Ok(())
    }

    fn start_execution_inner(
        &mut self,
        definition: &Definition,
        node_id: &str,
        exec_id: i64,
        from: Vec<crate::runtime::workflow::node::ExecutionRef>,
        require_ready: bool,
    ) -> Result<(), Error> {
        if exec_id <= 0 {
            return Err(Error::new("INVALID_ARGUMENTS", "exec_id must be positive"));
        }
        if require_ready && !self.ready_nodes(definition).iter().any(|id| id == node_id) {
            return Err(Error::new("INVALID_STATE", "workflow node is not ready"));
        }
        let expected = self
            .activation_sources(definition, node_id)
            .ok_or_else(|| {
                Error::new("INVALID_STATE", "workflow node is not ready in this step")
            })?;
        if from != expected {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "source executions do not match committed graph relations",
            ));
        }
        let step = self.step;
        let node = self.node_mut(node_id)?;
        node.status = crate::runtime::workflow::node::status::Status::Active;
        node.insert(crate::runtime::workflow::node::Execution {
            exec_id,
            idx: step,
            status: crate::runtime::workflow::node::status::execution::Status::Running,
            from,
            ..Default::default()
        })?;
        node.append_attempt(
            exec_id,
            crate::runtime::workflow::node::Attempt {
                idx: 0,
                status: crate::runtime::workflow::node::status::attempt::Status::Running,
                ..Default::default()
            },
        )
    }

    pub fn complete_execution(
        &mut self,
        node_id: &str,
        exec_id: i64,
        messages: Messages,
        to: Vec<String>,
    ) -> Result<(), Error> {
        let node = self.node_mut(node_id)?;
        let execution = node.execution_mut(exec_id)?;
        if !matches!(
            execution.status,
            crate::runtime::workflow::node::status::execution::Status::Running
                | crate::runtime::workflow::node::status::execution::Status::Waiting
        ) {
            return Err(Error::new("INVALID_STATE", "execution is not active"));
        }
        execution.status = crate::runtime::workflow::node::status::execution::Status::Succeeded;
        if let Some(attempt) = execution.attempts.last_mut() {
            attempt.status = crate::runtime::workflow::node::status::attempt::Status::Succeeded;
        }
        execution.messages = messages;
        execution.to = to;
        node.status = crate::runtime::workflow::node::status::Status::Succeeded;
        self.forget_wait_locations(node_id, exec_id);
        Ok(())
    }

    pub fn wait_execution(
        &mut self,
        node_id: &str,
        exec_id: i64,
        wait_id: String,
    ) -> Result<(), Error> {
        if self.pause(node_id)?.is_some() {
            return Err(Error::new("CONFLICT", "workflow node already has a wait"));
        }
        let node = self.node_mut(node_id)?;
        let execution = node.execution_mut(exec_id)?;
        // Native Agent/child progress can mark this execution waiting before
        // its final Wait signal registers the ordinary wait record. The pause
        // check above still rejects registering that record twice.
        if !matches!(
            execution.status,
            crate::runtime::workflow::node::status::execution::Status::Running
                | crate::runtime::workflow::node::status::execution::Status::Waiting
        ) {
            return Err(Error::new("INVALID_STATE", "execution is not active"));
        }
        if execution.waits.contains_key(&wait_id) {
            return Err(Error::new("CONFLICT", "execution wait already exists"));
        }
        execution.status = crate::runtime::workflow::node::status::execution::Status::Waiting;
        if let Some(attempt) = execution.attempts.last_mut() {
            attempt.status = crate::runtime::workflow::node::status::attempt::Status::Waiting;
        }
        execution.waits.insert(
            wait_id.clone(),
            crate::runtime::workflow::node::Wait {
                kind: "external".into(),
                status: crate::runtime::workflow::node::status::wait::Status::Waiting,
                created_time: 0,
                resolved_time: None,
                response: None,
            },
        );
        Ok(())
    }

    pub fn fail_execution(
        &mut self,
        node_id: &str,
        exec_id: i64,
        error: Error,
    ) -> Result<(), Error> {
        let node = self.node_mut(node_id)?;
        let execution = node.execution_mut(exec_id)?;
        if !matches!(
            execution.status,
            crate::runtime::workflow::node::status::execution::Status::Pending
                | crate::runtime::workflow::node::status::execution::Status::Running
                | crate::runtime::workflow::node::status::execution::Status::Waiting
        ) {
            return Err(Error::new("INVALID_STATE", "execution is not active"));
        }
        execution.status = crate::runtime::workflow::node::status::execution::Status::Failed;
        if let Some(attempt) = execution.attempts.last_mut()
            && matches!(
                attempt.status,
                crate::runtime::workflow::node::status::attempt::Status::Running
                    | crate::runtime::workflow::node::status::attempt::Status::Waiting
            )
        {
            attempt.status = crate::runtime::workflow::node::status::attempt::Status::Failed;
            attempt.error = Some(error.clone());
        }
        execution.error = Some(error);
        node.status = crate::runtime::workflow::node::status::Status::Failed;
        self.forget_wait_locations(node_id, exec_id);
        Ok(())
    }

    pub fn cancel_execution(&mut self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        let node = self.node_mut(node_id)?;
        let execution = node.execution_mut(exec_id)?;
        if !matches!(
            execution.status,
            crate::runtime::workflow::node::status::execution::Status::Pending
                | crate::runtime::workflow::node::status::execution::Status::Running
                | crate::runtime::workflow::node::status::execution::Status::Waiting
        ) {
            return Err(Error::new("INVALID_STATE", "execution is not active"));
        }
        execution.status = crate::runtime::workflow::node::status::execution::Status::Cancelled;
        if let Some(attempt) = execution.attempts.last_mut()
            && matches!(
                attempt.status,
                crate::runtime::workflow::node::status::attempt::Status::Running
                    | crate::runtime::workflow::node::status::attempt::Status::Waiting
            )
        {
            attempt.status = crate::runtime::workflow::node::status::attempt::Status::Cancelled;
        }
        node.status = crate::runtime::workflow::node::status::Status::Cancelled;
        self.forget_wait_locations(node_id, exec_id);
        Ok(())
    }

    fn forget_wait_locations(&mut self, node_id: &str, exec_id: i64) {
        self.waits.retain(|wait| {
            !wait
                .nodes
                .first()
                .is_some_and(|owner| owner.node_id == node_id && owner.exec_id == exec_id)
        });
    }

    /// Completes the current barrier and advances the graph step. A waiting,
    /// running, or pending execution keeps the current step unchanged.
    pub fn advance_step(&mut self) -> bool {
        let blocked = self.nodes.values().any(|node| {
            node.executions.values().any(|execution| {
                execution.idx == self.step
                    && matches!(
                        execution.status,
                        crate::runtime::workflow::node::status::execution::Status::Pending
                            | crate::runtime::workflow::node::status::execution::Status::Running
                            | crate::runtime::workflow::node::status::execution::Status::Waiting
                    )
            })
        });
        if blocked {
            return false;
        }
        self.step = self.step.saturating_add(1);
        true
    }
}
