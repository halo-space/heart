use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::components::message::Messages;
use crate::components::model::token::Usage;
use crate::components::plan::Plan;
use crate::runtime::agent::node;
use crate::runtime::agent::node::status::Status;

/// Mutable execution-graph state for one Agent plan version.
///
/// The graph only tracks logical node state and dependency activation. It does
/// not own a Planner, Evaluator, concurrency executor, Memory, or Checkpoint.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Graph {
    pub plan: Plan,
    pub nodes: BTreeMap<String, node::State>,
    pub history: Vec<History>,
}

/// Static plan snapshot embedded in the canonical Agent State.
///
/// This is deliberately named by its module rather than exported as a second
/// graph-state type.  The mutable [`Graph`] above owns execution facts;
/// `graph::State` only preserves the static plan in the Agent checkpoint JSON.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct State {
    pub plan: Plan,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct History {
    pub plan: Plan,
    pub nodes: BTreeMap<String, node::State>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Start {
    pub node_id: String,
    pub exec_id: i64,
}

impl Graph {
    pub fn new(plan: Plan) -> Result<Self, Error> {
        plan.validate_structure()?;
        Self::initialize(plan)
    }

    /// Internal construction path used by the Agent runtime. Plan names are
    /// task labels; Tool/MCP/Skill selection remains inside the Agent loop.
    pub(crate) fn from_plan(plan: Plan) -> Result<Self, Error> {
        plan.validate()?;
        Self::initialize(plan)
    }

    fn initialize(plan: Plan) -> Result<Self, Error> {
        let nodes = plan
            .nodes
            .keys()
            .map(|node_id| (node_id.clone(), node::State::default()))
            .collect();
        let mut graph = Self {
            plan,
            nodes,
            history: Vec::new(),
        };
        graph.activate_ready()?;
        Ok(graph)
    }

    /// Update the execution graph with the complete plan returned by Planner.
    /// Validation finishes before any current state is changed.
    pub(crate) fn update(&mut self, plan: Plan) -> Result<(), Error> {
        if self
            .nodes
            .values()
            .any(|state| matches!(state.status, Status::Running | Status::Waiting))
        {
            return Err(Error::new(
                "INVALID_STATE",
                "cannot update the plan while agent nodes are running or waiting",
            ));
        }
        let old_plan = self.plan.clone();
        let old_nodes = self.nodes.clone();
        let mut candidate = Graph::from_plan(plan)?;

        // A successful result is reusable only when its logical node definition
        // and all upstream dependencies are unchanged.  If a planner changes a
        // node or rewires an edge, invalidate that node and its complete
        // downstream closure; unrelated successful branches remain available.
        let changed: BTreeSet<String> = candidate
            .plan
            .nodes
            .keys()
            .filter(|node_id| old_plan.nodes.get(*node_id) != candidate.plan.nodes.get(*node_id))
            .cloned()
            .chain(
                old_plan
                    .edges
                    .iter()
                    .chain(candidate.plan.edges.iter())
                    .filter(|edge| {
                        !old_plan.edges.contains(edge) || !candidate.plan.edges.contains(edge)
                    })
                    .flat_map(|edge| [edge.from.clone(), edge.to.clone()]),
            )
            .collect();
        let mut invalidated = changed;
        let mut frontier: Vec<String> = invalidated.iter().cloned().collect();
        while let Some(source) = frontier.pop() {
            for edge in candidate
                .plan
                .edges
                .iter()
                .filter(|edge| edge.from == source)
            {
                if invalidated.insert(edge.to.clone()) {
                    frontier.push(edge.to.clone());
                }
            }
        }
        for (node_id, state) in old_nodes {
            if !invalidated.contains(&node_id)
                && candidate.plan.nodes.contains_key(&node_id)
                && state.status == Status::Succeeded
            {
                candidate.nodes.insert(node_id, state);
            }
        }
        candidate.activate_ready()?;
        candidate.history = self.history.clone();
        candidate.history.push(History {
            plan: self.plan.clone(),
            nodes: self.nodes.clone(),
        });
        *self = candidate;
        Ok(())
    }

    pub fn history(&self) -> &[History] {
        &self.history
    }

    pub fn state(&self, node_id: &str) -> Result<&node::State, Error> {
        self.nodes
            .get(node_id)
            .ok_or_else(|| Error::new("NOT_FOUND", "execution graph node was not found"))
    }

    pub fn state_mut(&mut self, node_id: &str) -> Result<&mut node::State, Error> {
        self.nodes
            .get_mut(node_id)
            .ok_or_else(|| Error::new("NOT_FOUND", "execution graph node was not found"))
    }

    /// Return the direct source node IDs for a logical node. The order is
    /// deterministic, but callers should use the IDs rather than positional
    /// order to read source data.
    pub fn predecessors(&self, node_id: &str) -> Result<Vec<String>, Error> {
        self.state(node_id)?;
        Ok(self
            .plan
            .edges
            .iter()
            .filter(|edge| edge.to == node_id)
            .map(|edge| edge.from.clone())
            .collect())
    }

    /// Read each directly connected predecessor's committed Messages without
    /// synthesizing a merged collection. A node can read this only after all
    /// its sources have successfully committed.
    pub fn source_messages(&self, node_id: &str) -> Result<BTreeMap<String, Messages>, Error> {
        let predecessors = self.predecessors(node_id)?;
        let mut messages = BTreeMap::new();
        for source in predecessors {
            let state = self.state(&source)?;
            if state.status != Status::Succeeded {
                return Err(Error::new(
                    "INVALID_STATE",
                    "execution graph source has not succeeded",
                ));
            }
            // A status flag alone is not a committed execution. Reject an
            // inconsistent restored or publicly modified source state.
            if state
                .execution
                .as_ref()
                .and_then(|execution| execution.attempts.last())
                .is_none_or(|attempt| attempt.status != Status::Succeeded)
            {
                return Err(Error::new(
                    "INVALID_STATE",
                    "execution graph source has no successful current attempt",
                ));
            }
            let source_messages = state.messages.clone().ok_or_else(|| {
                Error::new(
                    "INVALID_STATE",
                    "execution graph source has no committed messages",
                )
            })?;
            messages.insert(source, source_messages);
        }
        Ok(messages)
    }

    pub fn ready_nodes(&self) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|(_, state)| state.status == Status::Ready)
            .map(|(node_id, _)| node_id.clone())
            .collect()
    }

    /// Recompute dependency activation without introducing a global step or
    /// barrier. Every newly eligible node enters `ready` together.
    pub fn activate_ready(&mut self) -> Result<Vec<String>, Error> {
        let candidates: Vec<String> = self
            .nodes
            .iter()
            .filter(|(node_id, state)| {
                state.status == Status::Pending && self.predecessors_succeeded(node_id)
            })
            .map(|(node_id, _)| node_id.clone())
            .collect();

        for node_id in &candidates {
            self.transition(node_id, Status::Ready)?;
        }
        Ok(candidates)
    }

    pub fn start(&mut self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        self.validate_start(node_id)?;
        if self.execution_ids().contains(&exec_id) {
            return Err(Error::new(
                "CONFLICT",
                "agent execution identity is already used",
            ));
        }
        self.transition(node_id, Status::Running)?;
        self.state_mut(node_id)?.begin_execution(exec_id)
    }

    /// Atomically claim one executable batch. The caller supplies execution
    /// identities; this method only enforces readiness and concurrency.
    pub fn start_batch(
        &mut self,
        max_concurrency: usize,
        starts: &[Start],
    ) -> Result<Vec<String>, Error> {
        if max_concurrency == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "max_concurrency must be positive",
            ));
        }
        let running = self
            .nodes
            .values()
            .filter(|state| state.status == Status::Running)
            .count();
        if running.saturating_add(starts.len()) > max_concurrency {
            return Err(Error::new(
                "CONCURRENCY_LIMIT",
                "execution graph batch exceeds max_concurrency",
            ));
        }

        let mut seen = BTreeSet::new();
        let mut execution_ids = self.execution_ids();
        for start in starts {
            if !seen.insert(start.node_id.as_str()) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "execution graph batch contains a duplicate node",
                ));
            }
            self.validate_start(&start.node_id)?;
            if !execution_ids.insert(start.exec_id) {
                return Err(Error::new(
                    "CONFLICT",
                    "agent execution identity is already used",
                ));
            }
        }

        for start in starts {
            // All fallible validation precedes mutation; do not re-scan
            // history for every item of this already validated batch.
            let state = self.nodes.get_mut(&start.node_id).expect("validated node");
            state.begin_execution(start.exec_id)?;
            state.status = Status::Running;
        }
        Ok(starts.iter().map(|start| start.node_id.clone()).collect())
    }

    pub fn wait(&mut self, node_id: &str) -> Result<(), Error> {
        self.ensure_attempt(node_id)?;
        self.transition(node_id, Status::Waiting)?;
        self.state_mut(node_id)?
            .set_attempt_result(Status::Waiting, None, None)
    }

    /// Internal transition used after the waiting reply has been accepted.
    /// External callers submit feedback through the Agent `reply` entry point.
    #[doc(hidden)]
    pub fn resume(&mut self, node_id: &str) -> Result<(), Error> {
        self.ensure_attempt(node_id)?;
        self.transition(node_id, Status::Running)?;
        self.state_mut(node_id)?
            .current_attempt_mut()
            .map(|attempt| attempt.status = Status::Running)
    }

    pub fn cancel(&mut self, node_id: &str) -> Result<(), Error> {
        self.ensure_attempt(node_id)?;
        self.transition(node_id, Status::Cancelled)?;
        self.state_mut(node_id)?
            .set_attempt_result(Status::Cancelled, None, None)
    }

    pub fn fail(&mut self, node_id: &str) -> Result<(), Error> {
        self.fail_with_usage(node_id, None, None)
    }

    pub fn fail_with(&mut self, node_id: &str, error: Option<Error>) -> Result<(), Error> {
        self.fail_with_usage(node_id, error, None)
    }

    pub fn fail_with_usage(
        &mut self,
        node_id: &str,
        error: Option<Error>,
        usage: Option<Usage>,
    ) -> Result<(), Error> {
        self.ensure_attempt(node_id)?;
        self.transition(node_id, Status::Failed)?;
        self.state_mut(node_id)?
            .set_attempt_result(Status::Failed, usage, error)
    }

    /// Retry the current logical execution with a new Attempt. The caller is
    /// responsible for deciding whether retry policy permits this operation.
    pub fn retry(&mut self, node_id: &str) -> Result<(), Error> {
        let state = self.state_mut(node_id)?;
        if state.status != Status::Failed {
            return Err(Error::new(
                "INVALID_STATE",
                "only a failed agent node can be retried",
            ));
        }
        state.add_attempt()?;
        state.status = Status::Running;
        Ok(())
    }

    pub fn succeed(
        &mut self,
        node_id: &str,
        messages: Messages,
        usage: Option<Usage>,
    ) -> Result<Vec<String>, Error> {
        self.ensure_attempt(node_id)?;
        self.transition(node_id, Status::Succeeded)?;
        let state = self.state_mut(node_id)?;
        state.messages = Some(messages.clone());
        if let Some(execution) = state.execution.as_mut() {
            execution.messages = messages.clone();
            if let Some(attempt) = execution.attempts.last_mut() {
                attempt.messages = messages;
            }
        }
        state.set_attempt_result(Status::Succeeded, usage, None)?;
        self.activate_ready()
    }

    pub(crate) fn set_finish_reason(
        &mut self,
        node_id: &str,
        finish_reason: Option<String>,
    ) -> Result<(), Error> {
        self.state_mut(node_id)?
            .set_attempt_finish_reason(finish_reason)
    }

    /// Agent plans complete only when all leaves succeeded and no node remains
    /// active or unactivated. A failed branch therefore cannot be mistaken for
    /// successful completion.
    pub fn is_complete(&self) -> bool {
        let leaves = self
            .plan
            .nodes
            .keys()
            .filter(|node_id| self.successors(node_id).is_empty());
        let mut has_leaf = false;
        for node_id in leaves {
            has_leaf = true;
            if self
                .nodes
                .get(node_id)
                .is_none_or(|state| state.status != Status::Succeeded)
            {
                return false;
            }
        }
        has_leaf
            && self.nodes.values().all(|state| {
                matches!(
                    state.status,
                    Status::Succeeded | Status::Failed | Status::Cancelled
                )
            })
    }

    fn transition(&mut self, node_id: &str, next: Status) -> Result<(), Error> {
        let state = self.state_mut(node_id)?;
        if !state.can_transition_to(&next) {
            return Err(Error::new(
                "INVALID_STATE",
                "execution graph node state transition is not allowed",
            ));
        }
        state.status = next;
        Ok(())
    }

    fn ensure_attempt(&self, node_id: &str) -> Result<(), Error> {
        let state = self.state(node_id)?;
        if state
            .execution
            .as_ref()
            .and_then(|execution| execution.attempts.last())
            .is_none()
        {
            return Err(Error::new(
                "INVALID_STATE",
                "agent node has no current attempt",
            ));
        }
        Ok(())
    }

    fn validate_start(&self, node_id: &str) -> Result<(), Error> {
        let state = self.state(node_id)?;
        if state.execution.is_some() {
            return Err(Error::new(
                "CONFLICT",
                "agent node already has a current execution",
            ));
        }
        if state.status != Status::Ready {
            return Err(Error::new("INVALID_STATE", "agent node is not ready"));
        }
        Ok(())
    }

    fn execution_ids(&self) -> BTreeSet<i64> {
        self.nodes
            .values()
            .chain(
                self.history
                    .iter()
                    .flat_map(|history| history.nodes.values()),
            )
            .filter_map(|state| state.execution.as_ref().map(|execution| execution.exec_id))
            .collect()
    }

    fn predecessors_succeeded(&self, node_id: &str) -> bool {
        self.predecessor_refs(node_id).iter().all(|source| {
            self.nodes
                .get(*source)
                .is_some_and(|state| state.status == Status::Succeeded)
        })
    }

    fn predecessor_refs(&self, node_id: &str) -> Vec<&str> {
        self.plan
            .edges
            .iter()
            .filter(|edge| edge.to == node_id)
            .map(|edge| edge.from.as_str())
            .collect()
    }

    fn successors(&self, node_id: &str) -> BTreeSet<&str> {
        self.plan
            .edges
            .iter()
            .filter(|edge| edge.from == node_id)
            .map(|edge| edge.to.as_str())
            .collect()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_runtime_agent_graph_tests.rs"]
mod tests;
