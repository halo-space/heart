use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::components::message::Messages;
use crate::components::model::token::Usage;
use crate::components::plan::Plan;
use crate::runtime::agent::node::{State, Status};

/// Mutable AgentGraph execution state for one Plan version.
///
/// The graph only tracks logical node state and dependency activation. It does
/// not own a Planner, Evaluator, concurrency executor, Memory, or Checkpoint.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Graph {
    pub plan: Plan,
    pub nodes: BTreeMap<String, State>,
    pub history: Vec<History>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct History {
    pub plan: Plan,
    pub nodes: BTreeMap<String, State>,
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
            .map(|node_id| (node_id.clone(), State::default()))
            .collect();
        let mut graph = Self {
            plan,
            nodes,
            history: Vec::new(),
        };
        graph.activate_ready()?;
        Ok(graph)
    }

    /// Internal plan adoption used by Agent Runtime. Validation finishes
    /// before any current state is changed.
    pub(crate) fn adopt_plan(&mut self, plan: Plan) -> Result<(), Error> {
        if self
            .nodes
            .values()
            .any(|state| matches!(state.status, Status::Running | Status::Waiting))
        {
            return Err(Error::new(
                "INVALID_STATE",
                "cannot adopt a plan while agent nodes are running or waiting",
            ));
        }
        let mut candidate = Graph::from_plan(plan)?;
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

    pub fn state(&self, node_id: &str) -> Result<&State, Error> {
        self.nodes
            .get(node_id)
            .ok_or_else(|| Error::new("NOT_FOUND", "agent graph node was not found"))
    }

    pub fn state_mut(&mut self, node_id: &str) -> Result<&mut State, Error> {
        self.nodes
            .get_mut(node_id)
            .ok_or_else(|| Error::new("NOT_FOUND", "agent graph node was not found"))
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
                    "agent graph source has not succeeded",
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
                    "agent graph source has no successful current attempt",
                ));
            }
            let source_messages = state.messages.clone().ok_or_else(|| {
                Error::new(
                    "INVALID_STATE",
                    "agent graph source has no committed messages",
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
                "agent graph batch exceeds max_concurrency",
            ));
        }

        let mut seen = BTreeSet::new();
        let mut execution_ids = self.execution_ids();
        for start in starts {
            if !seen.insert(start.node_id.as_str()) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "agent graph batch contains a duplicate node",
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

    pub fn resume(&mut self, node_id: &str) -> Result<(), Error> {
        self.ensure_attempt(node_id)?;
        self.transition(node_id, Status::Running)?;
        self.state_mut(node_id)?
            .current_attempt_mut()
            .map(|attempt| attempt.status = Status::Running)
    }

    pub fn interrupt(&mut self, node_id: &str) -> Result<(), Error> {
        self.ensure_attempt(node_id)?;
        self.transition(node_id, Status::Interrupted)?;
        self.state_mut(node_id)?
            .set_attempt_result(Status::Interrupted, None, None)
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
        state.messages = Some(messages);
        state.set_attempt_result(Status::Succeeded, usage, None)?;
        self.activate_ready()
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
                    Status::Succeeded | Status::Failed | Status::Cancelled | Status::Interrupted
                )
            })
    }

    fn transition(&mut self, node_id: &str, next: Status) -> Result<(), Error> {
        let state = self.state_mut(node_id)?;
        if !state.can_transition_to(&next) {
            return Err(Error::new(
                "INVALID_STATE",
                "agent graph node state transition is not allowed",
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
mod tests {
    use super::*;
    use crate::plan::{Edge, PlanNode};

    fn plan(edges: &[(&str, &str)]) -> Plan {
        let mut nodes = BTreeMap::new();
        for id in ["a", "b", "c"] {
            nodes.insert(
                id.to_owned(),
                PlanNode {
                    node_id: id.to_owned(),
                    name: id.to_owned(),
                    objective: format!("run {id}"),
                },
            );
        }
        Plan {
            version: 1,
            nodes,
            edges: edges
                .iter()
                .map(|(from, to)| Edge {
                    from: (*from).to_owned(),
                    to: (*to).to_owned(),
                })
                .collect(),
        }
    }

    #[test]
    fn roots_are_ready_and_join_waits_for_all_predecessors() {
        let mut graph = Graph::new(plan(&[("a", "c"), ("b", "c")])).unwrap();
        assert_eq!(graph.ready_nodes(), vec!["a", "b"]);
        assert_eq!(graph.state("c").unwrap().status, Status::Pending);

        graph.start("a", 101).unwrap();
        assert!(
            graph
                .succeed("a", Messages::default(), None)
                .unwrap()
                .is_empty()
        );
        assert_eq!(graph.state("c").unwrap().status, Status::Pending);

        graph.start("b", 102).unwrap();
        assert_eq!(
            graph.succeed("b", Messages::default(), None).unwrap(),
            vec!["c"]
        );
        assert_eq!(graph.state("c").unwrap().status, Status::Ready);
    }

    #[test]
    fn failure_blocks_only_its_downstream_branch() {
        let mut graph = Graph::new(plan(&[("a", "c")])).unwrap();
        graph.start("a", 101).unwrap();
        graph.fail("a").unwrap();
        assert!(graph.ready_nodes().contains(&"b".to_owned()));
        assert_eq!(graph.state("c").unwrap().status, Status::Pending);
    }

    #[test]
    fn waiting_resumes_without_activating_downstream() {
        let mut graph = Graph::new(plan(&[("a", "c")])).unwrap();
        graph.start("a", 101).unwrap();
        graph.wait("a").unwrap();
        assert_eq!(graph.state("c").unwrap().status, Status::Pending);
        graph.resume("a").unwrap();
        assert_eq!(graph.state("a").unwrap().status, Status::Running);
    }

    #[test]
    fn invalid_transition_does_not_mutate_state() {
        let mut graph = Graph::new(plan(&[])).unwrap();
        graph.start("a", 101).unwrap();
        graph.succeed("a", Messages::default(), None).unwrap();
        let error = graph.resume("a").unwrap_err();
        assert_eq!(error.code, "INVALID_STATE");
        assert_eq!(graph.state("a").unwrap().status, Status::Succeeded);
    }

    #[test]
    fn completion_requires_successful_leaves() {
        let mut graph = Graph::new(plan(&[("a", "c"), ("b", "c")])).unwrap();
        assert!(!graph.is_complete());
        graph.start("a", 101).unwrap();
        graph.succeed("a", Messages::default(), None).unwrap();
        graph.start("b", 102).unwrap();
        graph.succeed("b", Messages::default(), None).unwrap();
        graph.start("c", 103).unwrap();
        graph.succeed("c", Messages::default(), None).unwrap();
        assert!(graph.is_complete());
    }

    #[test]
    fn retry_reuses_exec_id_and_keeps_attempt_history() {
        let mut graph = Graph::new(plan(&[])).unwrap();
        graph.start("a", 9001).unwrap();
        let first_error = Error::new("TEMPORARY", "provider unavailable");
        graph
            .fail_with_usage(
                "a",
                Some(first_error.clone()),
                Some(Usage {
                    input: Some(2),
                    output: Some(3),
                    total: Some(5),
                    input_details: None,
                    output_details: None,
                    details: Default::default(),
                }),
            )
            .unwrap();
        graph.retry("a").unwrap();
        graph
            .succeed(
                "a",
                Messages::default(),
                Some(Usage {
                    input: Some(4),
                    output: Some(6),
                    total: Some(10),
                    input_details: None,
                    output_details: None,
                    details: Default::default(),
                }),
            )
            .unwrap();

        let execution = graph.state("a").unwrap().execution.as_ref().unwrap();
        assert_eq!(execution.exec_id, 9001);
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(execution.attempts[0].error, Some(first_error));
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(15));
        assert_eq!(
            graph.state("a").unwrap().usage.as_ref().unwrap().total,
            Some(15)
        );
    }

    #[test]
    fn batch_start_is_atomic_and_enforces_concurrency() {
        let mut graph = Graph::new(plan(&[])).unwrap();
        let starts = vec![
            Start {
                node_id: "a".into(),
                exec_id: 1,
            },
            Start {
                node_id: "b".into(),
                exec_id: 2,
            },
        ];
        let error = graph.start_batch(1, &starts).unwrap_err();
        assert_eq!(error.code, "CONCURRENCY_LIMIT");
        assert_eq!(graph.state("a").unwrap().status, Status::Ready);
        assert_eq!(graph.state("b").unwrap().status, Status::Ready);

        assert_eq!(graph.start_batch(2, &starts).unwrap(), vec!["a", "b"]);
        assert_eq!(
            graph
                .state("a")
                .unwrap()
                .execution
                .as_ref()
                .unwrap()
                .exec_id,
            1
        );
    }

    #[test]
    fn waiting_releases_a_batch_slot() {
        let mut graph = Graph::new(plan(&[])).unwrap();
        graph
            .start_batch(
                1,
                &[Start {
                    node_id: "a".into(),
                    exec_id: 1,
                }],
            )
            .unwrap();
        graph.wait("a").unwrap();
        assert_eq!(
            graph
                .start_batch(
                    1,
                    &[Start {
                        node_id: "b".into(),
                        exec_id: 2,
                    }],
                )
                .unwrap(),
            vec!["b"]
        );
    }

    #[test]
    fn replacing_plan_keeps_old_execution_history_without_reuse() {
        let mut graph = Graph::new(plan(&[("a", "c")])).unwrap();
        graph.start("a", 101).unwrap();
        graph.succeed("a", Messages::default(), None).unwrap();

        let replacement = Plan {
            version: 2,
            nodes: BTreeMap::from([(
                "new".into(),
                PlanNode {
                    node_id: "new".into(),
                    name: "a".into(),
                    objective: "new objective".into(),
                },
            )]),
            edges: Vec::new(),
        };
        graph.adopt_plan(replacement).unwrap();

        assert_eq!(graph.history().len(), 1);
        assert_eq!(graph.history()[0].nodes["a"].status, Status::Succeeded);
        assert_eq!(graph.state("new").unwrap().status, Status::Ready);
        assert!(graph.state("new").unwrap().execution.is_none());
    }

    #[test]
    fn restored_graph_accepts_task_names_without_toolkit_validation() {
        let graph = Graph::new(plan(&[])).unwrap();
        let snapshot = serde_json::to_value(&graph).unwrap();
        let mut restored: Graph = serde_json::from_value(snapshot).unwrap();
        let replacement = plan(&[]);
        restored.adopt_plan(replacement).unwrap();

        let snapshot = serde_json::to_value(&restored).unwrap();
        let mut restored: Graph = serde_json::from_value(snapshot).unwrap();
        let before = restored.clone();
        let replacement = Plan {
            version: 3,
            nodes: BTreeMap::from([(
                "billing".into(),
                PlanNode {
                    node_id: "billing".into(),
                    name: "billing".into(),
                    objective: "charge the account".into(),
                },
            )]),
            edges: Vec::new(),
        };
        restored.adopt_plan(replacement).unwrap();
        assert_ne!(restored, before);
    }

    #[test]
    fn source_messages_are_read_by_node_id_without_merging() {
        let mut graph = Graph::new(plan(&[("a", "c"), ("b", "c")])).unwrap();
        graph.start("a", 101).unwrap();
        graph
            .succeed(
                "a",
                Messages::new([crate::Message::function(serde_json::json!({
                    "source": "a"
                }))]),
                None,
            )
            .unwrap();
        graph.start("b", 102).unwrap();
        graph
            .succeed(
                "b",
                Messages::new([crate::Message::function(serde_json::json!({
                    "source": "b"
                }))]),
                None,
            )
            .unwrap();

        let sources = graph.source_messages("c").unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources["a"].as_slice()[0].values["source"], "a");
        assert_eq!(sources["b"].as_slice()[0].values["source"], "b");
    }

    #[test]
    fn source_messages_require_all_sources_to_have_committed_success() {
        let mut graph = Graph::new(plan(&[("a", "c"), ("b", "c")])).unwrap();
        graph.start("a", 101).unwrap();
        graph.succeed("a", Messages::default(), None).unwrap();
        let error = graph.source_messages("c").unwrap_err();
        assert_eq!(error.code, "INVALID_STATE");
    }

    #[test]
    fn invalid_plan_replacement_is_atomic() {
        let mut graph = Graph::new(plan(&[])).unwrap();
        let invalid = Plan {
            version: 2,
            nodes: BTreeMap::from([(
                "a".into(),
                PlanNode {
                    node_id: "a".into(),
                    name: "a".into(),
                    objective: "a".into(),
                },
            )]),
            edges: vec![Edge {
                from: "a".into(),
                to: "missing".into(),
            }],
        };
        let error = graph.adopt_plan(invalid).unwrap_err();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
        assert_eq!(graph.plan.version, 1);
        assert!(graph.history().is_empty());
    }

    #[test]
    fn execution_identity_cannot_be_shared_by_different_nodes() {
        let mut graph = Graph::new(plan(&[])).unwrap();
        graph.start("a", 101).unwrap();
        let before = graph.clone();
        assert_eq!(graph.start("b", 101).unwrap_err().code, "CONFLICT");
        assert_eq!(graph, before);
        graph.wait("a").unwrap();
        let before = graph.clone();
        assert_eq!(graph.start("b", 101).unwrap_err().code, "CONFLICT");
        assert_eq!(graph, before);
    }

    #[test]
    fn duplicate_execution_identity_rejects_the_whole_batch() {
        let mut graph = Graph::new(plan(&[])).unwrap();
        let before = graph.clone();
        let starts = [
            Start {
                node_id: "a".into(),
                exec_id: 101,
            },
            Start {
                node_id: "b".into(),
                exec_id: 101,
            },
        ];
        assert_eq!(graph.start_batch(2, &starts).unwrap_err().code, "CONFLICT");
        assert_eq!(graph, before);
    }

    #[test]
    fn late_invalid_batch_item_does_not_start_earlier_nodes() {
        for invalid in ["a", "missing", "c"] {
            let mut graph = Graph::new(plan(&[("b", "c")])).unwrap();
            let before = graph.clone();
            let starts = [
                Start {
                    node_id: "a".into(),
                    exec_id: 101,
                },
                Start {
                    node_id: invalid.into(),
                    exec_id: 102,
                },
            ];
            assert!(graph.start_batch(2, &starts).is_err());
            assert_eq!(graph, before);
        }
    }

    #[test]
    fn historical_identity_remains_reserved_after_replan_and_restore() {
        let mut graph = Graph::from_plan(plan(&[])).unwrap();
        graph.start("a", 101).unwrap();
        graph.succeed("a", Messages::default(), None).unwrap();
        graph.adopt_plan(plan(&[])).unwrap();
        let mut graph: Graph =
            serde_json::from_value(serde_json::to_value(graph).unwrap()).unwrap();
        let before = graph.clone();
        assert_eq!(graph.start("a", 101).unwrap_err().code, "CONFLICT");
        assert_eq!(graph, before);
        let starts = [
            Start {
                node_id: "a".into(),
                exec_id: 102,
            },
            Start {
                node_id: "b".into(),
                exec_id: 101,
            },
        ];
        assert_eq!(graph.start_batch(2, &starts).unwrap_err().code, "CONFLICT");
        assert_eq!(graph, before);
        graph.start("a", 103).unwrap();
        assert_eq!(
            graph.history[0].nodes["a"]
                .execution
                .as_ref()
                .unwrap()
                .exec_id,
            101
        );
    }

    #[test]
    fn plan_adoption_never_replaces_running_or_waiting_nodes() {
        for waiting in [false, true] {
            let mut graph = Graph::from_plan(plan(&[])).unwrap();
            graph.start("a", 101).unwrap();
            if waiting {
                graph.wait("a").unwrap();
            }
            let before = graph.clone();
            assert_eq!(
                graph.adopt_plan(plan(&[])).unwrap_err().code,
                "INVALID_STATE"
            );
            assert_eq!(graph, before);
        }
    }

    #[test]
    fn source_success_requires_a_successful_current_attempt() {
        let mut graph = Graph::new(plan(&[("a", "c")])).unwrap();
        let state = graph.state_mut("a").unwrap();
        state.status = Status::Succeeded;
        state.messages = Some(Messages::new([crate::Message::function(
            serde_json::json!({"value": "old"}),
        )]));
        let before = graph.clone();
        assert_eq!(
            graph.source_messages("c").unwrap_err().code,
            "INVALID_STATE"
        );
        assert_eq!(graph, before);

        let state = graph.state_mut("a").unwrap();
        state.begin_execution(101).unwrap();
        state.execution.as_mut().unwrap().attempts[0].status = Status::Failed;
        let before = graph.clone();
        assert_eq!(
            graph.source_messages("c").unwrap_err().code,
            "INVALID_STATE"
        );
        assert_eq!(graph, before);
    }

    #[test]
    fn same_capability_keeps_results_separate_and_sources_are_owned_snapshots() {
        let mut input = plan(&[("a", "c"), ("b", "c")]);
        input.nodes.get_mut("a").unwrap().name = "search".into();
        input.nodes.get_mut("b").unwrap().name = "search".into();
        let mut graph = Graph::new(input).unwrap();
        for (node_id, exec_id, city) in [("a", 101, "北京"), ("b", 102, "上海")] {
            graph.start(node_id, exec_id).unwrap();
            graph
                .succeed(
                    node_id,
                    Messages::new([crate::Message::function(serde_json::json!({"city": city}))]),
                    None,
                )
                .unwrap();
        }
        let before = graph.clone();
        let mut sources = graph.source_messages("c").unwrap();
        assert_eq!(sources["a"].as_slice()[0].values["city"], "北京");
        assert_eq!(sources["b"].as_slice()[0].values["city"], "上海");
        sources.get_mut("a").unwrap().0[0]
            .values
            .insert("city".into(), serde_json::json!("changed"));
        assert_eq!(graph, before);
    }

    #[test]
    fn failed_or_waiting_sources_cannot_publish_saved_messages() {
        for waiting in [false, true] {
            let mut graph = Graph::new(plan(&[("a", "c")])).unwrap();
            graph.start("a", 101).unwrap();
            if waiting {
                graph.wait("a").unwrap();
            } else {
                graph.fail("a").unwrap();
            }
            graph.state_mut("a").unwrap().messages =
                Some(Messages::new([crate::Message::function(
                    serde_json::json!({"value": "uncommitted"}),
                )]));
            let before = graph.clone();
            assert_eq!(
                graph.source_messages("c").unwrap_err().code,
                "INVALID_STATE"
            );
            assert_eq!(graph, before);
        }
    }
}
