//! Static Workflow definitions and the executable graph-state boundary.
//! Relation declarations remain immutable after compilation; execution facts
//! are kept in the separate serializable `State`.

use std::any::TypeId;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::StreamExt;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::Error;
use crate::agent::{self, Agent};
use crate::components::event::Event;
use crate::components::message::Messages;
use crate::components::message::Role;
pub use crate::components::middleware::RetryPolicy;
use crate::components::model::token::Usage;
use crate::components::node::{Node, Signal};
use crate::runtime::cancellation::Cancellation;
use crate::runtime::workflow::node::{AttemptStatus, Step, StepAttempt};

pub const START: &str = "START";
pub const END: &str = "END";

type BoxFuture = Pin<Box<dyn Future<Output = Result<Signal<Value>, Error>> + Send + 'static>>;
type RouteFn = Arc<dyn Fn(&Messages) -> Result<Vec<String>, Error> + Send + Sync + 'static>;
type LoopFn = Arc<dyn Fn(&Messages) -> bool + Send + Sync + 'static>;
type AgentFuture = Pin<Box<dyn Future<Output = AgentRun> + 'static>>;

pub(crate) struct AgentRun {
    pub result: Result<Messages, Error>,
    pub messages: Messages,
    pub steps: Vec<Step>,
    pub usage: Option<Usage>,
}

pub(crate) trait ErasedAgent: Send + Sync + 'static {
    fn call(&self, input: agent::Input, cancellation: Cancellation) -> AgentFuture;
}

struct TypedAgent<A> {
    agent: Arc<A>,
}

impl<A: Agent + 'static> ErasedAgent for TypedAgent<A> {
    fn call(&self, input: agent::Input, cancellation: Cancellation) -> AgentFuture {
        let agent = Arc::clone(&self.agent);
        Box::pin(async move {
            let mut messages = Messages::default();
            let mut steps = Vec::new();
            let mut usage = None;
            let stream = match agent.run(input, cancellation.clone()).await {
                Ok(stream) => stream,
                Err(error) => {
                    return AgentRun {
                        result: Err(error),
                        messages,
                        steps,
                        usage,
                    };
                }
            };
            futures_util::pin_mut!(stream);
            let mut final_assistant = None;
            while let Some(event) = stream.next().await {
                match event {
                    Ok(Event::Delta { .. }) => {}
                    Ok(Event::Complete {
                        message,
                        usage: call_usage,
                        finish_reason,
                    }) => {
                        let cancelled = finish_reason.as_deref() == Some("cancelled");
                        let kind = match message.role {
                            Role::Assistant => "model",
                            Role::Tool => "tool",
                            Role::Function => "function",
                            _ => "agent",
                        };
                        let index = steps.len() + 1;
                        steps.push(Step {
                            id: format!("complete_{index}"),
                            kind: kind.into(),
                            status: if cancelled { "cancelled" } else { "succeeded" }.into(),
                            from: Vec::new(),
                            to: Vec::new(),
                            call_id: None,
                            attempts: vec![StepAttempt {
                                status: if cancelled {
                                    AttemptStatus::Cancelled
                                } else {
                                    AttemptStatus::Succeeded
                                },
                                started_time: None,
                                finished_time: None,
                                usage: call_usage.clone(),
                                finish_reason,
                                error: None,
                            }],
                            error: None,
                        });
                        usage = sum_usage(usage, call_usage);
                        if message.role == Role::Assistant {
                            final_assistant = Some(message.clone());
                        }
                        messages.push(message);
                        if cancelled {
                            return AgentRun {
                                result: Err(Error::new("CANCELLED", "agent run cancelled")),
                                messages,
                                steps,
                                usage,
                            };
                        }
                    }
                    Err(error) => {
                        return AgentRun {
                            result: Err(error),
                            messages,
                            steps,
                            usage,
                        };
                    }
                }
            }
            AgentRun {
                result: final_assistant
                    .map(|message| Messages::new([message]))
                    .ok_or_else(|| {
                        Error::new(
                            "INVALID_RESPONSE",
                            "agent produced no final Assistant Complete",
                        )
                    }),
                messages,
                steps,
                usage,
            }
        })
    }
}

fn sum_usage(left: Option<Usage>, right: Option<Usage>) -> Option<Usage> {
    crate::components::model::token::merge(left, right)
}

/// Internal heterogeneous execution boundary, not a second public Node API.
#[allow(dead_code)]
pub(crate) trait ErasedNode: Send + Sync + 'static {
    fn call(&self, input: Value, cancellation: Cancellation) -> BoxFuture;
}

#[allow(dead_code)]
struct TypedNode<N, I, O, E> {
    node: Arc<N>,
    input_schema: jsonschema::Validator,
    output_schema: jsonschema::Validator,
    marker: std::marker::PhantomData<fn(I, O, E)>,
}

impl<N, I, O, E> ErasedNode for TypedNode<N, I, O, E>
where
    N: Node<I, O, E>,
    I: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
    E: Into<Error> + Send + 'static,
{
    fn call(&self, input: Value, cancellation: Cancellation) -> BoxFuture {
        // Validate before deserialization and before calling user code; Serde
        // alone does not enforce JSON Schema ranges, lengths or patterns.
        if let Err(error) = crate::components::node::schema::validate(
            &self.input_schema,
            &input,
            "INVALID_ARGUMENTS",
            "node input violates its declared schema",
        ) {
            return Box::pin(async move { Err(error) });
        }
        let node = Arc::clone(&self.node);
        let output_schema = self.output_schema.clone();
        Box::pin(async move {
            let input: I = serde_json::from_value(input)
                .map_err(|error| Error::new("INVALID_ARGUMENTS", error.to_string()))?;
            match node.call(input, cancellation).await.map_err(Into::into)? {
                Signal::Complete(output) => {
                    let value = serde_json::to_value(output)
                        .map_err(|error| Error::new("INVALID_OUTPUT", error.to_string()))?;
                    crate::components::node::schema::validate(
                        &output_schema,
                        &value,
                        "INVALID_OUTPUT",
                        "node output violates its declared schema",
                    )?;
                    Ok(Signal::Complete(value))
                }
                Signal::Wait(wait) => Ok(Signal::Wait(wait)),
            }
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct IoSchema {
    pub input: Value,
    pub output: Value,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FieldSource {
    pub node_id: String,
    pub field: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct InputBinding {
    pub from: FieldSource,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NodeAttrs {
    #[serde(default)]
    pub input: BTreeMap<String, InputBinding>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NodeDefinition {
    pub node_id: String,
    pub node_name: Option<String>,
    pub io: IoSchema,
    pub attrs: NodeAttrs,
    pub retry: RetryPolicy,
    #[serde(default)]
    pub workflow: Option<GraphReference>,
    #[serde(default)]
    pub agent: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphReference {
    pub definition_id: String,
    pub version: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Endpoint {
    Node(String),
    Start,
    End,
}

impl Endpoint {
    pub(crate) fn node(&self) -> Option<&str> {
        match self {
            Self::Node(id) => Some(id),
            Self::Start | Self::End => None,
        }
    }
}

impl From<&str> for Endpoint {
    fn from(value: &str) -> Self {
        match value {
            START => Self::Start,
            END => Self::End,
            node_id => Self::Node(node_id.into()),
        }
    }
}

impl From<String> for Endpoint {
    fn from(value: String) -> Self {
        Self::from(value.as_str())
    }
}

impl Serialize for Endpoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(match self {
            Self::Node(id) => id,
            Self::Start => START,
            Self::End => END,
        })
    }
}

impl<'de> Deserialize<'de> for Endpoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Self::from(value))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Edge {
    pub from: Endpoint,
    pub to: Endpoint,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Join {
    pub from: Vec<String>,
    pub to: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Conditional {
    pub from: String,
    pub to: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FanOut {
    pub from: String,
    pub to: String,
    pub mode: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Loop {
    pub from: String,
    pub to: String,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct FailureRelation {
    pub from: Vec<String>,
    pub to: Vec<String>,
    pub on: String,
    #[serde(default)]
    pub codes: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphDefinition {
    pub ordinary_edges: Vec<Edge>,
    pub join_definitions: Vec<Join>,
    #[serde(default)]
    pub conditional_edges: Vec<Conditional>,
    #[serde(default)]
    pub fan_out_definitions: Vec<FanOut>,
    #[serde(default)]
    pub loop_definitions: Vec<Loop>,
    #[serde(default)]
    pub failure_relations: Vec<FailureRelation>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Definition {
    pub definition_id: String,
    pub version: String,
    pub nodes: BTreeMap<String, NodeDefinition>,
    pub graph: GraphDefinition,
}

/// Lifecycle of one persisted Workflow graph state.
///
/// This is state data, not a scheduler.  A Runtime may advance it while it
/// executes a compiled definition, but the state itself only describes the
/// current step, waits, and each logical node's execution facts.
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Wait {
    pub node_id: String,
    pub exec_id: i64,
    pub wait_id: String,
}

/// Serializable Workflow Runtime state for one `chat_id`.
///
/// Node execution data remains nested under `nodes[node_id]`; `waits` is a
/// graph-level index keyed by `node_id` so restoring a waiting node does not
/// require scanning every node.  This type intentionally does not run a
/// scheduler or persist an external Checkpoint.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct State {
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
    pub waits: BTreeMap<String, Wait>,
    pub nodes: BTreeMap<String, crate::runtime::workflow::node::State>,
}

impl State {
    pub fn new(definition: &Definition) -> Self {
        let nodes = definition
            .nodes
            .keys()
            .map(|node_id| {
                (
                    node_id.clone(),
                    crate::runtime::workflow::node::State::new(
                        crate::runtime::workflow::node::Status::Idle,
                    ),
                )
            })
            .collect();

        Self {
            definition_id: definition.definition_id.clone(),
            version: definition.version.clone(),
            step: 0,
            status: Status::Running,
            usage: None,
            initial: None,
            waits: BTreeMap::new(),
            nodes,
        }
    }

    /// Rebuilds the graph-level usage from committed node Attempt details.
    /// This is deliberately idempotent and safe to call after restore.
    pub(crate) fn recompute_usage(&mut self) {
        let mut usage = None;
        for state in self.nodes.values_mut() {
            for execution in state.executions.values_mut() {
                let mut execution_usage = None;
                for attempt in &mut execution.attempts {
                    // A Graph Node's Attempt is a query projection of the
                    // child's calls, not an extra model call of its own.
                    if let Some(child) = &mut attempt.graph {
                        child.recompute_usage();
                        attempt.usage = child.usage.clone();
                    }
                    execution_usage = sum_usage(execution_usage, attempt.usage.clone());
                }
                execution.usage = execution_usage;
                usage = sum_usage(usage, execution.usage.clone());
            }
        }
        self.usage = usage;
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

    pub fn add_wait(&mut self, wait: Wait) -> Result<(), Error> {
        self.node(&wait.node_id)?;
        if self.waits.contains_key(&wait.node_id) {
            return Err(Error::new("CONFLICT", "workflow node already has a wait"));
        }
        self.waits.insert(wait.node_id.clone(), wait);
        Ok(())
    }

    pub fn remove_wait(&mut self, node_id: &str) -> Option<Wait> {
        self.waits.remove(node_id)
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
                            crate::runtime::workflow::node::ExecutionStatus::Succeeded
                                | crate::runtime::workflow::node::ExecutionStatus::Failed
                                | crate::runtime::workflow::node::ExecutionStatus::Cancelled
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

    /// Resolves the single active wait for a node. The response is kept in
    /// the execution facts so a restored runtime can resume without a second
    /// mailbox or scheduler record.
    pub fn resolve_wait(&mut self, node_id: &str, response: Messages) -> Result<(), Error> {
        let wait = match self.waits.get(node_id).cloned() {
            Some(wait) => wait,
            None => {
                let already_resolved = self.nodes.get(node_id).and_then(|node| {
                    node.executions.values().find_map(|execution| {
                        execution.waits.values().find_map(|wait| {
                            (wait.status == "resolved").then(|| wait.response.clone())
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
        let node = self.node_mut(node_id)?;
        let execution = node.execution_mut(wait.exec_id)?;
        for attempt in &mut execution.attempts {
            if let Some(child) = attempt.graph.as_mut() {
                let child_waits = child.waits.keys().cloned().collect::<Vec<_>>();
                if child_waits.len() > 1 {
                    return Err(Error::new(
                        "CONFLICT",
                        "nested workflow has multiple active waits",
                    ));
                }
                if let Some(child_node_id) = child_waits.into_iter().next() {
                    child.resolve_wait(&child_node_id, response.clone())?;
                }
            }
        }
        let record = execution
            .waits
            .get_mut(&wait.wait_id)
            .ok_or_else(|| Error::new("INVALID_STATE", "active wait is missing from execution"))?;
        if record.status == "resolved" {
            if record.response.as_ref() == Some(&response) {
                return Ok(());
            }
            return Err(Error::new(
                "CONFLICT",
                "wait was already resolved differently",
            ));
        }
        if record.status != "waiting"
            || execution.status != crate::runtime::workflow::node::ExecutionStatus::Waiting
        {
            return Err(Error::new("INVALID_STATE", "execution is not waiting"));
        }
        record.status = "resolved".into();
        record.resolved_time = Some(0);
        record.response = Some(response);
        execution.status = crate::runtime::workflow::node::ExecutionStatus::Running;
        if let Some(attempt) = execution.attempts.last_mut() {
            attempt.status = crate::runtime::workflow::node::AttemptStatus::Running;
        }
        node.status = crate::runtime::workflow::node::Status::Active;
        self.waits.remove(node_id);
        self.status = Status::Running;
        Ok(())
    }

    /// Returns one resolved wait that must be resumed by the runtime.
    pub fn resolved_wait(&self) -> Option<(String, i64, String, Messages)> {
        self.nodes.iter().find_map(|(node_id, node)| {
            node.executions.values().find_map(|execution| {
                if execution.status != crate::runtime::workflow::node::ExecutionStatus::Running {
                    return None;
                }
                execution.waits.iter().find_map(|(wait_id, wait)| {
                    (wait.status == "resolved").then(|| {
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
                        crate::runtime::workflow::node::ExecutionStatus::Pending
                            | crate::runtime::workflow::node::ExecutionStatus::Running
                            | crate::runtime::workflow::node::ExecutionStatus::Waiting
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
        use crate::runtime::workflow::node::ExecutionStatus;

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
                                    ExecutionStatus::Succeeded
                                        | ExecutionStatus::Failed
                                        | ExecutionStatus::Cancelled
                                )
                        })
                        .max_by_key(|execution| (execution.idx, execution.exec_id))
                })?;
                any_failed |= execution.status == ExecutionStatus::Failed
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
            .join_definitions
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
                        let normal = execution.status == ExecutionStatus::Succeeded
                            && execution.to.iter().any(|target| target == node_id);
                        let failure = execution.status == ExecutionStatus::Failed
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
        node.status = crate::runtime::workflow::node::Status::Active;
        node.insert(crate::runtime::workflow::node::Execution {
            exec_id,
            idx: step,
            status: crate::runtime::workflow::node::ExecutionStatus::Running,
            from,
            ..Default::default()
        })?;
        node.append_attempt(
            exec_id,
            crate::runtime::workflow::node::Attempt {
                idx: 0,
                status: crate::runtime::workflow::node::AttemptStatus::Running,
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
            crate::runtime::workflow::node::ExecutionStatus::Running
                | crate::runtime::workflow::node::ExecutionStatus::Waiting
        ) {
            return Err(Error::new("INVALID_STATE", "execution is not active"));
        }
        execution.status = crate::runtime::workflow::node::ExecutionStatus::Succeeded;
        if let Some(attempt) = execution.attempts.last_mut() {
            attempt.status = crate::runtime::workflow::node::AttemptStatus::Succeeded;
        }
        execution.messages = messages;
        execution.to = to;
        node.status = crate::runtime::workflow::node::Status::Succeeded;
        Ok(())
    }

    pub fn wait_execution(
        &mut self,
        node_id: &str,
        exec_id: i64,
        wait_id: String,
    ) -> Result<(), Error> {
        if self.waits.contains_key(node_id) {
            return Err(Error::new("CONFLICT", "workflow node already has a wait"));
        }
        let node = self.node_mut(node_id)?;
        let execution = node.execution_mut(exec_id)?;
        if execution.status != crate::runtime::workflow::node::ExecutionStatus::Running {
            return Err(Error::new("INVALID_STATE", "execution is not running"));
        }
        if execution.waits.contains_key(&wait_id) {
            return Err(Error::new("CONFLICT", "execution wait already exists"));
        }
        execution.status = crate::runtime::workflow::node::ExecutionStatus::Waiting;
        if let Some(attempt) = execution.attempts.last_mut() {
            attempt.status = crate::runtime::workflow::node::AttemptStatus::Waiting;
        }
        execution.waits.insert(
            wait_id.clone(),
            crate::runtime::workflow::node::Wait {
                kind: "external".into(),
                status: "waiting".into(),
                created_time: 0,
                resolved_time: None,
                response: None,
            },
        );
        self.add_wait(Wait {
            node_id: node_id.into(),
            exec_id,
            wait_id,
        })
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
            crate::runtime::workflow::node::ExecutionStatus::Running
                | crate::runtime::workflow::node::ExecutionStatus::Waiting
        ) {
            return Err(Error::new("INVALID_STATE", "execution is not active"));
        }
        execution.status = crate::runtime::workflow::node::ExecutionStatus::Failed;
        if let Some(attempt) = execution.attempts.last_mut() {
            attempt.status = crate::runtime::workflow::node::AttemptStatus::Failed;
            attempt.error = Some(error.clone());
        }
        execution.error = Some(error);
        node.status = crate::runtime::workflow::node::Status::Failed;
        Ok(())
    }

    pub fn cancel_execution(&mut self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        let node = self.node_mut(node_id)?;
        let execution = node.execution_mut(exec_id)?;
        if !matches!(
            execution.status,
            crate::runtime::workflow::node::ExecutionStatus::Running
                | crate::runtime::workflow::node::ExecutionStatus::Waiting
        ) {
            return Err(Error::new("INVALID_STATE", "execution is not active"));
        }
        execution.status = crate::runtime::workflow::node::ExecutionStatus::Cancelled;
        if let Some(attempt) = execution.attempts.last_mut() {
            attempt.status = crate::runtime::workflow::node::AttemptStatus::Cancelled;
        }
        node.status = crate::runtime::workflow::node::Status::Cancelled;
        Ok(())
    }

    /// Completes the current barrier and advances the graph step. A waiting,
    /// running, or pending execution keeps the current step unchanged.
    pub fn advance_step(&mut self) -> bool {
        let blocked = self.nodes.values().any(|node| {
            node.executions.values().any(|execution| {
                execution.idx == self.step
                    && matches!(
                        execution.status,
                        crate::runtime::workflow::node::ExecutionStatus::Pending
                            | crate::runtime::workflow::node::ExecutionStatus::Running
                            | crate::runtime::workflow::node::ExecutionStatus::Waiting
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

#[derive(Clone)]
pub struct CompiledWorkflow {
    inner: Arc<CompiledInner>,
}

struct CompiledInner {
    definition: Definition,
    #[allow(dead_code)]
    executors: BTreeMap<String, Arc<dyn ErasedNode>>,
    graph_nodes: BTreeMap<String, Arc<CompiledWorkflow>>,
    agents: BTreeMap<String, Arc<dyn ErasedAgent>>,
    codecs: BTreeMap<String, (bool, bool)>,
    routes: BTreeMap<String, RouteFn>,
    loops: BTreeMap<String, LoopFn>,
}

impl std::fmt::Debug for CompiledWorkflow {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompiledWorkflow")
            .field("definition", &self.inner.definition)
            .finish()
    }
}

impl CompiledWorkflow {
    pub fn definition(&self) -> &Definition {
        &self.inner.definition
    }

    pub fn graph_node(&self, node_id: &str) -> Option<&CompiledWorkflow> {
        self.inner.graph_nodes.get(node_id).map(Arc::as_ref)
    }

    pub(crate) fn agent(&self, node_id: &str) -> Option<Arc<dyn ErasedAgent>> {
        self.inner.agents.get(node_id).cloned()
    }

    fn contains_definition(&self, definition_id: &str) -> bool {
        self.inner.definition.definition_id == definition_id
            || self
                .inner
                .graph_nodes
                .values()
                .any(|workflow| workflow.contains_definition(definition_id))
    }

    #[allow(dead_code)]
    pub(crate) fn executor(&self, node_id: &str) -> Option<Arc<dyn ErasedNode>> {
        self.inner.executors.get(node_id).cloned()
    }

    pub(crate) fn input_is_messages(&self, node_id: &str) -> bool {
        self.inner.codecs.get(node_id).is_some_and(|codec| codec.0)
    }

    pub(crate) fn output_is_messages(&self, node_id: &str) -> bool {
        self.inner.codecs.get(node_id).is_some_and(|codec| codec.1)
    }

    pub(crate) fn route(&self, node_id: &str, messages: &Messages) -> Result<Vec<String>, Error> {
        self.inner
            .routes
            .get(node_id)
            .map(|route| route(messages))
            .unwrap_or_else(|| Ok(Vec::new()))
    }

    pub(crate) fn loop_enabled(&self, node_id: &str, messages: &Messages) -> bool {
        self.inner
            .loops
            .get(node_id)
            .is_some_and(|condition| condition(messages))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GraphError {
    pub code: String,
    pub message: String,
    pub details: Option<String>,
}

impl GraphError {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for GraphError {}

pub struct WorkflowBuilder {
    definition_id: String,
    max_nodes: usize,
    nodes: BTreeMap<String, NodeDefinition>,
    codecs: BTreeMap<String, (bool, bool)>,
    executors: BTreeMap<String, Arc<dyn ErasedNode>>,
    graph_nodes: BTreeMap<String, Arc<CompiledWorkflow>>,
    agents: BTreeMap<String, Arc<dyn ErasedAgent>>,
    graph: GraphDefinition,
    routes: BTreeMap<String, RouteFn>,
    loops: BTreeMap<String, LoopFn>,
}

impl WorkflowBuilder {
    pub fn new(definition_id: impl Into<String>) -> Self {
        Self {
            definition_id: definition_id.into(),
            max_nodes: 100,
            nodes: BTreeMap::new(),
            codecs: BTreeMap::new(),
            executors: BTreeMap::new(),
            graph_nodes: BTreeMap::new(),
            agents: BTreeMap::new(),
            graph: GraphDefinition::default(),
            routes: BTreeMap::new(),
            loops: BTreeMap::new(),
        }
    }

    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    pub fn add_node<I, O, E, N>(
        &mut self,
        node_id: impl Into<String>,
        node: N,
    ) -> Result<&mut Self, GraphError>
    where
        N: Node<I, O, E>,
        I: DeserializeOwned + JsonSchema + Send + 'static,
        O: Serialize + JsonSchema + Send + 'static,
        E: Into<Error> + Send + 'static,
    {
        let node_id = node_id.into();
        validate_node_id(&node_id)?;
        if self.nodes.contains_key(&node_id) {
            return Err(GraphError::new(
                "DUPLICATE_NODE",
                format!("duplicate node: {node_id}"),
            ));
        }
        if self.nodes.len() >= self.max_nodes {
            return Err(GraphError::new("MAX_NODES", "workflow node limit exceeded"));
        }

        let input = serde_json::to_value(schema_for!(I))
            .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
        let output = serde_json::to_value(schema_for!(O))
            .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
        let input_messages = TypeId::of::<I>() == TypeId::of::<Messages>();
        let output_messages = TypeId::of::<O>() == TypeId::of::<Messages>();
        if (!input_messages && input["type"] != "object")
            || (!output_messages && output["type"] != "object")
        {
            return Err(GraphError::new(
                "INVALID_SCHEMA",
                "ordinary Node I/O schema roots must be objects; Messages is the only exception",
            ));
        }

        // Both validators must compile before changing the Builder. The
        // immutable schema JSON still participates in the definition version;
        // executable validators never enter State or Checkpoint.
        let input_schema = crate::components::node::schema::compile(&input).map_err(|error| {
            let mut graph_error = GraphError::new(&error.code, error.message);
            graph_error.details = error.details;
            graph_error
        })?;
        let output_schema = crate::components::node::schema::compile(&output).map_err(|error| {
            let mut graph_error = GraphError::new(&error.code, error.message);
            graph_error.details = error.details;
            graph_error
        })?;

        self.nodes.insert(
            node_id.clone(),
            NodeDefinition {
                node_id: node_id.clone(),
                node_name: None,
                io: IoSchema { input, output },
                attrs: NodeAttrs::default(),
                retry: RetryPolicy::default(),
                workflow: None,
                agent: false,
            },
        );
        self.codecs
            .insert(node_id.clone(), (input_messages, output_messages));
        self.executors.insert(
            node_id,
            Arc::new(TypedNode::<N, I, O, E> {
                node: Arc::new(node),
                input_schema,
                output_schema,
                marker: std::marker::PhantomData,
            }),
        );
        Ok(self)
    }

    /// Registers a framework Agent without making it implement ordinary Node.
    pub fn add_agent_node<A: Agent + 'static>(
        &mut self,
        node_id: impl Into<String>,
        agent: A,
    ) -> Result<&mut Self, GraphError> {
        let node_id = node_id.into();
        validate_node_id(&node_id)?;
        if self.nodes.contains_key(&node_id) {
            return Err(GraphError::new(
                "DUPLICATE_NODE",
                format!("duplicate node: {node_id}"),
            ));
        }
        if self.nodes.len() >= self.max_nodes {
            return Err(GraphError::new("MAX_NODES", "workflow node limit exceeded"));
        }
        let input = serde_json::to_value(schema_for!(agent::Input))
            .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
        let output = serde_json::to_value(schema_for!(Messages))
            .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
        self.nodes.insert(
            node_id.clone(),
            NodeDefinition {
                node_id: node_id.clone(),
                node_name: None,
                io: IoSchema { input, output },
                attrs: NodeAttrs::default(),
                retry: RetryPolicy::default(),
                workflow: None,
                agent: true,
            },
        );
        self.codecs.insert(node_id.clone(), (false, true));
        self.agents.insert(
            node_id,
            Arc::new(TypedAgent {
                agent: Arc::new(agent),
            }),
        );
        Ok(self)
    }

    /// Adds an already compiled Workflow as one explicit parent-graph node.
    /// The child definition remains a distinct graph boundary and is not
    /// erased into the ordinary Node executor map.
    pub fn add_graph_node(
        &mut self,
        node_id: impl Into<String>,
        workflow: &CompiledWorkflow,
    ) -> Result<&mut Self, GraphError> {
        let node_id = node_id.into();
        validate_node_id(&node_id)?;
        if self.nodes.contains_key(&node_id) {
            return Err(GraphError::new(
                "DUPLICATE_NODE",
                format!("duplicate node: {node_id}"),
            ));
        }
        if self.nodes.len() >= self.max_nodes {
            return Err(GraphError::new("MAX_NODES", "workflow node limit exceeded"));
        }
        if workflow.contains_definition(&self.definition_id) {
            return Err(GraphError::new(
                "RECURSIVE_GRAPH",
                "a Workflow cannot contain itself directly or indirectly",
            ));
        }

        let input = serde_json::to_value(schema_for!(Messages))
            .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
        let output = input.clone();
        let definition = GraphReference {
            definition_id: workflow.definition().definition_id.clone(),
            version: workflow.definition().version.clone(),
        };
        self.nodes.insert(
            node_id.clone(),
            NodeDefinition {
                node_id: node_id.clone(),
                node_name: None,
                io: IoSchema { input, output },
                attrs: NodeAttrs::default(),
                retry: RetryPolicy::default(),
                workflow: Some(definition),
                agent: false,
            },
        );
        self.codecs.insert(node_id.clone(), (true, true));
        self.graph_nodes.insert(node_id, Arc::new(workflow.clone()));
        Ok(self)
    }

    pub fn set_attrs(&mut self, node_id: &str, attrs: NodeAttrs) -> Result<&mut Self, GraphError> {
        self.node_mut(node_id)?.attrs = attrs;
        Ok(self)
    }

    pub fn set_retry(
        &mut self,
        node_id: &str,
        retry: RetryPolicy,
    ) -> Result<&mut Self, GraphError> {
        if retry.min_delay_ms > retry.max_delay_ms {
            return Err(GraphError::new(
                "INVALID_RETRY",
                "min_delay_ms exceeds max_delay_ms",
            ));
        }
        self.node_mut(node_id)?.retry = retry;
        Ok(self)
    }

    pub fn add_edge(
        &mut self,
        from: impl Into<Endpoint>,
        to: impl Into<Endpoint>,
    ) -> Result<&mut Self, GraphError> {
        let edge = Edge {
            from: from.into(),
            to: to.into(),
        };
        self.validate_edge(&edge)?;
        if self.graph.ordinary_edges.contains(&edge) {
            return Err(GraphError::new("DUPLICATE_EDGE", "duplicate ordinary edge"));
        }
        if self
            .graph
            .join_definitions
            .iter()
            .any(|join| edge.to.node() == Some(&join.to))
        {
            return Err(GraphError::new(
                "CONFLICTING_ACTIVATION",
                "Join target cannot have an ordinary incoming edge",
            ));
        }
        self.graph.ordinary_edges.push(edge);
        Ok(self)
    }

    pub fn add_join(
        &mut self,
        from: impl IntoIterator<Item = impl Into<String>>,
        to: impl Into<String>,
    ) -> Result<&mut Self, GraphError> {
        let join = Join {
            from: from.into_iter().map(Into::into).collect(),
            to: to.into(),
        };
        if join.from.len() < 2 || join.from.iter().collect::<BTreeSet<_>>().len() != join.from.len()
        {
            return Err(GraphError::new(
                "INVALID_JOIN",
                "Join requires at least two distinct sources",
            ));
        }
        self.require_node(&join.to)?;
        for source in &join.from {
            self.require_node(source)?;
        }
        if self
            .graph
            .join_definitions
            .iter()
            .any(|existing| existing.to == join.to)
        {
            return Err(GraphError::new(
                "DUPLICATE_JOIN",
                "target already has a Join",
            ));
        }
        if self
            .graph
            .ordinary_edges
            .iter()
            .any(|edge| edge.to.node() == Some(&join.to))
        {
            return Err(GraphError::new(
                "CONFLICTING_ACTIVATION",
                "Join target cannot have an ordinary incoming edge",
            ));
        }
        self.graph.join_definitions.push(join);
        Ok(self)
    }

    /// Declares the statically allowed targets of a conditional relation.
    /// The actual selected targets are still committed in NodeExecution.to.
    pub fn add_conditional_edges<F>(
        &mut self,
        from: impl Into<String>,
        route: F,
        to: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<&mut Self, GraphError>
    where
        F: Fn(&Messages) -> Result<Vec<String>, Error> + Send + Sync + 'static,
    {
        let source = from.into();
        let relation = Conditional {
            from: source.clone(),
            to: to.into_iter().map(Into::into).collect(),
        };
        self.require_node(&relation.from)?;
        if relation.to.is_empty()
            || relation.to.iter().collect::<BTreeSet<_>>().len() != relation.to.len()
        {
            return Err(GraphError::new(
                "INVALID_CONDITION",
                "conditional relation requires distinct targets",
            ));
        }
        for target in &relation.to {
            self.require_node(target)?;
        }
        if self
            .graph
            .conditional_edges
            .iter()
            .any(|existing| existing.from == relation.from && existing.to == relation.to)
        {
            return Err(GraphError::new(
                "DUPLICATE_EDGE",
                "duplicate conditional edge",
            ));
        }
        self.graph.conditional_edges.push(relation);
        self.routes.insert(source, Arc::new(route));
        Ok(self)
    }

    pub fn add_fan_out_edges(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
    ) -> Result<&mut Self, GraphError> {
        let relation = FanOut {
            from: from.into(),
            to: to.into(),
            mode: "multi".into(),
        };
        self.require_node(&relation.from)?;
        self.require_node(&relation.to)?;
        if self.graph.fan_out_definitions.contains(&relation) {
            return Err(GraphError::new("DUPLICATE_EDGE", "duplicate fan-out edge"));
        }
        self.graph.fan_out_definitions.push(relation);
        Ok(self)
    }

    pub fn add_loop_edge<F>(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: F,
    ) -> Result<&mut Self, GraphError>
    where
        F: Fn(&Messages) -> bool + Send + Sync + 'static,
    {
        let source = from.into();
        let relation = Loop {
            from: source.clone(),
            to: to.into(),
        };
        self.require_node(&relation.from)?;
        self.require_node(&relation.to)?;
        if relation.from == relation.to {
            return Err(GraphError::new("INVALID_LOOP", "self-loop is not allowed"));
        }
        if self.graph.loop_definitions.contains(&relation) {
            return Err(GraphError::new("DUPLICATE_EDGE", "duplicate loop edge"));
        }
        self.graph.loop_definitions.push(relation);
        self.loops.insert(source, Arc::new(condition));
        Ok(self)
    }

    pub fn add_failure_edges(
        &mut self,
        from: impl IntoIterator<Item = impl Into<String>>,
        to: impl IntoIterator<Item = impl Into<String>>,
        on: impl Into<String>,
    ) -> Result<&mut Self, GraphError> {
        let relation = FailureRelation {
            from: from.into_iter().map(Into::into).collect(),
            to: to.into_iter().map(Into::into).collect(),
            on: on.into(),
            codes: Vec::new(),
        };
        if relation.from.is_empty() || relation.to.is_empty() {
            return Err(GraphError::new(
                "INVALID_FAILURE_RELATION",
                "failure relation requires from and to",
            ));
        }
        if relation.on != "node_failed" && relation.on != "join_failed" {
            return Err(GraphError::new(
                "INVALID_FAILURE_RELATION",
                "failure relation on must be node_failed or join_failed",
            ));
        }
        if relation.on == "node_failed" && relation.from.len() != 1 {
            return Err(GraphError::new(
                "INVALID_FAILURE_RELATION",
                "node_failed relation requires exactly one source",
            ));
        }
        for node_id in relation.from.iter().chain(relation.to.iter()) {
            self.require_node(node_id)?;
        }
        if relation.on == "join_failed" {
            let sources: BTreeSet<&str> = relation.from.iter().map(String::as_str).collect();
            let matching_joins = self
                .graph
                .join_definitions
                .iter()
                .filter(|join| {
                    join.from.len() == sources.len()
                        && join
                            .from
                            .iter()
                            .map(String::as_str)
                            .collect::<BTreeSet<_>>()
                            == sources
                })
                .count();
            if matching_joins != 1 {
                return Err(GraphError::new(
                    "INVALID_FAILURE_RELATION",
                    "join_failed relation must match exactly one Join source set",
                ));
            }
        }
        if self.graph.failure_relations.contains(&relation) {
            return Err(GraphError::new(
                "DUPLICATE_FAILURE_RELATION",
                "duplicate failure relation",
            ));
        }
        self.graph.failure_relations.push(relation);
        Ok(self)
    }

    pub fn compile(self) -> Result<CompiledWorkflow, GraphError> {
        if self.definition_id.trim().is_empty() {
            return Err(GraphError::new(
                "INVALID_DEFINITION_ID",
                "definition_id must be non-empty",
            ));
        }
        if self.nodes.is_empty() {
            return Err(GraphError::new(
                "EMPTY_GRAPH",
                "workflow must contain a node",
            ));
        }
        self.validate_graph()?;
        let version = self.version()?;
        let definition = Definition {
            definition_id: self.definition_id,
            version,
            nodes: self.nodes,
            graph: self.graph,
        };
        Ok(CompiledWorkflow {
            inner: Arc::new(CompiledInner {
                definition,
                executors: self.executors,
                graph_nodes: self.graph_nodes,
                agents: self.agents,
                codecs: self.codecs,
                routes: self.routes,
                loops: self.loops,
            }),
        })
    }

    fn node_mut(&mut self, node_id: &str) -> Result<&mut NodeDefinition, GraphError> {
        self.nodes
            .get_mut(node_id)
            .ok_or_else(|| GraphError::new("UNKNOWN_NODE", format!("unknown node: {node_id}")))
    }

    fn require_node(&self, node_id: &str) -> Result<(), GraphError> {
        if self.nodes.contains_key(node_id) {
            Ok(())
        } else {
            Err(GraphError::new(
                "UNKNOWN_NODE",
                format!("unknown node: {node_id}"),
            ))
        }
    }

    fn validate_edge(&self, edge: &Edge) -> Result<(), GraphError> {
        if matches!(edge.from, Endpoint::End)
            || matches!(edge.to, Endpoint::Start)
            || matches!((&edge.from, &edge.to), (Endpoint::Start, Endpoint::End))
        {
            return Err(GraphError::new(
                "INVALID_ENDPOINT",
                "invalid START/END connection",
            ));
        }
        if let Some(node_id) = edge.from.node() {
            self.require_node(node_id)?;
        }
        if let Some(node_id) = edge.to.node() {
            self.require_node(node_id)?;
        }
        Ok(())
    }

    fn validate_graph(&self) -> Result<(), GraphError> {
        let mut entries = Vec::new();
        let mut exits = BTreeSet::new();
        let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for edge in &self.graph.ordinary_edges {
            match (&edge.from, &edge.to) {
                (Endpoint::Start, Endpoint::Node(target)) => entries.push(target.as_str()),
                (Endpoint::Node(source), Endpoint::End) => {
                    exits.insert(source.as_str());
                }
                (Endpoint::Node(source), Endpoint::Node(target)) => {
                    adjacency.entry(source).or_default().push(target);
                }
                _ => return Err(GraphError::new("INVALID_ENDPOINT", "invalid ordinary edge")),
            }
        }
        if entries.is_empty() || exits.is_empty() {
            return Err(GraphError::new(
                "MISSING_BOUNDARY",
                "workflow requires START and END connections",
            ));
        }
        for join in &self.graph.join_definitions {
            for source in &join.from {
                adjacency.entry(source).or_default().push(&join.to);
            }
        }
        for relation in &self.graph.conditional_edges {
            for target in &relation.to {
                adjacency
                    .entry(relation.from.as_str())
                    .or_default()
                    .push(target.as_str());
            }
        }
        for relation in &self.graph.fan_out_definitions {
            adjacency
                .entry(relation.from.as_str())
                .or_default()
                .push(relation.to.as_str());
        }
        for relation in &self.graph.loop_definitions {
            adjacency
                .entry(relation.from.as_str())
                .or_default()
                .push(relation.to.as_str());
        }
        for relation in &self.graph.failure_relations {
            for source in &relation.from {
                for target in &relation.to {
                    adjacency
                        .entry(source.as_str())
                        .or_default()
                        .push(target.as_str());
                }
            }
        }
        let mut reachable = BTreeSet::new();
        let mut queue: VecDeque<&str> = entries.into();
        while let Some(node) = queue.pop_front() {
            if reachable.insert(node) {
                queue.extend(adjacency.get(node).into_iter().flatten().copied());
            }
        }
        if self
            .nodes
            .keys()
            .any(|node| !reachable.contains(node.as_str()))
        {
            return Err(GraphError::new(
                "UNREACHABLE_NODE",
                "every Node must be reachable from START",
            ));
        }
        if !exits.iter().any(|exit| reachable.contains(exit)) {
            return Err(GraphError::new(
                "UNREACHABLE_END",
                "END is not reachable from START",
            ));
        }
        self.validate_bindings()
    }

    fn validate_bindings(&self) -> Result<(), GraphError> {
        for (target_id, target) in &self.nodes {
            let input_messages = self.codecs[target_id].0;
            if input_messages && !target.attrs.input.is_empty() {
                return Err(GraphError::new(
                    "INVALID_BINDING",
                    "Messages input cannot use attrs.input field bindings",
                ));
            }
            let properties = target.io.input["properties"].as_object();
            let incoming: BTreeSet<&str> = self
                .graph
                .ordinary_edges
                .iter()
                .filter(|edge| edge.to.node() == Some(target_id))
                .filter_map(|edge| edge.from.node())
                .chain(
                    self.graph
                        .join_definitions
                        .iter()
                        .filter(|join| &join.to == target_id)
                        .flat_map(|join| join.from.iter().map(String::as_str)),
                )
                .chain(
                    self.graph
                        .conditional_edges
                        .iter()
                        .filter(|relation| relation.to.iter().any(|target| target == target_id))
                        .map(|relation| relation.from.as_str()),
                )
                .chain(
                    self.graph
                        .fan_out_definitions
                        .iter()
                        .filter(|relation| relation.to == *target_id)
                        .map(|relation| relation.from.as_str()),
                )
                .chain(
                    self.graph
                        .loop_definitions
                        .iter()
                        .filter(|relation| relation.to == *target_id)
                        .map(|relation| relation.from.as_str()),
                )
                .collect();
            let entry = self.graph.ordinary_edges.iter().any(|edge| {
                matches!(edge.from, Endpoint::Start) && edge.to.node() == Some(target_id)
            });
            for (field, binding) in &target.attrs.input {
                if !properties.is_some_and(|properties| properties.contains_key(field)) {
                    return Err(GraphError::new(
                        "INVALID_BINDING",
                        format!("undeclared input field: {target_id}.{field}"),
                    ));
                }
                let source_id = binding.from.node_id.as_str();
                if !incoming.contains(source_id) {
                    return Err(GraphError::new(
                        "INVALID_BINDING",
                        format!("{source_id} is not an incoming source of {target_id}"),
                    ));
                }
                let source = &self.nodes[source_id];
                if !self.codecs[source_id].1
                    && source.io.output["properties"]
                        .get(&binding.from.field)
                        .is_none()
                {
                    return Err(GraphError::new(
                        "INVALID_BINDING",
                        format!(
                            "undeclared output field: {source_id}.{}",
                            binding.from.field
                        ),
                    ));
                }
            }
            if !entry && !input_messages && !target.agent {
                for field in target.io.input["required"].as_array().into_iter().flatten() {
                    if let Some(field) = field.as_str()
                        && !target.attrs.input.contains_key(field)
                    {
                        return Err(GraphError::new(
                            "MISSING_INPUT",
                            format!("required input field is not bound: {target_id}.{field}"),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn version(&self) -> Result<String, GraphError> {
        let value = serde_json::json!({
            "definition_id": self.definition_id,
            "nodes": self.nodes,
            "graph": self.graph,
            "codecs": self.codecs,
        });
        let canonical = canonicalize(value);
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| GraphError::new("SERIALIZATION_ERROR", error.to_string()))?;
        Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
    }
}

fn validate_node_id(node_id: &str) -> Result<(), GraphError> {
    if node_id.trim().is_empty() || node_id == START || node_id == END {
        Err(GraphError::new(
            "INVALID_NODE_ID",
            "node_id must be non-empty and cannot be START/END",
        ))
    } else {
        Ok(())
    }
}

fn canonicalize(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let sorted: BTreeMap<_, _> = object.into_iter().collect();
            Value::Object(
                sorted
                    .into_iter()
                    .map(|(key, value)| (key, canonicalize(value)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize).collect()),
        value => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_executor::block_on;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    #[derive(Deserialize, JsonSchema)]
    struct Input {
        query: String,
    }
    #[derive(JsonSchema, Serialize)]
    struct Output {
        answer: String,
    }

    async fn answer(input: Input, _: Cancellation) -> Result<Output, Error> {
        Ok(Output {
            answer: input.query,
        })
    }

    #[derive(Deserialize, JsonSchema)]
    struct LimitedInput {
        #[schemars(length(min = 3))]
        query: String,
    }

    async fn limited(input: LimitedInput, _: Cancellation) -> Result<Output, Error> {
        Ok(Output {
            answer: input.query,
        })
    }

    fn single() -> WorkflowBuilder {
        let mut builder = WorkflowBuilder::new("example");
        builder.add_node("answer", answer).unwrap();
        builder.add_edge(START, "answer").unwrap();
        builder.add_edge("answer", END).unwrap();
        builder
    }

    #[test]
    fn compile_preserves_schema_executor_and_stable_version() {
        let first = single().compile().unwrap();
        let second = single().compile().unwrap();
        assert_eq!(first.definition().version, second.definition().version);
        assert_eq!(
            first.definition().nodes["answer"].io.input["type"],
            "object"
        );
        assert_eq!(
            serde_json::to_value(&first.definition().graph.ordinary_edges[0]).unwrap(),
            serde_json::json!({"from": "START", "to": "answer"})
        );
        let result = block_on(
            first
                .executor("answer")
                .unwrap()
                .call(serde_json::json!({"query": "hello"}), Cancellation::new()),
        )
        .unwrap();
        assert_eq!(
            result,
            Signal::Complete(serde_json::json!({"answer": "hello"}))
        );
    }

    #[test]
    fn executor_enforces_json_schema_before_calling_node() {
        let mut builder = WorkflowBuilder::new("schema");
        builder.add_node("limited", limited).unwrap();
        builder.add_edge(START, "limited").unwrap();
        builder.add_edge("limited", END).unwrap();
        let workflow = builder.compile().unwrap();

        let invalid = block_on(
            workflow
                .executor("limited")
                .unwrap()
                .call(serde_json::json!({"query": "x"}), Cancellation::new()),
        )
        .unwrap_err();
        assert_eq!(invalid.code, "INVALID_ARGUMENTS");

        let valid = block_on(
            workflow
                .executor("limited")
                .unwrap()
                .call(serde_json::json!({"query": "abc"}), Cancellation::new()),
        )
        .unwrap();
        assert_eq!(
            valid,
            Signal::Complete(serde_json::json!({"answer": "abc"}))
        );
    }

    #[test]
    fn rejects_scalar_node_and_invalid_graph() {
        let mut builder = WorkflowBuilder::new("scalar");
        let scalar =
            async |_: String, _: Cancellation| -> Result<String, Error> { Ok(String::new()) };
        assert_eq!(
            builder.add_node("bad", scalar).err().unwrap().code,
            "INVALID_SCHEMA"
        );
        let mut builder = WorkflowBuilder::new("missing");
        builder.add_node("answer", answer).unwrap();
        assert_eq!(builder.compile().unwrap_err().code, "MISSING_BOUNDARY");
    }

    #[test]
    fn state_usage_is_recomputed_from_attempt_details() {
        let builder = single();
        let workflow = builder.compile().unwrap();
        let mut state = State::new(workflow.definition());
        state.nodes.get_mut("answer").unwrap().executions.insert(
            11,
            crate::runtime::workflow::node::Execution {
                exec_id: 11,
                attempts: vec![crate::runtime::workflow::node::Attempt {
                    usage: Some(Usage {
                        input: Some(12),
                        output: Some(5),
                        total: Some(17),
                        ..Usage::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        state.recompute_usage();
        assert_eq!(state.usage.as_ref().and_then(|usage| usage.input), Some(12));
        assert_eq!(state.usage.as_ref().and_then(|usage| usage.total), Some(17));
    }

    #[test]
    fn usage_recomputation_preserves_retry_details_without_double_counting_child() {
        use crate::model::token::{InputDetails, OutputDetails};
        use crate::runtime::workflow::node::{Attempt, AttemptStatus, Execution};

        let workflow = single().compile().unwrap();
        let mut child = State::new(workflow.definition());
        let make_usage = |total, private| Usage {
            input: Some(total - 2),
            output: Some(2),
            total: Some(total),
            input_details: Some(InputDetails { cached: Some(3) }),
            output_details: Some(OutputDetails { think: Some(1) }),
            details: serde_json::Map::from_iter([("provider".into(), serde_json::json!(private))]),
        };
        child.nodes.get_mut("answer").unwrap().executions.insert(
            12,
            Execution {
                exec_id: 12,
                attempts: vec![
                    Attempt {
                        idx: 0,
                        status: AttemptStatus::Failed,
                        usage: Some(make_usage(10, "failed")),
                        ..Default::default()
                    },
                    Attempt {
                        idx: 1,
                        status: AttemptStatus::Succeeded,
                        usage: Some(make_usage(20, "succeeded")),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        );
        // Deliberately stale persisted aggregates must not become another
        // billable record after restore or another consistency commit.
        child.usage = Some(make_usage(99, "stale"));
        let mut parent = State::new(workflow.definition());
        parent.nodes.get_mut("answer").unwrap().executions.insert(
            11,
            Execution {
                exec_id: 11,
                attempts: vec![Attempt {
                    usage: Some(make_usage(99, "stale")),
                    graph: Some(Box::new(child)),
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        for _ in 0..3 {
            parent.recompute_usage();
            let total = parent.usage.as_ref().unwrap();
            assert_eq!(total.total, Some(30));
            assert_eq!(total.input_details.as_ref().unwrap().cached, Some(6));
            assert_eq!(total.output_details.as_ref().unwrap().think, Some(2));
            assert!(total.details.is_empty());
            let execution = &parent.nodes["answer"].executions[&11];
            assert_eq!(execution.usage.as_ref().unwrap().total, Some(30));
            assert_eq!(
                execution.attempts[0].usage.as_ref().unwrap().total,
                Some(30)
            );
            let original = &execution.attempts[0].graph.as_ref().unwrap().nodes["answer"]
                .executions[&12]
                .attempts[0];
            assert_eq!(
                original.usage.as_ref().unwrap().details["provider"],
                "failed"
            );
        }
    }

    #[test]
    fn join_conflicts_with_ordinary_activation() {
        let mut builder = WorkflowBuilder::new("join");
        for id in ["a", "b", "merge"] {
            builder.add_node(id, crate::node::Echo).unwrap();
        }
        builder.add_edge(START, "a").unwrap();
        builder.add_edge(START, "b").unwrap();
        builder.add_join(["a", "b"], "merge").unwrap();
        assert_eq!(
            builder.add_edge("a", "merge").err().unwrap().code,
            "CONFLICTING_ACTIVATION"
        );
        builder.add_edge("merge", END).unwrap();
        assert_eq!(
            builder
                .compile()
                .unwrap()
                .definition()
                .graph
                .join_definitions
                .len(),
            1
        );
    }

    #[test]
    fn state_initializes_start_nodes_and_indexes_waits() {
        let compiled = single().compile().unwrap();
        let mut state = State::new(compiled.definition());
        assert_eq!(state.step, 0);
        assert_eq!(state.status, Status::Running);
        assert_eq!(
            state.nodes["answer"].status,
            crate::runtime::workflow::node::Status::Idle
        );

        state
            .add_wait(Wait {
                node_id: "answer".into(),
                exec_id: 1,
                wait_id: "wait_001".into(),
            })
            .unwrap();
        assert_eq!(state.waits["answer"].wait_id, "wait_001");
        assert_eq!(state.remove_wait("answer").unwrap().node_id, "answer");
    }

    #[test]
    fn state_rejects_wait_for_unknown_node_and_duplicate_node_wait() {
        let compiled = single().compile().unwrap();
        let mut state = State::new(compiled.definition());
        assert_eq!(
            state
                .add_wait(Wait {
                    node_id: "missing".into(),
                    exec_id: 1,
                    wait_id: "wait_001".into(),
                })
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
        state
            .add_wait(Wait {
                node_id: "answer".into(),
                exec_id: 1,
                wait_id: "wait_001".into(),
            })
            .unwrap();
        assert_eq!(
            state
                .add_wait(Wait {
                    node_id: "answer".into(),
                    exec_id: 1,
                    wait_id: "wait_002".into(),
                })
                .unwrap_err()
                .code,
            "CONFLICT"
        );
        assert_eq!(state.waits["answer"].wait_id, "wait_001");
    }

    #[test]
    fn add_graph_node_preserves_child_definition_boundary() {
        let child = single().compile().unwrap();
        let mut parent = WorkflowBuilder::new("parent");
        parent.add_graph_node("child", &child).unwrap();
        parent.add_edge(START, "child").unwrap();
        parent.add_edge("child", END).unwrap();
        let parent = parent.compile().unwrap();

        let node = &parent.definition().nodes["child"];
        assert_eq!(node.workflow.as_ref().unwrap().definition_id, "example");
        assert_eq!(
            parent.graph_node("child").unwrap().definition(),
            child.definition()
        );
    }

    #[test]
    fn add_graph_node_rejects_recursive_definition() {
        let child = single().compile().unwrap();
        let mut builder = WorkflowBuilder::new("example");
        assert_eq!(
            builder.add_graph_node("nested", &child).err().unwrap().code,
            "RECURSIVE_GRAPH"
        );
    }

    #[test]
    fn state_runs_one_barrier_and_activates_ordinary_successor() {
        let mut builder = WorkflowBuilder::new("barrier");
        builder.add_node("a", crate::node::Echo).unwrap();
        builder.add_node("b", crate::node::Echo).unwrap();
        builder.add_edge(START, "a").unwrap();
        builder.add_edge("a", "b").unwrap();
        builder.add_edge("b", END).unwrap();
        let compiled = builder.compile().unwrap();
        let mut state = State::new(compiled.definition());

        assert_eq!(state.ready_nodes(compiled.definition()), vec!["a"]);
        state
            .start_execution(compiled.definition(), "a", 1, Vec::new())
            .unwrap();
        assert!(!state.advance_step());
        state
            .complete_execution("a", 1, Messages::default(), vec!["b".into()])
            .unwrap();
        assert!(state.advance_step());
        assert_eq!(state.step, 1);
        assert_eq!(state.ready_nodes(compiled.definition()), vec!["b"]);
    }

    #[test]
    fn join_waits_until_all_sources_succeed() {
        let mut builder = WorkflowBuilder::new("join-state");
        for id in ["a", "b", "merge"] {
            builder.add_node(id, crate::node::Echo).unwrap();
        }
        builder.add_edge(START, "a").unwrap();
        builder.add_edge(START, "b").unwrap();
        builder.add_join(["a", "b"], "merge").unwrap();
        builder.add_edge("merge", END).unwrap();
        let compiled = builder.compile().unwrap();
        let mut state = State::new(compiled.definition());
        assert_eq!(state.ready_nodes(compiled.definition()), vec!["a", "b"]);
        state
            .start_batch(
                compiled.definition(),
                vec![("a".into(), 1, Vec::new()), ("b".into(), 2, Vec::new())],
            )
            .unwrap();
        state
            .complete_execution("a", 1, Messages::default(), vec!["merge".into()])
            .unwrap();
        assert!(
            !state
                .ready_nodes(compiled.definition())
                .iter()
                .any(|id| id == "merge")
        );
        state
            .complete_execution("b", 2, Messages::default(), vec!["merge".into()])
            .unwrap();
        assert!(state.advance_step());
        assert!(
            state
                .ready_nodes(compiled.definition())
                .iter()
                .any(|id| id == "merge")
        );
        state
            .start_execution(
                compiled.definition(),
                "merge",
                3,
                vec![
                    crate::runtime::workflow::node::ExecutionRef {
                        node_id: "a".into(),
                        exec_id: 1,
                    },
                    crate::runtime::workflow::node::ExecutionRef {
                        node_id: "b".into(),
                        exec_id: 2,
                    },
                ],
            )
            .unwrap();
    }

    #[test]
    fn start_batch_is_atomic_when_one_entry_is_invalid() {
        let mut builder = WorkflowBuilder::new("batch-atomic");
        builder.add_node("a", crate::node::Echo).unwrap();
        builder.add_node("b", crate::node::Echo).unwrap();
        builder.add_edge(START, "a").unwrap();
        builder.add_edge(START, "b").unwrap();
        builder.add_edge("a", END).unwrap();
        builder.add_edge("b", END).unwrap();
        let compiled = builder.compile().unwrap();
        let mut state = State::new(compiled.definition());
        let error = state
            .start_batch(
                compiled.definition(),
                vec![
                    ("a".into(), 1, Vec::new()),
                    ("missing".into(), 2, Vec::new()),
                ],
            )
            .unwrap_err();
        assert_eq!(error.code, "INVALID_STATE");
        assert!(state.nodes["a"].executions.is_empty());
        assert!(state.nodes["b"].executions.is_empty());
    }
}
