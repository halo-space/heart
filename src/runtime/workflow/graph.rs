//! Workflow graph definition and compilation.
//!
//! Executable adapters and mutable runtime State live in sibling modules so
//! this file remains the static graph/Builder boundary.

//! Static Workflow definitions and the executable graph-state boundary.
//! Relation declarations remain immutable after compilation; execution facts
//! are kept in the separate serializable `State`.

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::Cancellation;
use crate::Error;
use crate::components::agent::{self, Agent};
use crate::components::event::Event;
use crate::components::message::Messages;
use crate::components::message::Role;
pub use crate::components::middleware::RetryPolicy;
use crate::components::model::token::Usage;
use crate::components::node::{Node, Signal, View};
use crate::runtime::workflow::node::{AttemptStatus, Step, StepAttempt};

pub const START: &str = "START";
pub const END: &str = "END";

mod adapters;
mod state;

pub(crate) use adapters::{AgentAdapter, ErasedAgent, ErasedNode, NodeAdapter};
pub(crate) use state::ReplyExecution;
pub use state::{State, Status, Wait};

type RouteFn = Arc<dyn Fn(&Messages) -> Result<Vec<String>, Error> + Send + Sync + 'static>;
type LoopFn = Arc<dyn Fn(&Messages) -> bool + Send + Sync + 'static>;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Schema {
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
    pub schema: Schema,
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
/// A relation that activates an existing target only after every source has
/// succeeded. This is an edge declaration, not an executable node.
pub struct JoinEdge {
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
    pub join_edges: Vec<JoinEdge>,
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

/// Workflow APIs use the same public `code/message/details` error contract as
/// every other component.  This alias keeps the workflow namespace readable
/// without introducing a second error representation.
pub type GraphError = Error;

pub struct Builder {
    definition_id: String,
    max_nodes: usize,
    dsl_nodes: BTreeMap<String, crate::core::workflow::dsl::Node>,
    nodes: BTreeMap<String, NodeDefinition>,
    codecs: BTreeMap<String, (bool, bool)>,
    executors: BTreeMap<String, Arc<dyn ErasedNode>>,
    graph_nodes: BTreeMap<String, Arc<CompiledWorkflow>>,
    agents: BTreeMap<String, Arc<dyn ErasedAgent>>,
    graph: GraphDefinition,
    routes: BTreeMap<String, RouteFn>,
    loops: BTreeMap<String, LoopFn>,
}

impl Builder {
    pub fn new(definition_id: impl Into<String>) -> Self {
        Self {
            definition_id: definition_id.into(),
            max_nodes: 100,
            dsl_nodes: BTreeMap::new(),
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

    /// Start a builder from a validated serializable Native DSL document.
    ///
    /// The document supplies only static graph facts. Executable functions,
    /// Agents, Tools and child Workflows are bound explicitly through the
    /// `bind_*` methods below; no name lookup table or dynamic registry is
    /// created.
    pub fn from_dsl(document: crate::core::workflow::dsl::Document) -> Result<Self, GraphError> {
        document
            .validate()
            .map_err(|error| GraphError::new(&error.code, error.message))?;
        let mut builder = Self::new(document.definition_id.clone());
        if document.nodes.len() > builder.max_nodes {
            return Err(GraphError::new("MAX_NODES", "workflow node limit exceeded"));
        }
        for (node_id, node) in document.nodes {
            builder.dsl_nodes.insert(node_id.clone(), node.clone());
            let attrs = NodeAttrs {
                input: node
                    .attrs
                    .input
                    .into_iter()
                    .map(|(field, binding)| {
                        (
                            field,
                            InputBinding {
                                from: FieldSource {
                                    node_id: binding.from.node_id,
                                    field: binding.from.field,
                                },
                            },
                        )
                    })
                    .collect(),
            };
            let retry = RetryPolicy {
                max_retries: node.retry.max_retries,
                min_delay_ms: node.retry.min_delay_ms,
                max_delay_ms: node.retry.max_delay_ms,
            };
            builder.nodes.insert(
                node_id.clone(),
                NodeDefinition {
                    node_id,
                    node_name: Some(node.name),
                    schema: Schema {
                        input: node.schema.input,
                        output: node.schema.output,
                    },
                    attrs,
                    retry,
                    workflow: None,
                    agent: node.node_type == "agent",
                },
            );
        }
        builder.graph.ordinary_edges = document
            .edges
            .into_iter()
            .map(|edge| Edge {
                from: edge.from.into(),
                to: edge.to.into(),
            })
            .collect();
        builder.graph.join_edges = document
            .join_edges
            .into_iter()
            .map(|join| JoinEdge {
                from: join.from,
                to: join.to,
            })
            .collect();
        builder.graph.conditional_edges = document
            .conditions
            .into_iter()
            .map(|condition| Conditional {
                from: condition.from,
                to: condition.to,
            })
            .collect();
        builder.graph.fan_out_definitions = document
            .fan_outs
            .into_iter()
            .map(|fan_out| FanOut {
                from: fan_out.from,
                to: fan_out.to,
                mode: fan_out.mode,
            })
            .collect();
        builder.graph.loop_definitions = document
            .loops
            .into_iter()
            .map(|loop_edge| Loop {
                from: loop_edge.from,
                to: loop_edge.to,
            })
            .collect();
        builder.graph.failure_relations = document
            .failures
            .into_iter()
            .map(|failure| FailureRelation {
                from: failure.from,
                to: failure.to,
                on: failure.on,
                codes: failure.codes,
            })
            .collect();
        Ok(builder)
    }

    /// Bind an ordinary typed Node to a function node declared by the DSL.
    pub fn bind_node<I, O, E, N>(
        &mut self,
        node_id: impl Into<String>,
        node: N,
    ) -> Result<&mut Self, GraphError>
    where
        N: Node<I, O, E> + Any,
        I: DeserializeOwned + JsonSchema + Send + 'static,
        O: Serialize + JsonSchema + Send + 'static,
        E: Into<Error> + Send + 'static,
    {
        let node_id = node_id.into();
        self.require_dsl_type(&node_id, "function")?;
        let schema = self.dsl_schema(&node_id)?;
        self.add_node_with_schemas(node_id, node, schema.input, schema.output)
    }

    /// Bind a page-authored Python function node to the DSL declaration.
    pub fn bind_code<S: crate::sandbox::Sandbox + 'static>(
        &mut self,
        node_id: impl Into<String>,
        sandbox: S,
    ) -> Result<&mut Self, GraphError> {
        let node_id = node_id.into();
        self.require_dsl_type(&node_id, "function")?;
        let declaration = self.dsl_node(&node_id)?;
        let code = crate::core::workflow::code::Code::from_dsl(sandbox, declaration)
            .map_err(|error| GraphError::new(&error.code, error.message))?;
        self.add_node_with_schema(
            node_id,
            code,
            Schema {
                input: declaration.schema.input.clone(),
                output: declaration.schema.output.clone(),
            },
        )
    }

    /// Bind an Agent node declared by the DSL.
    pub fn bind_agent<A: Agent + 'static>(
        &mut self,
        node_id: impl Into<String>,
        agent: A,
    ) -> Result<&mut Self, GraphError> {
        let node_id = node_id.into();
        self.require_dsl_type(&node_id, "agent")?;
        self.add_agent_node_with_schema(node_id, agent)
    }

    /// Bind a ToolNode (or another Messages -> Messages node) declared as a
    /// `tool` node in the DSL.
    pub fn bind_tool<N>(
        &mut self,
        node_id: impl Into<String>,
        node: N,
    ) -> Result<&mut Self, GraphError>
    where
        N: Node<Messages, Messages, Error> + Any,
    {
        let node_id = node_id.into();
        self.require_dsl_type(&node_id, "tool")?;
        let schema = self.dsl_schema(&node_id)?;
        self.add_node_with_schemas(node_id, node, schema.input, schema.output)
    }

    /// Bind a compiled child Workflow declared as a `graph` node.
    pub fn bind_graph(
        &mut self,
        node_id: impl Into<String>,
        workflow: &CompiledWorkflow,
    ) -> Result<&mut Self, GraphError> {
        let node_id = node_id.into();
        self.require_dsl_type(&node_id, "graph")?;
        self.add_graph_node(node_id, workflow)
    }

    /// Bind the application-owned route function for one DSL condition.
    pub fn bind_condition<F>(
        &mut self,
        from: impl Into<String>,
        route: F,
    ) -> Result<&mut Self, GraphError>
    where
        F: Fn(&Messages) -> Result<Vec<String>, Error> + Send + Sync + 'static,
    {
        let from = from.into();
        if !self
            .graph
            .conditional_edges
            .iter()
            .any(|condition| condition.from == from)
        {
            return Err(GraphError::new(
                "UNKNOWN_CONDITION",
                "condition is not declared",
            ));
        }
        if self.routes.insert(from, Arc::new(route)).is_some() {
            return Err(GraphError::new(
                "DUPLICATE_CONDITION",
                "condition is already bound",
            ));
        }
        Ok(self)
    }

    /// Bind the application-owned loop predicate for one DSL loop.
    pub fn bind_loop<F>(
        &mut self,
        from: impl Into<String>,
        condition: F,
    ) -> Result<&mut Self, GraphError>
    where
        F: Fn(&Messages) -> bool + Send + Sync + 'static,
    {
        let from = from.into();
        if !self
            .graph
            .loop_definitions
            .iter()
            .any(|loop_edge| loop_edge.from == from)
        {
            return Err(GraphError::new("UNKNOWN_LOOP", "loop is not declared"));
        }
        if self.loops.insert(from, Arc::new(condition)).is_some() {
            return Err(GraphError::new("DUPLICATE_LOOP", "loop is already bound"));
        }
        Ok(self)
    }

    pub fn add_node<I, O, E, N>(
        &mut self,
        node_id: impl Into<String>,
        node: N,
    ) -> Result<&mut Self, GraphError>
    where
        N: Node<I, O, E> + Any,
        I: DeserializeOwned + JsonSchema + Send + 'static,
        O: Serialize + JsonSchema + Send + 'static,
        E: Into<Error> + Send + 'static,
    {
        let input = serde_json::to_value(schema_for!(I))
            .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
        let output = serde_json::to_value(schema_for!(O))
            .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
        self.add_node_with_schemas(node_id, node, input, output)
    }

    /// Adds an ordinary Node while preserving an explicitly supplied
    /// Definition input/output Schema. The Node implementation remains the
    /// same typed executor; the supplied schemas are used for its JSON
    /// boundary validation and are stored in the compiled Definition. This is
    /// used when a serializable DSL carries a more precise schema than the
    /// erased JSON-object wrapper.
    pub fn add_node_with_schema<I, O, E, N>(
        &mut self,
        node_id: impl Into<String>,
        node: N,
        schema: Schema,
    ) -> Result<&mut Self, GraphError>
    where
        N: Node<I, O, E> + Any,
        I: DeserializeOwned + JsonSchema + Send + 'static,
        O: Serialize + JsonSchema + Send + 'static,
        E: Into<Error> + Send + 'static,
    {
        self.add_node_with_schemas(node_id, node, schema.input, schema.output)
    }

    fn add_node_with_schemas<I, O, E, N>(
        &mut self,
        node_id: impl Into<String>,
        node: N,
        input: Value,
        output: Value,
    ) -> Result<&mut Self, GraphError>
    where
        N: Node<I, O, E> + Any,
        I: DeserializeOwned + JsonSchema + Send + 'static,
        O: Serialize + JsonSchema + Send + 'static,
        E: Into<Error> + Send + 'static,
    {
        let node_id = node_id.into();
        validate_node_id(&node_id)?;
        let already_declared = self.nodes.contains_key(&node_id);
        let already_bound = self.executors.contains_key(&node_id)
            || self.agents.contains_key(&node_id)
            || self.graph_nodes.contains_key(&node_id);
        if already_bound {
            return Err(GraphError::new(
                "DUPLICATE_NODE",
                format!("duplicate node: {node_id}"),
            ));
        }
        if !already_declared && self.nodes.len() >= self.max_nodes {
            return Err(GraphError::new("MAX_NODES", "workflow node limit exceeded"));
        }
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

        if let Some(declaration) = self.nodes.get(&node_id)
            && (declaration.schema.input != input || declaration.schema.output != output)
        {
            return Err(GraphError::new(
                "INVALID_SCHEMA",
                format!("bound Node schema does not match DSL node: {node_id}"),
            ));
        }
        if !already_declared {
            self.nodes.insert(
                node_id.clone(),
                NodeDefinition {
                    node_id: node_id.clone(),
                    node_name: None,
                    schema: Schema { input, output },
                    attrs: NodeAttrs::default(),
                    retry: RetryPolicy::default(),
                    workflow: None,
                    agent: false,
                },
            );
        }
        self.codecs
            .insert(node_id.clone(), (input_messages, output_messages));
        self.executors.insert(
            node_id,
            Arc::new(NodeAdapter::<N, I, O, E> {
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
        let input = serde_json::to_value(schema_for!(agent::Input))
            .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
        let output = serde_json::to_value(schema_for!(Messages))
            .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
        self.add_agent_node_with_schema_and_values(node_id, agent, input, output)
    }

    fn add_agent_node_with_schema<A: Agent + 'static>(
        &mut self,
        node_id: String,
        agent: A,
    ) -> Result<&mut Self, GraphError> {
        let input = serde_json::to_value(schema_for!(agent::Input))
            .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
        let output = serde_json::to_value(schema_for!(Messages))
            .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
        self.add_agent_node_with_schema_and_values(node_id, agent, input, output)
    }

    fn add_agent_node_with_schema_and_values<A: Agent + 'static>(
        &mut self,
        node_id: String,
        agent: A,
        input: Value,
        output: Value,
    ) -> Result<&mut Self, GraphError> {
        validate_node_id(&node_id)?;
        let already_declared = self.nodes.contains_key(&node_id);
        if self.agents.contains_key(&node_id)
            || self.executors.contains_key(&node_id)
            || self.graph_nodes.contains_key(&node_id)
        {
            return Err(GraphError::new(
                "DUPLICATE_NODE",
                format!("duplicate node: {node_id}"),
            ));
        }
        if !already_declared && self.nodes.len() >= self.max_nodes {
            return Err(GraphError::new("MAX_NODES", "workflow node limit exceeded"));
        }
        if let Some(declaration) = self.nodes.get(&node_id)
            && (declaration.schema.input != input || declaration.schema.output != output)
        {
            return Err(GraphError::new(
                "INVALID_SCHEMA",
                format!("bound Agent schema does not match DSL node: {node_id}"),
            ));
        }
        if !already_declared {
            self.nodes.insert(
                node_id.clone(),
                NodeDefinition {
                    node_id: node_id.clone(),
                    node_name: None,
                    schema: Schema { input, output },
                    attrs: NodeAttrs::default(),
                    retry: RetryPolicy::default(),
                    workflow: None,
                    agent: true,
                },
            );
        }
        self.codecs.insert(node_id.clone(), (false, true));
        self.agents.insert(
            node_id,
            Arc::new(AgentAdapter {
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
        let already_declared = self.nodes.contains_key(&node_id);
        if self.graph_nodes.contains_key(&node_id)
            || self.executors.contains_key(&node_id)
            || self.agents.contains_key(&node_id)
        {
            return Err(GraphError::new(
                "DUPLICATE_NODE",
                format!("duplicate node: {node_id}"),
            ));
        }
        if !already_declared && self.nodes.len() >= self.max_nodes {
            return Err(GraphError::new("MAX_NODES", "workflow node limit exceeded"));
        }
        if workflow.contains_definition(&self.definition_id) {
            return Err(GraphError::new(
                "RECURSIVE_GRAPH",
                "a Workflow cannot contain itself directly or indirectly",
            ));
        }

        let (input, output) = if let Some(declaration) = self.nodes.get(&node_id) {
            (
                declaration.schema.input.clone(),
                declaration.schema.output.clone(),
            )
        } else {
            let input = serde_json::to_value(schema_for!(Messages))
                .map_err(|error| GraphError::new("SCHEMA_ERROR", error.to_string()))?;
            (input.clone(), input)
        };
        let definition = GraphReference {
            definition_id: workflow.definition().definition_id.clone(),
            version: workflow.definition().version.clone(),
        };
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.workflow = Some(definition);
            node.agent = false;
        } else {
            self.nodes.insert(
                node_id.clone(),
                NodeDefinition {
                    node_id: node_id.clone(),
                    node_name: None,
                    schema: Schema { input, output },
                    attrs: NodeAttrs::default(),
                    retry: RetryPolicy::default(),
                    workflow: Some(definition),
                    agent: false,
                },
            );
        }
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
            .join_edges
            .iter()
            .any(|join| edge.to.node() == Some(&join.to))
        {
            return Err(GraphError::new(
                "CONFLICTING_ACTIVATION",
                "Join edge target cannot have an ordinary incoming edge",
            ));
        }
        self.graph.ordinary_edges.push(edge);
        Ok(self)
    }

    /// Add a Join edge between existing source nodes and an existing target.
    /// This does not create a node or an execution record for the relation.
    pub fn add_join_edge(
        &mut self,
        from: impl IntoIterator<Item = impl Into<String>>,
        to: impl Into<String>,
    ) -> Result<&mut Self, GraphError> {
        let join = JoinEdge {
            from: from.into_iter().map(Into::into).collect(),
            to: to.into(),
        };
        if join.from.len() < 2 || join.from.iter().collect::<BTreeSet<_>>().len() != join.from.len()
        {
            return Err(GraphError::new(
                "INVALID_JOIN",
                "Join edge requires at least two distinct sources",
            ));
        }
        self.require_node(&join.to)?;
        for source in &join.from {
            self.require_node(source)?;
        }
        if self
            .graph
            .join_edges
            .iter()
            .any(|existing| existing.to == join.to)
        {
            return Err(GraphError::new(
                "DUPLICATE_JOIN",
                "target already has a Join edge",
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
                "Join edge target cannot have an ordinary incoming edge",
            ));
        }
        self.graph.join_edges.push(join);
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
                .join_edges
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
                    "join_failed relation must match exactly one Join edge source set",
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
        self.validate_bindings_present()?;
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

    fn dsl_node(&self, node_id: &str) -> Result<&crate::core::workflow::dsl::Node, GraphError> {
        self.dsl_nodes.get(node_id).ok_or_else(|| {
            GraphError::new(
                "UNKNOWN_NODE",
                format!("node is not declared by the DSL: {node_id}"),
            )
        })
    }

    fn dsl_schema(&self, node_id: &str) -> Result<Schema, GraphError> {
        let node = self.dsl_node(node_id)?;
        Ok(Schema {
            input: node.schema.input.clone(),
            output: node.schema.output.clone(),
        })
    }

    fn require_dsl_type(&self, node_id: &str, expected: &str) -> Result<(), GraphError> {
        let node = self.dsl_node(node_id)?;
        if node.node_type != expected {
            return Err(GraphError::new(
                "INVALID_NODE_TYPE",
                format!("node {node_id} is not a {expected} node"),
            ));
        }
        Ok(())
    }

    fn validate_bindings_present(&self) -> Result<(), GraphError> {
        for node_id in self.nodes.keys() {
            let bound = self.executors.contains_key(node_id)
                || self.agents.contains_key(node_id)
                || self.graph_nodes.contains_key(node_id);
            if !bound {
                return Err(GraphError::new(
                    "UNBOUND_NODE",
                    format!("workflow node has no executable binding: {node_id}"),
                ));
            }
        }
        for condition in &self.graph.conditional_edges {
            if !self.routes.contains_key(&condition.from) {
                return Err(GraphError::new(
                    "UNBOUND_CONDITION",
                    format!("condition has no route binding: {}", condition.from),
                ));
            }
        }
        for loop_edge in &self.graph.loop_definitions {
            if !self.loops.contains_key(&loop_edge.from) {
                return Err(GraphError::new(
                    "UNBOUND_LOOP",
                    format!("loop has no predicate binding: {}", loop_edge.from),
                ));
            }
        }
        Ok(())
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
        for join in &self.graph.join_edges {
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
            let properties = target.schema.input["properties"].as_object();
            let incoming: BTreeSet<&str> = self
                .graph
                .ordinary_edges
                .iter()
                .filter(|edge| edge.to.node() == Some(target_id))
                .filter_map(|edge| edge.from.node())
                .chain(
                    self.graph
                        .join_edges
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
                if source_id == START {
                    if !entry {
                        return Err(GraphError::new(
                            "INVALID_BINDING",
                            format!("START is not an incoming source of {target_id}"),
                        ));
                    }
                    // START values come from the Workflow's initial Messages,
                    // so they have no compiled predecessor output schema.
                    continue;
                }
                if !incoming.contains(source_id) {
                    return Err(GraphError::new(
                        "INVALID_BINDING",
                        format!("{source_id} is not an incoming source of {target_id}"),
                    ));
                }
                let source = &self.nodes[source_id];
                if !self.codecs[source_id].1
                    && source.schema.output["properties"]
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
                for field in target.schema.input["required"]
                    .as_array()
                    .into_iter()
                    .flatten()
                {
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
            "dsl_nodes": &self.dsl_nodes,
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
#[path = "../../../tests/unit/workflow_graph.rs"]
mod tests;
