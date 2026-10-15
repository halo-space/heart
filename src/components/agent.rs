//! Shared Agent contract, input and default Chat implementation.
//!
//! Runtime-owned execution graphs and node state live under `runtime::agent`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::event::Event;
use crate::runtime::cancellation::Cancellation;

pub mod chat;
pub mod toolkit;

type ErasedStream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + 'static>>;
type RunFuture = Pin<Box<dyn Future<Output = Result<ErasedStream, Error>> + 'static>>;

trait ErasedAgent: Send + Sync {
    fn run(&self, input: Input, cancellation: Cancellation) -> RunFuture;
}

struct TypedAgent<A>(Arc<A>);

impl<A> ErasedAgent for TypedAgent<A>
where
    A: Agent + 'static,
{
    fn run(&self, input: Input, cancellation: Cancellation) -> RunFuture {
        let agent = Arc::clone(&self.0);
        Box::pin(async move {
            let stream = agent.run(input, cancellation).await?;
            Ok(Box::pin(stream) as ErasedStream)
        })
    }
}

/// One explicitly registered Agent that another Agent may delegate to.
///
/// SubAgents are kept separate from [`toolkit::Toolkit`]: Toolkit contains
/// Tool/MCP/Skill capabilities, while this type preserves a complete Agent
/// run boundary and its event stream.
#[derive(Clone)]
pub struct SubAgent {
    name: String,
    description: String,
    runner: Arc<dyn ErasedAgent>,
}

impl SubAgent {
    pub fn new<A>(name: impl Into<String>, agent: A) -> Result<Self, Error>
    where
        A: Agent + 'static,
    {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "subagent name must not be empty",
            ));
        }
        Ok(Self {
            name,
            description: String::new(),
            runner: Arc::new(TypedAgent(Arc::new(agent))),
        })
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description_text(&self) -> &str {
        &self.description
    }

    pub(crate) fn definition(&self) -> crate::components::tool::ToolDefinition {
        crate::components::tool::ToolDefinition {
            name: "agent_spawn".into(),
            description: "Delegate a child task to one of the registered SubAgents. The child input must be an agent::Input object.".into(),
            parameters: serde_json::from_value(serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "input": {"type": "object"}
                },
                "required": ["name", "input"],
                "additionalProperties": false
            }))
            .expect("static agent_spawn schema is an object"),
        }
    }

    pub(crate) fn has_name(&self, name: &str) -> bool {
        self.name == name
    }

    pub(crate) fn run(&self, input: Input, cancellation: Cancellation) -> RunFuture {
        self.runner.run(input, cancellation)
    }
}

impl std::fmt::Debug for SubAgent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SubAgent")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// Builder-shaped entry points keep application code concise while the
/// implementation remains split by Agent kind.
pub fn chat() -> chat::Builder {
    chat::builder()
}

pub type Error = crate::Error;

/// Public business input shared by ChatAgent and ReActAgent.
///
/// Runtime metadata is kept separate from the user payload. Execution source
/// identities stay in Input/Runtime, not internal Message metadata. Other
/// input metadata is retained while payload becomes ordered model content.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, JsonSchema)]
pub struct Input {
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub payload: Vec<Payload>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
pub struct Payload {
    #[serde(rename = "type")]
    pub r#type: String,
    pub data: Value,
    pub mime_type: Option<String>,
}

pub trait Agent: Send + Sync {
    type Stream: Stream<Item = Result<Event, Error>> + 'static;

    fn run(
        &self,
        input: Input,
        cancellation: Cancellation,
    ) -> impl Future<Output = Result<Self::Stream, Error>>;
}
