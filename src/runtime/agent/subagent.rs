//! SubAgent assembly plus private child execution and usage settlement.
use crate::runtime::agent::node::status::Status;
use crate::runtime::agent::node::{State, Step};
use crate::{
    Cancellation, Error, Message,
    components::agent::{Agent, Input},
    components::node::View,
    event::Event,
    message::Role,
    model::token::{InputDetails, OutputDetails, Usage},
    tool::Call,
};
use futures_util::task::AtomicWaker;
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

type ErasedStream =
    Pin<Box<dyn futures_core::Stream<Item = Result<Event, Error>> + Send + 'static>>;
type RunFuture = Pin<Box<dyn Future<Output = Result<ErasedStream, Error>> + Send + 'static>>;

struct SerializedStream {
    inner: ErasedStream,
    _guard: tokio::sync::OwnedMutexGuard<()>,
}

impl futures_core::Stream for SerializedStream {
    type Item = Result<Event, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

trait ErasedAgent: Send + Sync {
    fn run(&self, input: Input, cancellation: Cancellation, view: View) -> RunFuture;
}

mod adapter {
    use std::sync::Arc;

    pub(super) struct Agent<A>(pub(super) Arc<A>);
}

impl<A> ErasedAgent for adapter::Agent<A>
where
    A: Agent + 'static,
{
    fn run(&self, input: Input, cancellation: Cancellation, view: View) -> RunFuture {
        let agent = Arc::clone(&self.0);
        Box::pin(async move {
            // Parent run owns outward mode filtering for delegated events.
            let stream = agent
                .run_with_view(input, true, cancellation, &view)
                .await?;
            Ok(Box::pin(stream) as ErasedStream)
        })
    }
}

/// One explicitly registered Agent that another Agent may delegate to.
///
/// SubAgents are distinct from the internal Agent tool dispatcher: the
/// dispatcher contains Tool/MCP/Skill entries, while this retains a complete Agent
/// run boundary and its event stream.
#[derive(Clone)]
pub struct SubAgent {
    name: String,
    description: String,
    runner: Arc<dyn ErasedAgent>,
    serial: Arc<tokio::sync::Mutex<()>>,
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
            runner: Arc::new(adapter::Agent(Arc::new(agent))),
            serial: Arc::new(tokio::sync::Mutex::new(())),
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

    pub(crate) fn definition(subagents: &[SubAgent]) -> crate::components::tool::ToolDefinition {
        Self::definition_named(subagents, "agent_spawn", "Delegate a child task")
    }

    pub(crate) fn definition_named(
        subagents: &[SubAgent],
        tool_name: &str,
        operation: &str,
    ) -> crate::components::tool::ToolDefinition {
        let member_names = subagents
            .iter()
            .map(|subagent| subagent.name.clone())
            .collect::<Vec<_>>();
        Self::definition_for_names(subagents, tool_name, operation, &member_names)
    }

    pub(crate) fn definition_for_names(
        subagents: &[SubAgent],
        tool_name: &str,
        operation: &str,
        names: &[String],
    ) -> crate::components::tool::ToolDefinition {
        let members = subagents
            .iter()
            .filter(|subagent| names.iter().any(|name| name == subagent.name()))
            .map(|subagent| {
                if subagent.description.is_empty() {
                    format!("- {}", subagent.name)
                } else {
                    format!("- {}: {}", subagent.name, subagent.description)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let input_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "metadata": {"type": "object", "additionalProperties": true},
                "payload": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["text", "audio", "image", "file", "json"]},
                            "data": {
                                "type": "object",
                                "properties": {"value": {}},
                                "required": ["value"],
                                "additionalProperties": true
                            },
                            "mime_type": {"type": ["string", "null"]}
                        },
                        "required": ["type", "data"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["payload"],
            "additionalProperties": false
        });
        crate::components::tool::ToolDefinition {
            name: tool_name.into(),
            description: format!(
                "{operation} to one of the registered members. The member input must be an agent::Input object. Registered members:\n{members}"
            ),
            parameters: serde_json::from_value(serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": names},
                    "input": input_schema
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

    pub(crate) fn run_with_view(
        &self,
        input: Input,
        cancellation: Cancellation,
        view: View,
    ) -> RunFuture {
        let runner = Arc::clone(&self.runner);
        let serial = Arc::clone(&self.serial);
        Box::pin(async move {
            let guard = serial.lock_owned().await;
            let stream = runner.run(input, cancellation, view).await?;
            Ok(Box::pin(SerializedStream {
                inner: stream,
                _guard: guard,
            }) as ErasedStream)
        })
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

#[doc(hidden)]
pub mod relay;
#[doc(hidden)]
pub mod root;
mod run;

#[doc(hidden)]
pub use relay::{Observation, Observer, Relay, cancel_ready};
#[doc(hidden)]
pub use root::{Progress, Root};
pub(crate) use root::{
    cancel_pending, complete, message, observe, prepare, resume_calls, source, sync_waits,
};
pub(crate) use run::{Failure, inherit_identity, run};

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_runtime_agent_subagent_result_message_tests.rs"]
mod result_message_tests;

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_runtime_agent_subagent_observation_tests.rs"]
mod observation_tests;

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_runtime_agent_subagent_relay_tests.rs"]
mod relay_tests;

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_runtime_agent_subagent_definition_tests.rs"]
mod definition_tests;

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_runtime_agent_subagent_progress_tests.rs"]
mod progress_tests;
