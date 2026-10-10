//! Shared Agent contract, input and default Chat implementation.
//!
//! Runtime-owned execution graphs and node state live under `runtime::agent`.

use std::future::Future;

use futures_core::Stream;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::event::Event;
use crate::runtime::cancellation::Cancellation;

pub mod chat;
pub mod toolkit;

/// Builder-shaped entry points keep application code concise while the
/// implementation remains split by Agent kind.
pub fn chat() -> chat::Builder {
    chat::builder()
}

pub type Error = crate::Error;

/// Public business input shared by ChatAgent and ReActAgent.
///
/// Runtime metadata is kept separate from the user payload. It is copied to
/// the internal user message metadata, while payload values become ordered
/// model content parts.
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
