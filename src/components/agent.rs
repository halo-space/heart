//! Shared Agent contract and input types.
//!
//! Runtime-owned execution graphs and node state live under `runtime::agent`.

use futures_core::Stream;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Cancellation;
use crate::components::event::Event;

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

    /// Fix outward delivery for this invocation: true delivers Delta and
    /// Complete, false delivers Complete only. Errors and per-call Complete
    /// results are retained in both modes; the stream type never changes.
    ///
    /// `cancellation` is this invocation's cancellation scope. Create a fresh
    /// value for each independent run; clones are for work belonging to this
    /// run and share its cancellation state.
    fn run(
        &self,
        input: Input,
        stream: bool,
        cancellation: Cancellation,
    ) -> impl Future<Output = Result<Self::Stream, Error>>;

    /// Submit feedback to a live question. Business events stay on run's stream.
    /// The cancellation value scopes this reply submission, not other runs.
    fn reply(
        &self,
        input: Input,
        cancellation: Cancellation,
    ) -> impl Future<Output = Result<(), Error>>;
}
