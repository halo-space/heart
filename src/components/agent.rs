//! Shared Agent contract and input types.
//!
//! Runtime-owned execution graphs and node state live under `runtime::agent`.

use futures_core::Stream;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

use crate::Cancellation;
use crate::components::event::Event;

pub type Error = crate::Error;

/// Public business input shared by ChatAgent and ReActAgent.
///
/// Runtime metadata is kept separate from the user payload. Execution source
/// identities stay in Input/Runtime, not internal Message metadata. Other
/// input metadata is retained while payload becomes ordered model content.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Input {
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub payload: Vec<Payload>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    #[serde(rename = "type")]
    pub r#type: String,
    pub data: Value,
    pub mime_type: Option<String>,
}

/// Read-only facts visible to one Agent invocation.
///
/// The view belongs to the Agent contract. Runtime owns the source State and
/// gives the Agent a cloned projection, so the Agent cannot mutate execution
/// facts or introduce a second checkpoint representation.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    #[default]
    Pending,
    Ready,
    Running,
    Waiting,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct View {
    pub node_id: String,
    pub exec_id: Option<i64>,
    pub attempt_idx: Option<u32>,
    pub status: Status,
    pub input: Value,
    #[serde(default)]
    pub sources: BTreeMap<String, Value>,
    pub result: Option<Value>,
    #[serde(default)]
    pub attempts: Vec<Value>,
    pub error: Option<Value>,
    #[serde(default)]
    pub waiting: Vec<Value>,
    pub usage: Option<Value>,
    #[serde(default)]
    pub children: Vec<Value>,
}

impl View {
    /// Build the minimal view used for a standalone Agent invocation.
    pub fn agent(node_id: impl Into<String>, input: Value, state: Option<Value>) -> Self {
        let mut view = Self {
            node_id: node_id.into(),
            status: Status::Running,
            input,
            ..Self::default()
        };
        if let Some(state) = state {
            view.result = state.get("final_message").cloned();
            view.error = state.get("error").cloned();
            view.waiting = state
                .get("waits")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            view.usage = state.get("usage").cloned();
            view.children = state
                .get("nodes")
                .and_then(Value::as_object)
                .map(|nodes| {
                    nodes
                        .iter()
                        .map(|(node_id, state)| {
                            serde_json::json!({
                                "node_id": node_id,
                                "status": state.get("status"),
                                "messages": state.get("messages"),
                                "waits": state.get("waits"),
                                "usage": state.get("usage"),
                                "error": state.get("error"),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            view.attempts = state
                .get("execution")
                .and_then(|execution| execution.get("attempts"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            view.status = state
                .get("status")
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok())
                .unwrap_or(Status::Running);
        }
        view
    }
}

pub trait Agent: Send + Sync {
    type Stream: Stream<Item = Result<Event, Error>> + Send + 'static;

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
    ) -> impl Future<Output = Result<Self::Stream, Error>> + Send;

    /// Run with the read-only local view of the Agent node or SubAgent call.
    ///
    /// This default preserves the ordinary Agent contract. Implementations
    /// that need node-local diagnostics can override it; the owning runtime
    /// still retains sole ownership of State, execution and checkpoint facts.
    fn run_with_view(
        &self,
        input: Input,
        stream: bool,
        cancellation: Cancellation,
        _view: &View,
    ) -> impl Future<Output = Result<Self::Stream, Error>> + Send {
        self.run(input, stream, cancellation)
    }

    /// Submit feedback to a live question. Business events stay on run's stream.
    /// The cancellation value scopes this reply submission, not other runs.
    fn reply(
        &self,
        input: Input,
        cancellation: Cancellation,
    ) -> impl Future<Output = Result<(), Error>> + Send;
}
