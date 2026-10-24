//! Agent execution support: mutable run graphs, node state and history.
//!
//! Chat/ReAct definitions, builders and the public run contract live in
//! [`crate::agent`]. This module does not define an Agent kind or require
//! applications to construct a separate Runtime to call an Agent.
//! Agent-specific integrations are owned by this Runtime or by the Agent
//! implementation that composes the relevant Core components.

pub(crate) mod checkpoint;
pub mod completion;
pub mod graph;
pub(crate) mod model_steps;
pub mod node;
pub mod permission;
pub mod preparation;
pub(crate) mod reply;
pub(crate) mod scope;
pub(crate) mod tool_steps;
pub(crate) mod usage;
pub mod wait;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Error;
use crate::components::message::{Message, Messages};
use crate::components::model::token::Usage;

/// Canonical serialized state for one independent Agent run.
///
/// Checkpoint and Chat.state serialize independent copies of this state. The
/// type does not own persistence; Chat.state is an audit snapshot, not a
/// recovery source.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct State {
    pub chat_id: Option<i64>,
    #[serde(rename = "type")]
    pub r#type: String,
    pub status: String,
    /// Outward delivery mode fixed for this run and restored from Checkpoint.
    pub stream: bool,
    pub usage: Option<Usage>,
    /// Preparation of this actual Agent invocation (not any shared model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation: Option<preparation::State>,
    /// Agent's own after_execute result, not the internal model's after phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<completion::State>,
    /// Root Agent execution, created when it first delegates a child. Root
    /// messages/steps remain above the Attempt instead of being duplicated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<node::Execution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<graph::State>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub nodes: BTreeMap<String, node::State>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<graph::History>,
    #[serde(default)]
    pub messages: Messages,
    /// Internal steps of a graphless/root Agent. Planned task steps belong
    /// exclusively to their node's current Attempt instead.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<node::Step>,
    /// Ordered unresolved human-feedback positions; questions remain in Steps.
    #[serde(default)]
    pub waits: Vec<wait::Wait>,
    pub error: Option<Error>,
    pub final_message: Option<Message>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}
