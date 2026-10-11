//! Agent execution support: mutable run graphs, node state and history.
//!
//! Chat/ReAct definitions, builders and the public run contract live in
//! [`crate::agent`]. This module does not define an Agent kind or require
//! applications to construct a separate Runtime to call an Agent.
//! Component lifecycle integration is performed by [`crate::harness`].

pub mod graph;
pub mod node;
pub(crate) mod usage;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Error;
use crate::components::message::{Message, Messages};
use crate::components::model::token::Usage;

/// Static graph portion of an Agent runtime state.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GraphState {
    pub plan: crate::components::plan::Plan,
}

/// Canonical serialized state for one independent Agent run.
///
/// Checkpoint and Trace each serialize an independent copy of this state. The
/// type does not own persistence and does not make Trace part of recovery.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct State {
    pub chat_id: Option<i64>,
    #[serde(rename = "type")]
    pub r#type: String,
    pub status: String,
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph: Option<GraphState>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub nodes: BTreeMap<String, node::State>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<graph::History>,
    #[serde(default)]
    pub messages: Messages,
    pub error: Option<Error>,
    pub final_message: Option<Message>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}
