#![allow(unused_imports, ambiguous_glob_imported_traits)]
use futures_core::Stream;
use futures_executor::block_on;
use futures_util::stream;
use halo::agent::access::Access;
use halo::agent::agent_input::user_message;
use halo::agent::chat::runner::*;
use halo::agent::chat::*;
use halo::agent::core::dispatch::Dispatch;
use halo::agent::core::provider;
use halo::agent::hooks::Hooks;
use halo::agent::memory;
use halo::agent::model;
use halo::agent::model::chat::Request;
use halo::agent::model::content::part::Part;
use halo::agent::state::node::{Step, status::Status};
use halo::agent::subagent::Root;
use halo::agent::tool::{Call, Tool, ToolDefinition};
use halo::agent::trace::fastrace::Root as TraceRoot;
use halo::agent::*;
use serde_json::{Map, Value, json};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
mod unit {
    use super::*;
    include!("unit/agent_chat.rs");
}
