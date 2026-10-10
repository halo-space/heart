//! Halo Agent/Workflow execution kernel.
//!
//! This crate starts from the frozen public contracts. Runtime scheduling is
//! intentionally added after these contracts compile and are covered by tests.

pub mod agent;
mod components;
pub mod core;
pub mod runtime;

pub use components::checkpoint;
pub use components::compression;
/// Document contracts and the agreed concrete construction paths. Contract
/// definitions remain under components; implementations remain under core.
pub mod document {
    pub use crate::components::document::*;
    pub use crate::core::document::{LocalLoader, chunker, graph, parser, raptor, transformer};
}
pub use components::error::Error;
pub use components::evaluation;
pub use components::event;
pub use components::function;
pub mod harness;
pub use components::knowledge;
pub use components::mcp;
pub use components::memory;
pub use components::message::{self, Message, Messages};
pub mod middleware {
    pub use crate::components::middleware::*;
    pub use crate::core::middleware::Retry;
}
pub use components::model;
pub use components::node;
pub use components::permission;
pub use components::plan;
pub use components::prompt;
pub use components::reason;
pub use components::sandbox;
pub use components::skill;
pub use components::tool;
pub use components::workspace;
pub use core::provider;
pub use runtime::cancellation::Cancellation;
