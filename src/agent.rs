//! Public Agent entry points, independent of implementation directories.
//!
//! The shared contract lives in components; Chat and ReAct implementations
//! live in the Agent runtime. Core provides the concrete component instances
//! consumed by those implementations.

pub use crate::cancellation::Cancellation;
pub use crate::components::agent::{Agent, Error, Input, Payload};
pub use crate::components::message::{Message, Messages};
/// Concrete model providers are part of the Agent-facing implementation
/// namespace. The provider implementations themselves remain under `core`;
/// this re-export keeps application code on the single `halo::agent` entry.
pub use crate::core::provider;
pub use crate::runtime::agent::subagent::SubAgent;
pub use crate::runtime::agent::{chat, react};
pub use crate::runtime::trace;

/// Protocol adapters used by the outer HTTP or transport layer.
pub mod protocol {
    pub use crate::core::protocol::a2a;
    pub use crate::core::protocol::ag_ui;
}

/// External service dispatch built on top of Agent and Workflow runtimes.
/// Transport bindings are intentionally kept outside the Runtime itself.
pub mod server {
    pub use crate::server::a2a;
    pub use crate::server::axum;
    pub use crate::server::client;
    pub use crate::server::sse;
    pub use crate::server::tasks;
    pub use crate::server::webhook;
}

pub use crate::components::message;
pub use crate::components::{
    checkpoint, evaluation, event, function, knowledge, mcp, memory, model, permission, plan,
    prompt, reason, sandbox, skill, workspace,
};

pub mod compression {
    pub use crate::components::compression::*;
}

pub mod document {
    pub use crate::components::document::*;
    pub use crate::core::document::{LocalLoader, chunker, graph, parser, raptor, transformer};
}

pub mod middleware {
    pub use crate::components::middleware::*;
    pub use crate::core::middleware::Retry;
}

pub mod node {
    pub use crate::components::node::*;
    pub use crate::core::node::Echo;
}

pub mod tool {
    pub use crate::components::tool::*;
    pub use crate::core::tool::execute;
}

/// Public entry to the existing concrete implementation layer. The contents
/// and hierarchy are exactly the same as `src/core`; this module only places
/// that implementation layer below the stable `agent` namespace.
pub mod core {
    pub use crate::core::*;
}

/// Workflow orchestration exposed as part of the Agent domain.
pub mod workflow {
    pub use crate::runtime::workflow::*;
}

/// Runtime implementation details needed by low-level integrations.
///
/// The crate deliberately exposes them below `halo::agent`; runtime is not a
/// separate top-level public module. Most applications should use
/// `agent::chat`, `agent::react`, `agent::workflow`, and the Agent contract.
#[doc(hidden)]
pub mod runtime {
    pub mod agent {
        pub use crate::runtime::agent::{State, graph, node, react, wait};
    }
}

/// Construct the default Chat Agent implementation.
pub fn chat() -> chat::Builder {
    chat::builder()
}

/// Construct the ReAct implementation through the shared Agent namespace.
pub fn react() -> react::Builder {
    react::builder()
}
