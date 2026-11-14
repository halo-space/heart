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
pub use crate::runtime::agent::team;
pub use crate::runtime::agent::{chat, react};
#[doc(hidden)]
pub mod subagent {
    pub use crate::runtime::agent::subagent::*;
}
pub use crate::runtime::trace;

/// Protocol adapters used by the outer HTTP or transport layer.
pub mod protocol {
    pub use crate::core::protocol::a2a;
    pub use crate::core::protocol::ag_ui;
}

/// External messaging channel adapters.
pub mod channels {
    pub use crate::channels::dingtalk;
    pub use crate::channels::feishu;
}

/// External service dispatch built on top of Agent and Workflow runtimes.
/// Transport bindings are intentionally kept outside the Runtime itself.
pub mod server {
    pub use crate::server::a2a;
    pub use crate::server::axum;
    pub use crate::server::client;
    pub use crate::server::sse;
    pub use crate::server::task;
    pub use crate::server::webhook;
}

pub use crate::components::message;
pub use crate::components::{
    checkpoint, evaluation, event, function, knowledge, mcp, model, permission, plan, prompt,
    reason, sandbox, skill, workspace,
};

/// Memory contracts plus the Agent runtime's composition configuration.
pub mod memory {
    pub use crate::components::memory::*;
    pub use crate::runtime::agent::memory::Config;
}

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

/// Serializable Agent execution graph facts for integrations that need to
/// inspect or persist the runtime projection. Execution ownership remains in
/// `runtime`; this is the stable `halo::agent` namespace for the public data.
pub mod graph {
    pub use crate::runtime::agent::graph::*;
}

/// Public waiting data and reply helpers used by channel integrations.
pub mod wait {
    pub use crate::runtime::agent::wait::*;
}

/// Hidden runtime scope hooks used by integration tests for nested Agent
/// recovery. They are not part of the stable application API.
#[doc(hidden)]
pub mod scope {
    pub use crate::runtime::agent::scope::{capture, restore};
}

/// Hidden lifecycle helpers used by integration tests that exercise saved
/// Agent attempt facts through the public crate boundary.
#[doc(hidden)]
pub mod lifecycle {
    pub use crate::runtime::agent::lifecycle::*;
}

/// Hidden model-step helpers used by integration tests for partial output.
#[doc(hidden)]
pub mod model_steps {
    pub use crate::runtime::agent::model_steps::*;
}

#[doc(hidden)]
pub mod access {
    pub use crate::runtime::agent::access::*;
}

#[doc(hidden)]
pub mod hooks {
    pub use crate::runtime::agent::hooks::*;
}

#[doc(hidden)]
pub mod agent_input {
    pub use crate::components::agent_input::*;
}

/// Agent-local state data exposed for diagnostics and checkpoint tooling.
pub mod state {
    pub use crate::runtime::agent::State;
    pub use crate::runtime::agent::status::Status;
    pub mod node {
        pub use crate::runtime::agent::node::*;
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

/// Construct the first static Team composition around a ReAct Leader.
pub fn team() -> team::Builder {
    team::builder()
}
