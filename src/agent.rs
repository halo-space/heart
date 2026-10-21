//! Public Agent entry points, independent of implementation directories.
//!
//! The shared contract lives in components; Chat and ReAct implementations
//! live in core and use the same Agent contract. Runtime owns per-run state.

pub use crate::components::agent::{Agent, Error, Input, Payload};
// Kept only for in-repository compatibility tests while the implementation
// module remains private to the public Agent API. Applications mount concrete
// Tools, MCP servers and Skills with the methods below; they do not construct
// this internal dispatcher.
#[doc(hidden)]
pub use crate::core::agent::mounts as capabilities;
pub use crate::core::agent::subagent::SubAgent;
pub use crate::core::agent::{chat, react};

/// Construct the default Chat Agent implementation.
pub fn chat() -> chat::Builder {
    chat::builder()
}

/// Construct the ReAct implementation through the shared Agent namespace.
pub fn react() -> react::Builder {
    react::builder()
}
