//! Public Agent entry points, independent of implementation directories.
//!
//! The shared contract lives in components; Chat and ReAct implementations
//! live in core and use the same Agent contract. Runtime owns per-run state.

pub use crate::components::agent::{Agent, Error, Input, Payload};
// Internal heterogeneous Tool/MCP/Skill dispatch. Applications use the
// concrete `.tools(...)`, `.mcps(...)` and `.skills(...)` methods below.
#[doc(hidden)]
pub use crate::core::agent::mounts as toolkit;
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
