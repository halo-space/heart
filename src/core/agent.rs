//! Concrete Agent implementations built from shared Core dispatch.
//!
//! Chat and ReAct implement the same Agent component contract; they do not
//! define separate Runtime kinds.

pub(crate) mod access;
// Internal dispatcher for heterogeneous Tools, MCP servers and Skills. It is
// deliberately not part of the public Agent API; callers configure each kind
// through `.tools(...)`, `.mcps(...)` and `.skills(...)`.
pub mod chat;
mod completion;
#[doc(hidden)]
pub(crate) mod dispatch;
pub(crate) mod hooks;
pub(crate) mod input;
mod lifecycle;
mod preparation;
pub mod react;
pub mod subagent;
