//! Public Agent entry points, independent of implementation directories.
//!
//! The shared contract and default Chat live in components; ReAct is a core
//! implementation of that same contract. Runtime owns per-run execution state.

pub use crate::components::agent::{Agent, Error, Input, Payload, chat, toolkit};
pub use crate::core::agent::react;

/// Construct the ReAct implementation through the shared Agent namespace.
pub fn react() -> react::Builder {
    react::builder()
}
