//! Concrete Agent implementations built from shared Core capabilities.
//!
//! Chat and ReAct implement the same Agent component contract; they do not
//! define separate Runtime kinds.

pub(crate) mod access;
pub mod chat;
mod completion;
pub(crate) mod hooks;
pub(crate) mod input;
mod lifecycle;
mod preparation;
pub mod react;
pub mod subagent;
pub mod toolkit;
