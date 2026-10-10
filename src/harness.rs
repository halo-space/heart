//! Integration workflows for Agent components.
//!
//! Agents use these private workflows to dispatch middleware, apply permission,
//! recall history, persist Chats and schedule Session compression. Public
//! component contracts live in components; execution graphs, node state and
//! accounting live in runtime::agent. Applications configure an Agent and call
//! its existing run method; lifecycle integration is performed here.
//!
//! Configured instances are supplied by core implementations or application
//! implementations of the same component traits. Compression algorithms and
//! provider calls belong to those implementations; Harness triggers them and
//! coordinates the result with the configured memory stores.

pub(crate) mod access;
pub(crate) mod hooks;
pub(crate) mod memory;
