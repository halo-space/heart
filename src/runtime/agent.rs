//! Agent execution support: mutable run graphs, node state and history.
//!
//! Chat/ReAct definitions, builders and the public run contract live in
//! [`crate::agent`]. This module does not define an Agent kind or require
//! applications to construct a separate Runtime to call an Agent.
//! Component lifecycle integration is performed by [`crate::harness`].

pub mod graph;
pub mod node;
pub(crate) mod usage;
