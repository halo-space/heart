//! Workflow definition helpers that do not own Runtime state.
//!
//! The DSL parser is deliberately kept in Core. It accepts only the
//! serializable Heart Native DSL; executable Rust functions, Agents and
//! compiled subgraphs are bound later by the appropriate layer.

pub mod code;
pub mod dsl;
