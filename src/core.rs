//! Concrete implementations of component contracts.
//!
//! Runtime orchestration does not belong here. Implementations are added only
//! after the corresponding component contract is stable.
pub mod checkpoint;
pub mod compression;
pub(crate) mod dispatch;
pub mod document;
pub mod evaluation;
pub(crate) mod lifecycle;
pub mod mcp;
pub mod memory;
mod message_transform;
pub mod middleware;
pub mod node;
pub(crate) mod operation;
pub mod permission;
pub mod plan;
pub mod provider;
pub mod reason;
pub mod sandbox;
pub mod skill;
mod structured;
pub(crate) mod tool;
pub mod workspace;
