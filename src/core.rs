//! Concrete implementations of component contracts.
//!
//! Runtime orchestration does not belong here. Implementations are added only
//! after the corresponding component contract is stable.
pub mod agent;
pub mod checkpoint;
pub mod compression;
pub mod document;
pub mod evaluation;
pub mod mcp;
pub mod memory;
mod message_transform;
pub mod middleware;
mod operation;
pub mod permission;
pub mod plan;
pub mod provider;
pub mod reason;
pub mod sandbox;
pub mod skill;
mod structured;
pub mod workspace;

/// Compatibility namespace matching the design vocabulary. The idiomatic
/// Rust path is `core::memory::InMemory`; this namespace exposes the agreed
/// `core::Memory::memory::new()` construction spelling as well.
#[allow(non_snake_case)]
pub mod Memory {
    pub mod memory {
        pub use super::super::memory::InMemory;

        pub fn new() -> InMemory {
            InMemory::new()
        }
    }
}
