//! Concrete Sandbox implementations for the component contract.

/// Host-process execution. This is not an isolation mechanism.
pub mod local;
pub use local::{Config, Local};

/// Docker-backed execution.
pub mod docker;
