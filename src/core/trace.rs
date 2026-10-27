//! Trace projection contracts and concrete implementations.
//!
//! Runtime code owns execution state and emits observations. Concrete trace
//! backends live below this module and never replace State or Checkpoint.

pub mod fastrace;
