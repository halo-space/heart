//! Shared transport plus provider-specific Chat and Embed adapters.
//!
//! The modules keep connection setup, chat wire translation, and embedding wire
//! translation separate while preserving the existing provider entry points.

mod chat;
mod embed;
mod transport;

pub use chat::{Chat, Events};
pub use embed::Embed;
pub(crate) use transport::{Connection, Kind};

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_core_provider_compatible_tests.rs"]
mod tests;
