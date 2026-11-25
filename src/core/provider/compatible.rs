//! Shared transport plus provider-specific Chat, Embed and Multimodal adapters.
//!
//! The modules keep connection setup, chat wire translation, and embedding wire
//! translation separate while preserving the existing provider entry points.

mod chat;
mod embed;
mod multimodal;
mod transport;

pub use chat::{Chat, Events};
pub use embed::Embed;
pub use multimodal::Multimodal;
pub(crate) use transport::{Connection, Kind};

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_core_provider_compatible_tests.rs"]
mod tests;
