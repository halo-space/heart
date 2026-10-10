#![allow(async_fn_in_trait)]

//! Short-term conversation history contracts.
//!
//! The three records have independent responsibilities and live in their
//! nearest module. Each module exposes a small `Store` contract with the same
//! CRUD shape; there is no cross-record storage trait.

pub type Error = crate::Error;

/// Top-level owner of a short-term conversation record.
///
/// This is intentionally separate from `memory::Type`, whose values describe
/// long-term memory records (`summary`, `fact`, and `preference`).
#[derive(Clone, Debug, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Type {
    Agent,
    Workflow,
}

pub mod chat;
pub mod session;
pub mod trace;
