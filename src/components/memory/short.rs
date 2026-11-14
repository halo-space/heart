#![allow(async_fn_in_trait)]

//! Short-term conversation history contracts.
//!
//! The two records have independent responsibilities and live in their
//! nearest module. Each module exposes a small `Store` contract with the same
//! CRUD shape; there is no cross-record storage trait. Chat.state carries the
//! complete terminal Agent/Workflow state, so there is no third process record.

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

/// Unified short-term memory contract.
///
/// Session and Chat remain separate records internally, but callers configure
/// one short-memory store. A backend decides how those two records are stored
/// and exposes them only through this composite contract.
pub trait Store: Send + Sync {
    type Session: session::Store;
    type Chat: chat::Store;

    fn session(&self) -> &Self::Session;
    fn chat(&self) -> &Self::Chat;
}
