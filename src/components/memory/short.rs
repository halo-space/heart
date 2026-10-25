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

/// Small adapter for composing independent Session and Chat implementations.
/// Backends may use this adapter, but application code should normally use a
/// backend's own `short::...::new` constructor.
#[derive(Clone)]
pub struct Pair<S, C> {
    session: std::sync::Arc<S>,
    chat: std::sync::Arc<C>,
}

impl<S, C> Pair<S, C> {
    pub fn new(session: S, chat: C) -> Self {
        Self {
            session: std::sync::Arc::new(session),
            chat: std::sync::Arc::new(chat),
        }
    }
}

impl<S, C> Store for Pair<S, C>
where
    S: session::Store,
    C: chat::Store,
{
    type Session = S;
    type Chat = C;

    fn session(&self) -> &Self::Session {
        self.session.as_ref()
    }

    fn chat(&self) -> &Self::Chat {
        self.chat.as_ref()
    }
}
