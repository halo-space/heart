pub mod chat;
/// Elasticsearch implementations of the short-memory stores.
pub mod es {
    use elasticsearch::Elasticsearch;

    pub use super::super::es::short::Config;

    /// Construct one short-memory store. Session and Chat remain internal to
    /// the backend and are not split in the application-facing API.
    pub fn new(
        client: Elasticsearch,
        config: Config,
    ) -> Result<impl crate::components::memory::short::Store, crate::Error> {
        super::super::es::short::Memory::new(client, config)
    }
}
pub mod session;

/// Compose independent Session and Chat backends into one short-memory store.
/// This helper belongs to Core because it is a concrete implementation, not a
/// component contract.
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

impl<S, C> crate::components::memory::short::Store for Pair<S, C>
where
    S: crate::components::memory::short::session::Store,
    C: crate::components::memory::short::chat::Store,
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

/// Zero-configuration short-memory store. It owns the Session and Chat
/// records internally and is passed to `memory::Config` as one backend.
pub struct InMemory {
    inner: Pair<session::InMemory, chat::InMemory>,
}

impl InMemory {
    pub fn new() -> Self {
        Self {
            inner: Pair::new(session::InMemory::new(), chat::InMemory::new()),
        }
    }
}

impl Default for InMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::components::memory::short::Store for InMemory {
    type Session = session::InMemory;
    type Chat = chat::InMemory;

    fn session(&self) -> &Self::Session {
        self.inner.session()
    }

    fn chat(&self) -> &Self::Chat {
        self.inner.chat()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_core_memory_short_tests.rs"]
mod tests;
