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

/// Zero-configuration short-memory store. It owns the Session and Chat
/// records internally and is passed to `memory::Config` as one backend.
pub struct InMemory {
    inner: crate::components::memory::short::Pair<session::InMemory, chat::InMemory>,
}

impl InMemory {
    pub fn new() -> Self {
        Self {
            inner: crate::components::memory::short::Pair::new(
                session::InMemory::new(),
                chat::InMemory::new(),
            ),
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
mod tests {
    use std::collections::BTreeMap;

    use crate::Cancellation;
    use crate::components::memory::short::chat::Store as _;
    use crate::components::memory::short::session::Store as _;
    use crate::components::memory::short::{Type, chat, session};
    use futures_executor::block_on;

    #[test]
    fn short_memory_stores_keep_their_own_crud_boundaries() {
        let cancellation = Cancellation::new();

        let sessions = super::session::InMemory::new();
        let session = block_on(sessions.create(
            session::Session {
                id: 101,
                tenant_id: 1,
                user_id: 2,
                r#type: Type::Agent,
                title: Some("Weather".into()),
                summary: Some("recent context".into()),
                usage: None,
                status: session::Status::Active,
                metadata: BTreeMap::new().into_iter().collect(),
                created_time: 0,
                updated_time: 0,
            },
            &cancellation,
        ))
        .expect("session create");
        assert!(session.created_time > 0);
        assert_eq!(
            block_on(sessions.read(101, &cancellation))
                .expect("session read")
                .title
                .as_deref(),
            Some("Weather")
        );
        let mut changed_type = session.clone();
        changed_type.r#type = Type::Workflow;
        assert_eq!(
            block_on(sessions.update(changed_type, &cancellation))
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );

        let chats = super::chat::InMemory::new();
        let chat = block_on(chats.create(
            chat::Chat {
                id: 201,
                tenant_id: 1,
                user_id: 2,
                agent_id: 3,
                session_id: 101,
                r#type: Type::Agent,
                status: chat::Status::Completed,
                input: None,
                message: None,
                usage: None,
                graph: None,
                state: None,
                metadata: BTreeMap::new().into_iter().collect(),
                started_time: None,
                completed_time: None,
                created_time: 0,
                updated_time: 0,
            },
            &cancellation,
        ))
        .expect("chat create");
        assert!(chat.created_time > 0);
        block_on(chats.delete(201, &cancellation)).expect("chat delete");
        assert_eq!(
            block_on(chats.read(201, &cancellation)).unwrap_err().code,
            "NOT_FOUND"
        );
    }
}
