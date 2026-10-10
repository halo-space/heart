pub mod chat;
pub mod session;
pub mod trace;

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use futures_executor::block_on;
    use serde_json::Value;

    use crate::components::memory::short::chat::Store as _;
    use crate::components::memory::short::session::Store as _;
    use crate::components::memory::short::trace::Store as _;
    use crate::components::memory::short::{Type, chat, session, trace};
    use crate::runtime::cancellation::Cancellation;

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

        let traces = super::trace::InMemory::new();
        block_on(traces.create(
            trace::Trace {
                chat_id: 201,
                user_id: Some(2),
                session_id: Some(101),
                agent_id: Some(3),
                execution: Value::Object(Default::default()),
                created_time: 0,
                updated_time: 0,
            },
            &cancellation,
        ))
        .expect("trace create");
        block_on(traces.read(201, &cancellation)).expect("trace read");
    }
}
