//! Short memory keeps Session and Chat as the only persisted run records.
use std::sync::{Arc, Barrier};

use futures_executor::block_on;
use halo::agent::{Cancellation, Message};
use halo::{
    agent::memory::short::{Type, chat, session},
    agent::message::Role,
    agent::model::token::Usage,
    core,
};
use serde_json::json;

use chat::Store as _;
use session::Store as _;

fn session(id: i64) -> session::Session {
    session::Session {
        id,
        tenant_id: 1,
        user_id: 2,
        r#type: Type::Agent,
        title: Some("Weather".into()),
        summary: None,
        usage: None,
        status: session::Status::Active,
        metadata: json!({"custom":true}).as_object().unwrap().clone(),
        created_time: 0,
        updated_time: 0,
    }
}

fn chat() -> chat::Chat {
    chat::Chat {
        id: 201,
        tenant_id: 1,
        user_id: 2,
        agent_id: 3,
        session_id: 101,
        r#type: Type::Agent,
        status: chat::Status::Running,
        input: Some(chat::Request {
            metadata: Default::default(),
            payload: vec![chat::Payload {
                r#type: "text".into(),
                data: json!({"value":"weather?"}),
                mime_type: None,
            }],
        }),
        message: None,
        usage: None,
        graph: None,
        state: None,
        metadata: Default::default(),
        started_time: None,
        completed_time: None,
        created_time: 0,
        updated_time: 0,
    }
}

fn summary(value: &mut session::Session, label: &str, from: i64, to: i64) {
    value.summary = Some(label.into());
    value.metadata.insert("from".into(), json!(from));
    value.metadata.insert("to".into(), json!(to));
    value.usage = Some(Usage {
        input: Some(50),
        output: Some(10),
        total: Some(60),
        ..Default::default()
    });
}

#[test]
fn session_replaces_summary_coverage_and_usage_atomically() {
    let store = core::memory::short::session::InMemory::new();
    let cancellation = Cancellation::new();
    let mut snapshot = block_on(store.create(session(101), &cancellation)).unwrap();
    let created_time = snapshot.created_time;
    let previous_time = snapshot.updated_time;
    summary(&mut snapshot, "new summary", 201, 205);
    snapshot.created_time = 7;
    let updated = block_on(store.update(snapshot.clone(), &cancellation)).unwrap();
    assert_eq!(updated.created_time, created_time);
    assert!(updated.updated_time > previous_time);
    assert_eq!(updated.summary, snapshot.summary);
    assert_eq!(updated.metadata, snapshot.metadata);
    assert_eq!(updated.usage, snapshot.usage);
    let mut next = updated;
    summary(&mut next, "replacement", 201, 210);
    next.usage.as_mut().unwrap().total = Some(30);
    assert_eq!(
        block_on(store.update(next, &cancellation))
            .unwrap()
            .usage
            .unwrap()
            .total,
        Some(30)
    );
}

#[test]
fn stale_session_summary_cannot_overwrite_a_newer_snapshot() {
    let store = core::memory::short::session::InMemory::new();
    let cancellation = Cancellation::new();
    let original = block_on(store.create(session(101), &cancellation)).unwrap();
    let mut current = original.clone();
    summary(&mut current, "new", 201, 210);
    let current = block_on(store.update(current, &cancellation)).unwrap();
    let mut stale = original;
    summary(&mut stale, "old delayed result", 201, 205);
    assert_eq!(
        block_on(store.update(stale, &cancellation))
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    assert_eq!(block_on(store.read(101, &cancellation)).unwrap(), current);
}

#[test]
fn concurrent_session_updates_from_one_snapshot_have_one_winner() {
    let store = Arc::new(core::memory::short::session::InMemory::new());
    let original = block_on(store.create(session(101), &Cancellation::new())).unwrap();
    let barrier = Arc::new(Barrier::new(3));
    let handles: Vec<_> = (0..2)
        .map(|index| {
            let store = store.clone();
            let barrier = barrier.clone();
            let mut value = original.clone();
            summary(&mut value, &format!("summary {index}"), 201, 205 + index);
            std::thread::spawn(move || {
                barrier.wait();
                block_on(store.update(value, &Cancellation::new()))
            })
        })
        .collect();
    barrier.wait();
    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter_map(|result| result.as_ref().err())
            .next()
            .unwrap()
            .code,
        "CONFLICT"
    );
}

#[test]
fn chat_state_contains_complete_workflow_snapshot_at_closeout() {
    let store = core::memory::short::chat::InMemory::new();
    let cancellation = Cancellation::new();
    let mut value = block_on(store.create(chat(), &cancellation)).unwrap();
    value.status = chat::Status::Completed;
    value.message = Some(Message::new(Role::Assistant));
    value.usage = Some(Usage {
        total: Some(123),
        ..Default::default()
    });
    value.graph = Some(json!({"nodes":{"search":{}},"edges":[]}));
    value.state = Some(json!({"status":"succeeded","nodes":{}}));
    value.completed_time = Some(100);
    let value = block_on(store.update(value, &cancellation)).unwrap();
    assert_eq!(block_on(store.read(201, &cancellation)).unwrap(), value);
    let options =
        json!({"filter":{"tenant_id":1,"user_id":2,"agent_id":3,"session_id":101,"type":"agent"}});
    assert_eq!(
        block_on(store.list(options.as_object().unwrap().clone(), &cancellation)).unwrap(),
        vec![value]
    );
}

#[test]
fn chat_updates_and_deletes_are_physical_and_cancellation_safe() {
    let store = core::memory::short::chat::InMemory::new();
    let active = Cancellation::new();
    let value = block_on(store.create(chat(), &active)).unwrap();
    let cancelled = Cancellation::new();
    cancelled.cancel();
    assert_eq!(
        block_on(store.update(value.clone(), &cancelled))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(
        block_on(store.delete(value.id, &cancelled))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(block_on(store.read(value.id, &active)).unwrap(), value);
    block_on(store.delete(value.id, &active)).unwrap();
    assert_eq!(
        block_on(store.read(value.id, &active)).unwrap_err().code,
        "NOT_FOUND"
    );
}
