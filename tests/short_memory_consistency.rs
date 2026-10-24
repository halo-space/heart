//! Short Memory remains independent of Agent/Workflow execution and Checkpoint.
use std::sync::{Arc, Barrier};

use futures_executor::block_on;
use halo::agent::{Cancellation, Message};
use halo::{
    agent::memory::short::{Type, chat, session, trace},
    agent::message::Role,
    agent::model::token::Usage,
    core,
};
use serde_json::json;

use chat::Store as _;
use session::Store as _;
use trace::Store as _;

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
        metadata: Default::default(),
        started_time: None,
        completed_time: None,
        created_time: 0,
        updated_time: 0,
    }
}

fn trace() -> trace::Trace {
    trace::Trace {
        chat_id: 201,
        user_id: None,
        session_id: None,
        agent_id: None,
        state: json!({"nodes":{},"messages":[]}),
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
    // created_time is store-owned and cannot be changed by a caller.
    snapshot.created_time = 7;
    let updated = block_on(store.update(snapshot.clone(), &cancellation)).unwrap();
    assert_eq!(updated.created_time, created_time);
    assert!(updated.updated_time > previous_time);
    assert_eq!(updated.summary, snapshot.summary);
    assert_eq!(updated.metadata, snapshot.metadata);
    assert_eq!(updated.usage, snapshot.usage);
    assert_eq!(block_on(store.read(101, &cancellation)).unwrap(), updated);

    let mut next = updated;
    summary(&mut next, "replacement", 201, 210);
    next.usage.as_mut().unwrap().total = Some(30);
    let updated = block_on(store.update(next, &cancellation)).unwrap();
    assert_eq!(updated.usage.unwrap().total, Some(30)); // not accumulated
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
    // Conflict recovery is read latest -> recompute -> update, not retry stale.
    let mut fresh = current;
    fresh.title = Some("updated title".into());
    block_on(store.update(fresh, &cancellation)).unwrap();
}

#[test]
fn deleted_and_recreated_session_does_not_accept_its_old_snapshot() {
    let store = core::memory::short::session::InMemory::new();
    let cancellation = Cancellation::new();
    let mut stale = block_on(store.create(session(101), &cancellation)).unwrap();
    block_on(store.delete(101, &cancellation)).unwrap();
    let current = block_on(store.create(session(101), &cancellation)).unwrap();
    assert!(current.updated_time > stale.updated_time);
    summary(&mut stale, "old deleted session", 201, 205);
    assert_eq!(
        block_on(store.update(stale, &cancellation))
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    assert_eq!(block_on(store.read(101, &cancellation)).unwrap(), current);
}

#[test]
fn fast_session_updates_never_reuse_a_snapshot_marker() {
    let store = core::memory::short::session::InMemory::new();
    let cancellation = Cancellation::new();
    let mut current = block_on(store.create(session(101), &cancellation)).unwrap();
    for index in 0..256 {
        let mut next = current.clone();
        next.title = Some(format!("title {index}"));
        let next = block_on(store.update(next, &cancellation)).unwrap();
        assert!(next.updated_time > current.updated_time);
        assert_eq!(
            block_on(store.update(current, &cancellation))
                .unwrap_err()
                .code,
            "CONFLICT"
        );
        current = next;
    }
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
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    let winner = results
        .iter()
        .find_map(|result| result.as_ref().ok())
        .unwrap();
    let loser = results
        .iter()
        .find_map(|result| result.as_ref().err())
        .unwrap();
    assert_eq!(loser.code, "CONFLICT");
    assert_eq!(
        block_on(store.read(101, &Cancellation::new())).unwrap(),
        *winner
    );
}

#[test]
fn session_identity_cannot_be_changed_and_rejection_is_atomic() {
    let store = core::memory::short::session::InMemory::new();
    let cancellation = Cancellation::new();
    let original = block_on(store.create(session(101), &cancellation)).unwrap();
    for index in 0..3 {
        let mut value = original.clone();
        match index {
            0 => value.tenant_id = 9,
            1 => value.user_id = 9,
            _ => value.r#type = Type::Workflow,
        }
        summary(&mut value, "must not commit", 201, 205);
        assert_eq!(
            block_on(store.update(value, &cancellation))
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        assert_eq!(block_on(store.read(101, &cancellation)).unwrap(), original);
    }
}

#[test]
fn session_snapshots_are_detached_and_different_sessions_do_not_conflict() {
    let store = core::memory::short::session::InMemory::new();
    let cancellation = Cancellation::new();
    let mut a = block_on(store.create(session(101), &cancellation)).unwrap();
    let mut b = block_on(store.create(session(102), &cancellation)).unwrap();
    summary(&mut a, "a", 201, 205);
    assert!(
        block_on(store.read(101, &cancellation))
            .unwrap()
            .summary
            .is_none()
    );
    block_on(store.update(a, &cancellation)).unwrap();
    summary(&mut b, "b", 301, 305);
    block_on(store.update(b, &cancellation)).unwrap();
    assert_eq!(
        block_on(store.read(102, &cancellation))
            .unwrap()
            .summary
            .as_deref(),
        Some("b")
    );
}

#[test]
fn chat_ownership_and_original_input_cannot_be_reassigned() {
    let store = core::memory::short::chat::InMemory::new();
    let cancellation = Cancellation::new();
    let original = block_on(store.create(chat(), &cancellation)).unwrap();
    for index in 0..7 {
        let mut value = original.clone();
        match index {
            0 => value.tenant_id = 9,
            1 => value.user_id = 9,
            2 => value.agent_id = 9,
            3 => value.session_id = 9,
            4 => value.r#type = Type::Workflow,
            5 => value.input = None,
            _ => {
                value.input.as_mut().unwrap().payload[0].data = json!({"value":"different request"})
            }
        }
        value.status = chat::Status::Completed;
        assert_eq!(
            block_on(store.update(value, &cancellation))
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        assert_eq!(block_on(store.read(201, &cancellation)).unwrap(), original);
    }
}

#[test]
fn chat_final_message_usage_and_graph_remain_updatable() {
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
fn trace_optional_ownership_can_be_completed_but_not_changed_or_removed() {
    let store = core::memory::short::trace::InMemory::new();
    let cancellation = Cancellation::new();
    let mut value = block_on(store.create(trace(), &cancellation)).unwrap();
    value.user_id = Some(2);
    value.agent_id = Some(3);
    value.session_id = Some(101);
    let original = block_on(store.update(value, &cancellation)).unwrap();
    for index in 0..6 {
        let mut value = original.clone();
        match index {
            0 => value.user_id = Some(9),
            1 => value.agent_id = Some(9),
            2 => value.session_id = Some(9),
            3 => value.user_id = None,
            4 => value.agent_id = None,
            _ => value.session_id = None,
        }
        value.state = json!({"must_not_commit":true});
        assert_eq!(
            block_on(store.update(value, &cancellation))
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        assert_eq!(block_on(store.read(201, &cancellation)).unwrap(), original);
    }
}

#[test]
fn trace_keeps_complete_execution_json_without_a_second_projection_shape() {
    let store = core::memory::short::trace::InMemory::new();
    let cancellation = Cancellation::new();
    let mut value = block_on(store.create(trace(), &cancellation)).unwrap();
    let execution = json!({"chat_id":201,"status":"failed","graph":{"nodes":{"a":{"state":{"execution":{"exec_id":"exec_1","attempts":[{"error":{"code":"TRANSPORT","message":"lost","details":null}}]}}}},"edges":[]},"usage":{"total":14}});
    value.state = execution.clone();
    block_on(store.update(value, &cancellation)).unwrap();
    assert_eq!(
        block_on(store.read(201, &cancellation)).unwrap().state,
        execution
    );
}

#[test]
fn cancelled_updates_and_deletes_leave_all_short_memory_records_intact() {
    let sessions = core::memory::short::session::InMemory::new();
    let chats = core::memory::short::chat::InMemory::new();
    let traces = core::memory::short::trace::InMemory::new();
    let active = Cancellation::new();
    let s = block_on(sessions.create(session(101), &active)).unwrap();
    let c = block_on(chats.create(chat(), &active)).unwrap();
    let t = block_on(traces.create(trace(), &active)).unwrap();
    let cancelled = Cancellation::new();
    cancelled.cancel();
    assert_eq!(
        block_on(sessions.update(s.clone(), &cancelled))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(
        block_on(chats.update(c.clone(), &cancelled))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(
        block_on(traces.update(t.clone(), &cancelled))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(
        block_on(sessions.delete(101, &cancelled)).unwrap_err().code,
        "CANCELLED"
    );
    assert_eq!(
        block_on(chats.delete(201, &cancelled)).unwrap_err().code,
        "CANCELLED"
    );
    assert_eq!(
        block_on(traces.delete(201, &cancelled)).unwrap_err().code,
        "CANCELLED"
    );
    assert_eq!(block_on(sessions.read(101, &active)).unwrap(), s);
    assert_eq!(block_on(chats.read(201, &active)).unwrap(), c);
    assert_eq!(block_on(traces.read(201, &active)).unwrap(), t);
}

#[test]
fn update_does_not_upsert_missing_short_records() {
    let cancellation = Cancellation::new();
    let sessions = core::memory::short::session::InMemory::new();
    let chats = core::memory::short::chat::InMemory::new();
    let traces = core::memory::short::trace::InMemory::new();
    assert_eq!(
        block_on(sessions.update(session(101), &cancellation))
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert_eq!(
        block_on(chats.update(chat(), &cancellation))
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert_eq!(
        block_on(traces.update(trace(), &cancellation))
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert_eq!(
        block_on(sessions.read(101, &cancellation))
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert_eq!(
        block_on(chats.read(201, &cancellation)).unwrap_err().code,
        "NOT_FOUND"
    );
    assert_eq!(
        block_on(traces.read(201, &cancellation)).unwrap_err().code,
        "NOT_FOUND"
    );
}

#[test]
fn short_record_deletes_are_physical_and_do_not_cascade_across_modules() {
    let cancellation = Cancellation::new();
    let sessions = core::memory::short::session::InMemory::new();
    let chats = core::memory::short::chat::InMemory::new();
    let traces = core::memory::short::trace::InMemory::new();
    block_on(sessions.create(session(101), &cancellation)).unwrap();
    block_on(chats.create(chat(), &cancellation)).unwrap();
    block_on(traces.create(trace(), &cancellation)).unwrap();
    block_on(sessions.delete(101, &cancellation)).unwrap();
    assert_eq!(
        block_on(sessions.read(101, &cancellation))
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    block_on(chats.read(201, &cancellation)).unwrap();
    block_on(traces.read(201, &cancellation)).unwrap();
    // No hidden tombstone or foreign-key policy prevents explicit recreation.
    block_on(sessions.create(session(101), &cancellation)).unwrap();
    block_on(chats.delete(201, &cancellation)).unwrap();
    block_on(traces.read(201, &cancellation)).unwrap();
    block_on(traces.delete(201, &cancellation)).unwrap();
}
