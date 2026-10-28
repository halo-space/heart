//! Native waiting-fact collection, not the complete generic live handoff.
use std::collections::BTreeMap;

use halo::agent::{Cancellation, Message};
use halo::{
    agent::checkpoint::Checkpoint,
    agent::core,
    agent::message::Role,
    agent::model::content::part::Part,
    runtime::agent::{
        State,
        node::{Step, Wait},
        wait,
    },
};
use serde_json::json;

fn message(role: Role, text: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": text}).as_object().unwrap().clone(),
    });
    message
}

fn question() -> Message {
    let mut message = message(Role::Assistant, "请提供本次所需的反馈");
    message.metadata.insert("chat_id".into(), json!(1001));
    message.metadata.insert("reply_id".into(), json!(5001));
    message
}

#[test]
fn one_feedback_resolves_the_wait_but_not_the_step_or_its_business_result() {
    let mut step = Step {
        id: "human_0".into(),
        kind: "human".into(),
        status: "waiting".into(),
        waits: BTreeMap::from([(5001, Wait::new("input", question(), 1).unwrap())]),
        ..Default::default()
    };
    let feedback = message(Role::User, "收到后继续，不做内容质量评估");
    assert!(
        step.waits
            .get_mut(&5001)
            .unwrap()
            .reply(feedback.clone())
            .unwrap()
    );
    assert_eq!(step.waits[&5001].status, "resolved");
    assert_eq!(step.waits[&5001].feedback.as_slice(), &[feedback]);
    assert_eq!(step.status, "waiting", "only the owner resumes execution");
    assert!(
        step.message.is_none(),
        "collection is not a business result"
    );
}

#[tokio::test]
async fn three_feedback_parts_preserve_required_count_and_progress_through_checkpoint() {
    let checkpoint = core::checkpoint::InMemory::new();
    let cancellation = Cancellation::new();
    let original_question = question();
    let mut record = Wait::new("input", original_question.clone(), 3).unwrap();
    let first = message(Role::User, "第一份");
    let second = message(Role::User, "第二份");
    let third = message(Role::User, "第三份");
    assert!(!record.reply(first.clone()).unwrap());
    let state = State {
        chat_id: Some(1001),
        r#type: "agent".into(),
        status: "waiting".into(),
        steps: vec![Step {
            id: "human_0".into(),
            kind: "human".into(),
            status: "waiting".into(),
            waits: BTreeMap::from([(5001, record)]),
            ..Default::default()
        }],
        waits: vec![halo::runtime::agent::wait::Wait {
            nodes: vec![],
            step: "human_0".into(),
        }],
        ..Default::default()
    };
    checkpoint
        .save(1001, serde_json::to_value(&state).unwrap(), &cancellation)
        .await
        .unwrap();
    let saved = checkpoint.load(1001, &cancellation).await.unwrap();
    assert_eq!(saved["steps"][0]["waits"]["5001"]["count"], 3);
    let mut restored: State = serde_json::from_value(saved).unwrap();
    let record = restored.steps[0].waits.get_mut(&5001).unwrap();
    assert!(
        !record.reply(first.clone()).unwrap(),
        "retry must not count twice"
    );
    assert!(!record.reply(second.clone()).unwrap());
    assert_eq!(record.status, "waiting");
    assert_eq!(record.feedback.as_slice(), &[first.clone(), second.clone()]);
    assert!(record.reply(third.clone()).unwrap());
    assert_eq!(record.feedback.as_slice(), &[first, second, third.clone()]);
    assert_eq!(record.question, original_question);
    assert_eq!(record.count, 3);
    assert!(
        record.reply(third).unwrap(),
        "resolved duplicates remain idempotent"
    );
    assert_eq!(record.feedback.as_slice().len(), 3);
    assert_eq!(
        restored.status, "waiting",
        "owner still controls runtime state"
    );
    assert_eq!(
        restored.waits, state.waits,
        "collection does not reorder FIFO"
    );
}

#[test]
fn changed_metadata_never_recounts_a_part_or_changes_the_required_count() {
    let mut record = Wait::new("input", question(), 3).unwrap();
    let first = message(Role::User, "同一份反馈");
    assert!(!record.reply(first.clone()).unwrap());
    let mut retry = first;
    retry.metadata.insert("count".into(), json!(1));
    retry.metadata.insert("transport".into(), json!("retry"));
    assert!(!record.reply(retry).unwrap());
    assert_eq!(record.count, 3);
    assert_eq!(record.feedback.as_slice().len(), 1);
    assert_eq!(record.status, "waiting");
}

#[test]
fn completed_wait_rejects_another_part_without_changing_saved_facts() {
    let mut record = Wait::new("input", question(), 1).unwrap();
    assert!(record.reply(message(Role::User, "已提交")).unwrap());
    let saved = record.clone();
    assert_eq!(
        record
            .reply(message(Role::User, "另一份"))
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    assert_eq!(record, saved);
}

#[test]
fn different_structured_values_are_distinct_feedback_not_duplicate_empty_text() {
    let mut record = Wait::new("input", question(), 2).unwrap();
    let mut first = Message::new(Role::User);
    first.values.insert("answer".into(), json!("first"));
    let mut second = Message::new(Role::User);
    second.values.insert("answer".into(), json!("second"));
    assert!(!record.reply(first.clone()).unwrap());
    assert!(record.reply(second.clone()).unwrap());
    assert_eq!(record.feedback.as_slice(), &[first, second]);
}

#[test]
fn initiation_rejects_zero_count_empty_kind_and_multi_part_approval() {
    for (kind, count) in [("input", 0), ("  ", 1), ("approval", 3)] {
        assert_eq!(
            Wait::new(kind, question(), count).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
    }
    let record = Wait::new("custom-purpose", question(), 3).unwrap();
    assert_eq!(record.kind, "custom-purpose");
}

#[test]
fn invalid_restored_count_or_status_rejects_feedback_atomically() {
    let mut waiting = Wait::new("input", question(), 3).unwrap();
    assert!(!waiting.reply(message(Role::User, "已有内容")).unwrap());
    for (count, status) in [
        (0, "waiting"),
        (1, "waiting"),
        (3, "resolved"),
        (3, "received"),
    ] {
        let mut corrupt = waiting.clone();
        corrupt.count = count;
        corrupt.status = status.into();
        let saved = corrupt.clone();
        assert_eq!(
            corrupt
                .reply(message(Role::User, "新反馈"))
                .unwrap_err()
                .code,
            "INVALID_STATE"
        );
        assert_eq!(corrupt, saved);
    }
}

#[test]
fn omitted_count_uses_single_feedback_default_and_next_snapshot_is_explicit() {
    let mut value = serde_json::to_value(Wait::new("input", question(), 1).unwrap()).unwrap();
    value.as_object_mut().unwrap().remove("count");
    let mut record: Wait = serde_json::from_value(value).unwrap();
    assert_eq!(record.count, 1);
    assert!(record.reply(message(Role::User, "继续")).unwrap());
    assert_eq!(serde_json::to_value(record).unwrap()["count"], 1);
}

#[test]
fn same_payload_for_different_questions_does_not_share_feedback_or_status() {
    let mut first = Wait::new("input", question(), 1).unwrap();
    let mut other_question = question();
    other_question
        .metadata
        .insert("reply_id".into(), json!(5002));
    let mut second = Wait::new("input", other_question.clone(), 3).unwrap();
    let feedback = message(Role::User, "相同内容，不同问题");
    assert!(first.reply(feedback.clone()).unwrap());
    assert!(second.feedback.is_empty());
    assert!(!second.reply(feedback).unwrap());
    assert_eq!(second.count, 3);
    assert_eq!(second.status, "waiting");
    assert_eq!(second.question, other_question);
    assert_eq!(first.status, "resolved");
}

#[tokio::test]
async fn asking_for_feedback_without_an_active_agent_step_is_rejected() {
    assert_eq!(
        wait::ask(
            message(Role::Assistant, "请提供反馈"),
            1,
            &Cancellation::new()
        )
        .await
        .unwrap_err()
        .code,
        "INVALID_STATE"
    );
}
