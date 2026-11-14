use crate::{Cancellation, core::workspace::Local};
use serde_json::json;
use tempfile::tempdir;

use super::{Change, Goal, GoalStatus, Item, Kind, Scope, State};

fn item(id: &str) -> Item {
    Item {
        id: id.into(),
        kind: Kind::Prompt,
        title: "prompt".into(),
        content: json!("keep changes reviewable"),
        scope: Scope::Session,
        session_id: Some(1),
        source: Some("user".into()),
        version: 0,
        created_time: 1,
        updated_time: 1,
        metadata: Default::default(),
    }
}

fn change(id: &str) -> Change {
    Change {
        id: id.into(),
        trigger: "refine".into(),
        version: 0,
        changes: vec!["updated prompt".into()],
        evidence: vec!["run-1".into()],
        outcome: "accepted".into(),
        created_time: 1,
    }
}

#[test]
fn upsert_replaces_item_and_advances_version() {
    let mut state = State::new();
    state.upsert(item("prompt-1"), change("change-1")).unwrap();
    assert_eq!(state.version, 1);
    assert_eq!(state.get("prompt-1").unwrap().version, 1);
    assert!(state.validate().is_ok());

    let mut replacement = item("prompt-1");
    replacement.content = json!("review every change");
    state.upsert(replacement, change("change-2")).unwrap();
    assert_eq!(state.version, 2);
    assert_eq!(state.items.len(), 1);
    assert_eq!(state.get("prompt-1").unwrap().version, 2);
    assert_eq!(state.changes.len(), 2);
}

#[test]
fn delete_records_only_existing_items() {
    let mut state = State::new();
    assert!(!state.delete("missing", change("change-1")).unwrap());
    assert_eq!(state.version, 0);
    state.upsert(item("prompt-1"), change("change-2")).unwrap();
    assert!(state.delete("prompt-1", change("change-3")).unwrap());
    assert_eq!(state.version, 2);
    assert!(state.get("prompt-1").is_none());
    assert_eq!(state.changes.len(), 2);
}

#[test]
fn state_round_trips_as_json() {
    let mut state = State::new();
    state.upsert(item("prompt-1"), change("change-1")).unwrap();
    let json = serde_json::to_value(&state).unwrap();
    let restored: State = serde_json::from_value(json).unwrap();
    assert_eq!(restored, state);
}

#[test]
fn prompt_and_subagent_views_only_expose_selected_kinds() {
    let mut state = State::new();
    state.upsert(item("prompt-1"), change("change-1")).unwrap();
    let mut skill = item("skill-1");
    skill.kind = Kind::Skill;
    skill.content = json!("use the local skill");
    state.upsert(skill, change("change-2")).unwrap();
    let mut memory = item("memory-1");
    memory.kind = Kind::Memory;
    memory.content = json!("the user prefers concise answers");
    state.upsert(memory, change("change-3")).unwrap();
    let mut member = item("reviewer");
    member.kind = Kind::Subagent;
    member.content = json!({"description": "review"});
    state.upsert(member, change("change-4")).unwrap();
    assert_eq!(
        state
            .context_for(Some(1), &["reviewer".into()])
            .unwrap()
            .as_deref(),
        Some("keep changes reviewable\n\nuse the local skill\n\nthe user prefers concise answers")
    );
    assert_eq!(state.subagents(), vec!["reviewer"]);
}

#[tokio::test]
async fn local_store_persists_current_state_and_history() {
    let directory = tempdir().unwrap();
    let workspace = Local::new(directory.path()).unwrap();
    let store = super::Store::new(workspace.clone());
    let cancellation = Cancellation::new();
    assert!(store.load(&cancellation).await.unwrap().is_none());

    let mut state = State::new();
    state.upsert(item("prompt-1"), change("change-1")).unwrap();
    store.save(&state, &cancellation).await.unwrap();

    let restored = store.load(&cancellation).await.unwrap().unwrap();
    assert_eq!(restored, state);
    assert_eq!(
        workspace
            .list(".heart/harness/history", false, &cancellation)
            .await
            .unwrap(),
        vec![".heart/harness/history/1.json"]
    );

    let mut updated = item("prompt-1");
    updated.content = json!("use the evidence");
    let refined = store
        .refine(updated, change("change-2"), &cancellation)
        .await
        .unwrap();
    assert_eq!(refined.version, 2);
    assert_eq!(
        refined.get("prompt-1").unwrap().content,
        json!("use the evidence")
    );

    let rolled_back = store.rollback(1, &cancellation).await.unwrap();
    assert_eq!(rolled_back.version, 3);
    assert_eq!(
        rolled_back.get("prompt-1").unwrap().content,
        json!("keep changes reviewable")
    );
    assert!(
        workspace
            .read(".heart/harness/history/1.json", &cancellation)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn rejected_refinement_does_not_create_state() {
    let directory = tempdir().unwrap();
    let workspace = Local::new(directory.path()).unwrap();
    let store = super::Store::new(workspace);
    let cancellation = Cancellation::new();
    let mut rejected = change("rejected");
    rejected.outcome = "rejected".into();
    assert!(
        store
            .refine(item("prompt-1"), rejected, &cancellation)
            .await
            .is_err()
    );
    assert!(store.load(&cancellation).await.unwrap().is_none());
}

#[test]
fn active_goal_is_added_to_context_and_completed_goal_is_not() {
    let mut state = State::new();
    state
        .update_goal(
            Goal {
                id: "goal-1".into(),
                objective: "finish the report".into(),
                status: GoalStatus::Active,
                created_time: 1,
                updated_time: 1,
            },
            change("goal-1"),
        )
        .unwrap();
    assert_eq!(
        state.context_for(Some(1), &[]).unwrap().as_deref(),
        Some("Current Goal: finish the report")
    );
    state
        .update_goal(
            Goal {
                id: "goal-1".into(),
                objective: "finish the report".into(),
                status: GoalStatus::Completed,
                created_time: 1,
                updated_time: 2,
            },
            change("goal-1-complete"),
        )
        .unwrap();
    assert!(state.context_for(Some(1), &[]).unwrap().is_none());
}

#[tokio::test]
async fn concurrent_refinements_serialize_and_keep_both_items() {
    let directory = tempdir().unwrap();
    let workspace = Local::new(directory.path()).unwrap();
    let first = super::Store::new(workspace.clone());
    let second = super::Store::new(workspace);
    let cancellation = Cancellation::new();
    let first_result = first.refine(item("one"), change("change-1"), &cancellation);
    let second_result = second.refine(item("two"), change("change-2"), &cancellation);
    let (first, second) = tokio::join!(first_result, second_result);
    assert!(first.is_ok());
    assert!(second.is_ok());
    let final_state = super::Store::new(Local::new(directory.path()).unwrap())
        .load(&cancellation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(final_state.version, 2);
    assert!(final_state.get("one").is_some());
    assert!(final_state.get("two").is_some());
}
