//! Durable Harness state.
//!
//! This module describes the state that can outlive one Agent run. It is
//! deliberately independent from Runtime State and Checkpoint: Runtime owns
//! the current execution, while Harness owns reusable prompts, memories,
//! Skills and SubAgent descriptions.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Error;

/// A durable Harness item kind.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Prompt,
    Memory,
    Skill,
    Subagent,
}

/// The lifetime of an item inside a Workspace.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Available only to one session.
    Session,
    /// Available to sessions using the same Workspace.
    Workspace,
}

/// Minimal long-task goal state. Background scheduling is outside Harness.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GoalStatus {
    Active,
    Paused,
    Completed,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Goal {
    pub id: String,
    pub objective: String,
    pub status: GoalStatus,
    pub created_time: i64,
    pub updated_time: i64,
}

/// One reusable Harness item.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Item {
    pub id: String,
    pub kind: Kind,
    pub title: String,
    pub content: Value,
    pub scope: Scope,
    #[serde(default)]
    pub session_id: Option<i64>,
    pub source: Option<String>,
    pub version: u64,
    pub created_time: i64,
    pub updated_time: i64,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// A recorded Harness refinement operation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Change {
    pub id: String,
    pub trigger: String,
    pub version: u64,
    #[serde(default)]
    pub changes: Vec<String>,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub outcome: String,
    pub created_time: i64,
}

/// The current durable Harness state for one Workspace.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct State {
    pub version: u64,
    #[serde(default)]
    pub goal: Option<Goal>,
    #[serde(default)]
    pub items: Vec<Item>,
    #[serde(default)]
    pub changes: Vec<Change>,
}

pub use super::state_store::Store;

impl State {
    /// Create an empty Harness state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Find one item by its stable Harness id.
    pub fn get(&self, id: &str) -> Option<&Item> {
        self.items.iter().find(|item| item.id == id)
    }

    /// Apply a validated Goal update and advance the Harness version.
    pub fn update_goal(&mut self, goal: Goal, mut change: Change) -> Result<(), Error> {
        validate_goal(&goal)?;
        validate_change(&change)?;
        self.version = self.version.saturating_add(1);
        change.version = self.version;
        self.goal = Some(goal);
        self.changes.push(change);
        Ok(())
    }

    /// Build the durable instruction context used by a Harness run.
    ///
    /// Prompt, Skill and durable Memory items become contextual instructions.
    /// Subagent items describe reusable members rather than becoming an
    /// unvalidated prompt fragment.
    pub fn context(&self, session_id: Option<i64>) -> Option<String> {
        self.context_for(session_id, &[]).ok().flatten()
    }

    /// Build context and validate persisted SubAgent descriptions against the
    /// statically registered members supplied by the Harness Builder.
    pub fn context_for(
        &self,
        session_id: Option<i64>,
        registered_subagents: &[String],
    ) -> Result<Option<String>, Error> {
        let mut parts = self
            .items
            .iter()
            .filter(|item| match item.scope {
                Scope::Workspace => true,
                Scope::Session => item.session_id.is_some() && item.session_id == session_id,
            })
            .filter(|item| matches!(item.kind, Kind::Prompt | Kind::Memory | Kind::Skill))
            .filter_map(|item| item.content.as_str())
            .filter(|content| !content.trim().is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        for item in self.items.iter().filter(|item| {
            item.kind == Kind::Subagent
                && match item.scope {
                    Scope::Workspace => true,
                    Scope::Session => item.session_id.is_some() && item.session_id == session_id,
                }
        }) {
            if !registered_subagents.iter().any(|name| name == &item.id) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "Harness state references an unregistered SubAgent",
                ));
            }
            if let Some(content) = item
                .content
                .as_str()
                .filter(|content| !content.trim().is_empty())
            {
                parts.push(format!("SubAgent {}: {content}", item.id));
            }
        }
        if let Some(goal) = &self.goal
            && goal.status == GoalStatus::Active
        {
            parts.push(format!("Current Goal: {}", goal.objective));
        }
        Ok((!parts.is_empty()).then(|| parts.join("\n\n")))
    }

    /// Names of statically registered SubAgents selected by this state.
    /// Empty means that the complete Builder member set is used.
    pub fn subagents(&self) -> Vec<String> {
        self.items
            .iter()
            .filter(|item| item.kind == Kind::Subagent)
            .map(|item| item.id.clone())
            .collect()
    }

    /// Insert or replace an item and advance the state version.
    ///
    /// The caller supplies the item version and the refinement change. This
    /// keeps persistence policy outside the data model while guaranteeing
    /// that every accepted update advances the aggregate version exactly once.
    pub fn upsert(&mut self, mut item: Item, mut change: Change) -> Result<(), Error> {
        validate_item(&item)?;
        validate_change(&change)?;
        self.version = self.version.saturating_add(1);
        item.version = self.version;
        change.version = self.version;
        if let Some(existing) = self.items.iter_mut().find(|value| value.id == item.id) {
            *existing = item;
        } else {
            self.items.push(item);
        }
        self.changes.push(change);
        Ok(())
    }

    /// Remove an item and record the accepted change.
    pub fn delete(&mut self, id: &str, mut change: Change) -> bool {
        let before = self.items.len();
        self.items.retain(|item| item.id != id);
        if self.items.len() == before {
            return false;
        }
        self.version = self.version.saturating_add(1);
        change.version = self.version;
        self.changes.push(change);
        true
    }

    /// Validate the state before it is persisted.
    pub fn validate(&self) -> Result<(), crate::Error> {
        for item in &self.items {
            validate_item(item)?;
        }
        if let Some(goal) = &self.goal {
            validate_goal(goal)?;
        }
        let mut ids = std::collections::BTreeSet::new();
        if self.items.iter().any(|item| !ids.insert(&item.id)) {
            return Err(crate::Error::new(
                "INVALID_ARGUMENTS",
                "harness item ids must be unique",
            ));
        }
        if self
            .items
            .iter()
            .any(|item| item.version == 0 || item.version > self.version)
        {
            return Err(crate::Error::new(
                "INVALID_ARGUMENTS",
                "harness item version is outside state version",
            ));
        }
        for change in &self.changes {
            validate_change(change)?;
            if change.version == 0 || change.version > self.version {
                return Err(crate::Error::new(
                    "INVALID_ARGUMENTS",
                    "harness change version is outside state version",
                ));
            }
        }
        Ok(())
    }
}

fn validate_item(item: &Item) -> Result<(), Error> {
    if item.id.trim().is_empty() {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "Harness item id must not be empty",
        ));
    }
    if item.title.trim().is_empty() {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "Harness item title must not be empty",
        ));
    }
    if item.scope == Scope::Session && item.session_id.is_none() {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "session-scoped Harness item requires session_id",
        ));
    }
    Ok(())
}

fn validate_goal(goal: &Goal) -> Result<(), Error> {
    if goal.id.trim().is_empty() || goal.objective.trim().is_empty() {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "Harness Goal requires an id and objective",
        ));
    }
    Ok(())
}

fn validate_change(change: &Change) -> Result<(), Error> {
    if change.id.trim().is_empty()
        || change.trigger.trim().is_empty()
        || change.outcome != "accepted"
        || change.changes.is_empty()
        || change.changes.iter().any(|value| value.trim().is_empty())
        || change.evidence.is_empty()
        || change.evidence.iter().any(|value| value.trim().is_empty())
    {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "Harness refinement requires an accepted change with evidence",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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
        assert!(!state.delete("missing", change("change-1")));
        assert_eq!(state.version, 0);
        state.upsert(item("prompt-1"), change("change-2")).unwrap();
        assert!(state.delete("prompt-1", change("change-3")));
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
            Some(
                "keep changes reviewable\n\nuse the local skill\n\nthe user prefers concise answers"
            )
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
}
