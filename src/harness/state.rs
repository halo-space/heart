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
    pub fn context(&self, session_id: Option<i64>) -> Result<Option<String>, Error> {
        self.context_for(session_id, &[])
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
    pub fn delete(&mut self, id: &str, mut change: Change) -> Result<bool, Error> {
        validate_change(&change)?;
        let before = self.items.len();
        self.items.retain(|item| item.id != id);
        if self.items.len() == before {
            return Ok(false);
        }
        self.version = self.version.saturating_add(1);
        change.version = self.version;
        self.changes.push(change);
        Ok(true)
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
#[path = "../../tests/unit/embedded/src_harness_state_tests.rs"]
mod tests;
