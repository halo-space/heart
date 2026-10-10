use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::components::message::Messages;
use crate::components::model::token::Usage;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    #[default]
    Idle,
    Active,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionStatus {
    #[default]
    Pending,
    Running,
    Waiting,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AttemptStatus {
    #[default]
    Running,
    Waiting,
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Attempt {
    pub idx: u32,
    pub status: AttemptStatus,
    pub started_time: Option<i64>,
    pub finished_time: Option<i64>,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub steps: Vec<Step>,
    #[serde(default)]
    pub messages: Messages,
    pub graph: Option<Box<crate::runtime::workflow::graph::State>>,
    pub error: Option<Error>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Execution {
    pub exec_id: i64,
    pub idx: u64,
    pub status: ExecutionStatus,
    pub attempts: Vec<Attempt>,
    pub messages: Messages,
    pub usage: Option<Usage>,
    pub error: Option<Error>,
    pub from: Vec<ExecutionRef>,
    pub to: Vec<String>,
    #[serde(default)]
    pub waits: BTreeMap<String, Wait>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Step {
    pub id: String,
    pub kind: String,
    pub status: String,
    pub from: Vec<String>,
    pub to: Vec<String>,
    pub call_id: Option<String>,
    pub attempts: Vec<StepAttempt>,
    pub error: Option<Error>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StepAttempt {
    pub status: AttemptStatus,
    pub started_time: Option<i64>,
    pub finished_time: Option<i64>,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
    pub error: Option<Error>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Wait {
    pub kind: String,
    pub status: String,
    pub created_time: i64,
    pub resolved_time: Option<i64>,
    pub response: Option<Messages>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExecutionRef {
    pub node_id: String,
    pub exec_id: i64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct State {
    pub status: Status,
    pub executions: BTreeMap<i64, Execution>,
}

impl State {
    pub fn new(status: Status) -> Self {
        Self {
            status,
            executions: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, execution: Execution) -> Result<(), Error> {
        if execution.exec_id <= 0 {
            return Err(Error::new("INVALID_ARGUMENTS", "exec_id must be positive"));
        }
        if self.executions.contains_key(&execution.exec_id) {
            return Err(Error::new("CONFLICT", "execution already exists"));
        }
        self.executions.insert(execution.exec_id, execution);
        Ok(())
    }

    pub fn execution_mut(&mut self, exec_id: i64) -> Result<&mut Execution, Error> {
        self.executions
            .get_mut(&exec_id)
            .ok_or_else(|| Error::new("NOT_FOUND", "execution not found"))
    }

    pub fn append_attempt(&mut self, exec_id: i64, attempt: Attempt) -> Result<(), Error> {
        let execution = self.execution_mut(exec_id)?;
        let expected = u32::try_from(execution.attempts.len())
            .map_err(|_| Error::new("INVALID_STATE", "too many attempts"))?;
        if attempt.idx != expected {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "attempt index must be contiguous",
            ));
        }
        execution.attempts.push(attempt);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_attempt_history_and_rejects_gaps() {
        let mut state = State::new(Status::Active);
        state
            .insert(Execution {
                exec_id: 1,
                ..Execution::default()
            })
            .unwrap();
        state
            .append_attempt(
                1,
                Attempt {
                    idx: 0,
                    status: AttemptStatus::Failed,
                    ..Attempt::default()
                },
            )
            .unwrap();
        assert_eq!(state.executions[&1].attempts.len(), 1);
        assert_eq!(
            state
                .append_attempt(
                    1,
                    Attempt {
                        idx: 2,
                        status: AttemptStatus::Running,
                        ..Attempt::default()
                    },
                )
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
    }

    #[test]
    fn rejects_duplicate_and_non_positive_execution_ids() {
        let mut state = State::new(Status::Idle);
        assert_eq!(
            state
                .insert(Execution {
                    exec_id: 0,
                    ..Execution::default()
                })
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        state
            .insert(Execution {
                exec_id: 2,
                ..Execution::default()
            })
            .unwrap();
        assert_eq!(
            state
                .insert(Execution {
                    exec_id: 2,
                    ..Execution::default()
                })
                .unwrap_err()
                .code,
            "CONFLICT"
        );
    }
}
