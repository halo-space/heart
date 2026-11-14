use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::components::message::Messages;
use crate::components::model::token::Usage;

pub mod status;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Attempt {
    pub idx: u32,
    pub status: status::attempt::Status,
    pub created_time: Option<i64>,
    pub finished_time: Option<i64>,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub steps: Vec<Step>,
    #[serde(default)]
    pub messages: Messages,
    /// Embedded Agent runtime state owned by this Workflow Attempt.
    #[serde(default)]
    pub agent: Option<serde_json::Value>,
    pub graph: Option<Box<crate::runtime::workflow::graph::State>>,
    pub error: Option<Error>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Execution {
    pub exec_id: i64,
    pub idx: u64,
    pub status: status::execution::Status,
    pub attempts: Vec<Attempt>,
    pub messages: Messages,
    pub usage: Option<Usage>,
    pub error: Option<Error>,
    pub from: Vec<ExecutionRef>,
    pub to: Vec<String>,
    #[serde(default)]
    pub waits: BTreeMap<String, Wait>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Step {
    pub id: String,
    pub kind: String,
    pub status: status::step::Status,
    pub from: Vec<String>,
    pub to: Vec<String>,
    pub call_id: Option<String>,
    pub attempts: Vec<StepAttempt>,
    pub error: Option<Error>,
    /// Native Tool before/call/after/complete progress, not a separate Hook Step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    /// Last acknowledged arguments, independent of the original graph input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call: Option<crate::tool::Call>,
    /// Successful call result in after; successful transformed result in complete.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<crate::Message>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StepAttempt {
    pub status: status::attempt::Status,
    pub created_time: Option<i64>,
    pub finished_time: Option<i64>,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
    pub error: Option<Error>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Wait {
    pub kind: String,
    pub status: status::wait::Status,
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
    pub status: status::Status,
    pub executions: BTreeMap<i64, Execution>,
}

impl State {
    pub fn new(status: status::Status) -> Self {
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
#[path = "../../../tests/unit/embedded/src_runtime_workflow_node_tests.rs"]
mod tests;
