//! Native preparation facts of this Agent execution, never component-global.
use serde::{Deserialize, Serialize};

use crate::{Error, agent::Input};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// New invocation; no Memory Chat has been acknowledged yet.
    Start,
    Execute,
    BeforePrompt,
    AfterPrompt,
    History,
    Complete,
}

/// Successful working input and prompt are distinct from the original query.
/// The cursor acknowledges completed middleware methods in the current stage.
/// Embedded Agents keep this inside their actual parent's Attempt.agent.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct State {
    pub stage: Stage,
    pub idx: usize,
    pub input: Input,
    pub prompt: Option<String>,
}

impl State {
    pub(crate) fn validate(&self, middleware_count: usize, query: &Input) -> Result<(), Error> {
        if self.idx > middleware_count
            || (!matches!(
                self.stage,
                Stage::Start | Stage::Execute | Stage::BeforePrompt | Stage::AfterPrompt
            ) && self.idx != 0)
            || ["tenant_id", "user_id", "agent_id", "session_id", "chat_id"]
                .into_iter()
                .any(|key| self.input.metadata.get(key) != query.metadata.get(key))
        {
            return Err(Error::new(
                "INVALID_STATE",
                "Agent preparation does not match its execution",
            ));
        }
        Ok(())
    }

    pub(crate) fn new(input: Input, prompt: Option<String>) -> Self {
        Self {
            stage: Stage::Start,
            idx: 0,
            input,
            prompt,
        }
    }

    pub(crate) fn restore(snapshot: &serde_json::Value) -> Result<Option<Self>, Error> {
        snapshot
            .get("preparation")
            .filter(|value| !value.is_null())
            .map(|value| {
                serde_json::from_value(value.clone())
                    .map_err(|_| Error::new("INVALID_STATE", "invalid Agent preparation"))
            })
            .transpose()
    }

    pub(crate) fn next(&mut self, stage: Stage) {
        self.stage = stage;
        self.idx = 0;
    }
}
