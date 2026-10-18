//! Agent-owned final-result processing, separate from an internal Model Step.
use crate::{Error, Message};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct State {
    /// after, complete, or committed; no separate Hook execution is created.
    pub stage: String,
    /// Number of successfully completed middleware methods in this stage.
    pub idx: usize,
    /// Last successful transformation, never an unfinished Hook's working copy.
    pub message: Message,
    pub finish_reason: Option<String>,
}
impl State {
    pub(crate) fn new(message: Message, finish_reason: Option<String>) -> Self {
        Self {
            stage: "after".into(),
            idx: 0,
            message,
            finish_reason,
        }
    }
    pub(crate) fn validate(&self, count: usize) -> Result<(), Error> {
        if !matches!(self.stage.as_str(), "after" | "complete" | "committed")
            || self.idx > count
            || (self.stage != "after" && self.idx != count)
        {
            return Err(Error::new(
                "INVALID_STATE",
                "invalid Agent completion phase",
            ));
        }
        Ok(())
    }
    pub(crate) fn pending(&self) -> bool {
        self.stage != "committed"
    }
}
