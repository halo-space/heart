//! Permission method progress belongs to the actual Tool Step, not a grant cache.
use crate::{
    Error,
    permission::{Decision, Request},
    tool::Call,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct State {
    /// before, check, after, complete; independent of the Tool lifecycle stage.
    pub stage: String,
    /// Successfully acknowledged methods in the current before/after stage.
    pub idx: usize,
    /// Last successful before transformation; the policy checks this Request.
    pub input: Request,
    /// Last successful after transformation; never used as a policy input.
    pub output: Option<Request>,
    /// Last observed check result, never sufficient to authorize restoration.
    pub decision: Option<Decision>,
}

impl State {
    pub(crate) fn new(input: Request) -> Self {
        Self {
            stage: "before".into(),
            idx: 0,
            input,
            output: None,
            decision: None,
        }
    }
    pub(crate) fn call(&self) -> Result<Call, Error> {
        serde_json::from_value(
            self.input
                .metadata
                .get("call")
                .cloned()
                .ok_or_else(invalid)?,
        )
        .map_err(|_| invalid())
    }
    pub(crate) fn validate(&self, call: &Call, count: usize) -> Result<(), Error> {
        let checked = self.call()?;
        if self.idx > count
            || !matches!(
                self.stage.as_str(),
                "before" | "check" | "after" | "complete"
            )
            || self.input.action != "call"
            || self.input.resource != checked.name
            || checked.name != call.name
            || checked.call_id != call.call_id
            || (self.stage != "complete" && checked != *call)
            || (self.stage == "check" && self.idx != 0)
            || (self.stage == "complete" && self.idx != count)
            || (matches!(self.stage.as_str(), "before" | "check")
                && (self.output.is_some() || self.decision.is_some()))
            || (matches!(self.stage.as_str(), "after" | "complete")
                && (self.output.is_none() || self.decision.is_none()))
        {
            return Err(invalid());
        }
        if let Some(output) = &self.output {
            identity(&self.input, output).map_err(|_| invalid())?;
        }
        Ok(())
    }
}

pub(crate) fn identity(original: &Request, request: &Request) -> Result<(), Error> {
    if request.action != original.action
        || request.resource != original.resource
        || request.metadata.get("call") != original.metadata.get("call")
    {
        Err(Error::new(
            "INVALID_ARGUMENTS",
            "permission middleware cannot change the checked operation or call",
        ))
    } else {
        Ok(())
    }
}
fn invalid() -> Error {
    Error::new("INVALID_STATE", "invalid native permission facts")
}
