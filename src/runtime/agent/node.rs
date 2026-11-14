use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Error;
use crate::components::message::{Message, Messages};
use crate::components::model::token::Usage;
use crate::components::tool::{Call, Evaluation};

pub mod status;
use status::Status;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Attempt {
    pub idx: usize,
    pub status: Status,
    pub created_time: Option<i64>,
    pub finished_time: Option<i64>,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub steps: Vec<Step>,
    /// Messages retained for this attempt, including input and internal
    /// evaluation feedback. They need not equal the node's downstream output.
    #[serde(default)]
    pub messages: Messages,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<super::completion::State>,
    /// Dynamic child nodes of this Attempt; not members of the static Plan.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub nodes: std::collections::BTreeMap<String, State>,
    /// Complete Agent scene for a child execution's current Attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<serde_json::Value>,
    pub error: Option<Error>,
}

/// One question owned by an actual Agent Step, keyed by its i64 reply_id.
/// Resolving this wait does not mark the Step or its node successful.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Wait {
    /// Purpose of the feedback, such as approval, user input or external data.
    pub kind: String,
    /// Resolved does not imply successful node execution.
    pub status: status::wait::Status,
    /// Required contributions, fixed when the wait starts. Missing input uses
    /// the ordinary one-feedback default; serialized scenes retain the value.
    #[serde(default = "Wait::default_count")]
    pub count: usize,
    /// The actual question, not the Tool's successful business result.
    pub question: Message,
    /// Contributions retained for this question; not automatic model context.
    pub feedback: Messages,
}

impl Wait {
    fn default_count() -> usize {
        1
    }

    /// Start one question's collection. This creates waiting facts only; its
    /// owning execution still registers/presents the question and resumes work.
    pub fn new(kind: impl Into<String>, question: Message, count: usize) -> Result<Self, Error> {
        let record = Self {
            kind: kind.into(),
            status: status::wait::Status::Waiting,
            count,
            question,
            feedback: Messages::default(),
        };
        record
            .validate()
            .map_err(|error| Error::new("INVALID_ARGUMENTS", error.message))?;
        Ok(record)
    }

    /// Collect feedback for this question and report whether it is resolved.
    /// This does not evaluate content, grant permission, mark a node successful,
    /// emit events or wake an execution. Those remain the owner's responsibility.
    /// The role/content/values payload identifies a retransmission; changes to
    /// transport metadata or usage do not turn it into another contribution.
    pub fn reply(&mut self, feedback: Message) -> Result<bool, Error> {
        self.validate()?;
        if self
            .feedback
            .as_slice()
            .iter()
            .any(|saved| Self::feedback_match(saved, &feedback))
        {
            return Ok(self.status == status::wait::Status::Resolved);
        }
        if self.status == status::wait::Status::Resolved {
            return Err(Error::new(
                "CONFLICT",
                "this question already received all required feedback",
            ));
        }
        self.feedback.push(feedback);
        let resolved = self.feedback.as_slice().len() == self.count;
        if resolved {
            self.status = status::wait::Status::Resolved;
        }
        Ok(resolved)
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.kind.trim().is_empty()
            || self.count == 0
            || (self.kind == "approval" && self.count != 1)
            || !matches!(
                self.status,
                status::wait::Status::Waiting | status::wait::Status::Resolved
            )
        {
            return Err(Error::new("INVALID_STATE", "invalid feedback wait"));
        }
        let feedback = self.feedback.as_slice();
        if feedback.len() > self.count
            || (self.status == status::wait::Status::Resolved) != (feedback.len() == self.count)
            || feedback.iter().enumerate().any(|(idx, message)| {
                feedback[..idx]
                    .iter()
                    .any(|previous| Self::feedback_match(previous, message))
            })
        {
            return Err(Error::new(
                "INVALID_STATE",
                "feedback does not match the saved wait count and status",
            ));
        }
        Ok(())
    }

    /// Compare only the submitted feedback payload. Transport metadata such
    /// as `reply_id` does not make a repeated message a new contribution.
    fn feedback_match(left: &Message, right: &Message) -> bool {
        left.role == right.role && left.content == right.content && left.values == right.values
    }
}

/// One logical internal Agent operation belonging to an Attempt.
///
/// The runtime owns the shape and persistence of a step. Concrete Agent
/// implementations decide when to append model, tool, evaluation or child
/// execution steps; an empty list is valid for a node that has no internal
/// sub-steps.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Step {
    pub id: String,
    pub kind: String,
    pub status: status::step::Status,
    #[serde(default)]
    pub from: Vec<String>,
    #[serde(default)]
    pub to: Vec<String>,
    pub call_id: Option<String>,
    #[serde(default)]
    pub attempts: Vec<StepAttempt>,
    pub error: Option<Error>,
    /// Current operation lifecycle stage, independent of execution status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    /// Successful middleware methods in this Model Step's current before/after stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idx: Option<usize>,
    /// Selected Call in before; validated effective arguments from call onward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call: Option<Call>,
    /// Last acknowledged Model input; completed middleware transformations are
    /// retained here before dispatch as well as after a successful call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<Box<crate::model::chat::Request>>,
    /// Raw call result in after; completed after result from eval onward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,
    /// Completed business evaluation, including a rejection's feedback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval: Option<Evaluation>,
    /// Current permission method progress for this exact Tool invocation.
    /// Completed decisions are observations, not reusable authorization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission: Option<Box<super::permission::State>>,
    /// Actual questions and accumulated feedback. Historical resolved records
    /// remain here; this is not the top-level ordered waiting locator list.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub waits: BTreeMap<i64, Wait>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct StepAttempt {
    pub status: Status,
    pub created_time: Option<i64>,
    pub finished_time: Option<i64>,
    pub usage: Option<Usage>,
    pub finish_reason: Option<String>,
    /// Received output of an unfinished Model call. This is not a successful
    /// Complete, downstream output or automatic input to another Model call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<Box<Message>>,
    pub error: Option<Error>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Execution {
    pub exec_id: i64,
    pub attempts: Vec<Attempt>,
    #[serde(default)]
    pub messages: Messages,
    pub usage: Option<Usage>,
    pub error: Option<Error>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct State {
    pub status: Status,
    pub messages: Option<Messages>,
    pub usage: Option<Usage>,
    pub execution: Option<Execution>,
}

impl State {
    pub(crate) fn can_transition_to(&self, next: &Status) -> bool {
        matches!(
            (&self.status, next),
            (Status::Pending, Status::Ready)
                | (Status::Ready, Status::Running)
                | (Status::Running, Status::Waiting)
                | (Status::Running, Status::Succeeded)
                | (Status::Running, Status::Failed)
                | (Status::Running, Status::Cancelled)
                | (Status::Waiting, Status::Running)
        )
    }

    pub(crate) fn begin_execution(&mut self, exec_id: i64) -> Result<(), Error> {
        if self.execution.is_some() {
            return Err(Error::new(
                "CONFLICT",
                "agent node already has a current execution",
            ));
        }
        self.execution = Some(Execution {
            exec_id,
            attempts: vec![Attempt {
                completion: None,
                idx: 0,
                status: Status::Running,
                created_time: Some(now_millis()),
                finished_time: None,
                usage: None,
                finish_reason: None,
                steps: Vec::new(),
                messages: Messages::default(),
                nodes: Default::default(),
                agent: None,
                error: None,
            }],
            messages: Messages::default(),
            usage: None,
            error: None,
        });
        Ok(())
    }

    pub(crate) fn current_attempt_mut(&mut self) -> Result<&mut Attempt, Error> {
        self.execution
            .as_mut()
            .and_then(|execution| execution.attempts.last_mut())
            .ok_or_else(|| Error::new("INVALID_STATE", "agent node has no current attempt"))
    }

    pub(crate) fn add_attempt(&mut self) -> Result<(), Error> {
        let execution = self
            .execution
            .as_mut()
            .ok_or_else(|| Error::new("INVALID_STATE", "agent node has no current execution"))?;
        let idx = execution.attempts.len();
        execution.attempts.push(Attempt {
            completion: None,
            idx,
            status: Status::Running,
            created_time: Some(now_millis()),
            finished_time: None,
            usage: None,
            finish_reason: None,
            steps: Vec::new(),
            messages: Messages::default(),
            nodes: Default::default(),
            agent: None,
            error: None,
        });
        execution.error = None;
        Ok(())
    }

    pub(crate) fn set_attempt_result(
        &mut self,
        status: Status,
        usage: Option<Usage>,
        error: Option<Error>,
    ) -> Result<(), Error> {
        {
            let attempt = self.current_attempt_mut()?;
            let terminal = matches!(
                &status,
                Status::Succeeded | Status::Failed | Status::Cancelled
            );
            attempt.status = status;
            if terminal {
                attempt.finished_time = Some(now_millis());
            }
            attempt.usage = usage.clone();
            attempt.error = error.clone();
        }
        let aggregate = {
            let execution = self.execution.as_mut().ok_or_else(|| {
                Error::new("INVALID_STATE", "agent node has no current execution")
            })?;
            execution.usage = execution.attempts.iter().fold(None, |aggregate, attempt| {
                crate::components::model::token::merge(aggregate, attempt.usage.clone())
            });
            execution.error = error;
            execution.usage.clone()
        };
        self.usage = aggregate;
        Ok(())
    }

    pub(crate) fn set_attempt_finish_reason(
        &mut self,
        finish_reason: Option<String>,
    ) -> Result<(), Error> {
        self.current_attempt_mut()?.finish_reason = finish_reason;
        Ok(())
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::model::token::{InputDetails, OutputDetails};
    use serde_json::Map;
    use serde_json::json;

    #[test]
    fn usage_is_recomputable_and_keeps_standard_details_only_in_aggregate() {
        let mut state = State::default();
        state.begin_execution(1).unwrap();
        let first = Usage {
            input: Some(10),
            output: Some(4),
            total: Some(14),
            input_details: Some(InputDetails { cached: Some(3) }),
            output_details: Some(OutputDetails { think: Some(2) }),
            details: Map::from_iter([("provider_private".into(), json!(99))]),
        };
        state
            .set_attempt_result(
                Status::Failed,
                Some(first.clone()),
                Some(Error::new("TIMEOUT", "failed")),
            )
            .unwrap();
        assert!(state.usage.as_ref().unwrap().details.is_empty());
        state.add_attempt().unwrap();
        let second = Usage {
            input: Some(20),
            output: Some(6),
            total: Some(26),
            input_details: Some(InputDetails { cached: Some(5) }),
            output_details: Some(OutputDetails { think: Some(3) }),
            details: Map::from_iter([("provider_private".into(), json!(100))]),
        };
        for _ in 0..2 {
            state
                .set_attempt_result(Status::Succeeded, Some(second.clone()), None)
                .unwrap();
            let usage = state.usage.as_ref().unwrap();
            assert_eq!(usage.total, Some(40));
            assert_eq!(usage.input_details.as_ref().unwrap().cached, Some(8));
            assert_eq!(usage.output_details.as_ref().unwrap().think, Some(5));
            assert!(usage.details.is_empty());
        }
        let execution = state.execution.as_ref().unwrap();
        assert_eq!(execution.attempts[0].usage, Some(first));
        assert_eq!(execution.attempts[1].usage, Some(second));
        assert!(execution.attempts[0].created_time.is_some());
        assert!(execution.attempts[0].finished_time.is_some());
        let missing =
            crate::components::model::token::merge(Some(Usage::default()), Some(Usage::default()))
                .unwrap();
        assert_eq!(missing.input, None);
        assert_eq!(missing.input_details, None);
        assert_eq!(missing.output_details, None);
    }
}
