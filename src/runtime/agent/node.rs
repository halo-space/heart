use serde::{Deserialize, Serialize};
use serde_json::Map;

use crate::Error;
use crate::components::message::Messages;
use crate::components::model::token::{InputDetails, OutputDetails, Usage};

/// Runtime state of one logical Agent Plan node.
///
/// This is intentionally separate from `workflow::node::State`: Agent plans
/// have no Pregel step or Workflow Join state.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    #[default]
    Pending,
    Ready,
    Running,
    Waiting,
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Attempt {
    pub idx: usize,
    pub status: Status,
    pub usage: Option<Usage>,
    pub error: Option<Error>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Execution {
    pub exec_id: i64,
    pub attempts: Vec<Attempt>,
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
                | (Status::Running, Status::Interrupted)
                | (Status::Waiting, Status::Running)
                | (Status::Interrupted, Status::Running)
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
                idx: 0,
                status: Status::Running,
                usage: None,
                error: None,
            }],
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
            idx,
            status: Status::Running,
            usage: None,
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
            attempt.status = status;
            attempt.usage = usage.clone();
            attempt.error = error.clone();
        }
        let aggregate = {
            let execution = self.execution.as_mut().ok_or_else(|| {
                Error::new("INVALID_STATE", "agent node has no current execution")
            })?;
            execution.usage = execution.attempts.iter().fold(None, |aggregate, attempt| {
                merge_usage(aggregate, attempt.usage.clone())
            });
            execution.error = error;
            execution.usage.clone()
        };
        self.usage = aggregate;
        Ok(())
    }
}

fn merge_usage(left: Option<Usage>, right: Option<Usage>) -> Option<Usage> {
    match (left, right) {
        (Some(left), Some(right)) => Some(Usage {
            input: add_count(left.input, right.input),
            output: add_count(left.output, right.output),
            total: add_count(left.total, right.total),
            input_details: if left.input_details.is_some() || right.input_details.is_some() {
                Some(InputDetails {
                    cached: add_count(
                        left.input_details.and_then(|value| value.cached),
                        right.input_details.and_then(|value| value.cached),
                    ),
                })
            } else {
                None
            },
            output_details: if left.output_details.is_some() || right.output_details.is_some() {
                Some(OutputDetails {
                    think: add_count(
                        left.output_details.and_then(|value| value.think),
                        right.output_details.and_then(|value| value.think),
                    ),
                })
            } else {
                None
            },
            details: Map::new(),
        }),
        (left, right) => left.or(right).map(|mut usage| {
            usage.details.clear();
            usage
        }),
    }
}

fn add_count(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => left.checked_add(right),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let missing = merge_usage(Some(Usage::default()), Some(Usage::default())).unwrap();
        assert_eq!(missing.input, None);
        assert_eq!(missing.input_details, None);
        assert_eq!(missing.output_details, None);
    }
}
