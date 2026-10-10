//! Agent-level final-result evaluation, separate from Tool::eval.

use crate::{Cancellation, Messages, agent::Input};
use serde::{Deserialize, Serialize};

pub type Error = crate::Error;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub passed: bool,
    pub feedback: Option<Messages>,
}

#[allow(async_fn_in_trait)]
/// Judge the original input against the final Agent result. A business
/// rejection returns passed=false/feedback; technical failure returns Error.
/// This contract does not identify plan nodes, retry tools or replan a graph.
pub trait Evaluator: Send + Sync {
    async fn evaluate(
        &self,
        input: &Input,
        result: &Messages,
        cancellation: &Cancellation,
    ) -> Result<Evaluation, Error>;
}
