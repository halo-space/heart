use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Cancellation;

pub type Error = crate::Error;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Behavior {
    Allow,
    Deny,
    Ask,
}

/// A completed policy decision, not a business Message or a check failure.
/// Rule chaining is internal to the policy; no unresolved passthrough escapes.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Decision {
    pub behavior: Behavior,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Request {
    pub action: String,
    pub resource: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// Permission only decides whether an operation may proceed. It does not
/// create a Message, NodeExecution, Attempt, or Runtime state.
pub trait Permission: Send + Sync {
    fn check(
        &self,
        request: &Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Decision, Error>> + Send;
}
