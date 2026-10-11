use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::runtime::cancellation::Cancellation;

pub type Error = crate::Error;

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
    ) -> impl std::future::Future<Output = Result<(), Error>> + Send;
}
