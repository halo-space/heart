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
#[allow(async_fn_in_trait)]
pub trait Permission: Send + Sync {
    async fn check(&self, request: &Request, cancellation: &Cancellation) -> Result<(), Error>;
}
