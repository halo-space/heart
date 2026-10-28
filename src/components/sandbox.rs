use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Cancellation;

pub type Error = crate::Error;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Request {
    pub operation: String,
    #[serde(default)]
    pub input: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Response {
    pub data: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// Execution boundary, not an isolation guarantee. Concrete backends live in
/// `core::sandbox`; selecting one never silently enables host execution.
pub trait Sandbox: Send + Sync {
    fn execute(
        &self,
        request: Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Response, Error>> + Send;
}
