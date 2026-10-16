use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Cancellation;

pub type Error = crate::Error;

pub mod local;

pub use local::Local;

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

/// Execution boundary, not an isolation guarantee. `Local` runs a host process
/// with the application's permissions. Isolated backends must be explicitly
/// selected; they must never silently fall back to host execution. Merely
/// exporting a default implementation does not enable Agent execution.
pub trait Sandbox: Send + Sync {
    fn execute(
        &self,
        request: Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Response, Error>> + Send;
}
