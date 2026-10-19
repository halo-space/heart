use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Cancellation;

pub type Error = crate::Error;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Definition {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// Instructions and resource information for an Agent, not an executable
/// function. Reading a Skill never grants access to tools or owns Runtime state.
pub trait Skill: Send + Sync {
    fn definition(&self) -> &Definition;

    fn read(
        &self,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<String, Error>> + Send;
}
