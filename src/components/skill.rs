use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::message::Messages;
use crate::runtime::cancellation::Cancellation;

pub type Error = crate::Error;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Definition {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// A Skill is an agent-facing capability bundle. It may internally compose
/// Prompt, Tool, Agent, MCP, or ordinary functions, but exposes one message
/// input/output contract and does not own Runtime state.
pub trait Skill: Send + Sync {
    fn definition(&self) -> &Definition;

    fn run(
        &self,
        input: Messages,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Messages, Error>> + Send;
}
