use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::message::Message;
use crate::components::tool::{Call, ToolDefinition};
use crate::runtime::cancellation::Cancellation;

pub type Error = crate::Error;

/// Non-secret MCP server metadata. Connection/authentication details belong to
/// the concrete application integration and are not serialized here.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Definition {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// MCP protocol boundary. Discovery produces normal Tool definitions; MCP is
/// not itself a Workflow Node or a second Tool contract.
#[allow(async_fn_in_trait)]
pub trait Server: Send + Sync {
    fn definition(&self) -> &Definition;

    async fn list_tools(&self, cancellation: &Cancellation) -> Result<Vec<ToolDefinition>, Error>;

    async fn call(&self, call: Call, cancellation: &Cancellation) -> Result<Message, Error>;
}
