use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Cancellation;
use crate::components::message::Message;
use crate::components::tool::{Call, ToolDefinition};

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
pub trait Server: Send + Sync {
    fn definition(&self) -> &Definition;

    /// Return the tools discovered during MCP server initialization.
    ///
    /// MCP discovery is deliberately synchronous at this boundary: an MCP
    /// server is only attachable to an Agent or Workspace after its connection
    /// and tool listing have succeeded. Agents therefore never rediscover
    /// tools while running.
    fn tools(&self) -> &[ToolDefinition];

    fn call(
        &self,
        call: Call,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Message, Error>> + Send;
}
