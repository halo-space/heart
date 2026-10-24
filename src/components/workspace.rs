use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{future::Future, pin::Pin};

use crate::{
    Cancellation,
    Message,
    mcp::Definition as McpDefinition,
    skill::Definition as SkillDefinition,
    tool::{Call, Evaluation, ToolDefinition},
};

pub type Error = crate::Error;

/// Future returned by a Workspace Tool execution. Keeping the erased return
/// type in one public alias avoids repeating a complex trait-object signature
/// in every concrete Workspace implementation.
pub type ToolFuture<'a> =
    Pin<Box<dyn Future<Output = Result<(Message, Evaluation), Error>> + Send + 'a>>;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Local,
    Server,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Definition {
    pub mode: Mode,
    /// Required only by local implementations. Server implementations keep
    /// resources behind their service and leave this unset.
    pub path: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// Common workspace identity, catalog and execution boundary. The selected
/// concrete implementation owns file or database operations. Its catalog is
/// deliberately expressed as the three capabilities the Agent can mount:
/// ordinary Tools, connected MCP servers and Skills. The public names are
/// simply `tool`, `mcp` and `skill`; the heterogeneous dispatch table stays
/// inside the concrete implementation.
pub trait Workspace: Send + Sync {
    fn definition(&self) -> &Definition;

    /// Tool schemas exposed by this Workspace, including its built-in tools.
    fn tool(&self) -> Vec<ToolDefinition>;

    /// Connected MCP server metadata. The server's discovered Tool schemas
    /// are included in `tools`; this list preserves the MCP server identity.
    fn mcp(&self) -> Vec<McpDefinition>;

    /// Skill metadata exposed by this Workspace. Instructions stay lazy and
    /// are read through the ordinary `skill` Tool call.
    fn skill(&self) -> Vec<SkillDefinition>;

    fn execute<'a>(
        &'a self,
        call: Call,
        cancellation: &'a Cancellation,
    ) -> ToolFuture<'a>;
}

impl<T> Workspace for std::sync::Arc<T>
where
    T: Workspace + ?Sized,
{
    fn definition(&self) -> &Definition {
        (**self).definition()
    }

    fn tool(&self) -> Vec<ToolDefinition> {
        (**self).tool()
    }

    fn mcp(&self) -> Vec<McpDefinition> {
        (**self).mcp()
    }

    fn skill(&self) -> Vec<SkillDefinition> {
        (**self).skill()
    }

    fn execute<'a>(
        &'a self,
        call: Call,
        cancellation: &'a Cancellation,
    ) -> ToolFuture<'a> {
        (**self).execute(call, cancellation)
    }
}
