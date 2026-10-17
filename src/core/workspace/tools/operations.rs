use std::sync::Arc;

use serde_json::json;
use tokio::sync::RwLock;

use super::{
    Call, Files, Message, Tool, ToolDefinition, definition, string_argument, tool_message,
};
use crate::core::operation::cancellable;
use crate::{Cancellation, Error};

pub(super) struct Delete {
    workspace: Arc<Files>,
    access: Arc<RwLock<()>>,
    definition: ToolDefinition,
}

impl Delete {
    pub(super) fn new(workspace: Arc<Files>, access: Arc<RwLock<()>>) -> Self {
        Self {
            workspace,
            access,
            definition: definition(
                "workspace_delete",
                "Physically delete one file at a workspace-relative path. Does not recursively delete directories.",
                json!({"type":"object","properties":{"path":{"type":"string","minLength":1}},"required":["path"],"additionalProperties":false}),
            ),
        }
    }
}

impl Tool for Delete {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let path = string_argument(call, "path")?;
        let _guard = cancellable(async { Ok(self.access.write().await) }, cancellation).await?;
        self.workspace.delete(path, cancellation).await?;
        Ok(tool_message(json!({"path":path})))
    }
}

pub(super) struct Command {
    workspace: Arc<Files>,
    access: Arc<RwLock<()>>,
    definition: ToolDefinition,
}

impl Command {
    pub(super) fn new(workspace: Arc<Files>, access: Arc<RwLock<()>>) -> Self {
        let description = match workspace.as_ref() {
            Files::Local(_) => {
                "Execute literal argv on the host starting in the configured workspace. A shell runs only when explicitly selected in args. Not an isolation boundary; host privileges apply. Returns stdout, stderr and exit_code."
            }
            Files::Docker(_) => {
                "Execute literal argv in the configured persistent Docker workspace, sharing files with its file tools. A shell runs only when explicitly selected in args. Returns stdout, stderr and exit_code."
            }
        };
        Self {
            workspace,
            access,
            definition: definition(
                "workspace_exec_shell",
                description,
                json!({"type":"object","properties":{"args":{"type":"array","items":{"type":"string"},"minItems":1}},"required":["args"],"additionalProperties":false}),
            ),
        }
    }
}

impl Tool for Command {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        // A command may mutate files. Share the existing writer lock with
        // edits/deletion and readers without creating another execution loop.
        let _guard = cancellable(async { Ok(self.access.write().await) }, cancellation).await?;
        let response = self
            .workspace
            .command(
                serde_json::Value::Object(call.arguments.clone()),
                cancellation,
            )
            .await?;
        Ok(tool_message(response))
    }
}
