use std::sync::Arc;

use serde::Deserialize;
use serde_json::json;
use tokio::sync::RwLock;

use super::{Call, Files, Message, Tool, ToolDefinition, arguments, definition, tool_message};
use crate::{Cancellation, Error, core::operation::cancellable};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Arguments {
    path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

pub(super) struct Edit {
    workspace: Arc<Files>,
    access: Arc<RwLock<()>>,
    definition: ToolDefinition,
}

impl Edit {
    pub(super) fn new(workspace: Arc<Files>, access: Arc<RwLock<()>>) -> Self {
        Self {
            workspace,
            access,
            definition: definition(
                "workspace_edit",
                "Replace exact UTF-8 text in a workspace file. old_string must match exactly and uniquely unless replace_all is true. No match or an ambiguous match leaves the file unchanged.",
                json!({
                    "type":"object", "properties":{
                        "path":{"type":"string"},
                        "old_string":{"type":"string","minLength":1},
                        "new_string":{"type":"string"},
                        "replace_all":{"type":"boolean","default":false}
                    }, "required":["path","old_string","new_string"], "additionalProperties":false
                }),
            ),
        }
    }
}

impl Tool for Edit {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let args: Arguments = arguments(call)?;
        if args.old_string.is_empty() || args.old_string == args.new_string {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "edit requires nonempty old_string and different new_string",
            ));
        }
        // Serialize workspace Tool mutations across the whole read/modify/write,
        // not just the final write. Raw external writers remain application-owned.
        let _guard = cancellable(async { Ok(self.access.write().await) }, cancellation).await?;
        let bytes = self.workspace.read(&args.path, cancellation).await?;
        let content = String::from_utf8(bytes)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "workspace file is not UTF-8 text"))?;
        let count = content.matches(&args.old_string).count();
        if count == 0 {
            return Err(Error::new("NOT_FOUND", "edit text was not found"));
        }
        if count > 1 && !args.replace_all {
            return Err(Error::new(
                "CONFLICT",
                "edit text is ambiguous; use unique text or replace_all",
            ));
        }
        let updated = if args.replace_all {
            content.replace(&args.old_string, &args.new_string)
        } else {
            content.replacen(&args.old_string, &args.new_string, 1)
        };
        self.workspace
            .write(&args.path, updated.into_bytes(), cancellation)
            .await?;
        Ok(tool_message(json!({"path":args.path,"replacements":count})))
    }
}
