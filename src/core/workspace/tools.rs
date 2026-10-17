//! Core Tool adapters bound to a concrete file resource backend.
//!
//! These tools share the concrete Local file implementation, not the
//! common Workspace contract. Database workspaces do not use this adapter.
//! No host paths are accepted, and destructive deletion is intentionally not
//! included in the default toolset.

use std::sync::Arc;
use tokio::sync::{RwLock, RwLockReadGuard};

use serde_json::{Map, Value, json};

use super::Local;
use crate::components::message::{Message, Role};
use crate::components::model::content::part::{Part, object as part_data};
use crate::components::tool::{Call, Tool, ToolDefinition};
use crate::core::agent::toolkit::Toolkit;
use crate::{Cancellation, Error};

mod edit;
mod search;

/// Bind the default file tools to one shared backend and the existing Toolkit.
pub(super) fn attach(workspace: Local, toolkit: Toolkit) -> Toolkit {
    let workspace = Arc::new(workspace);
    let access = Arc::new(RwLock::new(()));
    toolkit
        .tool(Read {
            workspace: Arc::clone(&workspace),
            access: Arc::clone(&access),
            definition: read_definition(),
        })
        .tool(Write {
            workspace: Arc::clone(&workspace),
            access: Arc::clone(&access),
            definition: write_definition(),
        })
        .tool(search::List::new(
            Arc::clone(&workspace),
            Arc::clone(&access),
        ))
        .tool(edit::Edit::new(Arc::clone(&workspace), Arc::clone(&access)))
        .tool(search::Glob::new(
            Arc::clone(&workspace),
            Arc::clone(&access),
        ))
        .tool(search::Grep::new(workspace, access))
}

struct Read {
    workspace: Arc<Local>,
    access: Arc<RwLock<()>>,
    definition: ToolDefinition,
}

impl Tool for Read {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let path = string_argument(call, "path")?;
        let _guard = reading(&self.access, cancellation).await?;
        let bytes = self.workspace.read(path, cancellation).await?;
        let content = String::from_utf8(bytes)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "workspace file is not UTF-8 text"))?;
        Ok(tool_message(json!({ "path": path, "content": content })))
    }
}

struct Write {
    workspace: Arc<Local>,
    access: Arc<RwLock<()>>,
    definition: ToolDefinition,
}

impl Tool for Write {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let path = string_argument(call, "path")?;
        let content = string_argument(call, "content")?;
        let _guard = super::super::operation::cancellable(
            async { Ok(self.access.write().await) },
            cancellation,
        )
        .await?;
        self.workspace
            .write(path, content.as_bytes().to_vec(), cancellation)
            .await?;
        Ok(tool_message(json!({ "path": path })))
    }
}

fn arguments<T: serde::de::DeserializeOwned>(call: &Call) -> Result<T, Error> {
    serde_json::from_value(Value::Object(call.arguments.clone())).map_err(|error| {
        Error::new("INVALID_ARGUMENTS", "invalid workspace tool arguments")
            .with_details(error.to_string())
    })
}

async fn reading<'a>(
    access: &'a RwLock<()>,
    cancellation: &Cancellation,
) -> Result<RwLockReadGuard<'a, ()>, Error> {
    super::super::operation::cancellable(async { Ok(access.read().await) }, cancellation).await
}

fn string_argument<'a>(call: &'a Call, key: &str) -> Result<&'a str, Error> {
    call.arguments
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "workspace tool argument is invalid"))
}

pub(super) fn tool_message(value: Value) -> Message {
    let text = value.to_string();
    let mut message = Message::new(Role::Tool);
    message.content.push(Part {
        r#type: "text".into(),
        data: part_data(json!({ "value": text })),
    });
    message.values = value.as_object().cloned().unwrap_or_else(Map::new);
    message
}

fn read_definition() -> ToolDefinition {
    definition(
        "workspace_read",
        "Read a UTF-8 text file from the configured workspace using a relative path.",
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
            "additionalProperties": false
        }),
    )
}

fn write_definition() -> ToolDefinition {
    definition(
        "workspace_write",
        "Write UTF-8 text to a relative file path in the configured workspace.",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"],
            "additionalProperties": false
        }),
    )
}

pub(super) fn definition(name: &str, description: &str, parameters: Value) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: description.into(),
        parameters: parameters.as_object().cloned().unwrap_or_else(Map::new),
    }
}
