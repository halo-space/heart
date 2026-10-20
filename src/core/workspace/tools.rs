//! Core Tool adapters bound to a concrete file resource backend.
//!
//! These tools share the concrete Local/Docker file implementation, not the
//! common Workspace contract. Database workspaces do not use this adapter.
//! File arguments are relative. Single-file deletion is built in; host
//! commands require an explicitly selected execution environment (configured
//! Local Sandbox or Docker Workspace).

use std::sync::Arc;
use tokio::sync::{RwLock, RwLockReadGuard};

use serde_json::{Map, Value, json};

use super::{Docker, Local};
use crate::components::message::{Message, Role};
use crate::components::model::content::part::{Part, object as part_data};
use crate::components::skill::{Definition as SkillDefinition, Skill as _, Source as SkillSource};
use crate::components::tool::{Call, Tool, ToolDefinition};
use crate::core::agent::toolkit::Toolkit;
use crate::{Cancellation, Error};

mod edit;
mod operations;
mod search;

/// Private dispatch over the two actual file environments. This is not a
/// public backend trait, registry, or a file requirement on Server Workspace.
pub(super) enum Files {
    Local(Local),
    Docker(Docker),
}

impl From<Local> for Files {
    fn from(local: Local) -> Self {
        Self::Local(local)
    }
}

impl Files {
    fn command_enabled(&self) -> bool {
        match self {
            Self::Local(local) => local.sandbox.is_some(),
            Self::Docker(_) => true,
        }
    }

    async fn list(
        &self,
        path: &str,
        recursive: bool,
        cancellation: &Cancellation,
    ) -> Result<Vec<String>, Error> {
        match self {
            Self::Local(local) => local.list(path, recursive, cancellation).await,
            Self::Docker(docker) => docker.list(path, recursive, cancellation).await,
        }
    }

    async fn read(&self, path: &str, cancellation: &Cancellation) -> Result<Vec<u8>, Error> {
        match self {
            Self::Local(local) => local.read(path, cancellation).await,
            Self::Docker(docker) => docker.read(path, cancellation).await,
        }
    }

    async fn write(
        &self,
        path: &str,
        data: Vec<u8>,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        match self {
            Self::Local(local) => local.write(path, data, cancellation).await,
            Self::Docker(docker) => docker.write(path, data, cancellation).await,
        }
    }

    async fn delete(&self, path: &str, cancellation: &Cancellation) -> Result<(), Error> {
        match self {
            Self::Local(local) => local.delete(path, cancellation).await,
            Self::Docker(docker) => docker.delete(path, cancellation).await,
        }
    }

    async fn command(&self, input: Value, cancellation: &Cancellation) -> Result<Value, Error> {
        match self {
            Self::Local(local) => {
                let sandbox = local.sandbox.as_ref().ok_or_else(|| {
                    Error::new(
                        "UNAVAILABLE",
                        "workspace has no command execution environment",
                    )
                })?;
                let response = sandbox
                    .execute_at(
                        crate::sandbox::Request {
                            operation: "command".into(),
                            input,
                            metadata: Default::default(),
                        },
                        Some(&local.root),
                        cancellation,
                    )
                    .await?;
                Ok(response.data)
            }
            Self::Docker(docker) => docker.command(input, cancellation).await,
        }
    }
}

/// Bind the default file tools to one shared backend and the existing Toolkit.
pub(super) fn attach(workspace: Files, toolkit: Toolkit) -> Toolkit {
    let command_enabled = workspace.command_enabled();
    let discover_skills = match &workspace {
        Files::Local(_) => true,
        Files::Docker(docker) => docker.discover_skills,
    };
    let workspace = Arc::new(workspace);
    let access = Arc::new(RwLock::new(()));
    let toolkit = if discover_skills {
        toolkit.skill_source(DirectorySkills {
            workspace: Arc::clone(&workspace),
        })
    } else {
        toolkit
    };
    let toolkit = toolkit
        .tools(Read {
            workspace: Arc::clone(&workspace),
            access: Arc::clone(&access),
            definition: read_definition(),
        })
        .tools(Write {
            workspace: Arc::clone(&workspace),
            access: Arc::clone(&access),
            definition: write_definition(),
        })
        .tools(search::List::new(
            Arc::clone(&workspace),
            Arc::clone(&access),
        ))
        .tools(edit::Edit::new(Arc::clone(&workspace), Arc::clone(&access)))
        .tools(search::Glob::new(
            Arc::clone(&workspace),
            Arc::clone(&access),
        ))
        .tools(search::Grep::new(
            Arc::clone(&workspace),
            Arc::clone(&access),
        ))
        .tools(operations::Delete::new(
            Arc::clone(&workspace),
            Arc::clone(&access),
        ));
    if command_enabled {
        toolkit.tools(operations::Command::new(workspace, access))
    } else {
        toolkit
    }
}

/// Discover standard `skills/<name>/SKILL.md` directories in the same file
/// environment as the Workspace tools. No second host path is introduced for
/// Docker, and resource reads remain confined by the Workspace path checks.
struct DirectorySkills {
    workspace: Arc<Files>,
}

impl DirectorySkills {
    async fn load(
        &self,
        name: &str,
        cancellation: &Cancellation,
    ) -> Result<super::super::skill::Skill, Error> {
        let path = format!("skills/{name}/SKILL.md");
        let bytes = self.workspace.read(&path, cancellation).await?;
        let markdown = std::str::from_utf8(&bytes)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "Skill document must be UTF-8"))?;
        let skill = super::super::skill::Skill::from_markdown(markdown)?;
        if skill.definition().name != name {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "Skill name must match its directory",
            ));
        }
        Ok(skill)
    }
}

impl SkillSource for DirectorySkills {
    async fn list(&self, cancellation: &Cancellation) -> Result<Vec<SkillDefinition>, Error> {
        let paths = match self.workspace.list("skills", false, cancellation).await {
            Ok(paths) => paths,
            Err(error) if error.code == "NOT_FOUND" => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let mut definitions = Vec::new();
        for path in paths {
            let Some(name) = path.strip_prefix("skills/") else {
                continue;
            };
            if name.is_empty() || name.contains('/') {
                continue;
            }
            match self.load(name, cancellation).await {
                Ok(skill) => definitions.push(skill.definition().clone()),
                Err(error) if error.code == "NOT_FOUND" => continue,
                Err(error) => return Err(error),
            }
        }
        Ok(definitions)
    }

    async fn read(&self, name: &str, cancellation: &Cancellation) -> Result<String, Error> {
        self.load(name, cancellation)
            .await?
            .read(cancellation)
            .await
    }

    async fn read_resource(
        &self,
        name: &str,
        path: &str,
        cancellation: &Cancellation,
    ) -> Result<Vec<u8>, Error> {
        super::validate_path(path)?;
        // Validate the Skill directory before reading a related resource.
        self.load(name, cancellation).await?;
        self.workspace
            .read(&format!("skills/{name}/{path}"), cancellation)
            .await
    }
}

struct Read {
    workspace: Arc<Files>,
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
    workspace: Arc<Files>,
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
