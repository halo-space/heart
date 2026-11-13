use std::sync::Arc;

use crate::Cancellation;
use crate::components::agent::Error;
use crate::components::message::Message;
use crate::components::tool::{Call, Tool, ToolDefinition};

use super::entry::{READER_NAME, SkillView};

/// The single model-visible Tool used to read a selected Skill's instructions
/// or a normalized resource path. It never executes Skill scripts.
pub(crate) struct Reader {
    skills: Vec<Arc<dyn SkillView>>,
    definition: ToolDefinition,
}

impl Reader {
    pub(crate) fn new(skills: Vec<Arc<dyn SkillView>>) -> Self {
        Self {
            skills,
            definition: ToolDefinition {
                name: READER_NAME.into(),
                description: "Read a Skill's full instructions by exact name, or a text resource relative to that Skill with `resource`. Skills explain how to use available Tools and MCP servers; reading does not execute them.".into(),
                parameters: serde_json::from_value(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "skill": {
                            "type": "string",
                            "description": "The exact name of the registered Skill to read."
                        },
                        "resource": {
                            "type": "string",
                            "description": "Optional path to a text resource inside that Skill."
                        }
                    },
                    "required": ["skill"],
                    "additionalProperties": false
                }))
                .expect("static Skill parameters are an object"),
            },
        }
    }
}

impl Tool for Reader {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let name = call
            .arguments
            .get("skill")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "skill name is required"))?;
        let skill = self
            .skills
            .iter()
            .find(|skill| skill.definition().name == name)
            .ok_or_else(|| Error::new("NOT_FOUND", "skill is not registered"))?;
        if let Some(path) = call
            .arguments
            .get("resource")
            .and_then(serde_json::Value::as_str)
        {
            if path.is_empty()
                || path.contains(['\\', '\0'])
                || path
                    .split('/')
                    .any(|part| part.is_empty() || part == "." || part == "..")
            {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "Skill resource requires a normalized relative path",
                ));
            }
            let data = crate::cancellation::cancellable(
                skill.read_resource(path.to_owned(), cancellation.clone()),
                cancellation,
            )
            .await?;
            let content = String::from_utf8(data).map_err(|_| {
                Error::new("INVALID_ARGUMENTS", "Skill resource must be UTF-8 text")
            })?;
            Ok(Message::function(serde_json::json!({
                "name": name, "resource": path, "content": content
            })))
        } else {
            let instructions =
                crate::cancellation::cancellable(skill.read(cancellation.clone()), cancellation)
                    .await?;
            Ok(Message::function(serde_json::json!({
                "name": name, "instructions": instructions
            })))
        }
    }
}
