use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::message::Messages;
use crate::components::model::Error;
pub use crate::components::tool::ToolDefinition;

/// Compatibility export; the canonical public path is `event::Event`.
pub use crate::components::event::Event;

pub type Options = Map<String, Value>;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Config {
    pub model_name: String,
    #[serde(default)]
    pub options: Options,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Named { name: String },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ResponseFormat {
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
    },
}

impl ResponseFormat {
    pub fn validate(&self) -> Result<(), Error> {
        match self {
            Self::JsonObject => Ok(()),
            Self::JsonSchema { name, schema, .. } => {
                if name.trim().is_empty() {
                    return Err(Error::new(
                        "INVALID_ARGUMENTS",
                        "response schema name is empty",
                    ));
                }
                if schema.get("type") != Some(&Value::String("object".into())) {
                    return Err(Error::new(
                        "INVALID_ARGUMENTS",
                        "response schema root must be object",
                    ));
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Request {
    pub messages: Messages,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<ToolChoice>,
    pub stream: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub options: Options,
}

impl Request {
    pub fn validate(&self) -> Result<(), Error> {
        let mut names = std::collections::BTreeSet::new();
        for tool in &self.tools {
            tool.validate()?;
            if !names.insert(tool.name.as_str()) {
                return Err(Error::new("INVALID_ARGUMENTS", "tool names must be unique"));
            }
        }
        match &self.tool_choice {
            Some(ToolChoice::Required) if self.tools.is_empty() => {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "required tool choice needs tools",
                ));
            }
            Some(ToolChoice::Named { name }) if !names.contains(name.as_str()) => {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "named tool is not in tools",
                ));
            }
            _ => {}
        }
        if let Some(format) = &self.response_format {
            format.validate()?;
        }
        Ok(())
    }
}

/// A Provider chooses its concrete stream type. The public contract remains
/// the associated stream and `Result<Event, model::Error>` items.
#[allow(async_fn_in_trait)]
pub trait Chat: Send + Sync {
    type Stream: Stream<Item = Result<Event, Error>> + Send + 'static;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: "test tool".into(),
            parameters: serde_json::from_value(json!({"type": "object", "properties": {}}))
                .unwrap(),
        }
    }

    #[test]
    fn request_validates_tool_choice_and_response_format() {
        let mut request = Request {
            tools: vec![tool("weather")],
            tool_choice: Some(ToolChoice::Named {
                name: "weather".into(),
            }),
            response_format: Some(ResponseFormat::JsonSchema {
                name: "answer".into(),
                schema: json!({"type": "object", "properties": {}}),
                strict: true,
            }),
            ..Request::default()
        };
        assert!(request.validate().is_ok());
        request.tool_choice = Some(ToolChoice::Named {
            name: "missing".into(),
        });
        assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");
    }

    #[test]
    fn config_keeps_model_defaults_separate_from_request_options() {
        let config = Config {
            model_name: "demo".into(),
            options: Map::from_iter([("thinking".into(), json!(true))]),
        };
        let request = Request {
            options: Map::from_iter([("temperature_mode".into(), json!("fast"))]),
            ..Request::default()
        };
        assert_eq!(config.model_name, "demo");
        assert!(config.options.contains_key("thinking"));
        assert!(request.options.contains_key("temperature_mode"));
    }
}
