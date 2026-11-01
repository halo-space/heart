use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::event::Event;
use crate::components::message::Messages;
use crate::components::model::Error;
use crate::components::tool::ToolDefinition;

pub type Options = Map<String, Value>;

/// Multimodal tool selection is the same public behavior as Chat, but it is
/// owned by this model capability and does not reuse a Chat type.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Named { name: String },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Config {
    pub model_name: String,
    #[serde(default)]
    pub context_limit: Option<u64>,
    #[serde(default)]
    pub options: Options,
}

impl Config {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            context_limit: None,
            options: Options::new(),
        }
    }

    pub fn options(mut self, options: Options) -> Self {
        self.options = options;
        self
    }
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
    #[serde(default)]
    pub modalities: Vec<String>,
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub options: Options,
}

impl Request {
    pub fn validate(&self) -> Result<(), Error> {
        validate_tools(&self.tools, self.tool_choice.as_ref())?;
        if let Some(format) = &self.response_format {
            format.validate()?;
        }
        if self.modalities.iter().any(|value| value.trim().is_empty()) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "modalities must contain nonempty values",
            ));
        }
        Ok(())
    }
}

pub trait Multimodal: Send + Sync {
    type Stream: Stream<Item = Result<Event, Error>> + Send;

    fn stream(
        &self,
        request: Request,
    ) -> impl std::future::Future<Output = Result<Self::Stream, Error>> + Send;
}

fn validate_tools(tools: &[ToolDefinition], choice: Option<&ToolChoice>) -> Result<(), Error> {
    let mut names = std::collections::BTreeSet::new();
    for tool in tools {
        tool.validate()?;
        if !names.insert(tool.name.as_str()) {
            return Err(Error::new("INVALID_ARGUMENTS", "tool names must be unique"));
        }
    }
    match choice {
        Some(ToolChoice::Required) if tools.is_empty() => Err(Error::new(
            "INVALID_ARGUMENTS",
            "required tool choice needs tools",
        )),
        Some(ToolChoice::Named { name }) if !names.contains(name.as_str()) => Err(Error::new(
            "INVALID_ARGUMENTS",
            "named tool is not in tools",
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, Request, ResponseFormat, ToolChoice};
    use crate::components::tool::ToolDefinition;
    use serde_json::json;

    #[test]
    fn request_validates_modalities_and_schema() {
        let mut request = Request {
            modalities: vec!["text".into(), "image".into()],
            response_format: Some(ResponseFormat::JsonSchema {
                name: "answer".into(),
                schema: json!({"type":"object","properties":{}}),
                strict: true,
            }),
            ..Request::default()
        };
        request.validate().unwrap();
        request.modalities.push("  ".into());
        assert!(request.validate().is_err());
    }

    #[test]
    fn config_keeps_model_defaults_separate() {
        let config = Config::new("vision-model");
        assert_eq!(config.model_name, "vision-model");
        assert!(config.options.is_empty());
    }

    #[test]
    fn tool_choice_is_owned_by_multimodal_contract() {
        let request = Request {
            tools: vec![ToolDefinition {
                name: "describe".into(),
                description: "Describe the image".into(),
                parameters: json!({"type":"object"}).as_object().unwrap().clone(),
            }],
            tool_choice: Some(ToolChoice::Named {
                name: "describe".into(),
            }),
            ..Request::default()
        };
        request.validate().unwrap();
    }
}
