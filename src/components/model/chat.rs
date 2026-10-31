use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::event::Event;
use crate::components::message::{Message, Messages, Role};
use crate::components::model::Error;
use crate::components::model::content::part::Part;
pub use crate::components::tool::ToolDefinition;

pub type Options = Map<String, Value>;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Config {
    pub model_name: String,
    /// Optional model context window in estimated input tokens.
    ///
    /// This is a common model capability, so it is kept separate from
    /// provider-specific `options` and never sent as a wire parameter.
    #[serde(default)]
    pub context_limit: Option<u64>,
    #[serde(default)]
    pub options: Options,
}

impl Config {
    /// Create a model binding with no provider-specific defaults.
    ///
    /// Provider initialization remains separate: this value only selects the
    /// model and its model-level defaults for one `Chat` instance.
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            context_limit: None,
            options: Options::new(),
        }
    }

    /// Set provider-specific model defaults while keeping the common contract
    /// independent of any one provider's option names.
    pub fn options(mut self, options: Options) -> Self {
        self.options = options;
        self
    }
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
pub(crate) fn estimate_input_tokens(request: &Request) -> u64 {
    let message_bytes: usize = request
        .messages
        .as_slice()
        .iter()
        .map(|message| {
            let role_bytes = format!("{:?}", message.role).len();
            let content_bytes = message.content.iter().map(part_bytes).sum::<usize>();
            // values and metadata are part of the model input contract. They
            // carry structured Function/Tool results, call identities and
            // media references even when the content parts contain little text.
            let values_bytes = serde_json::to_vec(&message.values)
                .map(|value| value.len())
                .unwrap_or(0);
            let metadata_bytes = serde_json::to_vec(&message.metadata)
                .map(|value| value.len())
                .unwrap_or(0);
            role_bytes
                .saturating_add(content_bytes)
                .saturating_add(values_bytes)
                .saturating_add(metadata_bytes)
        })
        .sum();
    let tool_bytes: usize = request
        .tools
        .iter()
        .map(|tool| {
            tool.name.len()
                + tool.description.len()
                + serde_json::Value::Object(tool.parameters.clone())
                    .to_string()
                    .len()
        })
        .sum();
    estimate_tokens(message_bytes.saturating_add(tool_bytes))
}

/// Fit a model request to the context window advertised by its concrete Chat.
///
/// The compressor's `with_limit` controls only the history sent to a summary
/// call. This helper is the separate Runtime guard for ordinary model calls:
/// it keeps system/developer messages, the current summary and the newest
/// complete conversation groups, then removes the oldest removable groups.
/// Tool results stay attached to their preceding group so trimming cannot
/// create an orphaned tool result.
pub(crate) fn fit_request(request: Request, context_limit: Option<u64>) -> Result<Request, Error> {
    let Some(context_limit) = context_limit.filter(|limit| *limit > 0) else {
        return Ok(request);
    };
    let output_reserve = request.max_tokens.map(u64::from).unwrap_or(0);
    let input_limit = context_limit.saturating_sub(output_reserve);
    if estimate_input_tokens(&request) <= input_limit {
        return Ok(request);
    }

    let groups = message_groups(&request.messages);
    if groups.is_empty() {
        return Err(context_limit_error());
    }
    let mut keep = vec![false; groups.len()];
    for (index, group) in groups.iter().enumerate() {
        if group.iter().any(|message| {
            matches!(message.role, Role::System | Role::Developer) || is_summary(message)
        }) {
            keep[index] = true;
        }
    }
    keep[groups.len() - 1] = true;

    let estimate = |keep: &[bool]| {
        let messages = Messages::new(
            groups
                .iter()
                .zip(keep)
                .filter(|(_, keep)| **keep)
                .flat_map(|(group, _)| group.iter().cloned()),
        );
        let mut candidate = request.clone();
        candidate.messages = messages;
        estimate_input_tokens(&candidate)
    };

    // Add older groups only when they fit. Iterating from newest to oldest
    // preserves the most recent context when the complete history is large.
    for index in (0..groups.len()).rev() {
        if keep[index] {
            continue;
        }
        keep[index] = true;
        if estimate(&keep) > input_limit {
            keep[index] = false;
        }
    }

    let mut fitted = request;
    fitted.messages = Messages::new(
        groups
            .iter()
            .zip(&keep)
            .filter(|(_, keep)| **keep)
            .flat_map(|(group, _)| group.iter().cloned()),
    );
    if estimate_input_tokens(&fitted) > input_limit {
        return Err(context_limit_error());
    }
    Ok(fitted)
}

fn context_limit_error() -> Error {
    Error::new(
        "CONTEXT_LENGTH_EXCEEDED",
        "model input exceeds the configured context limit",
    )
}

fn message_groups(messages: &Messages) -> Vec<Vec<Message>> {
    let mut groups: Vec<Vec<Message>> = Vec::new();
    for message in messages.as_slice() {
        // A Tool/Function result belongs to the immediately preceding model
        // group. Keeping it with that group preserves provider call/result
        // ordering during history trimming.
        if matches!(message.role, Role::Tool | Role::Function)
            && let Some(group) = groups.last_mut()
        {
            group.push(message.clone());
        } else {
            groups.push(vec![message.clone()]);
        }
    }
    groups
}

fn is_summary(message: &Message) -> bool {
    message.content.iter().any(|part| {
        part.r#type == "text"
            && part
                .data
                .get("value")
                .and_then(Value::as_str)
                .is_some_and(|value| value.starts_with("[Session summary]\n"))
    })
}

pub(crate) fn part_bytes(part: &Part) -> usize {
    let body = part
        .data
        .get("value")
        .and_then(Value::as_str)
        .map(str::len)
        .unwrap_or_else(|| Value::Object(part.data.clone()).to_string().len());
    part.r#type.len().saturating_add(body)
}

pub(crate) fn estimate_tokens(bytes: usize) -> u64 {
    u64::try_from(bytes.saturating_add(3) / 4).unwrap_or(u64::MAX)
}

pub trait Chat: Send + Sync {
    type Stream: Stream<Item = Result<Event, Error>> + Send + 'static;

    /// Optional model context window, measured in estimated input tokens.
    ///
    /// Runtime uses this only as a safety limit before dispatching a request;
    /// providers that do not expose a limit keep the existing full-history
    /// behavior. The value belongs to the concrete model binding, not to a
    /// second Runtime configuration object.
    fn context_limit(&self) -> Option<u64> {
        None
    }

    fn stream(
        &self,
        request: Request,
    ) -> impl std::future::Future<Output = Result<Self::Stream, Error>> + Send;
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
            context_limit: None,
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

    #[test]
    fn config_constructor_keeps_provider_and_model_binding_separate() {
        let config =
            Config::new("demo").options(Map::from_iter([("thinking".into(), json!(true))]));
        assert_eq!(config.model_name, "demo");
        assert_eq!(config.options["thinking"], json!(true));
    }
}
