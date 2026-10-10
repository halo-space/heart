use std::collections::{BTreeMap, BTreeSet};
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use eventsource_stream::{EventStreamError, Eventsource};
use futures_core::Stream;
use futures_util::{StreamExt, stream};
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde_json::{Map, Value, json};

use crate::event::Event;
use crate::message::{Message, Role};
use crate::model::{
    self,
    chat::{Request, ResponseFormat, ToolChoice},
    content::Part,
    token::{InputDetails, OutputDetails, Usage},
};

type Error = model::Error;

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Kind {
    OpenAI,
    DeepSeek,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::OpenAI => "openai",
            Self::DeepSeek => "deepseek",
        }
    }
}

#[derive(Clone)]
pub(super) struct Connection {
    client: reqwest::Client,
    base_url: String,
    provider_id: i64,
    kind: Kind,
}

impl Connection {
    pub(super) fn new(
        kind: Kind,
        provider_id: i64,
        name: &str,
        base_url: &str,
        api_key: &str,
        timeout: u64,
    ) -> Result<Self, Error> {
        if name != kind.name() || timeout == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "provider name or timeout is invalid",
            ));
        }
        let url = reqwest::Url::parse(base_url)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid provider base URL"))?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "provider URL must be HTTP(S) without credentials, query or fragment",
            ));
        }
        if api_key.trim().is_empty() {
            return Err(Error::new("AUTHENTICATION", "provider API key is required"));
        }
        let mut authorization = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|_| Error::new("AUTHENTICATION", "provider API key is not a valid header"))?;
        authorization.set_sensitive(true);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, authorization);
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(timeout))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| Error::new("TRANSPORT", "provider HTTP client initialization failed"))?;
        Ok(Self {
            client,
            base_url: url.as_str().trim_end_matches('/').into(),
            provider_id,
            kind,
        })
    }

    pub(super) fn chat(&self, config: model::chat::Config) -> Result<Chat, Error> {
        validate_model(&config.model_name)?;
        Ok(Chat {
            connection: self.clone(),
            config,
        })
    }

    pub(super) fn embed(&self, config: model::embed::Config) -> Result<Embed, Error> {
        validate_model(&config.model_name)?;
        Ok(Embed {
            connection: self.clone(),
            config,
        })
    }

    async fn post(&self, endpoint: &str, body: Value) -> Result<reqwest::Response, Error> {
        let response = self
            .client
            .post(format!("{}/{endpoint}", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(transport_error)?;
        if response.status().is_success() {
            return Ok(response);
        }
        let status = response.status().as_u16();
        let body = response.json::<Value>().await.unwrap_or(Value::Null);
        Err(api_error(status, &body))
    }

    fn metadata(&self, model_name: &str) -> Map<String, Value> {
        Map::from_iter([
            ("provider_id".into(), self.provider_id.into()),
            ("provider_name".into(), self.kind.name().into()),
            ("model_name".into(), model_name.into()),
        ])
    }
}

fn validate_model(name: &str) -> Result<(), Error> {
    if name.trim().is_empty() {
        Err(Error::new("INVALID_ARGUMENTS", "model name is empty"))
    } else {
        Ok(())
    }
}

fn transport_error(error: reqwest::Error) -> Error {
    if error.is_timeout() {
        Error::new("TIMEOUT", "provider request timed out")
    } else if error.is_decode() {
        Error::new("INVALID_RESPONSE", "provider response cannot be decoded")
    } else {
        Error::new("TRANSPORT", "provider HTTP transport failed")
    }
}

fn api_error(status: u16, body: &Value) -> Error {
    let raw_code = body
        .pointer("/error/code")
        .and_then(Value::as_str)
        .unwrap_or("");
    let code = match raw_code {
        "context_length_exceeded" => "CONTEXT_LENGTH_EXCEEDED",
        "content_filter" | "content_policy_violation" => "CONTENT_FILTERED",
        "unsupported_parameter" | "unsupported_value" => "UNSUPPORTED",
        _ => match status {
            400 | 422 => "INVALID_ARGUMENTS",
            401 => "AUTHENTICATION",
            403 => "PERMISSION_DENIED",
            408 => "TIMEOUT",
            429 => "RATE_LIMITED",
            500..=599 => "UNAVAILABLE",
            _ => "PROVIDER",
        },
    };
    // Do not copy raw Provider messages/body: they can echo credentials or prompts.
    Error::new(code, "provider request failed").with_details(format!("http_status={status}"))
}

#[derive(Clone)]
pub struct Chat {
    connection: Connection,
    config: model::chat::Config,
}

/// Provider-owned concrete stream; public items remain the shared Event.
pub struct Events {
    inner: Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>,
}

impl Stream for Events {
    type Item = Result<Event, Error>;
    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(context)
    }
}

impl model::chat::Chat for Chat {
    type Stream = Events;
    async fn stream(&self, request: Request) -> Result<Events, Error> {
        let body = encode_request(self.connection.kind, &self.config, &request)?;
        let format = request.response_format.clone();
        let response = self.connection.post("chat/completions", body).await?;
        let metadata = self.connection.metadata(&self.config.model_name);
        if !request.stream {
            let value: Value = response.json().await.map_err(transport_error)?;
            let event = complete_response(&value, metadata, format)?;
            return Ok(Events {
                inner: Box::pin(stream::iter([Ok(event)]).fuse()),
            });
        }
        if !response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.split(';').next() == Some("text/event-stream"))
        {
            return Err(Error::new(
                "INVALID_RESPONSE",
                "streaming response must be text/event-stream",
            ));
        }
        let wire = response.bytes_stream().eventsource().map(|event| {
            event.map(|event| event.data).map_err(|error| match error {
                EventStreamError::Transport(error) => transport_error(error),
                _ => Error::new("INVALID_RESPONSE", "invalid SSE response framing"),
            })
        });
        let state = WireState {
            wire: Box::pin(wire),
            accumulator: Accumulator::new(metadata),
            format,
            done: false,
        };
        Ok(Events {
            inner: Box::pin(stream::unfold(state, next_event).fuse()),
        })
    }
}

fn encode_request(
    kind: Kind,
    config: &model::chat::Config,
    request: &Request,
) -> Result<Value, Error> {
    request.validate()?;
    if request.messages.is_empty() {
        return Err(Error::new("INVALID_ARGUMENTS", "chat messages are empty"));
    }
    if request
        .temperature
        .is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value))
        || request
            .top_p
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        || request.max_tokens == Some(0)
    {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "invalid chat sampling parameters",
        ));
    }
    let mut options = config.options.clone();
    options.extend(request.options.clone());
    if kind == Kind::DeepSeek {
        if matches!(
            request.response_format,
            Some(ResponseFormat::JsonSchema { .. })
        ) {
            return Err(Error::new(
                "UNSUPPORTED",
                "DeepSeek Chat does not support this JSON Schema response format",
            ));
        }
        // Current DeepSeek protocol enables thinking by default. An explicit
        // effort selects the mode; otherwise use its explicit thinking switch.
        let thinking = match options.get("reasoning_effort").and_then(Value::as_str) {
            Some("none") => false,
            Some("minimal" | "low" | "medium" | "high" | "max") => true,
            _ => {
                options
                    .get("thinking")
                    .and_then(|value| value.get("type"))
                    .and_then(Value::as_str)
                    != Some("disabled")
            }
        };
        if thinking
            && matches!(
                request.tool_choice,
                Some(ToolChoice::Required | ToolChoice::Named { .. })
            )
        {
            return Err(Error::new(
                "UNSUPPORTED",
                "DeepSeek thinking does not support required or named tool choice",
            ));
        }
    }
    let mut body = Map::new();
    let allowed: &[&str] = match kind {
        Kind::OpenAI => &[
            "reasoning_effort",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "user",
            "service_tier",
            "store",
            "logprobs",
            "top_logprobs",
        ],
        Kind::DeepSeek => &[
            "thinking",
            "reasoning_effort",
            "frequency_penalty",
            "presence_penalty",
            "logprobs",
            "top_logprobs",
        ],
    };
    for key in allowed {
        if let Some(value) = options.get(*key) {
            body.insert((*key).into(), value.clone());
        }
    }
    body.insert("model".into(), config.model_name.clone().into());
    body.insert(
        "messages".into(),
        Value::Array(
            request
                .messages
                .as_slice()
                .iter()
                .map(|message| encode_message(kind, message))
                .collect::<Result<_, _>>()?,
        ),
    );
    body.insert("stream".into(), request.stream.into());
    if request.stream {
        body.insert("stream_options".into(), json!({"include_usage": true}));
    }
    if let Some(value) = request.temperature {
        body.insert("temperature".into(), json!(value));
    }
    if let Some(value) = request.top_p {
        body.insert("top_p".into(), json!(value));
    }
    if let Some(value) = request.max_tokens {
        body.insert(
            match kind {
                Kind::OpenAI => "max_completion_tokens",
                Kind::DeepSeek => "max_tokens",
            }
            .into(),
            value.into(),
        );
    }
    if let Some(value) = &request.stop {
        body.insert("stop".into(), json!(value));
    }
    if !request.tools.is_empty() {
        body.insert("tools".into(), json!(request.tools.iter().map(|tool| json!({"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters}})).collect::<Vec<_>>()));
    }
    if let Some(choice) = &request.tool_choice {
        body.insert(
            "tool_choice".into(),
            match choice {
                ToolChoice::None => json!("none"),
                ToolChoice::Auto => json!("auto"),
                ToolChoice::Required => json!("required"),
                ToolChoice::Named { name } => {
                    json!({"type": "function", "function": {"name": name}})
                }
            },
        );
    }
    if let Some(format) = &request.response_format {
        body.insert("response_format".into(), match format {
            ResponseFormat::JsonObject => json!({"type": "json_object"}),
            ResponseFormat::JsonSchema { name, schema, strict } => {
                jsonschema::meta::validate(schema).map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid response JSON Schema"))?;
                jsonschema::validator_for(schema).map_err(|_| Error::new("INVALID_ARGUMENTS", "response JSON Schema cannot be compiled"))?;
                json!({"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": strict}})
            }
        });
    }
    Ok(Value::Object(body))
}

fn part_value(part: &Part) -> Result<&str, Error> {
    part.data
        .get("value")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "content part requires a text value"))
}

fn encode_message(kind: Kind, message: &Message) -> Result<Value, Error> {
    let call_id = message
        .metadata
        .get("call_id")
        .or_else(|| message.values.get("call_id"))
        .and_then(Value::as_str);
    let is_tool =
        message.role == Role::Tool || (message.role == Role::Function && call_id.is_some());
    if is_tool {
        let call_id = call_id
            .filter(|id| !id.is_empty())
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "tool message requires call_id"))?;
        let content = if message.content.is_empty() {
            let mut values = message.values.clone();
            values.remove("call_id");
            values.remove("name");
            serde_json::to_string(&values)
                .map_err(|_| Error::new("INVALID_ARGUMENTS", "tool result cannot be encoded"))?
        } else {
            message
                .content
                .iter()
                .map(|part| {
                    if part.r#type != "text" {
                        return Err(Error::new(
                            "UNSUPPORTED",
                            "tool multimodal input is not implemented",
                        ));
                    }
                    part_value(part)
                })
                .collect::<Result<Vec<_>, _>>()?
                .join("")
        };
        return Ok(json!({"role": "tool", "tool_call_id": call_id, "content": content}));
    }
    if message.role == Role::Function {
        return Err(Error::new(
            "UNSUPPORTED",
            "standalone Function values must be explicitly converted to model input",
        ));
    }
    if kind == Kind::DeepSeek && message.role == Role::Developer {
        return Err(Error::new(
            "UNSUPPORTED",
            "DeepSeek Chat does not support developer role",
        ));
    }
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();
    let mut call_ids = BTreeSet::new();
    let mut think = String::new();
    let mut refusal = String::new();
    for part in &message.content {
        match part.r#type.as_str() {
            "text" => content.push(json!({"type": "text", "text": part_value(part)?})),
            "think" if kind == Kind::DeepSeek && message.role == Role::Assistant => {
                think.push_str(part_value(part)?)
            }
            "refusal" if message.role == Role::Assistant => refusal.push_str(part_value(part)?),
            "tool_call" if message.role == Role::Assistant => {
                let id = part
                    .data
                    .get("call_id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "tool_call requires call_id"))?;
                let name = part
                    .data
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "tool_call requires name"))?;
                if !call_ids.insert(id) {
                    return Err(Error::new(
                        "INVALID_ARGUMENTS",
                        "duplicate tool call identity",
                    ));
                }
                let arguments = match part.data.get("arguments") {
                    Some(Value::String(raw)) => raw.clone(),
                    Some(Value::Object(object)) => json!(object).to_string(),
                    _ => {
                        return Err(Error::new(
                            "INVALID_ARGUMENTS",
                            "tool_call arguments must be object or original text",
                        ));
                    }
                };
                tool_calls.push(json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}));
            }
            "image" if kind == Kind::OpenAI && message.role == Role::User => {
                let source = part
                    .data
                    .get("source")
                    .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "image source is missing"))?;
                let url = source.get("value").and_then(Value::as_str).ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", "image source value is missing")
                })?;
                if url.trim().is_empty() {
                    return Err(Error::new(
                        "INVALID_ARGUMENTS",
                        "image source value is empty",
                    ));
                }
                if !matches!(
                    source.get("kind").and_then(Value::as_str),
                    Some("url" | "base64")
                ) {
                    return Err(Error::new(
                        "UNSUPPORTED",
                        "image source kind is unsupported",
                    ));
                }
                let mut image = json!({"url": url});
                if let Some(detail) = source.get("detail") {
                    if !matches!(detail.as_str(), Some("auto" | "low" | "high")) {
                        return Err(Error::new("INVALID_ARGUMENTS", "invalid image detail"));
                    }
                    image["detail"] = detail.clone();
                }
                content.push(json!({"type": "image_url", "image_url": image}));
            }
            _ => {
                return Err(Error::new(
                    "UNSUPPORTED",
                    "content part is unsupported by this Provider",
                ));
            }
        }
    }
    let role = serde_json::to_value(&message.role)
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid message role"))?;
    let mut output = json!({"role": role, "content": content});
    if !tool_calls.is_empty() {
        output["tool_calls"] = json!(tool_calls);
    }
    if !think.is_empty() {
        output["reasoning_content"] = think.into();
    }
    if !refusal.is_empty() {
        output["refusal"] = refusal.into();
    }
    if kind == Kind::DeepSeek {
        // DeepSeek accepts text content, not OpenAI multimodal content arrays.
        output["content"] = message
            .content
            .iter()
            .filter(|part| part.r#type == "text")
            .map(part_value)
            .collect::<Result<Vec<_>, _>>()?
            .join("")
            .into();
    }
    Ok(output)
}

fn count(value: Option<&Value>) -> Result<Option<u64>, Error> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "provider usage count is invalid")),
    }
}

fn parse_usage(value: Option<&Value>) -> Result<Option<Usage>, Error> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let object = value
        .as_object()
        .ok_or_else(|| Error::new("INVALID_RESPONSE", "provider usage must be an object"))?;
    let input = count(object.get("prompt_tokens"))?;
    let output = count(object.get("completion_tokens"))?;
    let total = count(object.get("total_tokens"))?.or_else(|| {
        input
            .zip(output)
            .and_then(|(input, output)| input.checked_add(output))
    });
    let cached = count(
        value
            .pointer("/prompt_tokens_details/cached_tokens")
            .or_else(|| object.get("prompt_cache_hit_tokens")),
    )?;
    let think = count(value.pointer("/completion_tokens_details/reasoning_tokens"))?;
    let details = object
        .iter()
        .filter(|(key, value)| {
            !matches!(
                key.as_str(),
                "prompt_tokens" | "completion_tokens" | "total_tokens"
            ) && value.is_number()
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    Ok(Some(Usage {
        input,
        output,
        total,
        input_details: cached.map(|cached| InputDetails {
            cached: Some(cached),
        }),
        output_details: think.map(|think| OutputDetails { think: Some(think) }),
        details,
    }))
}

#[derive(Default)]
struct CallParts {
    id: String,
    name: String,
    arguments: String,
    position: Option<usize>,
}

struct Accumulator {
    message: Message,
    calls: BTreeMap<u64, CallParts>,
    usage: Option<Usage>,
    finish: Option<String>,
}

impl Accumulator {
    fn new(metadata: Map<String, Value>) -> Self {
        let mut message = Message::new(Role::Assistant);
        message.metadata = metadata;
        Self {
            message,
            calls: BTreeMap::new(),
            usage: None,
            finish: None,
        }
    }

    fn push_text(&mut self, kind: &str, text: &str) {
        if let Some(last) = self
            .message
            .content
            .last_mut()
            .filter(|part| part.r#type == kind)
        {
            let previous = last.data.get("value").and_then(Value::as_str).unwrap_or("");
            last.data["value"] = format!("{previous}{text}").into();
        } else {
            self.message.content.push(Part {
                r#type: kind.into(),
                data: json!({"value": text}),
            });
        }
    }

    fn delta(&mut self, value: &Value) -> Result<Option<Message>, Error> {
        if value.get("error").is_some() {
            return Err(api_error(0, value));
        }
        if let Some(usage) = parse_usage(value.get("usage"))? {
            self.usage = Some(usage);
        }
        let choices = value
            .get("choices")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "chat choices are missing"))?;
        if choices.is_empty() {
            return Ok(None);
        }
        if choices.len() != 1 || choices[0].get("index").and_then(Value::as_u64).unwrap_or(0) != 0 {
            return Err(Error::new(
                "UNSUPPORTED",
                "multiple chat choices are unsupported",
            ));
        }
        let choice = &choices[0];
        if let Some(reason) = choice
            .get("finish_reason")
            .filter(|reason| !reason.is_null())
        {
            self.finish = Some(
                reason
                    .as_str()
                    .ok_or_else(|| Error::new("INVALID_RESPONSE", "invalid finish reason"))?
                    .into(),
            );
        }
        let delta = choice
            .get("delta")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "chat delta is missing"))?;
        if delta.get("role").is_some_and(|role| role != "assistant") {
            return Err(Error::new(
                "INVALID_RESPONSE",
                "chat response role is invalid",
            ));
        }
        if delta.contains_key("function_call") {
            return Err(Error::new(
                "UNSUPPORTED",
                "legacy function_call output is unsupported",
            ));
        }
        let mut observation = Message::new(Role::Assistant);
        observation.metadata = self.message.metadata.clone();
        for (field, kind) in [
            ("reasoning_content", "think"),
            ("content", "text"),
            ("refusal", "refusal"),
        ] {
            if let Some(value) = delta.get(field).filter(|value| !value.is_null()) {
                let text = value
                    .as_str()
                    .ok_or_else(|| Error::new("INVALID_RESPONSE", "chat text delta is invalid"))?;
                if !text.is_empty() {
                    self.push_text(kind, text);
                    observation.content.push(Part {
                        r#type: kind.into(),
                        data: json!({"value": text}),
                    });
                }
            }
        }
        if let Some(calls) = delta.get("tool_calls").filter(|value| !value.is_null()) {
            for call in calls
                .as_array()
                .ok_or_else(|| Error::new("INVALID_RESPONSE", "tool_calls delta is invalid"))?
            {
                let index = call
                    .get("index")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| Error::new("INVALID_RESPONSE", "tool call index is missing"))?;
                if call.get("type").is_some_and(|kind| kind != "function") {
                    return Err(Error::new(
                        "UNSUPPORTED",
                        "non-function Tool call is unsupported",
                    ));
                }
                let entry = self.calls.entry(index).or_default();
                if let Some(id) = call.get("id").filter(|value| !value.is_null()) {
                    let id = id.as_str().ok_or_else(|| {
                        Error::new("INVALID_RESPONSE", "tool call identity is invalid")
                    })?;
                    if !entry.id.is_empty() && entry.id != id {
                        return Err(Error::new("INVALID_RESPONSE", "tool call identity changed"));
                    }
                    entry.id = id.into();
                }
                if call.get("function").is_some_and(|value| !value.is_object()) {
                    return Err(Error::new(
                        "INVALID_RESPONSE",
                        "tool call function is invalid",
                    ));
                }
                if let Some(name) = call
                    .pointer("/function/name")
                    .filter(|value| !value.is_null())
                {
                    entry.name.push_str(name.as_str().ok_or_else(|| {
                        Error::new("INVALID_RESPONSE", "tool call name is invalid")
                    })?);
                }
                let arguments = match call
                    .pointer("/function/arguments")
                    .filter(|value| !value.is_null())
                {
                    Some(value) => value.as_str().ok_or_else(|| {
                        Error::new("INVALID_RESPONSE", "tool call arguments are invalid")
                    })?,
                    None => "",
                };
                entry.arguments.push_str(arguments);
                let part = Part {
                    r#type: "tool_call".into(),
                    data: json!({"call_id": entry.id, "name": entry.name, "arguments": entry.arguments}),
                };
                let position = *entry.position.get_or_insert_with(|| {
                    self.message.content.push(part.clone());
                    self.message.content.len() - 1
                });
                self.message.content[position] = part;
                if !entry.id.is_empty() && !entry.name.is_empty() {
                    observation.content.push(Part { r#type: "tool_call".into(), data: json!({"call_id": entry.id, "name": entry.name, "arguments": arguments}) });
                }
            }
        }
        Ok((!observation.content.is_empty()).then_some(observation))
    }

    fn complete(mut self, format: Option<ResponseFormat>) -> Result<Event, Error> {
        let finish = self.finish.ok_or_else(|| {
            Error::new(
                "INVALID_RESPONSE",
                "model stream ended without finish reason",
            )
        })?;
        let mut ids = BTreeSet::new();
        for entry in self.calls.values() {
            if entry.id.trim().is_empty() || entry.name.trim().is_empty() {
                return Err(Error::new("INVALID_RESPONSE", "incomplete tool call"));
            }
            if !ids.insert(&entry.id) {
                return Err(Error::new(
                    "INVALID_RESPONSE",
                    "duplicate tool call identity",
                ));
            }
            let arguments = serde_json::from_str::<Map<String, Value>>(&entry.arguments)
                .map(Value::Object)
                .unwrap_or_else(|_| entry.arguments.clone().into());
            self.message.content[entry.position.expect("tool call has a position")].data["arguments"] =
                arguments;
        }
        if let Some(format) = format {
            let text = self
                .message
                .content
                .iter()
                .filter(|part| part.r#type == "text")
                .filter_map(|part| part.data.get("value").and_then(Value::as_str))
                .collect::<String>();
            if let Ok(values) = serde_json::from_str::<Map<String, Value>>(&text) {
                let valid = match format {
                    ResponseFormat::JsonObject => true,
                    ResponseFormat::JsonSchema { schema, .. } => jsonschema::validator_for(&schema)
                        .is_ok_and(|validator| validator.is_valid(&Value::Object(values.clone()))),
                };
                if valid {
                    self.message.values = values;
                }
            }
        }
        self.message.usage = self.usage.clone();
        Ok(Event::Complete {
            message: self.message,
            usage: self.usage,
            finish_reason: Some(finish),
        })
    }
}

fn complete_response(
    value: &Value,
    metadata: Map<String, Value>,
    format: Option<ResponseFormat>,
) -> Result<Event, Error> {
    let choices = value
        .get("choices")
        .and_then(Value::as_array)
        .filter(|choices| choices.len() == 1)
        .ok_or_else(|| Error::new("INVALID_RESPONSE", "expected one complete chat choice"))?;
    let mut delta = choices[0]
        .get("message")
        .cloned()
        .ok_or_else(|| Error::new("INVALID_RESPONSE", "chat message is missing"))?;
    if let Some(calls) = delta.get_mut("tool_calls").and_then(Value::as_array_mut) {
        for (index, call) in calls.iter_mut().enumerate() {
            let call = call
                .as_object_mut()
                .ok_or_else(|| Error::new("INVALID_RESPONSE", "tool call must be an object"))?;
            call.insert("index".into(), index.into());
        }
    }
    let mut accumulator = Accumulator::new(metadata);
    accumulator.delta(&json!({"choices": [{"index": 0, "delta": delta, "finish_reason": choices[0].get("finish_reason")}], "usage": value.get("usage")}))?;
    accumulator.complete(format)
}

struct WireState {
    wire: Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>>,
    accumulator: Accumulator,
    format: Option<ResponseFormat>,
    done: bool,
}

async fn next_event(mut state: WireState) -> Option<(Result<Event, Error>, WireState)> {
    if state.done {
        return None;
    }
    loop {
        let result = match state.wire.next().await {
            Some(Ok(data)) if data == "[DONE]" => {
                state.done = true;
                let metadata = state.accumulator.message.metadata.clone();
                let accumulator =
                    std::mem::replace(&mut state.accumulator, Accumulator::new(metadata));
                return Some((accumulator.complete(state.format.take()), state));
            }
            Some(Ok(data)) => match serde_json::from_str::<Value>(&data) {
                Ok(value) => state.accumulator.delta(&value),
                Err(_) => Err(Error::new("INVALID_RESPONSE", "invalid JSON in SSE event")),
            },
            Some(Err(error)) => Err(error),
            None => Err(Error::new(
                "INVALID_RESPONSE",
                "SSE stream ended before DONE",
            )),
        };
        match result {
            Ok(Some(message)) => return Some((Ok(Event::Delta { message }), state)),
            Ok(None) => {}
            Err(error) => {
                state.done = true;
                return Some((Err(error), state));
            }
        }
    }
}

#[derive(Clone)]
pub struct Embed {
    connection: Connection,
    config: model::embed::Config,
}

impl model::embed::Embed for Embed {
    async fn embed(&self, request: model::embed::Request) -> Result<Message, Error> {
        request.validate()?;
        let mut options = self.config.options.clone();
        options.extend(request.options);
        let mut body = json!({"model": self.config.model_name, "input": request.input, "encoding_format": "float"});
        if let Some(dimensions) = request.dimensions {
            body["dimensions"] = dimensions.into();
        }
        if let Some(user) = options.get("user") {
            body["user"] = user.clone();
        }
        let value: Value = self
            .connection
            .post("embeddings", body)
            .await?
            .json()
            .await
            .map_err(transport_error)?;
        let data = value
            .get("data")
            .and_then(Value::as_array)
            .filter(|data| data.len() == 1)
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "expected one embedding"))?;
        if data[0].get("index").and_then(Value::as_u64) != Some(0) {
            return Err(Error::new("INVALID_RESPONSE", "embedding index is invalid"));
        }
        let vector = data[0]
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "embedding vector is missing"))?
            .iter()
            .map(|value| value.as_f64().map(|value| value as f32))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "embedding components are invalid"))?;
        let mut message = model::embed::message(
            vector,
            self.connection.kind.name(),
            &self.config.model_name,
            request.dimensions,
        )?;
        message
            .metadata
            .insert("provider_id".into(), self.connection.provider_id.into());
        message.usage = parse_usage(value.get("usage"))?;
        Ok(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Messages;

    #[test]
    fn direct_tool_messages_do_not_acquire_a_tool_result_envelope() {
        let mut message = Message::function(json!({"temperature": 22}));
        message.metadata.insert("call_id".into(), json!("call_1"));
        let wire = encode_message(Kind::OpenAI, &message).unwrap();
        assert_eq!(wire["role"], "tool");
        assert_eq!(wire["tool_call_id"], "call_1");
        assert_eq!(
            serde_json::from_str::<Value>(wire["content"].as_str().unwrap()).unwrap(),
            json!({"temperature": 22})
        );
        let mut error = Message::new(Role::Tool);
        error.metadata.insert("call_id".into(), json!("call_1"));
        error.content.push(Part {
            r#type: "text".into(),
            data: json!({"value": "city is missing"}),
        });
        assert_eq!(
            encode_message(Kind::DeepSeek, &error).unwrap()["content"],
            "city is missing"
        );
        assert!(!wire.to_string().contains("tool_result"));
        assert_eq!(
            encode_message(Kind::OpenAI, &Message::function(json!({"query":"x"})))
                .unwrap_err()
                .code,
            "UNSUPPORTED"
        );
    }

    #[test]
    fn image_uses_nested_source_detail_and_does_not_invent_id_resolution() {
        let mut message = Message::new(Role::User);
        message.content.push(Part { r#type: "image".into(), data: json!({"source":{"kind":"url","value":"https://example.com/image.png","detail":"low"}}) });
        let wire = encode_message(Kind::OpenAI, &message).unwrap();
        assert_eq!(wire["content"][0]["image_url"]["detail"], "low");
        assert_eq!(
            encode_message(Kind::DeepSeek, &message).unwrap_err().code,
            "UNSUPPORTED"
        );
        message.content[0].data["source"]["kind"] = json!("id");
        assert_eq!(
            encode_message(Kind::OpenAI, &message).unwrap_err().code,
            "UNSUPPORTED"
        );
    }

    #[test]
    fn deepseek_required_tool_choice_uses_explicit_thinking_options() {
        let config = model::chat::Config {
            model_name: "test".into(),
            options: Map::new(),
        };
        let mut request = Request {
            messages: Messages::new([Message::new(Role::User)]),
            tools: vec![model::chat::ToolDefinition {
                name: "weather".into(),
                description: "weather".into(),
                parameters: serde_json::from_value(json!({"type":"object"})).unwrap(),
            }],
            tool_choice: Some(ToolChoice::Required),
            ..Default::default()
        };
        assert_eq!(
            encode_request(Kind::DeepSeek, &config, &request)
                .unwrap_err()
                .code,
            "UNSUPPORTED"
        );
        request
            .options
            .insert("thinking".into(), json!({"type":"disabled"}));
        assert_eq!(
            encode_request(Kind::DeepSeek, &config, &request).unwrap()["tool_choice"],
            "required"
        );
        request
            .options
            .insert("reasoning_effort".into(), json!("high"));
        assert_eq!(
            encode_request(Kind::DeepSeek, &config, &request)
                .unwrap_err()
                .code,
            "UNSUPPORTED"
        );
        request
            .options
            .insert("reasoning_effort".into(), json!("none"));
        assert!(encode_request(Kind::DeepSeek, &config, &request).is_ok());
    }

    #[test]
    fn malformed_stream_tool_fields_and_nonassistant_role_are_rejected() {
        for delta in [
            json!({"role":"user"}),
            json!({"tool_calls":[{"index":0,"id":7}]}),
            json!({"tool_calls":[{"index":0,"function":[]}]}),
            json!({"tool_calls":[{"index":0,"function":{"arguments":7}}]}),
        ] {
            assert_eq!(
                Accumulator::new(Map::new())
                    .delta(&json!({"choices":[{"index":0,"delta":delta}]}))
                    .unwrap_err()
                    .code,
                "INVALID_RESPONSE"
            );
        }
    }

    #[test]
    fn json_object_fallback_and_usage_missing_values_remain_explicit() {
        for (content, expected) in [
            ("{\"x\":1}", json!({"x":1})),
            ("[1]", json!({})),
            ("invalid", json!({})),
        ] {
            let value = json!({"choices":[{"message":{"content":content},"finish_reason":"stop"}]});
            let Event::Complete { message, usage, .. } =
                complete_response(&value, Map::new(), Some(ResponseFormat::JsonObject)).unwrap()
            else {
                panic!()
            };
            assert_eq!(json!(message.values), expected);
            assert_eq!(message.content[0].data["value"], content);
            assert!(usage.is_none());
        }
        let usage = parse_usage(Some(&json!({"prompt_tokens":3,"completion_tokens":2})))
            .unwrap()
            .unwrap();
        assert_eq!(usage.total, Some(5));
        assert_eq!(
            parse_usage(Some(&json!({"prompt_tokens":-1})))
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
    }

    #[test]
    fn named_tool_choice_and_deepseek_reasoning_history_are_translated() {
        let mut history = Message::new(Role::Assistant);
        history.content = vec![
            Part {
                r#type: "think".into(),
                data: json!({"value":"先查数据"}),
            },
            Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"c","name":"weather","arguments":{"city":"北京"}}),
            },
        ];
        let wire = encode_message(Kind::DeepSeek, &history).unwrap();
        assert_eq!(wire["reasoning_content"], "先查数据");
        assert_eq!(
            wire["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"北京\"}"
        );
        let config = model::chat::Config {
            model_name: "test".into(),
            options: Map::new(),
        };
        let request = Request {
            messages: Messages::new([Message::new(Role::User)]),
            tools: vec![model::chat::ToolDefinition {
                name: "weather".into(),
                description: "weather".into(),
                parameters: serde_json::from_value(json!({"type":"object"})).unwrap(),
            }],
            tool_choice: Some(ToolChoice::Named {
                name: "weather".into(),
            }),
            ..Default::default()
        };
        assert_eq!(
            encode_request(Kind::OpenAI, &config, &request).unwrap()["tool_choice"],
            json!({"type":"function","function":{"name":"weather"}})
        );
    }
}
