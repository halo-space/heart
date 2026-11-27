//! Gemini provider implementations for the V2 TTS and multimodal contracts.
//!
//! Gemini's current Interactions API is kept behind this module.  The rest of
//! the crate only sees the model contracts and the shared `event::Event` stream.

use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use eventsource_stream::{EventStreamError, Eventsource};
use futures_core::Stream;
use futures_util::{StreamExt, stream};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::event::Event;
use crate::message::{Message, Messages, Role};
use crate::model::{self, content::part::Part, token::Usage};

type Error = model::Error;

#[derive(Clone, Deserialize)]
pub struct Config {
    pub provider_id: i64,
    pub provider_name: String,
    pub base_url: String,
    pub api_key: String,
    /// Total HTTP request timeout, in seconds.
    pub timeout: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider_id: 0,
            provider_name: "gemini".into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".into(),
            api_key: String::new(),
            timeout: 120,
        }
    }
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("provider_id", &self.provider_id)
            .field("provider_name", &self.provider_name)
            .field("api_key", &"[redacted]")
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct Gemini {
    connection: Connection,
}

impl Gemini {
    pub fn builder() -> Builder {
        Builder {
            config: Config::default(),
        }
    }

    pub fn new(config: Config) -> Result<Self, Error> {
        Ok(Self {
            connection: Connection::new(config)?,
        })
    }

    pub fn tts(&self, config: model::tts::Config) -> Result<tts::Tts, Error> {
        validate_model(&config.model_name)?;
        Ok(tts::Tts {
            connection: self.connection.clone(),
            config,
        })
    }

    pub fn chat(&self, config: model::chat::Config) -> Result<chat::Chat, Error> {
        validate_model(&config.model_name)?;
        Ok(chat::Chat {
            connection: self.connection.clone(),
            config,
        })
    }

    pub fn multimodal(
        &self,
        config: model::multimodal::Config,
    ) -> Result<multimodal::Multimodal, Error> {
        validate_model(&config.model_name)?;
        Ok(multimodal::Multimodal {
            connection: self.connection.clone(),
            config,
        })
    }
}

pub struct Builder {
    config: Config,
}

impl Builder {
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.config.base_url = base_url.into();
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.config.api_key = api_key.into();
        self
    }

    pub fn timeout(mut self, timeout: u64) -> Self {
        self.config.timeout = timeout;
        self
    }

    pub fn build(self) -> Result<Gemini, Error> {
        Gemini::new(self.config)
    }
}

#[derive(Clone)]
struct Connection {
    client: reqwest::Client,
    base_url: String,
    provider_id: i64,
}

impl Connection {
    fn new(config: Config) -> Result<Self, Error> {
        if config.provider_name != "gemini" || config.timeout == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "provider name or timeout is invalid",
            ));
        }
        let url = reqwest::Url::parse(&config.base_url)
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
        if config.api_key.trim().is_empty() {
            return Err(Error::new("AUTHENTICATION", "provider API key is required"));
        }

        let mut key = HeaderValue::from_str(&config.api_key)
            .map_err(|_| Error::new("AUTHENTICATION", "provider API key is invalid"))?;
        key.set_sensitive(true);
        let mut headers = HeaderMap::new();
        headers.insert(HeaderName::from_static("x-goog-api-key"), key);
        headers.insert(
            HeaderName::from_static("api-revision"),
            HeaderValue::from_static("2026-05-20"),
        );
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(config.timeout))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| Error::new("TRANSPORT", "provider HTTP client initialization failed"))?;

        Ok(Self {
            client,
            base_url: url.as_str().trim_end_matches('/').into(),
            provider_id: config.provider_id,
        })
    }

    async fn post(&self, body: Value, stream: bool) -> Result<reqwest::Response, Error> {
        let endpoint = if stream {
            "/interactions?alt=sse"
        } else {
            "/interactions"
        };
        let request = self
            .client
            .post(format!("{}{endpoint}", self.base_url))
            .json(&body);
        let response = request.send().await.map_err(transport_error)?;
        if response.status().is_success() {
            Ok(response)
        } else {
            let status = response.status().as_u16();
            let value = response.json::<Value>().await.unwrap_or(Value::Null);
            Err(api_error(status, &value))
        }
    }

    fn metadata(&self, model_name: &str) -> Map<String, Value> {
        Map::from_iter([
            ("provider_id".into(), self.provider_id.into()),
            ("provider_name".into(), "gemini".into()),
            ("model_name".into(), model_name.into()),
        ])
    }
}

fn validate_model(model_name: &str) -> Result<(), Error> {
    if model_name.trim().is_empty() {
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
    let provider_status = body
        .pointer("/error/status")
        .and_then(Value::as_str)
        .unwrap_or("");
    let code = match provider_status {
        "UNAUTHENTICATED" => "AUTHENTICATION",
        "PERMISSION_DENIED" => "PERMISSION_DENIED",
        "RESOURCE_EXHAUSTED" => "RATE_LIMITED",
        "INVALID_ARGUMENT" => "INVALID_ARGUMENTS",
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
    let details = if provider_status.is_empty() {
        format!("http_status={status}")
    } else {
        format!("http_status={status}; provider_status={provider_status}")
    };
    Error::new(code, "provider request failed").with_details(details)
}

#[derive(Clone, Copy)]
enum OutputKind {
    Tts,
    Chat,
    Multimodal,
}

pub struct Events {
    inner: Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>,
}

impl Stream for Events {
    type Item = Result<Event, Error>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(context)
    }
}

fn complete_events(
    value: Value,
    mut metadata: Map<String, Value>,
    kind: OutputKind,
) -> Result<Events, Error> {
    let (message, usage) = output_message(&value, kind)?;
    if let Some(turn_id) = value.get("id").and_then(Value::as_str) {
        metadata.insert("turn_id".into(), Value::String(turn_id.into()));
    }
    let finish_reason = value
        .get("status")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let mut complete = message;
    complete.metadata.extend(metadata);
    Ok(Events {
        inner: Box::pin(stream::iter([Ok(Event::Complete {
            message: complete,
            usage,
            finish_reason,
        })])),
    })
}

struct StreamState {
    wire: Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>>,
    message: Message,
    metadata: Map<String, Value>,
    kind: OutputKind,
    completed: bool,
}

fn streamed_events(
    response: reqwest::Response,
    metadata: Map<String, Value>,
    kind: OutputKind,
) -> Events {
    let wire = response.bytes_stream().eventsource().map(|event| {
        event.map(|event| event.data).map_err(|error| match error {
            EventStreamError::Transport(error) => transport_error(error),
            _ => Error::new("INVALID_RESPONSE", "invalid Gemini SSE framing"),
        })
    });
    let state = StreamState {
        wire: Box::pin(wire),
        message: Message::new(Role::Assistant),
        metadata,
        kind,
        completed: false,
    };
    Events {
        inner: Box::pin(stream::unfold(state, next_stream_event).fuse()),
    }
}

async fn next_stream_event(mut state: StreamState) -> Option<(Result<Event, Error>, StreamState)> {
    while let Some(item) = state.wire.next().await {
        let data = match item {
            Ok(data) if data.trim().is_empty() => continue,
            Ok(data) => data,
            Err(error) => return Some((Err(error), state)),
        };
        let value: Value = match serde_json::from_str(&data) {
            Ok(value) => value,
            Err(_) => {
                return Some((
                    Err(Error::new("INVALID_RESPONSE", "invalid Gemini SSE JSON")),
                    state,
                ));
            }
        };
        let event_type = value
            .get("event_type")
            .and_then(Value::as_str)
            .unwrap_or("");
        if event_type == "step.delta" {
            match delta_part(value.get("delta"), state.kind) {
                Ok(Some(part)) => {
                    state.message.content.push(part.clone());
                    let mut delta = Message::new(Role::Assistant);
                    delta.content.push(part);
                    return Some((Ok(Event::Delta { message: delta }), state));
                }
                Ok(None) => {}
                Err(error) => return Some((Err(error), state)),
            }
            continue;
        }
        if event_type == "interaction.completed" {
            state.completed = true;
            if let Some(turn_id) = value
                .get("interaction")
                .and_then(|interaction| interaction.get("id"))
                .and_then(Value::as_str)
                .or_else(|| value.get("id").and_then(Value::as_str))
            {
                state
                    .metadata
                    .insert("turn_id".into(), Value::String(turn_id.into()));
            }
            let usage = value.get("interaction").and_then(parse_usage);
            state.message.metadata.extend(state.metadata.clone());
            return Some((
                Ok(Event::Complete {
                    message: state.message.clone(),
                    usage,
                    finish_reason: Some("completed".into()),
                }),
                state,
            ));
        }
    }
    if !state.completed {
        state.message.metadata.extend(state.metadata.clone());
        state.completed = true;
        return Some((
            Ok(Event::Complete {
                message: state.message.clone(),
                usage: None,
                finish_reason: Some("stop".into()),
            }),
            state,
        ));
    }
    None
}

fn output_message(value: &Value, kind: OutputKind) -> Result<(Message, Option<Usage>), Error> {
    let mut message = Message::new(Role::Assistant);
    let steps = value
        .get("steps")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("INVALID_RESPONSE", "Gemini response has no steps"))?;
    for step in steps {
        if let Some(content) = step.get("content").and_then(Value::as_array) {
            for item in content {
                if let Some(part) = output_part(item, kind)? {
                    message.content.push(part);
                }
            }
        }
    }
    if message.content.is_empty() {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "Gemini response has no output",
        ));
    }
    Ok((message, value.get("usage").and_then(parse_usage)))
}

fn output_part(value: &Value, output_kind: OutputKind) -> Result<Option<Part>, Error> {
    let Some(part_kind) = value.get("type").and_then(Value::as_str) else {
        return Ok(None);
    };
    match part_kind {
        "text" if matches!(output_kind, OutputKind::Chat | OutputKind::Multimodal) => {
            Ok(Some(Part {
                r#type: "text".into(),
                data: object(json!({
                    "value": value.get("text").ok_or_else(|| {
                        Error::new("INVALID_RESPONSE", "Gemini text output has no text")
                    })?
                })),
                mime_type: None,
            }))
        }
        "audio" => {
            let data = value
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::new("INVALID_RESPONSE", "Gemini audio output has no data"))?;
            let mime = value
                .get("mime_type")
                .and_then(Value::as_str)
                .unwrap_or("audio/wav");
            Ok(Some(media_part(part_kind, data, mime)))
        }
        "image" | "video" | "file" => Err(Error::new(
            "UNSUPPORTED",
            "Gemini output media type is outside the current V3 scope",
        )),
        _ => Ok(None),
    }
}

fn delta_part(value: Option<&Value>, kind: OutputKind) -> Result<Option<Part>, Error> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(part_kind) = value.get("type").and_then(Value::as_str) else {
        return Ok(None);
    };
    match part_kind {
        "text" if matches!(kind, OutputKind::Chat | OutputKind::Multimodal) => Ok(Some(Part {
            r#type: "text".into(),
            data: object(json!({
                "value": value.get("text").ok_or_else(|| {
                    Error::new("INVALID_RESPONSE", "Gemini text output has no text")
                })?
            })),
            mime_type: None,
        })),
        "audio" => {
            let data = value
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::new("INVALID_RESPONSE", "Gemini audio output has no data"))?;
            let mime = value
                .get("mime_type")
                .and_then(Value::as_str)
                .unwrap_or("audio/wav");
            Ok(Some(media_part(part_kind, data, mime)))
        }
        "image" | "video" | "file" => Err(Error::new(
            "UNSUPPORTED",
            "Gemini output media type is outside the current V3 scope",
        )),
        _ => Ok(None),
    }
}

fn media_part(kind: &str, data: &str, mime: &str) -> Part {
    Part {
        r#type: kind.into(),
        data: object(json!({
            "source": {
                "kind": "base64",
                "value": data,
            },
        })),
        mime_type: Some(mime.to_owned()),
    }
}

fn parse_usage(value: &Value) -> Option<Usage> {
    let usage = value.get("usage")?;
    Some(Usage {
        input: usage
            .get("total_input_tokens")
            .or_else(|| usage.get("prompt_token_count"))
            .and_then(Value::as_u64),
        output: usage
            .get("total_output_tokens")
            .or_else(|| usage.get("candidates_token_count"))
            .and_then(Value::as_u64),
        total: usage.get("total_tokens").and_then(Value::as_u64),
        input_details: None,
        output_details: None,
        details: Map::new(),
    })
}

fn object(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

fn merge_options(config: &Map<String, Value>, request: &Map<String, Value>) -> Map<String, Value> {
    let mut merged = config.clone();
    merged.extend(request.clone());
    merged
}

fn text_part(part: &Part) -> Result<String, Error> {
    part.data
        .get("value")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "text part value is required"))
}

fn messages_to_input(messages: &Messages) -> Result<Value, Error> {
    if messages.is_empty() {
        return Err(Error::new("INVALID_ARGUMENTS", "model messages are empty"));
    }
    let mut input = Vec::new();
    for message in messages.as_slice() {
        let call_id = message
            .metadata
            .get("call_id")
            .or_else(|| message.values.get("call_id"))
            .and_then(Value::as_str);
        let name = message
            .metadata
            .get("name")
            .or_else(|| message.values.get("name"))
            .and_then(Value::as_str);
        if matches!(message.role, Role::Tool)
            || (message.role == Role::Function && call_id.is_some())
        {
            let call_id = call_id
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", "function result call_id is missing")
                })?;
            let name = name
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", "function result name is missing")
                })?;
            let result = message
                .content
                .iter()
                .map(part_to_input)
                .collect::<Result<Vec<_>, _>>()?;
            input.push(json!({
                "type": "function_result",
                "name": name,
                "call_id": call_id,
                "result": result
            }));
            continue;
        }
        for part in &message.content {
            if part.r#type == "tool_call" && message.role == Role::Assistant {
                let call_id = part
                    .data
                    .get("call_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| {
                        Error::new("INVALID_ARGUMENTS", "function call_id is missing")
                    })?;
                let name = part
                    .data
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "function name is missing"))?;
                let arguments = part
                    .data
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                input.push(json!({
                    "type": "function_call",
                    "name": name,
                    "call_id": call_id,
                    "arguments": arguments
                }));
            } else {
                input.push(part_to_input(part)?);
            }
        }
    }
    if input.is_empty() {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "model messages have no content",
        ));
    }
    Ok(Value::Array(input))
}

fn last_turn_id(messages: &Messages) -> Option<&str> {
    messages.as_slice().iter().rev().find_map(|message| {
        message
            .metadata
            .get("turn_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
    })
}

fn part_to_input(part: &Part) -> Result<Value, Error> {
    match part.r#type.as_str() {
        "text" => Ok(json!({"type":"text", "text": text_part(part)?})),
        "image" | "audio" | "video" | "file" => {
            let source = part
                .data
                .get("source")
                .and_then(Value::as_object)
                .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "media source is required"))?;
            let kind = source.get("kind").and_then(Value::as_str).unwrap_or("");
            let value = source.get("value").and_then(Value::as_str).unwrap_or("");
            let mime = part.mime_type.as_deref().ok_or_else(|| {
                Error::new(
                    "INVALID_ARGUMENTS",
                    format!("{} part requires mime_type", part.r#type),
                )
            })?;
            if value.trim().is_empty() {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "media source value is empty",
                ));
            }
            let wire_type = if part.r#type == "file" {
                "document"
            } else {
                part.r#type.as_str()
            };
            match kind {
                "url" => Ok(json!({"type":wire_type,"uri":value,"mime_type":mime})),
                "base64" => Ok(json!({"type":wire_type,"data":value,"mime_type":mime})),
                "id" => Err(Error::new(
                    "UNSUPPORTED",
                    "Gemini media id sources require application resolution",
                )),
                _ => Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "media source kind is invalid",
                )),
            }
        }
        _ => Err(Error::new(
            "UNSUPPORTED",
            "Gemini does not support this content part",
        )),
    }
}

fn chat_body(config: &model::chat::Config, request: &model::chat::Request) -> Result<Value, Error> {
    request.validate()?;
    if matches!(
        request.tool_choice,
        Some(model::chat::ToolChoice::Required | model::chat::ToolChoice::Named { .. })
    ) {
        return Err(Error::new(
            "UNSUPPORTED",
            "Gemini Chat does not support required or named tool choice in this adapter",
        ));
    }
    let multimodal = model::multimodal::Request {
        messages: request.messages.clone(),
        tools: request.tools.clone(),
        tool_choice: request.tool_choice.clone().map(|choice| match choice {
            model::chat::ToolChoice::None => model::multimodal::ToolChoice::None,
            model::chat::ToolChoice::Auto => model::multimodal::ToolChoice::Auto,
            model::chat::ToolChoice::Required => model::multimodal::ToolChoice::Required,
            model::chat::ToolChoice::Named { name } => {
                model::multimodal::ToolChoice::Named { name }
            }
        }),
        stream: request.stream,
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        top_p: request.top_p,
        stop: request.stop.clone(),
        modalities: Vec::new(),
        response_format: request.response_format.clone().map(|format| match format {
            model::chat::ResponseFormat::JsonObject => {
                model::multimodal::ResponseFormat::JsonObject
            }
            model::chat::ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            } => model::multimodal::ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            },
        }),
        options: request.options.clone(),
    };
    let mut body = multimodal_body(
        &model::multimodal::Config {
            model_name: config.model_name.clone(),
            context_limit: config.context_limit,
            options: config.options.clone(),
        },
        &multimodal,
    )?;
    if request.temperature.is_some()
        || request.max_tokens.is_some()
        || request.top_p.is_some()
        || request.stop.is_some()
    {
        let generation = body
            .as_object_mut()
            .expect("Gemini request body is an object")
            .entry("generation_config")
            .or_insert_with(|| json!({}));
        let generation = generation.as_object_mut().ok_or_else(|| {
            Error::new("INVALID_ARGUMENTS", "generation_config must be an object")
        })?;
        if let Some(value) = request.temperature {
            generation.insert("temperature".into(), json!(value));
        }
        if let Some(value) = request.max_tokens {
            generation.insert("max_output_tokens".into(), json!(value));
        }
        if let Some(value) = request.top_p {
            generation.insert("top_p".into(), json!(value));
        }
        if let Some(value) = &request.stop {
            generation.insert("stop_sequences".into(), json!(value));
        }
    }
    Ok(body)
}

pub mod chat {
    use super::*;

    pub struct Chat {
        pub(super) connection: Connection,
        pub(super) config: model::chat::Config,
    }

    impl model::chat::Chat for Chat {
        type Stream = Events;

        fn context_limit(&self) -> Option<u64> {
            self.config.context_limit.filter(|limit| *limit > 0)
        }

        async fn stream(&self, request: model::chat::Request) -> Result<Self::Stream, Error> {
            let stream = request.stream;
            let response = self
                .connection
                .post(chat_body(&self.config, &request)?, stream)
                .await?;
            let metadata = self.connection.metadata(&self.config.model_name);
            if stream {
                Ok(streamed_events(response, metadata, OutputKind::Chat))
            } else {
                complete_events(
                    response.json().await.map_err(transport_error)?,
                    metadata,
                    OutputKind::Chat,
                )
            }
        }
    }
}

fn tts_body(config: &model::tts::Config, request: &model::tts::Request) -> Result<Value, Error> {
    request.validate()?;
    let mut text = String::new();
    for message in request.messages.as_slice() {
        for part in &message.content {
            if part.r#type == "text" {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&text_part(part)?);
            }
        }
    }
    if text.trim().is_empty() {
        return Err(Error::new("INVALID_ARGUMENTS", "tts text is empty"));
    }
    let options = merge_options(&config.options, &request.options);
    let mut body = Value::Object(options);
    body["model"] = Value::String(config.model_name.clone());
    body["input"] = json!([{"type":"user_input","content":[{"type":"text","text":text}]}]);
    body["response_format"] = json!({
        "type": "audio",
        "mime_type": request.format.as_deref().unwrap_or("audio/wav")
    });
    body["generation_config"] = json!({
        "speech_config": [{"voice": request.voice.as_deref().unwrap_or("Kore")}]
    });
    // The Interactions API defaults to unary responses. Only send `stream`
    // when the caller explicitly requests SSE, matching the documented REST
    // shape and avoiding an unnecessary false-valued field for TTS.
    if request.stream {
        body["stream"] = Value::Bool(true);
    }
    Ok(body)
}

pub mod tts {
    use super::*;

    pub struct Tts {
        pub(super) connection: Connection,
        pub(super) config: model::tts::Config,
    }

    impl model::tts::Tts for Tts {
        type Stream = Events;

        async fn stream(&self, request: model::tts::Request) -> Result<Self::Stream, Error> {
            let stream = request.stream;
            let response = self
                .connection
                .post(tts_body(&self.config, &request)?, stream)
                .await?;
            let metadata = self.connection.metadata(&self.config.model_name);
            if stream {
                Ok(streamed_events(response, metadata, OutputKind::Tts))
            } else {
                complete_events(
                    response.json().await.map_err(transport_error)?,
                    metadata,
                    OutputKind::Tts,
                )
            }
        }
    }
}

fn multimodal_body(
    config: &model::multimodal::Config,
    request: &model::multimodal::Request,
) -> Result<Value, Error> {
    request.validate()?;
    validate_modalities(&request.modalities)?;
    let options = merge_options(&config.options, &request.options);
    let mut body = Value::Object(options);
    body["model"] = Value::String(config.model_name.clone());
    body["input"] = messages_to_input(&request.messages)?;
    body["stream"] = Value::Bool(request.stream);
    if let Some(turn_id) = last_turn_id(&request.messages) {
        body["previous_interaction_id"] = Value::String(turn_id.into());
    }
    if !request.modalities.is_empty() {
        body["modalities"] = Value::Array(
            request
                .modalities
                .iter()
                .cloned()
                .map(Value::String)
                .collect(),
        );
    }
    if let Some(format) = &request.response_format {
        body["response_format"] = match format {
            model::multimodal::ResponseFormat::JsonObject => {
                json!({"type":"text","mime_type":"application/json"})
            }
            model::multimodal::ResponseFormat::JsonSchema { schema, .. } => json!({
                "type":"text",
                "mime_type":"application/json",
                "schema":schema
            }),
        };
    }
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(
            request
                .tools
                .iter()
                .map(|tool| {
                    json!({
                        "type":"function",
                        "name":tool.name,
                        "description":tool.description,
                        "parameters":tool.parameters
                    })
                })
                .collect(),
        );
    }
    Ok(body)
}

fn validate_modalities(modalities: &[String]) -> Result<(), Error> {
    if let Some(value) = modalities
        .iter()
        .find(|value| !matches!(value.as_str(), "text" | "audio"))
    {
        return Err(Error::new(
            "UNSUPPORTED",
            format!("Gemini output modality is not supported in the current V3 scope: {value}"),
        ));
    }
    Ok(())
}

pub mod multimodal {
    use super::*;

    pub struct Multimodal {
        pub(super) connection: Connection,
        pub(super) config: model::multimodal::Config,
    }

    impl model::multimodal::Multimodal for Multimodal {
        type Stream = Events;

        async fn stream(&self, request: model::multimodal::Request) -> Result<Self::Stream, Error> {
            let stream = request.stream;
            let response = self
                .connection
                .post(multimodal_body(&self.config, &request)?, stream)
                .await?;
            let metadata = self.connection.metadata(&self.config.model_name);
            if stream {
                Ok(streamed_events(response, metadata, OutputKind::Multimodal))
            } else {
                complete_events(
                    response.json().await.map_err(transport_error)?,
                    metadata,
                    OutputKind::Multimodal,
                )
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_core_provider_gemini_tests.rs"]
mod tests;
