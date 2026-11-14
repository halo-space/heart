//! Gemini provider implementations for the V2 TTS and multimodal contracts.
//!
//! Gemini's current Interactions API is kept behind this module.  The rest of
//! the crate only sees `model::tts::Tts`, `model::multimodal::Multimodal` and
//! the shared `event::Event` stream.

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
    let code = body
        .pointer("/error/status")
        .and_then(Value::as_str)
        .map(|value| match value {
            "UNAUTHENTICATED" => "AUTHENTICATION",
            "PERMISSION_DENIED" => "PERMISSION_DENIED",
            "RESOURCE_EXHAUSTED" => "RATE_LIMITED",
            "INVALID_ARGUMENT" => "INVALID_ARGUMENTS",
            _ => "PROVIDER",
        })
        .unwrap_or(match status {
            400 | 422 => "INVALID_ARGUMENTS",
            401 => "AUTHENTICATION",
            403 => "PERMISSION_DENIED",
            408 => "TIMEOUT",
            429 => "RATE_LIMITED",
            500..=599 => "UNAVAILABLE",
            _ => "PROVIDER",
        });
    Error::new(code, "provider request failed").with_details(format!("http_status={status}"))
}

#[derive(Clone, Copy)]
enum OutputKind {
    Tts,
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
    metadata: Map<String, Value>,
    kind: OutputKind,
) -> Result<Events, Error> {
    let (message, usage) = output_message(&value, kind)?;
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
            if let Some(part) = delta_part(value.get("delta"), state.kind) {
                state.message.content.push(part.clone());
                let mut delta = Message::new(Role::Assistant);
                delta.content.push(part);
                return Some((Ok(Event::Delta { message: delta }), state));
            }
            continue;
        }
        if event_type == "interaction.completed" {
            state.completed = true;
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
                if let Some(part) = output_part(item, kind) {
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

fn output_part(value: &Value, _kind: OutputKind) -> Option<Part> {
    let kind = value.get("type")?.as_str()?;
    match kind {
        "text" => Some(Part {
            r#type: "text".into(),
            data: object(json!({"value": value.get("text")?})),
        }),
        "audio" | "image" | "video" | "file" => {
            let data = value.get("data")?.as_str()?;
            let mime = value
                .get("mime_type")
                .and_then(Value::as_str)
                .unwrap_or(match kind {
                    "audio" => "audio/wav",
                    "image" => "image/png",
                    "video" => "video/mp4",
                    _ => "application/octet-stream",
                });
            Some(media_part(kind, data, mime))
        }
        _ => None,
    }
}

fn delta_part(value: Option<&Value>, _kind: OutputKind) -> Option<Part> {
    let value = value?;
    let kind = value.get("type")?.as_str()?;
    match kind {
        "text" => Some(Part {
            r#type: "text".into(),
            data: object(json!({"value": value.get("text")?})),
        }),
        "audio" | "image" | "video" | "file" => {
            let data = value.get("data")?.as_str()?;
            let mime = value
                .get("mime_type")
                .and_then(Value::as_str)
                .unwrap_or("application/octet-stream");
            Some(media_part(kind, data, mime))
        }
        _ => None,
    }
}

fn media_part(kind: &str, data: &str, mime: &str) -> Part {
    Part {
        r#type: kind.into(),
        data: object(json!({
            "source": {
                "kind": "base64",
                "value": format!("data:{mime};base64,{data}"),
                "mime_type": mime,
            },
            "mime_type": mime,
        })),
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
        input.extend(
            message
                .content
                .iter()
                .map(part_to_input)
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    if input.is_empty() {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "model messages have no content",
        ));
    }
    Ok(Value::Array(input))
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
            let mime = source
                .get("mime_type")
                .or_else(|| part.data.get("mime_type"))
                .and_then(Value::as_str)
                .unwrap_or("application/octet-stream");
            if value.trim().is_empty() {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "media source value is empty",
                ));
            }
            match kind {
                "url" => Ok(json!({"type":part.r#type,"uri":value,"mime_type":mime})),
                "base64" => {
                    let raw = value
                        .split_once(",")
                        .map(|(_, payload)| payload)
                        .unwrap_or(value);
                    Ok(json!({"type":part.r#type,"data":raw,"mime_type":mime}))
                }
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
    body["stream"] = Value::Bool(request.stream);
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
    let options = merge_options(&config.options, &request.options);
    let mut body = Value::Object(options);
    body["model"] = Value::String(config.model_name.clone());
    body["input"] = messages_to_input(&request.messages)?;
    body["stream"] = Value::Bool(request.stream);
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
