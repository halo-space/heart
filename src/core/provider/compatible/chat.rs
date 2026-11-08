//! OpenAI-compatible Chat wire encoding and Event stream decoding.

use std::collections::{BTreeMap, BTreeSet};
use std::pin::Pin;
use std::task::{Context, Poll};

use eventsource_stream::{EventStreamError, Eventsource};
use futures_core::Stream;
use futures_util::{StreamExt, stream};
use serde_json::{Map, Value, json};

use super::common::{self, Connection, Kind};
use crate::event::Event;
use crate::message::{Message, Role};
use crate::model::token::Usage;
use crate::model::{
    self,
    chat::{Request, ResponseFormat, ToolChoice},
    content::part::{Part, object as part_data},
};

type Error = model::Error;

#[derive(Clone)]
pub struct Chat {
    pub(super) connection: Connection,
    pub(super) config: model::chat::Config,
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

    fn context_limit(&self) -> Option<u64> {
        // The limit is a common model capability, not a provider wire option.
        // Runtime sees it only through Chat::context_limit().
        self.config.context_limit.filter(|limit| *limit > 0)
    }

    async fn stream(&self, request: Request) -> Result<Events, Error> {
        let body = encode_request(self.connection.kind, &self.config, &request)?;
        let format = request.response_format.clone();
        let response = self.connection.post("chat/completions", body).await?;
        let metadata = self.connection.metadata(&self.config.model_name);
        if !request.stream {
            let value: Value = response.json().await.map_err(common::transport_error)?;
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
                EventStreamError::Transport(error) => common::transport_error(error),
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

pub(super) fn encode_request(
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
    let reserved = [
        "model",
        "messages",
        "stream",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
        "stream_options",
        "input",
        "dimensions",
        "encoding_format",
    ];
    common::validate_options(&options, &reserved)?;
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
    let mut body = options;
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
                Kind::DeepSeek | Kind::Vllm | Kind::Zhipu => "max_tokens",
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

pub(super) fn encode_message(kind: Kind, message: &Message) -> Result<Value, Error> {
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
            "image" if matches!(kind, Kind::OpenAI | Kind::Vllm) && message.role == Role::User => {
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
                match source.get("kind").and_then(Value::as_str) {
                    Some("url") => {}
                    Some("base64") => {
                        let valid_data_url = url
                            .strip_prefix("data:")
                            .and_then(|value| value.split_once(";base64,"))
                            .is_some_and(|(_, payload)| !payload.is_empty());
                        if !valid_data_url {
                            return Err(Error::new(
                                "INVALID_ARGUMENTS",
                                "base64 image source must be a nonempty data URL",
                            ));
                        }
                    }
                    _ => {
                        return Err(Error::new(
                            "UNSUPPORTED",
                            "image source kind is unsupported",
                        ));
                    }
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

#[derive(Default)]
struct CallParts {
    id: String,
    name: String,
    arguments: String,
    position: Option<usize>,
}

pub(super) struct Accumulator {
    message: Message,
    calls: BTreeMap<u64, CallParts>,
    usage: Option<Usage>,
    finish: Option<String>,
}

impl Accumulator {
    pub(super) fn new(metadata: Map<String, Value>) -> Self {
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
                data: part_data(json!({"value": text})),
            });
        }
    }

    pub(super) fn delta(&mut self, value: &Value) -> Result<Option<Message>, Error> {
        if value.get("error").is_some() {
            return Err(common::api_error(0, value));
        }
        if let Some(usage) = common::parse_usage(value.get("usage"))? {
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
        // Providers use different wire names for the same reasoning stream.
        // Keep the public Message contract stable and expose all of them as
        // the canonical `think` content part.
        for (value, kind) in [
            (
                delta
                    .get("reasoning")
                    .or_else(|| delta.get("reasoning_content")),
                "think",
            ),
            (delta.get("content"), "text"),
            (delta.get("refusal"), "refusal"),
        ] {
            if let Some(value) = value.filter(|value| !value.is_null()) {
                let text = value
                    .as_str()
                    .ok_or_else(|| Error::new("INVALID_RESPONSE", "chat text delta is invalid"))?;
                if !text.is_empty() {
                    self.push_text(kind, text);
                    observation.content.push(Part {
                        r#type: kind.into(),
                        data: part_data(json!({"value": text})),
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
                if call
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind != "function")
                {
                    return Err(Error::new(
                        "UNSUPPORTED",
                        format!(
                            "non-function Tool call type `{}` is unsupported",
                            call.get("type")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown")
                        ),
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
                    data: part_data(
                        json!({"call_id": entry.id, "name": entry.name, "arguments": entry.arguments}),
                    ),
                };
                let position = *entry.position.get_or_insert_with(|| {
                    self.message.content.push(part.clone());
                    self.message.content.len() - 1
                });
                self.message.content[position] = part;
                if !entry.id.is_empty() && !entry.name.is_empty() {
                    observation.content.push(Part { r#type: "tool_call".into(), data: part_data(json!({"call_id": entry.id, "name": entry.name, "arguments": arguments})) });
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

pub(super) fn complete_response(
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
