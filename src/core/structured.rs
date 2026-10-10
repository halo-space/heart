//! Private single-call structured model output support. Never executes tools,
//! assembles a result from Delta, retries a call, or touches Runtime state.

use futures_util::StreamExt;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::model::{
    chat::{Chat, Request, ResponseFormat},
    content::Part,
};
use crate::{Cancellation, Error, Message, Messages, event::Event, message::Role};

pub(super) fn text(role: Role, value: String) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": value}),
    });
    message
}

pub(super) fn prepare(mut request: Request, instruction: &str) -> Result<Request, Error> {
    request.validate()?;
    // Preserve explicit model options and caller-supplied JSON Schema.
    request
        .response_format
        .get_or_insert(ResponseFormat::JsonObject);
    request
        .messages
        .0
        .push(text(Role::System, instruction.to_owned()));
    Ok(request)
}

pub(super) async fn run<C: Chat, T: DeserializeOwned>(
    chat: &C,
    base: &Request,
    input: Messages,
    cancellation: &Cancellation,
) -> Result<T, Error> {
    let message = run_message(chat, base, input, cancellation).await?;
    let result = decode(&message)?;
    super::operation::check(cancellation)?;
    Ok(result)
}

pub(super) async fn run_message<C: Chat>(
    chat: &C,
    base: &Request,
    input: Messages,
    cancellation: &Cancellation,
) -> Result<Message, Error> {
    super::operation::check(cancellation)?;
    let mut request = base.clone();
    request.messages.0.extend(input.0);
    let stream = super::operation::cancellable(chat.stream(request), cancellation).await?;
    let mut stream = std::pin::pin!(stream);
    loop {
        let event =
            super::operation::cancellable(async { Ok(stream.next().await) }, cancellation).await?;
        match event {
            Some(Ok(Event::Delta { .. })) => {}
            Some(Ok(Event::Complete {
                mut message,
                usage,
                finish_reason,
                ..
            })) => {
                if finish_reason.as_deref() == Some("cancelled") {
                    return Err(Error::new("CANCELLED", "structured model call cancelled"));
                }
                // Preserve Complete accounting on the transformed Message.
                if usage.is_some() {
                    message.usage = usage;
                }
                super::operation::check(cancellation)?;
                return Ok(message);
            }
            Some(Err(error)) => return Err(error),
            None => {
                return Err(Error::new(
                    "INVALID_RESPONSE",
                    "model stream ended without Complete",
                ));
            }
        }
    }
}

pub(super) fn decode<T: DeserializeOwned>(message: &Message) -> Result<T, Error> {
    if message.role != Role::Assistant {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "structured model output must be assistant Message",
        ));
    }
    let mut content = String::new();
    for part in &message.content {
        match part.r#type.as_str() {
            "think" => {}
            "text" => {
                let value = part
                    .data
                    .get("value")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::new("INVALID_RESPONSE", "model text is not a string"))?;
                content.push_str(value);
            }
            _ => {
                return Err(Error::new(
                    "INVALID_RESPONSE",
                    "expected structured data, not tool call or media output",
                ));
            }
        }
    }
    let invalid = || {
        Error::new(
            "INVALID_RESPONSE",
            "model output does not match the required JSON structure",
        )
    };
    if !message.values.is_empty() {
        serde_json::from_value(Value::Object(message.values.clone())).map_err(|_| invalid())
    } else {
        // Parse the exact complete JSON. Do not strip fences or repair a patch.
        serde_json::from_str(&content).map_err(|_| invalid())
    }
}
