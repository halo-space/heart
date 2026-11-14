//! Message views used only while preparing a compression request.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::message::Role;
use crate::{Error, Message, Messages};

pub(crate) fn summary(output: &Messages) -> Result<String, Error> {
    let invalid = || {
        Error::new(
            "INVALID_RESPONSE",
            "compression must return nonempty assistant text",
        )
    };
    let mut messages = Vec::new();
    for message in output.as_slice() {
        if message.role != Role::Assistant {
            return Err(invalid());
        }
        let mut text = String::new();
        for part in &message.content {
            match part.r#type.as_str() {
                "think" => (),
                "text" => text.push_str(
                    part.data
                        .get("value")
                        .and_then(Value::as_str)
                        .ok_or_else(invalid)?,
                ),
                _ => return Err(invalid()),
            }
        }
        if !text.trim().is_empty() {
            messages.push(text);
        }
    }
    if messages.is_empty() {
        return Err(invalid());
    }
    Ok(messages.join("\n"))
}

/// Keep tool-call groups intact; unfinished calls stay in the live scene.
pub(crate) fn safe_messages(messages: Vec<Message>) -> Vec<Message> {
    let mut pending = BTreeSet::new();
    let mut group_start = None;
    let mut result = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        let calls = if message.role == Role::Assistant {
            message
                .content
                .iter()
                .filter(|part| part.r#type == "tool_call")
                .filter_map(|part| part.data.get("call_id").and_then(Value::as_str))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if !calls.is_empty() {
            group_start.get_or_insert(index);
            pending.extend(calls.into_iter().map(str::to_owned));
        }
        if matches!(message.role, Role::Tool | Role::Function)
            && let Some(call_id) = message_call_id(message)
        {
            pending.remove(call_id);
        }
        result.push(message.clone());
        if pending.is_empty() {
            group_start = None;
        }
    }
    if let Some(index) = group_start {
        result.truncate(index);
    }
    result
}

pub(crate) const MAX_TOOL_RESULT_CHARS: usize = 50_000;

pub(crate) fn truncate_tool_results(mut messages: Vec<Message>) -> Vec<Message> {
    for message in &mut messages {
        if !matches!(message.role, Role::Tool | Role::Function) {
            continue;
        }
        for part in &mut message.content {
            for value in part.data.values_mut() {
                truncate_value(value);
            }
        }
        for value in message.values.values_mut() {
            truncate_value(value);
        }
    }
    messages
}

fn truncate_value(value: &mut Value) {
    match value {
        Value::String(text) if text.chars().count() > MAX_TOOL_RESULT_CHARS => {
            let chars = text.chars().collect::<Vec<_>>();
            let keep = MAX_TOOL_RESULT_CHARS / 2;
            let removed = chars.len().saturating_sub(keep * 2);
            let prefix = chars[..keep].iter().collect::<String>();
            let suffix = chars[chars.len() - keep..].iter().collect::<String>();
            *text = format!(
                "{prefix}\n...[tool result truncated for compression input: {removed} chars omitted]...\n{suffix}"
            );
        }
        Value::Array(values) => values.iter_mut().for_each(truncate_value),
        Value::Object(values) => values.values_mut().for_each(truncate_value),
        _ => {}
    }
}

fn message_call_id(message: &Message) -> Option<&str> {
    message
        .metadata
        .get("call_id")
        .or_else(|| message.values.get("call_id"))
        .and_then(Value::as_str)
}
