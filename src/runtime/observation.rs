//! Private projection helpers, not a second event or execution model.

use crate::components::message::Role;
use crate::model::token::Usage;
use crate::runtime::trace::fastrace::Root;
use crate::{Error, Message, event::Event};

/// Record settled model usage with the same field names for foreground Events
/// and background Memory work. The audit is a trace projection only; it never
/// changes Runtime usage or the Chat billing record.
pub(crate) fn observe_usage(trace: &Root, event: &'static str, usage: Option<&Usage>) {
    trace.event_properties(event, usage_properties(usage));
}

fn usage_properties(usage: Option<&Usage>) -> Vec<(String, String)> {
    let mut properties = Vec::new();
    if let Some(usage) = usage {
        if let Some(input) = usage.input {
            properties.push(("usage.input".into(), input.to_string()));
        }
        if let Some(output) = usage.output {
            properties.push(("usage.output".into(), output.to_string()));
        }
        if let Some(total) = usage.total {
            properties.push(("usage.total".into(), total.to_string()));
        }
    }
    properties
}

/// Project one public Event result into the configured fastrace span.
///
/// Keep Trace as a compact projection instead of copying the Event Stream.
/// AgentScope records content on Agent/Model/Tool spans; streaming deltas stay
/// in the public Event Stream and are not duplicated into the trace backend.
pub(crate) fn observe_trace(trace: &Root, result: &Result<Event, Error>) {
    match result {
        Ok(Event::Delta { .. }) => {}
        Ok(Event::Complete {
            message,
            usage,
            finish_reason,
        }) => {
            let mut properties = vec![("status".into(), "completed".into())];
            if let Some(reason) = finish_reason {
                properties.push(("finish_reason".into(), reason.clone()));
            }
            properties.extend(usage_properties(usage.as_ref()));
            if let Some(reply_id) = message.metadata.get("reply_id").and_then(|v| v.as_i64()) {
                properties.push(("reply_id".into(), reply_id.to_string()));
            }
            // A non-tool assistant completion is the public Agent result. The
            // model span still retains every individual model response.
            if matches!(message.role, Role::Assistant)
                && !message
                    .content
                    .iter()
                    .any(|part| part.r#type == "tool_call")
            {
                trace.property_json("agent.output", message);
            }
            trace.event_properties("event.complete", properties);
        }
        Err(error) => {
            trace.event_properties(
                "event.error",
                vec![
                    ("status".into(), "failed".into()),
                    ("error_code".into(), error.code.clone()),
                    ("error_message".into(), error.message.clone()),
                ],
            );
        }
    }
}

pub(crate) fn clear_source(message: &mut Message) {
    // reply_id belongs to the runtime's question/feedback association, just
    // as the other fields belong to execution observation. Neither is model
    // context. Strip only top-level metadata: business content/values and
    // the authoritative Input/Step waiting facts must remain untouched.
    for key in ["chat_id", "node_id", "exec_id", "reply_id"] {
        message.metadata.remove(key);
    }
}

pub(crate) fn original_message(message: &Message) -> Message {
    let mut original = message.clone();
    clear_source(&mut original);
    original
}

/// Annotate only an outgoing event. Preserve a real child execution's source
/// when a containing Agent/Graph forwards it; the run identity is authoritative.
pub(crate) fn associate(event: &mut Event, chat_id: Option<i64>, source: Option<(&str, i64)>) {
    let message = match event {
        Event::Delta { message } | Event::Complete { message, .. } => message,
    };
    if let Some(chat_id) = chat_id {
        message.metadata.insert("chat_id".into(), chat_id.into());
    }
    let has_source = message
        .metadata
        .get("node_id")
        .and_then(|v| v.as_str())
        .is_some()
        && message
            .metadata
            .get("exec_id")
            .and_then(|v| v.as_i64())
            .is_some();
    if !has_source && let Some((node_id, exec_id)) = source {
        message.metadata.insert("node_id".into(), node_id.into());
        message.metadata.insert("exec_id".into(), exec_id.into());
    }
}

#[cfg(test)]
mod tests {
    use super::associate;
    use crate::{Message, event::Event, message::Role};
    use serde_json::json;

    #[test]
    fn associate_preserves_descendant_execution_source() {
        let mut message = Message::new(Role::Assistant);
        message.metadata = json!({"node_id":"child_node","exec_id":778})
            .as_object()
            .unwrap()
            .clone();
        let mut event = Event::Complete {
            message,
            usage: None,
            finish_reason: None,
        };

        associate(&mut event, Some(501), Some(("container_node", 901)));

        let Event::Complete { message, .. } = event else {
            unreachable!();
        };
        assert_eq!(message.metadata["chat_id"], 501);
        assert_eq!(message.metadata["node_id"], "child_node");
        assert_eq!(message.metadata["exec_id"], 778);
    }

    #[test]
    fn associate_fills_missing_execution_source() {
        let mut event = Event::Delta {
            message: Message::new(Role::Assistant),
        };

        associate(&mut event, Some(501), Some(("child_node", 778)));

        let Event::Delta { message } = event else {
            unreachable!();
        };
        assert_eq!(message.metadata["chat_id"], 501);
        assert_eq!(message.metadata["node_id"], "child_node");
        assert_eq!(message.metadata["exec_id"], 778);
    }
}
