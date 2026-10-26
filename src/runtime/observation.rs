//! Private projection helpers, not a second event or execution model.

use crate::{Message, event::Event};

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
