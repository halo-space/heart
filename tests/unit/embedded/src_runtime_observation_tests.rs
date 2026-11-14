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
