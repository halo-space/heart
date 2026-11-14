use futures_util::StreamExt;
use serde_json::json;

use super::adapt;
use crate::components::event::Event as HeartEvent;
use crate::components::message::{Message, Role};
use crate::components::model::content::part::Part;
use crate::components::protocol::ag_ui::{Context, Event};

#[tokio::test]
async fn adapts_text_reasoning_and_run_lifecycle() {
    let mut delta = Message::new(Role::Assistant);
    delta.content.push(Part {
        r#type: "think".into(),
        data: json!({"value": "先判断"}).as_object().unwrap().clone(),
    });
    delta.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "北京晴天"}).as_object().unwrap().clone(),
    });
    let events = futures_util::stream::iter(vec![Ok(HeartEvent::Delta { message: delta })]);
    let output = adapt(events, Context::new("session-1", "chat-1"))
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(output[0], Ok(Event::RunStarted { .. })));
    assert!(output
            .iter()
            .any(|event| matches!(event, Ok(Event::ReasoningMessageContent { delta, .. }) if delta == "先判断")));
    assert!(output.iter().any(
        |event| matches!(event, Ok(Event::TextMessageContent { delta, .. }) if delta == "北京晴天")
    ));
    assert!(matches!(output.last(), Some(Ok(Event::RunFinished { .. }))));
}

#[tokio::test]
async fn maps_tool_call_and_result_without_internal_wrapper() {
    let mut call = Message::new(Role::Assistant);
    call.content.push(Part {
        r#type: "tool_call".into(),
        data: json!({
            "call_id": "call-1",
            "name": "weather",
            "arguments": {"city": "北京"}
        })
        .as_object()
        .unwrap()
        .clone(),
    });
    let mut result = Message::new(Role::Function);
    result.metadata.insert("call_id".into(), json!("call-1"));
    result.values.insert("data".into(), json!("晴天"));
    let events = futures_util::stream::iter(vec![
        Ok(HeartEvent::Complete {
            message: call,
            usage: None,
            finish_reason: None,
        }),
        Ok(HeartEvent::Complete {
            message: result,
            usage: None,
            finish_reason: None,
        }),
    ]);
    let output = adapt(events, Context::new("session-1", "chat-1"))
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().any(|event| matches!(
        event,
        Ok(Event::ToolCallStart { tool_call_id, tool_call_name })
            if tool_call_id == "call-1" && tool_call_name == "weather"
    )));
    assert!(output.iter().any(|event| matches!(
        event,
        Ok(Event::ToolCallResult { tool_call_id, .. }) if tool_call_id == "call-1"
    )));
}

#[tokio::test]
async fn reuses_message_id_and_does_not_repeat_complete_text() {
    fn text(value: &str) -> Message {
        let mut message = Message::new(Role::Assistant);
        message.content.push(Part {
            r#type: "text".into(),
            data: json!({"value": value}).as_object().unwrap().clone(),
        });
        message
    }

    let events = futures_util::stream::iter(vec![
        Ok(HeartEvent::Delta {
            message: text("北"),
        }),
        Ok(HeartEvent::Delta {
            message: text("京"),
        }),
        Ok(HeartEvent::Complete {
            message: text("北京"),
            usage: None,
            finish_reason: Some("stop".into()),
        }),
    ]);
    let output = adapt(events, Context::new("session-1", "chat-1"))
        .collect::<Vec<_>>()
        .await;
    let ids = output
        .iter()
        .filter_map(|event| match event {
            Ok(Event::TextMessageStart { message_id, .. })
            | Ok(Event::TextMessageContent { message_id, .. })
            | Ok(Event::TextMessageEnd { message_id }) => Some(message_id),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(!ids.is_empty());
    assert!(ids.windows(2).all(|window| window[0] == window[1]));
    assert_eq!(
        output
            .iter()
            .filter(|event| matches!(event, Ok(Event::TextMessageStart { .. })))
            .count(),
        1
    );
    assert_eq!(
        output
            .iter()
            .filter(|event| matches!(event, Ok(Event::TextMessageContent { .. })))
            .count(),
        2
    );
    assert_eq!(
        output
            .iter()
            .filter(|event| matches!(event, Ok(Event::TextMessageEnd { .. })))
            .count(),
        1
    );
}

#[tokio::test]
async fn maps_source_error_to_run_error_without_finished() {
    let events = futures_util::stream::iter(vec![Err(crate::Error::new("FAILED", "network"))]);
    let output = adapt(events, Context::new("session-1", "chat-1"))
        .collect::<Vec<_>>()
        .await;
    assert!(
        output
            .iter()
            .any(|event| matches!(event, Ok(Event::RunError { .. })))
    );
    assert!(
        !output
            .iter()
            .any(|event| matches!(event, Ok(Event::RunFinished { outcome: None, .. })))
    );
}

#[tokio::test]
async fn maps_waiting_to_interrupt_outcome() {
    let mut question = Message::new(Role::Assistant);
    question.metadata.insert("reply_id".into(), json!(42));
    question.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "请确认"}).as_object().unwrap().clone(),
    });
    let events = futures_util::stream::iter(vec![
        Ok(HeartEvent::Complete {
            message: question,
            usage: None,
            finish_reason: None,
        }),
        Err(crate::Error::new("WAITING", "waiting for reply")),
    ]);
    let output = adapt(events, Context::new("session-1", "chat-1"))
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().any(|event| matches!(
        event,
        Ok(Event::RunFinished { outcome: Some(outcome), .. })
            if outcome["type"] == "interrupt" && outcome["interrupt_id"] == "42"
    )));
    assert!(
        !output
            .iter()
            .any(|event| matches!(event, Ok(Event::RunError { .. })))
    );
}

#[test]
fn serializes_ag_ui_field_names() {
    let event = Event::TextMessageContent {
        message_id: "message-1".into(),
        delta: "hello".into(),
    };
    let value = serde_json::to_value(event).unwrap();
    assert_eq!(value["type"], "TEXT_MESSAGE_CONTENT");
    assert_eq!(value["messageId"], "message-1");
    assert_eq!(value["delta"], "hello");
}
