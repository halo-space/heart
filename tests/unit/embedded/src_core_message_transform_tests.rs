use futures_util::stream;

use super::consume;
use crate::components::{
    event::Event,
    message::{Message, Messages, Role},
    model::{content::part::Part, token::Usage},
};
use crate::{Cancellation, Error};
use serde_json::json;

fn message(value: &str) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "text".into(),
        data: crate::components::model::content::part::object(json!({"value":value})),
    });
    message
}

#[test]
fn event_consumer_uses_complete_not_deltas_and_copies_complete_usage() {
    let usage = Usage {
        input: Some(10),
        output: Some(4),
        total: Some(14),
        ..Default::default()
    };
    let events = stream::iter(vec![
        Ok(Event::Delta {
            message: message("delta that must not be returned"),
        }),
        Ok(Event::Complete {
            message: message("provider complete"),
            usage: Some(usage.clone()),
            finish_reason: Some("stop".into()),
        }),
    ]);
    let output = futures_executor::block_on(consume(events, &Cancellation::new())).unwrap();
    assert_eq!(
        output,
        Messages::new([{
            let mut complete = message("provider complete");
            complete.usage = Some(usage);
            complete
        }])
    );
}

#[test]
fn event_consumer_preserves_message_usage_when_complete_usage_is_absent() {
    let mut complete = message("provider complete");
    complete.usage = Some(Usage {
        total: Some(12),
        ..Default::default()
    });
    let events = stream::iter(vec![Ok(Event::Complete {
        message: complete.clone(),
        usage: None,
        finish_reason: None,
    })]);
    let output = futures_executor::block_on(consume(events, &Cancellation::new())).unwrap();
    assert_eq!(output, Messages::new([complete]));
}

#[test]
fn event_consumer_returns_transport_error_and_rejects_missing_complete() {
    let expected = Error::new("TRANSPORT", "provider stream failed");
    let events = stream::iter(vec![
        Ok(Event::Delta {
            message: message("partial"),
        }),
        Err(expected.clone()),
    ]);
    assert_eq!(
        futures_executor::block_on(consume(events, &Cancellation::new())).unwrap_err(),
        expected
    );
    let events = stream::iter(vec![Ok(Event::Delta {
        message: message("partial only"),
    })]);
    assert_eq!(
        futures_executor::block_on(consume(events, &Cancellation::new()))
            .unwrap_err()
            .code,
        "INVALID_RESPONSE"
    );
}

#[test]
fn event_consumer_cancellation_stops_a_pending_event_stream() {
    let cancellation = Cancellation::new();
    let token = cancellation.clone();
    let events = stream::once(async move {
        token.cancel();
        Ok::<_, Error>(Event::Delta {
            message: message("not delivered"),
        })
    });
    assert_eq!(
        futures_executor::block_on(consume(events, &cancellation))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
}
