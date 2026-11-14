use futures_util::{StreamExt, stream};
use serde_json::json;

use super::{Observation, Observed};
use crate::{
    Error,
    agent::event::Event,
    agent::message::{Message, Role},
    agent::model::token::Usage,
};

#[test]
fn observes_events_without_changing_them() {
    let mut delta = Message::new(Role::Assistant);
    delta.metadata.insert("kind".into(), json!("delta"));
    let complete = Message::new(Role::Assistant);
    let source = stream::iter(vec![
        Ok(Event::Delta {
            message: delta.clone(),
        }),
        Ok(Event::Complete {
            message: complete.clone(),
            usage: Some(Usage {
                input: Some(2),
                output: Some(3),
                total: Some(5),
                ..Usage::default()
            }),
            finish_reason: Some("stop".into()),
        }),
        Err(Error::new("NETWORK", "connection closed")),
    ]);
    let mut observed = Observed::new(source);
    let handle = observed.handle();
    let values = futures_executor::block_on(async { observed.by_ref().collect::<Vec<_>>().await });

    assert!(matches!(values[0], Ok(Event::Delta { ref message }) if message == &delta));
    assert!(matches!(values[1], Ok(Event::Complete { ref message, .. }) if message == &complete));
    assert!(matches!(values[2], Err(Error { ref code, .. }) if code == "NETWORK"));
    assert_eq!(
        handle.snapshot(),
        Observation {
            delta: 1,
            complete: 1,
            error: 1,
            usage: Some(Usage {
                input: Some(2),
                output: Some(3),
                total: Some(5),
                ..Usage::default()
            }),
            last_error: Some("NETWORK".into()),
            last_finish_reason: Some("stop".into()),
        }
    );
}

#[test]
fn usage_is_aggregated_across_complete_events() {
    let source = stream::iter(vec![
        Ok(Event::Complete {
            message: Message::new(Role::Assistant),
            usage: Some(Usage {
                input: Some(2),
                output: Some(1),
                total: Some(3),
                ..Usage::default()
            }),
            finish_reason: None,
        }),
        Ok(Event::Complete {
            message: Message::new(Role::Assistant),
            usage: Some(Usage {
                input: Some(4),
                output: Some(2),
                total: Some(6),
                ..Usage::default()
            }),
            finish_reason: None,
        }),
    ]);
    let mut observed = Observed::new(source);
    futures_executor::block_on(async { while observed.next().await.is_some() {} });

    assert_eq!(
        observed.snapshot().usage.as_ref().and_then(|u| u.total),
        Some(9)
    );
}
