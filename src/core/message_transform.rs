use std::future::Future;
use std::task::Poll;

use futures_core::Stream;
use futures_util::{StreamExt, future::poll_fn};

use crate::{
    Cancellation, Error, Messages,
    event::Event,
    model::chat::{Chat, Request},
};

pub(super) async fn run<C: Chat>(
    chat: &C,
    base: &Request,
    input: Messages,
    cancellation: &Cancellation,
) -> Result<Messages, Error> {
    check(cancellation)?;
    let mut request = base.clone();
    request.messages.0.extend(input.0);
    let stream = cancellable(chat.stream(request), cancellation).await?;
    consume(stream, cancellation).await
}

/// Consume a Provider Event stream without manufacturing Chat responses.
/// Keeping this boundary separate lets protocol tests exercise Event handling
/// directly while real model behavior is covered by live Provider acceptance.
pub(super) async fn consume<S>(stream: S, cancellation: &Cancellation) -> Result<Messages, Error>
where
    S: Stream<Item = Result<Event, Error>>,
{
    let mut stream = std::pin::pin!(stream);
    loop {
        let event = cancellable(async { Ok(stream.next().await) }, cancellation).await?;
        match event {
            Some(Ok(Event::Delta { .. })) => {}
            Some(Ok(Event::Complete {
                mut message, usage, ..
            })) => {
                check(cancellation)?;
                if let Some(usage) = usage {
                    message.usage = Some(usage);
                }
                return Ok(Messages::new([message]));
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

async fn cancellable<T>(
    future: impl Future<Output = Result<T, Error>>,
    cancellation: &Cancellation,
) -> Result<T, Error> {
    let mut future = std::pin::pin!(future);
    poll_fn(|cx| {
        cancellation.register(cx.waker());
        if let Err(error) = check(cancellation) {
            return Poll::Ready(Err(error));
        }
        let result = future.as_mut().poll(cx);
        // A user implementation may cancel synchronously during its poll.
        if let Err(error) = check(cancellation) {
            return Poll::Ready(Err(error));
        }
        result
    })
    .await
}

fn check(cancellation: &Cancellation) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        Err(Error::new("CANCELLED", "message transformation cancelled"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
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
        let expected = Error::new("NETWORK_ERROR", "provider stream failed");
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
}
