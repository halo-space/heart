use futures_core::Stream;
use futures_util::StreamExt;

use crate::{
    Cancellation, Error, Messages,
    cancellation::{cancellable, check},
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
    let request = crate::components::model::chat::fit_request(request, chat.context_limit())?;
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

#[cfg(test)]
#[path = "../../tests/unit/embedded/src_core_message_transform_tests.rs"]
mod tests;
