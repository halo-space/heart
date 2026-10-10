use std::future::Future;
use std::task::Poll;

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
