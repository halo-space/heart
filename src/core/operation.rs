use std::{future::Future, task::Poll};

use futures_util::future::poll_fn;

use crate::{Cancellation, Error};

pub(super) fn check(cancellation: &Cancellation) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        Err(Error::new("CANCELLED", "operation cancelled"))
    } else {
        Ok(())
    }
}

pub(super) async fn cancellable<T>(
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
        if let Err(error) = check(cancellation) {
            return Poll::Ready(Err(error));
        }
        result
    })
    .await
}
