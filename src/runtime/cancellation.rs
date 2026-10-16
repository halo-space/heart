use std::{future::Future, task::Poll};

use crate::Cancellation;
use futures_util::future::poll_fn;

/// Poll an asynchronous operation while making cancellation observable even
/// when the operation itself does not produce another wake-up.
pub(crate) async fn cancellable<T>(
    future: impl Future<Output = Result<T, crate::Error>>,
    cancellation: &Cancellation,
) -> Result<T, crate::Error> {
    let mut future = std::pin::pin!(future);
    poll_fn(|cx| {
        cancellation.register(cx.waker());
        if cancellation.is_cancelled() {
            return Poll::Ready(Err(crate::Error::new("CANCELLED", "operation cancelled")));
        }
        let result = future.as_mut().poll(cx);
        if cancellation.is_cancelled() {
            return Poll::Ready(Err(crate::Error::new("CANCELLED", "operation cancelled")));
        }
        result
    })
    .await
}
