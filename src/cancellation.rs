use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::{
    future::Future,
    task::{Poll, Waker},
};

use futures_util::future::poll_fn;

use crate::Error;

#[derive(Debug, Default)]
struct State {
    cancelled: AtomicBool,
    waiters: Mutex<Vec<Waker>>,
}

/// A cooperative cancellation signal scoped to one execution.
///
/// Clones share cancellation state so one run can propagate cancellation to
/// its own child operations. Independent runs must use independently-created
/// values; cancelling one value never affects another value created with
/// [`Cancellation::new`].
#[derive(Clone, Debug, Default)]
pub struct Cancellation {
    state: Arc<State>,
}

impl Cancellation {
    /// Create a fresh cancellation scope for one execution.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        if !self.state.cancelled.swap(true, Ordering::AcqRel) {
            let waiters =
                std::mem::take(&mut *self.state.waiters.lock().expect("cancel lock poisoned"));
            for waiter in waiters {
                waiter.wake();
            }
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    /// Register a stream poller so cancellation wakes it even if its provider
    /// stream has no new item. Rechecking under the lock avoids a lost wake-up.
    pub(crate) fn register(&self, waker: &Waker) {
        let mut waiters = self.state.waiters.lock().expect("cancel lock poisoned");
        if self.is_cancelled() {
            waker.wake_by_ref();
        } else if !waiters.iter().any(|waiter| waiter.will_wake(waker)) {
            waiters.push(waker.clone());
        }
    }
}

/// Check the shared cancellation signal using the common runtime error shape.
pub(crate) fn check(cancellation: &Cancellation) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        Err(Error::new("CANCELLED", "operation cancelled"))
    } else {
        Ok(())
    }
}

/// Poll an async operation while making cancellation observable even when
/// the operation itself does not produce another wake-up.
pub(crate) async fn cancellable<T, E: From<Error>>(
    future: impl Future<Output = Result<T, E>>,
    cancellation: &Cancellation,
) -> Result<T, E> {
    let mut future = std::pin::pin!(future);
    poll_fn(|cx| {
        cancellation.register(cx.waker());
        if let Err(error) = check(cancellation) {
            return Poll::Ready(Err(error.into()));
        }
        let result = future.as_mut().poll(cx);
        if let Err(error) = check(cancellation) {
            return Poll::Ready(Err(error.into()));
        }
        result
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::Cancellation;

    #[test]
    fn cancellation_is_shared_within_a_run_but_isolated_between_runs() {
        let first_run = Cancellation::new();
        let first_run_child = first_run.clone();
        let second_run = Cancellation::new();

        first_run.cancel();

        assert!(first_run_child.is_cancelled());
        assert!(!second_run.is_cancelled());
    }
}
