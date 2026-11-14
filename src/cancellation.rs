use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicUsize, Ordering},
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
    next_waiter: AtomicUsize,
    waiters: Mutex<Vec<(usize, Waker)>>,
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
            for (_, waiter) in waiters {
                waiter.wake();
            }
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    /// Register a stream poller so cancellation wakes it even if its provider
    /// stream has no new item. Rechecking under the lock avoids a lost wake-up.
    pub(crate) fn register(&self, waker: &Waker) -> Registration {
        let mut waiters = self.state.waiters.lock().expect("cancel lock poisoned");
        if self.is_cancelled() {
            waker.wake_by_ref();
            return Registration {
                state: Weak::new(),
                id: None,
            };
        }
        let id = self.state.next_waiter.fetch_add(1, Ordering::Relaxed);
        waiters.push((id, waker.clone()));
        Registration {
            state: Arc::downgrade(&self.state),
            id: Some(id),
        }
    }
}

/// A cancellation waiter that unregisters itself when the owning poller is
/// finished. Keeping this guard with a stream prevents completed runs from
/// retaining their task wakers in the cancellation scope.
pub(crate) struct Registration {
    state: Weak<State>,
    id: Option<usize>,
}

impl Registration {
    pub(crate) fn update(&self, waker: &Waker) {
        let Some(id) = self.id else {
            waker.wake_by_ref();
            return;
        };
        let Some(state) = self.state.upgrade() else {
            return;
        };
        let mut waiters = state.waiters.lock().expect("cancel lock poisoned");
        if state.cancelled.load(Ordering::Acquire) {
            waker.wake_by_ref();
        } else if let Some((_, saved)) = waiters.iter_mut().find(|(saved_id, _)| *saved_id == id)
            && !saved.will_wake(waker)
        {
            *saved = waker.clone();
        }
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let Some(id) = self.id.take() else {
            return;
        };
        if let Some(state) = self.state.upgrade() {
            let mut waiters = state.waiters.lock().expect("cancel lock poisoned");
            waiters.retain(|(saved_id, _)| *saved_id != id);
        }
    }
}

tokio::task_local! {
    static CURRENT: Cancellation;
}

pub(crate) async fn scope<F>(cancellation: Cancellation, future: F) -> F::Output
where
    F: Future,
{
    CURRENT.scope(cancellation, future).await
}

pub(crate) fn current() -> Option<Cancellation> {
    CURRENT.try_with(Clone::clone).ok()
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
    let cancellation_scope = cancellation.clone();
    scope(cancellation_scope, async move {
        let mut future = std::pin::pin!(future);
        let mut registration: Option<Registration> = None;
        poll_fn(move |cx| {
            if let Some(waiter) = &registration {
                waiter.update(cx.waker());
            } else {
                registration = Some(cancellation.register(cx.waker()));
            }
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
    })
    .await
}

#[cfg(test)]
#[path = "../tests/unit/embedded/src_cancellation_tests.rs"]
mod tests;
