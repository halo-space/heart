use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::task::Waker;

#[derive(Debug, Default)]
struct State {
    cancelled: AtomicBool,
    waiters: Mutex<Vec<Waker>>,
}

/// Cooperative cancellation shared by a running operation and its children.
#[derive(Clone, Debug, Default)]
pub struct Cancellation {
    state: Arc<State>,
}

impl Cancellation {
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
