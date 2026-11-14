//! Event observation for Harness callers.
//!
//! `Observed` is a projection around an existing Agent or Workflow event
//! stream. It does not create another execution loop, alter events, or own
//! Runtime state. The snapshot advances only when the wrapped stream is
//! consumed.

use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use futures_core::Stream;

use crate::Error;
use crate::components::event::Event;
use crate::components::model::token::Usage;

/// Counters and settled model usage observed from one event stream.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Observation {
    pub delta: u64,
    pub complete: u64,
    pub error: u64,
    pub usage: Option<Usage>,
    pub last_error: Option<String>,
    pub last_finish_reason: Option<String>,
}

/// A stream wrapper that updates an observation snapshot as events are read.
pub struct Observed<S> {
    source: S,
    observation: Arc<Mutex<Observation>>,
}

impl<S> Observed<S> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            observation: Arc::new(Mutex::new(Observation::default())),
        }
    }

    /// Return the facts observed up to the last consumed item.
    pub fn snapshot(&self) -> Observation {
        self.observation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Share the observation snapshot with a caller that consumes the stream
    /// elsewhere, without exposing the wrapped source or Runtime state.
    pub fn handle(&self) -> Handle {
        Handle {
            observation: Arc::clone(&self.observation),
        }
    }
}

/// Read-only access to an [`Observed`] stream's current snapshot.
#[derive(Clone)]
pub struct Handle {
    observation: Arc<Mutex<Observation>>,
}

impl Handle {
    pub fn snapshot(&self) -> Observation {
        self.observation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl<S> Stream for Observed<S>
where
    S: Stream<Item = Result<Event, Error>>,
{
    type Item = Result<Event, Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // SAFETY: `source` is a structurally pinned field of `Observed`. The
        // wrapper never moves it after polling starts, and this method creates
        // only a shorter-lived pin to that field before delegating the poll.
        let this = unsafe { self.get_unchecked_mut() };
        let item = unsafe { Pin::new_unchecked(&mut this.source) }.poll_next(cx);
        let Poll::Ready(Some(ref result)) = item else {
            return item;
        };

        let mut observation = this
            .observation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match result {
            Ok(Event::Delta { .. }) => observation.delta = observation.delta.saturating_add(1),
            Ok(Event::Complete {
                usage,
                finish_reason,
                ..
            }) => {
                observation.complete = observation.complete.saturating_add(1);
                observation.usage =
                    crate::components::model::token::merge(observation.usage.take(), usage.clone());
                observation.last_finish_reason = finish_reason.clone();
            }
            Err(error) => {
                observation.error = observation.error.saturating_add(1);
                observation.last_error = Some(error.code.clone());
            }
        }
        item
    }
}

#[cfg(test)]
#[path = "../../tests/unit/embedded/src_harness_observe_tests.rs"]
mod tests;
