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

use crate::{Error, agent::event::Event, agent::model::token::Usage};

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
        // `source` is pinned together with this wrapper and is never moved
        // after polling starts, so projecting the pin is sound.
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
                    crate::agent::model::token::merge(observation.usage.take(), usage.clone());
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
mod tests {
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
        let values =
            futures_executor::block_on(async { observed.by_ref().collect::<Vec<_>>().await });

        assert!(matches!(values[0], Ok(Event::Delta { ref message }) if message == &delta));
        assert!(
            matches!(values[1], Ok(Event::Complete { ref message, .. }) if message == &complete)
        );
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
}
