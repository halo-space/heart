//! Private observation transport; execution facts remain in Runtime.state.

use std::{
    collections::VecDeque,
    future::Future,
    sync::{Arc, Mutex},
    task::Poll,
};

use futures_core::Stream;
use futures_util::task::AtomicWaker;

use crate::{Error, Message, Messages, event::Event};

pub(crate) fn project_message(message: &Message) -> Message {
    // Do not deep-copy potentially large business arrays only to discard them.
    Message {
        role: message.role.clone(),
        content: message.content.clone(),
        values: Default::default(),
        metadata: message.metadata.clone(),
        usage: message.usage.clone(),
    }
}

#[derive(Clone, Default)]
pub(super) struct Output(Arc<Mutex<VecDeque<Result<Event, Error>>>>);

/// Backpressured native facts from an active invocation to its graph owner.
/// Notifications are transient; only the existing Attempt is persisted.
#[derive(Clone, Default)]
pub(super) struct Relay(Arc<RelayInner>);
#[derive(Default)]
struct RelayInner {
    pending: Mutex<VecDeque<Progress>>,
    consumer: AtomicWaker,
}
pub(super) struct Progress {
    pub node_id: String,
    pub exec_id: i64,
    pub attempt: super::node::Attempt,
    pub event: Option<Event>,
    /// Admission changes only status, never replaces saved Attempt facts.
    pub entered: bool,
    pub ack: tokio::sync::oneshot::Sender<Result<(), Error>>,
}
impl Relay {
    pub(super) fn poll(&self, cx: &mut std::task::Context<'_>) -> Poll<Progress> {
        self.0.consumer.register(cx.waker());
        self.0
            .pending
            .lock()
            .expect("Workflow progress lock poisoned")
            .pop_front()
            .map_or(Poll::Pending, Poll::Ready)
    }
}

impl Output {
    fn push(&self, event: Result<Event, Error>) {
        // Observation transport failure must not turn into execution failure.
        if let Ok(mut queue) = self.0.lock() {
            queue.push_back(event);
        }
    }

    fn next(&self) -> Option<Result<Event, Error>> {
        self.0.lock().ok()?.pop_front()
    }
}

tokio::task_local! {
    static CURRENT: Output;
    static SOURCE: (i64, String, i64);
    static INTERACTION: crate::runtime::agent::reply::Run;
    static RELAY:Relay;
    static ATTEMPT:u32;
}

pub(super) async fn with_interaction<F: Future>(
    interaction: crate::runtime::agent::reply::Run,
    future: F,
) -> F::Output {
    INTERACTION.scope(interaction, future).await
}
pub(super) async fn with_optional_interaction<F: Future>(
    interaction: Option<crate::runtime::agent::reply::Run>,
    future: F,
) -> F::Output {
    match interaction {
        Some(interaction) => with_interaction(interaction, future).await,
        None => future.await,
    }
}
pub(super) fn interaction() -> Option<crate::runtime::agent::reply::Run> {
    INTERACTION.try_with(Clone::clone).ok()
}
pub(super) async fn with_relay<F: Future>(relay: Relay, future: F) -> F::Output {
    RELAY.scope(relay, future).await
}
pub(super) async fn with_attempt<F: Future>(idx: u32, future: F) -> F::Output {
    ATTEMPT.scope(idx, future).await
}
pub(super) fn attempt_index() -> u32 {
    ATTEMPT.try_with(|idx| *idx).unwrap_or(0)
}
pub(super) fn has_parent() -> bool {
    RELAY.try_with(|_| ()).is_ok() && SOURCE.try_with(|_| ()).is_ok()
}
pub(super) async fn progress(
    attempt: super::node::Attempt,
    event: Option<Event>,
) -> Result<(), Error> {
    Box::pin(handoff(attempt, event, false)).await
}

pub(super) async fn entered(idx: u32) -> Result<(), Error> {
    Box::pin(handoff(
        super::node::Attempt {
            idx,
            ..Default::default()
        },
        None,
        true,
    ))
    .await
}

async fn handoff(
    attempt: super::node::Attempt,
    mut event: Option<Event>,
    entered: bool,
) -> Result<(), Error> {
    let (Ok(relay), Ok((_, node_id, exec_id))) =
        (RELAY.try_with(Clone::clone), SOURCE.try_with(Clone::clone))
    else {
        if let Some(event) = event {
            emit(event).await;
        }
        return Ok(());
    };
    if let Some(event) = &mut event {
        let chat_id = SOURCE
            .try_with(|source| source.0)
            .expect("current Workflow source");
        crate::runtime::observation::associate(event, Some(chat_id), Some((&node_id, exec_id)));
    }
    let (ack, received) = tokio::sync::oneshot::channel();
    relay
        .0
        .pending
        .lock()
        .expect("Workflow progress lock poisoned")
        .push_back(Progress {
            node_id,
            exec_id,
            attempt,
            event,
            entered,
            ack,
        });
    relay.0.consumer.wake();
    received
        .await
        .map_err(|_| Error::new("CANCELLED", "Workflow execution owner ended"))?
}

pub(super) async fn with_source<F: Future>(
    chat_id: i64,
    node_id: &str,
    exec_id: i64,
    future: F,
) -> F::Output {
    SOURCE
        .scope((chat_id, node_id.to_owned(), exec_id), future)
        .await
}

/// Publish an existing fact and yield to the owning stream's caller. Nested
/// graphs inherit this scope instead of creating a second observation stream.
pub(crate) async fn emit(mut event: Event) {
    let Ok(output) = CURRENT.try_with(Clone::clone) else {
        return;
    };
    let _ = SOURCE.try_with(|(chat_id, node_id, exec_id)| {
        crate::runtime::observation::associate(
            &mut event,
            Some(*chat_id),
            Some((node_id, *exec_id)),
        );
    });
    match &mut event {
        Event::Delta { message } | Event::Complete { message, .. } => {
            message.values.clear();
        }
    }
    output.push(Ok(event));
    let mut yielded = false;
    std::future::poll_fn(|context| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            context.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await;
}

pub(super) fn stream<F>(output: Output, execution: F) -> impl Stream<Item = Result<Event, Error>>
where
    F: Future<Output = Result<Messages, Error>>,
{
    let mut execution = Box::pin(CURRENT.scope(output.clone(), execution));
    let mut finished = false;
    futures_util::stream::poll_fn(move |context| {
        if let Some(event) = output.next() {
            return Poll::Ready(Some(event));
        }
        if !finished {
            match execution.as_mut().poll(context) {
                Poll::Ready(result) => {
                    finished = true;
                    if let Err(error) = result {
                        // The execution owner has already committed its state
                        // and performed Checkpoint/Memory closeout.
                        output.push(Err(error));
                    }
                }
                Poll::Pending => {}
            }
        }
        if let Some(event) = output.next() {
            Poll::Ready(Some(event))
        } else if finished {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    })
}
