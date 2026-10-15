//! Private observation transport; execution facts remain in Runtime.state.

use std::{
    collections::VecDeque,
    future::Future,
    sync::{Arc, Mutex},
    task::Poll,
};

use futures_core::Stream;

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

pub(super) async fn emit_at(chat_id: i64, node_id: &str, exec_id: i64, mut event: Event) {
    crate::harness::observation::associate(&mut event, Some(chat_id), Some((node_id, exec_id)));
    emit(event).await;
}

/// Publish an existing fact and yield to the owning stream's caller. Nested
/// graphs inherit this scope instead of creating a second observation stream.
pub(crate) async fn emit(mut event: Event) {
    let Ok(output) = CURRENT.try_with(Clone::clone) else {
        return;
    };
    let _ = SOURCE.try_with(|(chat_id, node_id, exec_id)| {
        crate::harness::observation::associate(
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
