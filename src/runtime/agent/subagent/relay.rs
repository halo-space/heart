//! Child event backpressure and observation forwarding.

use super::*;

/// Private, consumer-driven forwarding. Each running child can enqueue only
/// one event or native scene update, then waits for the parent to take it. No producer task, persisted
/// queue, replay protocol or second public stream is created.
#[derive(Clone, Default)]
#[doc(hidden)]
pub struct Relay(Arc<RelayState>);

#[derive(Default)]
struct RelayState {
    events: Mutex<std::collections::VecDeque<(Observation, tokio::sync::oneshot::Sender<()>)>>,
    consumer: AtomicWaker,
}

impl Relay {
    pub(crate) fn poll_event(&self, cx: &mut Context<'_>) -> Poll<Observation> {
        self.0.consumer.register(cx.waker());
        if let Some((event, acknowledgement)) = self
            .0
            .events
            .lock()
            .expect("child event lock poisoned")
            .pop_front()
        {
            let _ = acknowledgement.send(());
            Poll::Ready(event)
        } else {
            Poll::Pending
        }
    }

    pub(crate) fn observer(
        &self,
        chat_id: Option<i64>,
        source: crate::runtime::agent::wait::Node,
    ) -> Observer {
        Observer {
            relay: self.clone(),
            chat_id,
            source: (source.node_id, source.exec_id),
            interaction: None,
            suppress_terminal: false,
        }
    }
}

/// Cancellation is cooperative, not an asynchronous Drop. Acknowledge queued
/// facts first, then poll only immediately ready work. Fresh native scenes may
/// release another backpressured handoff; arbitrary non-cooperating streams
/// cannot keep the owner alive merely by emitting more text.
#[inline(never)]
#[doc(hidden)]
pub fn cancel_ready<T>(
    cx: &mut Context<'_>,
    relay: &Relay,
    mut poll: impl FnMut(&mut Context<'_>) -> Poll<Option<T>>,
    mut commit: impl FnMut(&Observation) -> Result<(), Error>,
) -> Result<(Vec<T>, Vec<Event>), Error> {
    // A ready Tokio channel can otherwise yield Pending solely because an
    // enclosing poll spent its cooperative budget. Disable that artificial
    // yield for this synchronous drain only; actual Pending still stops it.
    let mut drain = Box::pin(tokio::task::unconstrained(std::future::poll_fn(|cx| {
        Poll::Ready(drain_ready(cx, relay, &mut poll, &mut commit))
    })));
    match drain.as_mut().poll(cx) {
        Poll::Ready(result) => result,
        Poll::Pending => unreachable!("synchronous cancellation drain"),
    }
}

fn drain_ready<T>(
    cx: &mut Context<'_>,
    relay: &Relay,
    mut poll: impl FnMut(&mut Context<'_>) -> Poll<Option<T>>,
    mut commit: impl FnMut(&Observation) -> Result<(), Error>,
) -> Result<(Vec<T>, Vec<Event>), Error> {
    let mut completed = Vec::new();
    let mut events = Vec::new();
    while let Poll::Ready(mut observation) = relay.poll_event(cx) {
        commit(&observation)?;
        events.extend(
            observation
                .event
                .take()
                .filter(|event| matches!(event, Event::Complete { message, .. } if !message.metadata.contains_key("reply_id"))),
        );
    }
    loop {
        let result = poll(cx);
        let mut native = false;
        while let Poll::Ready(mut observation) = relay.poll_event(cx) {
            native |= observation.scene.is_some();
            commit(&observation)?;
            events.extend(
                observation
                    .event
                    .take()
                    .filter(|event| matches!(event, Event::Complete { message, .. } if !message.metadata.contains_key("reply_id"))),
            );
        }
        match result {
            Poll::Ready(Some(result)) => completed.push(result),
            Poll::Ready(None) => break,
            Poll::Pending if native => continue,
            Poll::Pending => break,
        }
    }
    Ok((completed, events))
}

#[doc(hidden)]
pub struct Observer {
    pub(crate) relay: Relay,
    pub(crate) chat_id: Option<i64>,
    pub(crate) source: (String, i64),
    pub(crate) interaction: Option<crate::runtime::agent::reply::Handle>,
    pub(crate) suppress_terminal: bool,
}

#[doc(hidden)]
pub struct Observation {
    pub event: Option<Event>,
    pub feedback: bool,
    pub source: (String, i64),
    pub scene: Option<Arc<Value>>,
    pub usage: Option<Usage>,
    pub message: Option<Message>,
}

impl Observation {
    pub(crate) fn commit_feedback(
        &self,
        owner: Option<&crate::runtime::agent::reply::Handle>,
    ) -> Result<(), Error> {
        if let (Some(owner), Some(scene)) = (owner, &self.scene) {
            owner.commit_scene_feedback(scene)?;
        }
        Ok(())
    }
    pub(crate) fn has_facts(&self) -> bool {
        self.scene.is_some() || self.usage.is_some() || self.message.is_some()
    }
    pub(crate) fn question(&self) -> Result<bool, Error> {
        let Some(Event::Complete { message, .. }) = &self.event else {
            return Ok(false);
        };
        let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64) else {
            return Ok(false);
        };
        let scene = self
            .scene
            .as_deref()
            .ok_or_else(|| Error::new("INVALID_STATE", "child question has no saved scene"))?;
        if !crate::runtime::agent::wait::question(scene, id)? {
            return Err(Error::new(
                "INVALID_STATE",
                "child question has no actual waiting Step",
            ));
        }
        Ok(true)
    }
}

impl Observer {
    pub(crate) fn with_interaction(
        mut self,
        interaction: Option<crate::runtime::agent::reply::Handle>,
    ) -> Self {
        self.interaction = interaction;
        self
    }

    pub(crate) fn suppress_terminal(mut self, suppress: bool) -> Self {
        self.suppress_terminal = suppress;
        self
    }
    pub(crate) async fn emit(
        &self,
        mut event: Option<Event>,
        scene: Option<Arc<Value>>,
        usage: Option<Usage>,
        message: Option<Message>,
        feedback: bool,
    ) {
        if let Some(event) = &mut event {
            crate::runtime::observation::associate(
                event,
                self.chat_id,
                Some((&self.source.0, self.source.1)),
            );
        }
        if self.suppress_terminal
            && event.as_ref().is_some_and(|event| {
                matches!(
                    event,
                    Event::Complete { message, .. }
                        if !message.metadata.contains_key("reply_id")
                )
            })
        {
            event = None;
        }
        let (acknowledgement, consumed) = tokio::sync::oneshot::channel();
        self.relay
            .0
            .events
            .lock()
            .expect("child event lock poisoned")
            .push_back((
                Observation {
                    event,
                    feedback,
                    source: self.source.clone(),
                    scene,
                    usage,
                    message,
                },
                acknowledgement,
            ));
        self.relay.0.consumer.wake();
        let _ = consumed.await;
    }
}
