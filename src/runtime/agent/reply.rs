//! Transient handoff for a live run. Only the actual Steps/waits are saved.
use crate::{
    Cancellation, Error, Message,
    components::agent::Input,
    event::Event,
    message::Role,
    model::content::part::Part,
    runtime::agent::{
        node::{Step, Wait},
        wait,
    },
    tool::Call,
};
use futures_util::task::AtomicWaker;
use serde_json::{Map, Value};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
};
use tokio::sync::oneshot;

mod apply;
pub(crate) mod feedback;
mod restore;
mod slot;

pub(crate) use apply::{Change, apply};
pub(crate) use restore::consume as consume_restore;
pub(crate) use restore::duplicate as reply_duplicate;
pub(crate) use restore::pending as prepare_pending;
pub(crate) use restore::prepare as prepare_restore;
pub(crate) use slot::Slot;

#[derive(Default)]
pub(crate) struct Hub(Mutex<BTreeMap<i64, Weak<RunState>>>);

#[derive(Clone)]
#[doc(hidden)]
pub struct Handle(Arc<RunState>);

/// Owned only by the original stream state, never by a notification.
pub(crate) struct Owner(Handle);
impl Drop for Owner {
    fn drop(&mut self) {
        self.0.close();
    }
}

struct RunState {
    metadata: Map<String, Value>,
    questions: Arc<Mutex<BTreeMap<i64, Question>>>,
    /// Active question order is the business-assembled FIFO, shared by the
    /// root and all nested consumers. IDs identify questions; they do not sort it.
    order: Arc<Mutex<VecDeque<i64>>>,
    /// The root owns submission identity; each task polls its own handoff.
    parent: Option<Handle>,
    slot: Option<Slot>,
    /// Enclosing planned tasks retain ownership of these permits. A child
    /// reacquires them before continuing, but must never release them itself.
    enclosing: Vec<Slot>,
    children: Mutex<Vec<Weak<RunState>>>,
    suspended: AtomicBool,
    changes: Mutex<VecDeque<Change>>,
    consumer: AtomicWaker,
    closed: AtomicBool,
    restored: Mutex<BTreeMap<String, restore::Question>>,
    restored_tasks: Mutex<BTreeMap<wait::Node, restore::Consumer>>,
    restoring: AtomicBool,
    feedback: Mutex<BTreeMap<String, feedback::Saved>>,
}

struct Question {
    active: bool,
    owner: Weak<RunState>,
    sender: Option<oneshot::Sender<Input>>,
    submitted: Option<Value>,
    feedback: Option<feedback::Delivery>,
}

/// Transient task admission, not another runtime scene or persisted queue.
fn approved(input: &Input) -> Result<bool, Error> {
    let mut decisions = input
        .payload
        .iter()
        .filter(|part| part.r#type == "json")
        .filter_map(|part| {
            part.data
                .get("value")
                .and_then(|value| value.get("approved"))
        });
    let decision = decisions.next().and_then(Value::as_bool).ok_or_else(|| {
        Error::new(
            "INVALID_ARGUMENTS",
            "approval requires a boolean data.value.approved",
        )
    })?;
    if decisions.next().is_some() {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "approval must contain one unambiguous decision",
        ));
    }
    Ok(decision)
}

fn feedback_signature(input: &Input) -> Value {
    // Feedback is saved as Messages, not the transport Input envelope. Use
    // that same projection live and after recovery (including MIME layout).
    serde_json::to_value(crate::components::agent_input::user_message(input.clone()).content)
        .expect("feedback serialization")
}

impl Handle {
    /// Seed only from a validated outer State's existing ordered locators.
    /// Consumer ownership/authorization still comes from the actual Steps.
    pub(crate) fn restore_order(&self, ids: Vec<i64>) {
        *self.0.order.lock().expect("question order lock poisoned") = ids.into_iter().collect();
    }
    /// Same run/question directory, separate actual execution consumer. Not
    /// another public run, lifecycle, snapshot or output channel.
    pub(crate) fn branch(&self, slot: Slot) -> Self {
        self.consumer(Some(slot))
    }
    pub(crate) fn child(&self) -> Self {
        self.consumer(None)
    }
    fn consumer(&self, slot: Option<Slot>) -> Self {
        let mut enclosing = self.0.enclosing.clone();
        if let Some(slot) = &self.0.slot {
            enclosing.push(slot.clone());
        }
        let run = Self(Arc::new(RunState {
            metadata: self.0.metadata.clone(),
            questions: self.0.questions.clone(),
            order: self.0.order.clone(),
            parent: Some(self.clone()),
            slot,
            enclosing,
            children: Default::default(),
            suspended: AtomicBool::new(false),
            changes: Default::default(),
            consumer: AtomicWaker::new(),
            closed: AtomicBool::new(false),
            restored: Default::default(),
            restored_tasks: Default::default(),
            restoring: AtomicBool::new(false),
            feedback: Default::default(),
        }));
        self.0
            .children
            .lock()
            .expect("consumer tree lock poisoned")
            .push(Arc::downgrade(&run.0));
        run
    }
    fn closed(&self) -> bool {
        self.0.closed.load(Ordering::Acquire) || self.0.parent.as_ref().is_some_and(Handle::closed)
    }
    pub(crate) fn suspend(&self) {
        self.0.suspended.store(true, Ordering::Release);
        if self.children_blocked()
            && let Some(slot) = &self.0.slot
        {
            slot.release();
        }
    }
    pub(crate) fn children_blocked(&self) -> bool {
        let children = self
            .0
            .children
            .lock()
            .expect("consumer tree lock poisoned")
            .clone();
        children.into_iter().all(|child| {
            child.upgrade().is_none_or(|child| {
                child.closed.load(Ordering::Acquire)
                    || (child.suspended.load(Ordering::Acquire) && Handle(child).children_blocked())
            })
        })
    }
    pub(crate) async fn continue_execution(
        &self,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        // Publish runnable intent before awaiting capacity, so an enclosing
        // consumer cannot release a permit acquired by a resumed descendant.
        self.0.suspended.store(false, Ordering::Release);
        for slot in &self.0.enclosing {
            slot.acquire(cancellation).await?;
        }
        if let Some(slot) = &self.0.slot {
            slot.acquire(cancellation).await?;
        }
        Ok(())
    }
    pub(crate) fn owner(&self) -> Owner {
        Owner(self.clone())
    }
    pub(crate) fn close(&self) {
        if let Some(slot) = &self.0.slot {
            slot.release();
        }
        let (questions, senders, closed_ids) = {
            let mut questions = self
                .0
                .questions
                .lock()
                .expect("live question lock poisoned");
            self.0.closed.store(true, Ordering::Release);
            if self.0.parent.is_none() {
                (std::mem::take(&mut *questions), Vec::new(), Vec::new())
            } else {
                // Accepted IDs retain duplicate semantics; cancelled pending
                // IDs remain reserved so a late answer cannot authorize a
                // later question if an application supplies a repeated ID.
                let mut senders = Vec::new();
                let mut closed_ids = Vec::new();
                for (id, question) in questions.iter_mut() {
                    if question.owner.ptr_eq(&Arc::downgrade(&self.0))
                        && question.submitted.is_none()
                        && question
                            .feedback
                            .as_ref()
                            .is_none_or(|delivery| delivery.submitted.len() < delivery.count)
                    {
                        closed_ids.push(*id);
                        question.active = false;
                        senders.push((question.sender.take(), question.feedback.take()));
                    }
                }
                (BTreeMap::new(), senders, closed_ids)
            }
        };
        let changes =
            std::mem::take(&mut *self.0.changes.lock().expect("live change lock poisoned"));
        // Dropping channel senders can wake consumers; never do it under a lock.
        drop(questions);
        if !closed_ids.is_empty() {
            let mut order = self.0.order.lock().expect("question order lock poisoned");
            order.retain(|id| !closed_ids.contains(id));
        }
        drop(senders);
        drop(changes);
        let children = self
            .0
            .children
            .lock()
            .expect("consumer tree lock poisoned")
            .iter()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        for child in children {
            Handle(child).close();
        }
    }
    fn push(&self, change: Change) {
        self.0
            .changes
            .lock()
            .expect("live change lock poisoned")
            .push_back(change);
        self.0.consumer.wake();
    }
    pub(crate) async fn tool_stage(
        &self,
        call: Call,
        message: Option<Message>,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        if self.closed() {
            return Err(Error::new("CANCELLED", "Tool stage owner ended"));
        }
        let (ack, applied) = oneshot::channel();
        self.push(Change::ToolStage {
            call,
            message: message.map(Box::new),
            ack,
        });
        crate::cancellation::cancellable(
            async {
                applied
                    .await
                    .map_err(|_| Error::new("CANCELLED", "Tool stage owner ended"))?
            },
            cancellation,
        )
        .await
    }
    pub(crate) async fn permission_stage(
        &self,
        call: Call,
        state: super::permission::State,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        if self.closed() {
            return Err(Error::new("CANCELLED", "permission owner ended"));
        }
        let (ack, applied) = oneshot::channel();
        self.push(Change::PermissionStage {
            call,
            state: Box::new(state),
            ack,
        });
        crate::cancellation::cancellable(
            async {
                applied
                    .await
                    .map_err(|_| Error::new("CANCELLED", "permission owner ended"))?
            },
            cancellation,
        )
        .await
    }
    pub(crate) fn poll_change(&self, cx: &mut Context<'_>) -> Poll<Change> {
        self.0.consumer.register(cx.waker());
        match self
            .0
            .changes
            .lock()
            .expect("live change lock poisoned")
            .pop_front()
        {
            Some(change) => Poll::Ready(change),
            None => Poll::Pending,
        }
    }
    pub(crate) async fn ask(
        &self,
        call: &Call,
        reason: &str,
        id_fn: fn() -> i64,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        let chat_id = self
            .0
            .metadata
            .get("chat_id")
            .and_then(Value::as_i64)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "live approval requires chat_id"))?;
        let id = id_fn();
        let (sender, receiver) = oneshot::channel();
        {
            let mut questions = self
                .0
                .questions
                .lock()
                .expect("live question lock poisoned");
            if self.closed() {
                return Err(Error::new("CANCELLED", "question owner has ended"));
            }
            if questions.contains_key(&id) {
                return Err(Error::new(
                    "CONFLICT",
                    "id function reused an existing reply_id",
                ));
            }
            questions.insert(
                id,
                Question {
                    active: false,
                    owner: Arc::downgrade(&self.0),
                    sender: Some(sender),
                    submitted: None,
                    feedback: None,
                },
            );
        }
        let mut question = Message::new(Role::Assistant);
        question.content.push(Part { r#type: "text".into(), data: serde_json::json!({
            "value": format!("{reason}\nTool: {}\nArguments: {}", call.name, serde_json::to_string(&call.arguments).expect("Call serialization"))
        }).as_object().expect("object").clone() });
        question.metadata.insert("chat_id".into(), chat_id.into());
        question.metadata.insert("reply_id".into(), id.into());
        let (ack, presented) = oneshot::channel();
        self.push(Change::Present {
            call: call.clone(),
            id,
            question: Box::new(question),
            ack,
        });
        crate::cancellation::cancellable(
            async {
                presented
                    .await
                    .map_err(|_| Error::new("CANCELLED", "question owner ended"))?
            },
            cancellation,
        )
        .await?;
        feedback::release_slot();
        let input = crate::cancellation::cancellable(
            async {
                receiver
                    .await
                    .map_err(|_| Error::new("CANCELLED", "question consumer ended"))
            },
            cancellation,
        )
        .await?;
        let decision = approved(&input)?;
        let (ack, consumed) = oneshot::channel();
        self.push(Change::Answered {
            call: call.clone(),
            id,
            input,
            ack,
        });
        crate::cancellation::cancellable(
            async {
                consumed
                    .await
                    .map_err(|_| Error::new("CANCELLED", "feedback owner ended"))?
            },
            cancellation,
        )
        .await?;
        self.continue_execution(cancellation).await?;
        feedback::acquire_slot(cancellation).await?;
        if decision {
            Ok(())
        } else {
            Err(Error::new("PERMISSION_DENIED", "operation denied by user"))
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_runtime_agent_reply_tests.rs"]
mod tests;
