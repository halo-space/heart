//! Transient handoff for a live run. Only the actual Steps/waits are saved.
use crate::{
    Cancellation, Error, Message,
    agent::Input,
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

mod restore;
pub(crate) use restore::consume as consume_restore;
pub(crate) use restore::duplicate as saved_duplicate;
pub(crate) use restore::prepare as prepare_restore;

#[derive(Default)]
pub(crate) struct Hub(Mutex<BTreeMap<i64, Weak<Inner>>>);

#[derive(Clone)]
pub(crate) struct Run(Arc<Inner>);

/// Owned only by the original stream state, never by a notification.
pub(crate) struct Owner(Run);
impl Drop for Owner {
    fn drop(&mut self) {
        self.0.close();
    }
}

struct Inner {
    metadata: Map<String, Value>,
    questions: Arc<Mutex<BTreeMap<i64, Question>>>,
    /// Active question order is the business-assembled FIFO, shared by the
    /// root and all nested consumers. IDs identify questions; they do not sort it.
    order: Arc<Mutex<VecDeque<i64>>>,
    /// The root owns submission identity; each task polls its own handoff.
    parent: Option<Run>,
    slot: Option<Slot>,
    /// Enclosing planned tasks retain ownership of these permits. A child
    /// reacquires them before continuing, but must never release them itself.
    enclosing: Vec<Slot>,
    children: Mutex<Vec<Weak<Inner>>>,
    suspended: AtomicBool,
    changes: Mutex<VecDeque<Change>>,
    consumer: AtomicWaker,
    closed: AtomicBool,
    restored: Mutex<BTreeMap<String, restore::Question>>,
    restored_tasks: Mutex<BTreeMap<wait::Node, restore::Consumer>>,
    restoring: AtomicBool,
}

struct Question {
    active: bool,
    owner: Weak<Inner>,
    sender: Option<oneshot::Sender<Input>>,
    submitted: Option<Value>,
}

/// Transient task admission, not another runtime scene or persisted queue.
#[derive(Clone)]
pub(crate) struct Slot(Arc<Admission>);
struct Admission {
    slots: Arc<tokio::sync::Semaphore>,
    permit: Mutex<Option<tokio::sync::OwnedSemaphorePermit>>,
    entering: tokio::sync::Mutex<()>,
    changed: Arc<AtomicWaker>,
}
impl Slot {
    pub(crate) fn waiting(slots: Arc<tokio::sync::Semaphore>, changed: Arc<AtomicWaker>) -> Self {
        Self(Arc::new(Admission {
            slots,
            permit: Mutex::new(None),
            entering: tokio::sync::Mutex::new(()),
            changed,
        }))
    }
    pub(crate) fn new(
        slots: Arc<tokio::sync::Semaphore>,
        changed: Arc<AtomicWaker>,
    ) -> Result<Self, Error> {
        let permit = slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::new("CONCURRENCY_LIMIT", "no planned task slot available"))?;
        Ok(Self::with_permit(slots, changed, permit))
    }
    fn with_permit(
        slots: Arc<tokio::sync::Semaphore>,
        changed: Arc<AtomicWaker>,
        permit: tokio::sync::OwnedSemaphorePermit,
    ) -> Self {
        let slot = Self::waiting(slots, changed);
        *slot.0.permit.lock().expect("task slot lock poisoned") = Some(permit);
        slot
    }
    pub(crate) fn held(&self) -> bool {
        self.0
            .permit
            .lock()
            .expect("task slot lock poisoned")
            .is_some()
    }
    pub(crate) fn release(&self) {
        let permit = self
            .0
            .permit
            .lock()
            .expect("task slot lock poisoned")
            .take();
        if permit.is_some() {
            drop(permit);
            self.0.changed.wake();
        }
    }
    pub(crate) async fn acquire(&self, cancellation: &Cancellation) -> Result<(), Error> {
        crate::runtime::cancellation::cancellable(
            async {
                let _entering = self.0.entering.lock().await;
                if !self.held() {
                    let permit =
                        self.0.slots.clone().acquire_owned().await.map_err(|_| {
                            Error::new("CANCELLED", "planned task admission closed")
                        })?;
                    *self.0.permit.lock().expect("task slot lock poisoned") = Some(permit);
                    self.0.changed.wake();
                }
                Ok(())
            },
            cancellation,
        )
        .await
    }
}

pub(crate) enum Change {
    Present {
        call: Call,
        id: i64,
        question: Box<Message>,
        ack: oneshot::Sender<Result<(), Error>>,
    },
    Answered {
        call: Call,
        id: i64,
        input: Input,
        ack: oneshot::Sender<Result<(), Error>>,
    },
}

impl Hub {
    pub(crate) fn ensure_idle(&self, metadata: &Map<String, Value>) -> Result<(), Error> {
        let live = metadata
            .get("chat_id")
            .and_then(Value::as_i64)
            .and_then(|id| {
                self.0
                    .lock()
                    .expect("live reply lock poisoned")
                    .get(&id)
                    .and_then(Weak::upgrade)
            })
            .is_some_and(|run| !run.closed.load(Ordering::Acquire));
        if live {
            Err(Error::new(
                "CONFLICT",
                "this Agent already has a live run for chat_id",
            ))
        } else {
            Ok(())
        }
    }
    pub(crate) fn start(&self, metadata: Map<String, Value>) -> Result<Run, Error> {
        let mut runs = self.0.lock().expect("live reply lock poisoned");
        runs.retain(|_, run| {
            run.upgrade()
                .is_some_and(|run| !run.closed.load(Ordering::Acquire))
        });
        let id = metadata.get("chat_id").and_then(Value::as_i64);
        if id.is_some_and(|id| runs.contains_key(&id)) {
            return Err(Error::new(
                "CONFLICT",
                "this Agent already has a live run for chat_id",
            ));
        }
        let run = Run(Arc::new(Inner {
            metadata,
            questions: Default::default(),
            order: Default::default(),
            parent: None,
            slot: None,
            enclosing: Vec::new(),
            children: Default::default(),
            suspended: AtomicBool::new(false),
            changes: Default::default(),
            consumer: AtomicWaker::new(),
            closed: AtomicBool::new(false),
            restored: Default::default(),
            restored_tasks: Default::default(),
            restoring: AtomicBool::new(false),
        }));
        if let Some(id) = id {
            runs.insert(id, Arc::downgrade(&run.0));
        }
        Ok(run)
    }

    pub(crate) fn submit(&self, input: Input, cancellation: &Cancellation) -> Result<(), Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "reply submission cancelled"));
        }
        let chat = input
            .metadata
            .get("chat_id")
            .and_then(Value::as_i64)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "reply requires chat_id"))?;
        let id = input
            .metadata
            .get("reply_id")
            .and_then(Value::as_i64)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "reply requires reply_id"))?;
        let run = self
            .0
            .lock()
            .expect("live reply lock poisoned")
            .get(&chat)
            .and_then(Weak::upgrade)
            .filter(|run| !run.closed.load(Ordering::Acquire))
            .ok_or_else(|| Error::new("NOT_FOUND", "no live Agent run for this reply"))?;
        for key in ["tenant_id", "user_id", "agent_id", "session_id", "chat_id"] {
            if run
                .metadata
                .get(key)
                .is_some_and(|saved| input.metadata.get(key) != Some(saved))
            {
                return Err(Error::new(
                    "CONFLICT",
                    "reply belongs to a different Agent identity",
                ));
            }
        }
        let mut questions = run.questions.lock().expect("live question lock poisoned");
        if run.closed.load(Ordering::Acquire) {
            return Err(Error::new("NOT_FOUND", "Agent run has ended"));
        }
        let question = questions.get(&id).filter(|q| q.active).ok_or_else(|| {
            Error::new("NOT_FOUND", "reply does not identify a presented question")
        })?;
        // This handoff entry is installed by the actual approval consumer;
        // it does not classify arbitrary user inputs as permission decisions.
        let payload = feedback_signature(&input);
        if let Some(previous) = &question.submitted {
            return if previous == &payload {
                Ok(())
            } else {
                Err(Error::new(
                    "CONFLICT",
                    "this question already received a different reply",
                ))
            };
        }
        let order = run.order.lock().expect("question order lock poisoned");
        let Some(head) = order.front() else {
            return Err(Error::new("NOT_FOUND", "reply is not in the waiting order"));
        };
        if *head != id {
            return Err(Error::new(
                "CONFLICT",
                "reply must target the first waiting question",
            ));
        }
        drop(order);
        approved(&input)?;
        let question = questions.get_mut(&id).expect("validated active question");
        let sender = question
            .sender
            .take()
            .ok_or_else(|| Error::new("NOT_FOUND", "question is no longer receiving feedback"))?;
        question.submitted = Some(payload);
        // Reserve the single submission atomically, but wake the consumer
        // outside the lock: it may immediately request another confirmation.
        drop(questions);
        if sender.send(input).is_err() {
            if let Some(question) = run
                .questions
                .lock()
                .expect("live question lock poisoned")
                .get_mut(&id)
            {
                question.active = false;
                question.submitted = None;
            }
            return Err(Error::new("NOT_FOUND", "question consumer has ended"));
        }
        Ok(())
    }
}

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
    serde_json::to_value(crate::core::agent::input::user_message(input.clone()).content)
        .expect("feedback serialization")
}

impl Run {
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
        let run = Self(Arc::new(Inner {
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
        }));
        self.0
            .children
            .lock()
            .expect("consumer tree lock poisoned")
            .push(Arc::downgrade(&run.0));
        run
    }
    fn closed(&self) -> bool {
        self.0.closed.load(Ordering::Acquire) || self.0.parent.as_ref().is_some_and(Run::closed)
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
                    || (child.suspended.load(Ordering::Acquire) && Run(child).children_blocked())
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
        let (questions, senders) = {
            let mut questions = self
                .0
                .questions
                .lock()
                .expect("live question lock poisoned");
            self.0.closed.store(true, Ordering::Release);
            if self.0.parent.is_none() {
                (std::mem::take(&mut *questions), Vec::new())
            } else {
                // Accepted IDs retain duplicate semantics; cancelled pending
                // IDs remain reserved so a late answer cannot authorize a
                // later question if an application supplies a repeated ID.
                let senders = questions
                    .values_mut()
                    .filter(|question| {
                        question.owner.ptr_eq(&Arc::downgrade(&self.0))
                            && question.submitted.is_none()
                    })
                    .filter_map(|question| {
                        question.active = false;
                        question.sender.take()
                    })
                    .collect::<Vec<_>>();
                (BTreeMap::new(), senders)
            }
        };
        let changes =
            std::mem::take(&mut *self.0.changes.lock().expect("live change lock poisoned"));
        // Dropping channel senders can wake consumers; never do it under a lock.
        drop(questions);
        drop(senders);
        drop(changes);
    }
    fn push(&self, change: Change) {
        self.0
            .changes
            .lock()
            .expect("live change lock poisoned")
            .push_back(change);
        self.0.consumer.wake();
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
        crate::runtime::cancellation::cancellable(
            async {
                presented
                    .await
                    .map_err(|_| Error::new("CANCELLED", "question owner ended"))?
            },
            cancellation,
        )
        .await?;
        let input = crate::runtime::cancellation::cancellable(
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
        crate::runtime::cancellation::cancellable(
            async {
                consumed
                    .await
                    .map_err(|_| Error::new("CANCELLED", "feedback owner ended"))?
            },
            cancellation,
        )
        .await?;
        self.continue_execution(cancellation).await?;
        if decision {
            Ok(())
        } else {
            Err(Error::new("PERMISSION_DENIED", "operation denied by user"))
        }
    }
}

pub(crate) fn apply(
    change: Change,
    steps: &mut [Step],
    waits: &mut Vec<wait::Wait>,
    owner: &Run,
) -> Result<Option<Event>, Error> {
    match change {
        Change::Present {
            call,
            id,
            question,
            ack,
        } => {
            let result = (|| {
                let step = crate::runtime::agent::tool_steps::current(steps, &call.call_id)
                    .ok_or_else(|| {
                        Error::new("INVALID_STATE", "approval has no current Tool Step")
                    })?;
                if let Some(record) = step.waits.get(&id)
                    && (record.kind != "approval"
                        || record.status != "waiting"
                        || record.question != *question
                        || step.call.as_ref() != Some(&call))
                {
                    return Err(Error::new(
                        "CONFLICT",
                        "question identity already exists in Step",
                    ));
                }
                step.call = Some(call);
                step.status = "waiting".into();
                step.waits.entry(id).or_insert_with(|| Wait {
                    kind: "approval".into(),
                    status: "waiting".into(),
                    question: (*question).clone(),
                    feedback: Default::default(),
                });
                let position = wait::Wait {
                    nodes: Vec::new(),
                    step: step.id.clone(),
                };
                if !waits.contains(&position) {
                    waits.push(position);
                }
                {
                    let mut questions = owner
                        .0
                        .questions
                        .lock()
                        .expect("live question lock poisoned");
                    let question = questions.get_mut(&id).expect("registered question");
                    question.active = true;
                    let mut order = owner.0.order.lock().expect("question order lock poisoned");
                    if !order.contains(&id) {
                        order.push_back(id);
                    }
                }
                Ok(Some(Event::Complete {
                    message: *question,
                    usage: None,
                    finish_reason: None,
                }))
            })();
            let _ = ack.send(result.as_ref().map(|_| ()).map_err(Clone::clone));
            result
        }
        Change::Answered {
            call,
            id,
            input,
            ack,
        } => {
            let result = (|| {
                let step = crate::runtime::agent::tool_steps::current(steps, &call.call_id)
                    .filter(|step| step.call.as_ref() == Some(&call))
                    .ok_or_else(|| {
                        Error::new("INVALID_STATE", "approved operation is no longer current")
                    })?;
                let record = step
                    .waits
                    .get_mut(&id)
                    .filter(|wait| wait.kind == "approval" && wait.status == "waiting")
                    .ok_or_else(|| {
                        Error::new("INVALID_STATE", "approval question is no longer waiting")
                    })?;
                if owner
                    .0
                    .order
                    .lock()
                    .expect("question order lock poisoned")
                    .front()
                    != Some(&id)
                {
                    return Err(Error::new(
                        "CONFLICT",
                        "waiting questions must be consumed in order",
                    ));
                }
                record
                    .feedback
                    .push(crate::core::agent::input::user_message(input));
                record.status = "resolved".into();
                step.status = "running".into();
                waits.retain(|position| !position.nodes.is_empty() || position.step != step.id);
                owner
                    .0
                    .order
                    .lock()
                    .expect("question order lock poisoned")
                    .pop_front();
                Ok(None)
            })();
            let _ = ack.send(result.as_ref().map(|_| ()).map_err(Clone::clone));
            result
        }
    }
}
