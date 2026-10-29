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

pub(crate) mod feedback;
mod restore;
pub(crate) use restore::consume as consume_restore;
pub(crate) use restore::duplicate as saved_duplicate;
pub(crate) use restore::pending as prepare_pending;
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
    feedback: Mutex<BTreeMap<String, feedback::Saved>>,
}

struct Question {
    active: bool,
    owner: Weak<Inner>,
    sender: Option<oneshot::Sender<Input>>,
    submitted: Option<Value>,
    feedback: Option<feedback::Delivery>,
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
    pub(crate) fn with_permit(
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
    PermissionStage {
        call: Call,
        state: Box<super::permission::State>,
        ack: oneshot::Sender<Result<(), Error>>,
    },
    // Acknowledged in-memory lifecycle facts, never observation Events or
    // independent persistence requests. None means before completed; Some is
    // the raw call result to retain before entering after.
    ToolStage {
        call: Call,
        message: Option<Box<Message>>,
        ack: oneshot::Sender<Result<(), Error>>,
    },
    Collect {
        call: Call,
        id: i64,
        record: Box<Wait>,
        present: bool,
        ack: oneshot::Sender<Result<(), Error>>,
    },
    Feedback {
        call: Call,
        id: i64,
        input: Input,
        ack: oneshot::Sender<Result<bool, Error>>,
    },
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
            feedback: Default::default(),
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
        if question.feedback.is_some() {
            return feedback::submit(&run, questions, id, input, payload);
        }
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
    serde_json::to_value(crate::components::agent_input::user_message(input.clone()).content)
        .expect("feedback serialization")
}

impl Run {
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
            Run(child).close();
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
        crate::runtime::cancellation::cancellable(
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
        crate::runtime::cancellation::cancellable(
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
        crate::runtime::cancellation::cancellable(
            async {
                presented
                    .await
                    .map_err(|_| Error::new("CANCELLED", "question owner ended"))?
            },
            cancellation,
        )
        .await?;
        feedback::release_slot();
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
        feedback::acquire_slot(cancellation).await?;
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
        Change::PermissionStage { call, state, ack } => {
            let result = (|| {
                let step = crate::runtime::agent::step::current(steps, &call.call_id)
                    .ok_or_else(|| Error::new("INVALID_STATE", "permission has no Tool owner"))?;
                if !matches!(step.stage.as_deref(), Some("before" | "call"))
                    || step.call.as_ref() != Some(&call)
                    || state.call()? != call
                {
                    return Err(Error::new(
                        "INVALID_STATE",
                        "permission does not match its Tool owner",
                    ));
                }
                step.permission = Some(state);
                Ok(None)
            })();
            let _ = ack.send(result.as_ref().map(|_| ()).map_err(Clone::clone));
            result
        }
        Change::ToolStage { call, message, ack } => {
            let result = (|| {
                let step = crate::runtime::agent::step::current(steps, &call.call_id)
                    .ok_or_else(|| Error::new("INVALID_STATE", "Tool stage has no current Step"))?;
                let expected = if message.is_some() { "call" } else { "before" };
                if step.stage.as_deref() != Some(expected)
                    || step.call.as_ref().is_none_or(|saved| {
                        saved.name != call.name || (message.is_some() && saved != &call)
                    })
                {
                    return Err(Error::new(
                        "INVALID_STATE",
                        "Tool stage does not match its operation",
                    ));
                }
                if let Some(message) = message {
                    crate::runtime::agent::step::returned(steps, &call, &message);
                } else {
                    crate::runtime::agent::step::prepared(steps, &call);
                }
                Ok(None)
            })();
            let _ = ack.send(result.as_ref().map(|_| ()).map_err(Clone::clone));
            result
        }
        change @ (Change::Collect { .. } | Change::Feedback { .. }) => {
            feedback::apply(change, steps, waits, owner)
        }
        Change::Present {
            call,
            id,
            question,
            ack,
        } => {
            let result = (|| {
                let step = crate::runtime::agent::step::current(steps, &call.call_id).ok_or_else(
                    || Error::new("INVALID_STATE", "approval has no current Tool Step"),
                )?;
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
                if let Some(record) = step.waits.get(&id) {
                    record.validate()?;
                }
                step.call = Some(call);
                step.status = "waiting".into();
                if let std::collections::btree_map::Entry::Vacant(entry) = step.waits.entry(id) {
                    entry.insert(Wait::new("approval", (*question).clone(), 1)?);
                }
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
                let step = crate::runtime::agent::step::current(steps, &call.call_id)
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
                if !record.reply(crate::components::agent_input::user_message(input))? {
                    return Err(Error::new(
                        "INVALID_STATE",
                        "approval feedback is incomplete",
                    ));
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Payload;
    use serde_json::json;
    fn metadata() -> Map<String, Value> {
        Map::from_iter([
            ("chat_id".into(), json!(101)),
            ("session_id".into(), json!(202)),
            ("agent_id".into(), json!(303)),
            ("user_id".into(), json!(404)),
        ])
    }

    fn reply(reply_id: i64, approved: bool) -> Input {
        Input {
            metadata: Map::from_iter([
                ("chat_id".into(), json!(101)),
                ("session_id".into(), json!(202)),
                ("agent_id".into(), json!(303)),
                ("user_id".into(), json!(404)),
                ("reply_id".into(), json!(reply_id)),
            ]),
            payload: vec![Payload {
                r#type: "json".into(),
                data: json!({"value":{"approved":approved}}),
                mime_type: None,
            }],
        }
    }

    fn register(run: &Run, id: i64) -> oneshot::Receiver<Input> {
        let (sender, receiver) = oneshot::channel();
        run.0.questions.lock().unwrap().insert(
            id,
            Question {
                active: true,
                owner: Arc::downgrade(&run.0),
                sender: Some(sender),
                submitted: None,
                feedback: None,
            },
        );
        run.0.order.lock().unwrap().push_back(id);
        receiver
    }

    fn call() -> Call {
        Call {
            call_id: "call-1".into(),
            name: "deploy".into(),
            arguments: Map::from_iter([("version".into(), json!(2))]),
        }
    }

    fn tool_step(call: &Call) -> Step {
        Step {
            id: "tool-0".into(),
            kind: "tool".into(),
            status: "running".into(),
            call_id: Some(call.call_id.clone()),
            stage: Some("before".into()),
            call: Some(call.clone()),
            ..Step::default()
        }
    }

    fn approval_question(reply_id: i64) -> Message {
        let mut question = Message::new(Role::Assistant);
        question.metadata.insert("chat_id".into(), json!(101));
        question.metadata.insert("reply_id".into(), json!(reply_id));
        question
    }

    #[test]
    fn live_reply_submission_enforces_identity_fifo_and_same_reply_idempotency() {
        let hub = Hub::default();
        let run = hub.start(metadata()).unwrap();
        let mut first = register(&run, 501);
        let _second = register(&run, 502);

        let out_of_order = hub.submit(reply(502, true), &Cancellation::new());
        assert_eq!(out_of_order.unwrap_err().code, "CONFLICT");
        assert!(first.try_recv().is_err());

        let mut wrong_owner = reply(501, true);
        wrong_owner.metadata.insert("user_id".into(), json!(999));
        assert_eq!(
            hub.submit(wrong_owner, &Cancellation::new())
                .unwrap_err()
                .code,
            "CONFLICT"
        );

        let accepted = reply(501, true);
        hub.submit(accepted.clone(), &Cancellation::new()).unwrap();
        assert_eq!(first.try_recv().unwrap(), accepted);
        hub.submit(accepted, &Cancellation::new()).unwrap();
        assert_eq!(
            hub.submit(reply(501, false), &Cancellation::new())
                .unwrap_err()
                .code,
            "CONFLICT"
        );
    }

    #[test]
    fn invalid_approval_does_not_consume_the_presented_question() {
        let hub = Hub::default();
        let run = hub.start(metadata()).unwrap();
        let mut receiver = register(&run, 501);
        let mut invalid = reply(501, true);
        invalid.payload[0].data = json!({"value":{"approved":"yes"}});

        assert_eq!(
            hub.submit(invalid, &Cancellation::new()).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
        assert!(receiver.try_recv().is_err());
        assert_eq!(run.0.order.lock().unwrap().front(), Some(&501));
    }

    #[test]
    fn closed_live_run_rejects_late_replies() {
        let hub = Hub::default();
        let run = hub.start(metadata()).unwrap();
        let _receiver = register(&run, 501);
        run.close();

        assert_eq!(
            hub.submit(reply(501, true), &Cancellation::new())
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
        assert!(hub.ensure_idle(&metadata()).is_ok());
    }

    #[test]
    fn one_chat_id_cannot_own_two_live_runs() {
        let hub = Hub::default();
        let first = hub.start(metadata()).unwrap();
        assert_eq!(hub.ensure_idle(&metadata()).unwrap_err().code, "CONFLICT");
        assert_eq!(hub.start(metadata()).err().unwrap().code, "CONFLICT");

        first.close();
        assert!(hub.ensure_idle(&metadata()).is_ok());
        assert!(hub.start(metadata()).is_ok());
    }

    #[tokio::test]
    async fn a_run_rejects_reusing_a_reply_id_for_a_second_question() {
        fn same_id() -> i64 {
            501
        }

        let hub = Hub::default();
        let run = hub.start(metadata()).unwrap();
        let call = call();
        let cancellation = Cancellation::new();
        let mut first = Box::pin(run.ask(&call, "approve", same_id, &cancellation));
        assert!(futures_util::poll!(first.as_mut()).is_pending());

        assert_eq!(
            run.ask(
                &call,
                "approve another operation",
                same_id,
                &Cancellation::new()
            )
            .await
            .unwrap_err()
            .code,
            "CONFLICT"
        );

        let mut context = Context::from_waker(futures_util::task::noop_waker_ref());
        let Poll::Ready(Change::Present {
            id,
            question: presented,
            ack,
            ..
        }) = run.poll_change(&mut context)
        else {
            panic!("the first approval question should be queued for its owner");
        };
        assert_eq!(id, 501);
        assert_eq!(presented.metadata["reply_id"], json!(501));
        drop(ack);
        drop(first);
    }

    #[tokio::test]
    async fn closing_the_root_releases_a_nested_question_consumer() {
        fn id() -> i64 {
            701
        }

        let hub = Hub::default();
        let root = hub.start(metadata()).unwrap();
        let child = root.child();
        let call = call();
        let cancellation = Cancellation::new();
        let mut waiting = Box::pin(child.ask(&call, "approve", id, &cancellation));
        assert!(futures_util::poll!(waiting.as_mut()).is_pending());

        let mut context = Context::from_waker(futures_util::task::noop_waker_ref());
        let Poll::Ready(Change::Present { ack, .. }) = child.poll_change(&mut context) else {
            panic!("the nested question should be queued to its owner");
        };
        ack.send(Ok(())).unwrap();
        assert!(futures_util::poll!(waiting.as_mut()).is_pending());

        root.close();
        assert_eq!(waiting.await.unwrap_err().code, "CANCELLED");
    }

    #[tokio::test]
    async fn closing_one_child_does_not_cancel_its_sibling_question() {
        fn left_id() -> i64 {
            711
        }
        fn right_id() -> i64 {
            712
        }

        let hub = Hub::default();
        let root = hub.start(metadata()).unwrap();
        let left = root.child();
        let left_nested = left.child();
        let right = root.child();
        let call = call();
        let cancellation = Cancellation::new();
        let mut left_wait =
            Box::pin(left_nested.ask(&call, "nested left operation", left_id, &cancellation));
        let mut right_wait = Box::pin(right.ask(&call, "right operation", right_id, &cancellation));
        assert!(futures_util::poll!(left_wait.as_mut()).is_pending());
        assert!(futures_util::poll!(right_wait.as_mut()).is_pending());

        let mut context = Context::from_waker(futures_util::task::noop_waker_ref());
        let Poll::Ready(left_change) = left_nested.poll_change(&mut context) else {
            panic!("left child question should be queued");
        };
        let Poll::Ready(right_change) = right.poll_change(&mut context) else {
            panic!("right child question should be queued");
        };
        let mut left_steps = vec![tool_step(&call)];
        let mut right_steps = vec![tool_step(&call)];
        let mut left_waits = Vec::new();
        let mut right_waits = Vec::new();
        apply(left_change, &mut left_steps, &mut left_waits, &left_nested).unwrap();
        apply(right_change, &mut right_steps, &mut right_waits, &right).unwrap();
        assert!(futures_util::poll!(left_wait.as_mut()).is_pending());
        assert!(futures_util::poll!(right_wait.as_mut()).is_pending());

        left.close();
        assert_eq!(left_wait.await.unwrap_err().code, "CANCELLED");
        hub.submit(reply(712, true), &cancellation).unwrap();
        assert!(futures_util::poll!(right_wait.as_mut()).is_pending());
        let Poll::Ready(Change::Answered { ack, .. }) = right.poll_change(&mut context) else {
            panic!("right child's own reply should reach its owner");
        };
        ack.send(Ok(())).unwrap();
        assert!(right_wait.await.is_ok());
    }

    #[test]
    fn closing_a_child_preserves_duplicate_semantics_for_an_accepted_reply() {
        let hub = Hub::default();
        let root = hub.start(metadata()).unwrap();
        let child = root.child();
        let mut receiver = register(&child, 713);
        let accepted = reply(713, true);

        hub.submit(accepted.clone(), &Cancellation::new()).unwrap();
        assert_eq!(receiver.try_recv().unwrap(), accepted);
        child.close();

        hub.submit(reply(713, true), &Cancellation::new())
            .expect("the same accepted reply remains idempotent after child close");
        assert_eq!(
            hub.submit(reply(713, false), &Cancellation::new())
                .unwrap_err()
                .code,
            "CONFLICT"
        );
        root.close();
    }

    #[tokio::test]
    async fn interleaved_nested_questions_share_fifo_without_crossing_owners() {
        fn first_id() -> i64 {
            721
        }
        fn second_id() -> i64 {
            722
        }
        fn third_id() -> i64 {
            723
        }

        let hub = Hub::default();
        let root = hub.start(metadata()).unwrap();
        let left = root.child();
        let left_nested = left.child();
        let right = root.child();
        let cancellation = Cancellation::new();
        let call = call();
        let mut first = Box::pin(left.ask(&call, "left", first_id, &cancellation));
        let mut second = Box::pin(right.ask(&call, "right", second_id, &cancellation));
        let mut third = Box::pin(left_nested.ask(&call, "nested left", third_id, &cancellation));
        assert!(futures_util::poll!(first.as_mut()).is_pending());
        assert!(futures_util::poll!(second.as_mut()).is_pending());
        assert!(futures_util::poll!(third.as_mut()).is_pending());

        let mut left_steps = vec![tool_step(&call)];
        let mut right_steps = vec![tool_step(&call)];
        let mut nested_steps = vec![tool_step(&call)];
        let mut left_waits = Vec::new();
        let mut right_waits = Vec::new();
        let mut nested_waits = Vec::new();
        let mut context = Context::from_waker(futures_util::task::noop_waker_ref());
        for (owner, steps, waits) in [
            (&left, &mut left_steps, &mut left_waits),
            (&right, &mut right_steps, &mut right_waits),
            (&left_nested, &mut nested_steps, &mut nested_waits),
        ] {
            let Poll::Ready(change) = owner.poll_change(&mut context) else {
                panic!("each nested question should reach its native owner");
            };
            let event = apply(change, steps, waits, owner).unwrap();
            assert!(
                matches!(event, Some(Event::Complete { message, .. }) if message.metadata.contains_key("reply_id"))
            );
        }
        assert_eq!(
            *root.0.order.lock().unwrap(),
            VecDeque::from([721, 722, 723])
        );

        assert_eq!(
            hub.submit(reply(722, true), &cancellation)
                .unwrap_err()
                .code,
            "CONFLICT",
            "a later sibling cannot skip the first question in the shared FIFO"
        );
        hub.submit(reply(721, true), &cancellation).unwrap();
        assert!(futures_util::poll!(first.as_mut()).is_pending());
        let Poll::Ready(answered) = left.poll_change(&mut context) else {
            panic!("the first reply should be delivered only to its left owner");
        };
        assert!(matches!(answered, Change::Answered { id: 721, .. }));
        apply(answered, &mut left_steps, &mut left_waits, &left).unwrap();
        assert!(first.await.is_ok());
        assert!(futures_util::poll!(second.as_mut()).is_pending());
        assert!(futures_util::poll!(third.as_mut()).is_pending());

        hub.submit(reply(722, true), &cancellation).unwrap();
        assert!(futures_util::poll!(second.as_mut()).is_pending());
        let Poll::Ready(answered) = right.poll_change(&mut context) else {
            panic!("the second reply should be delivered only to its right owner");
        };
        assert!(matches!(answered, Change::Answered { id: 722, .. }));
        apply(answered, &mut right_steps, &mut right_waits, &right).unwrap();
        assert!(second.await.is_ok());
        hub.submit(reply(723, true), &cancellation).unwrap();
        assert!(futures_util::poll!(third.as_mut()).is_pending());
        let Poll::Ready(answered) = left_nested.poll_change(&mut context) else {
            panic!("the third reply should be delivered only to its nested left owner");
        };
        assert!(matches!(answered, Change::Answered { id: 723, .. }));
        apply(answered, &mut nested_steps, &mut nested_waits, &left_nested).unwrap();
        assert!(third.await.is_ok());
        assert!(root.0.order.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn planned_task_slot_is_released_while_waiting_and_reacquired_before_continuation() {
        let permits = Arc::new(tokio::sync::Semaphore::new(1));
        let changed = Arc::new(AtomicWaker::new());
        let slot = Slot::new(Arc::clone(&permits), Arc::clone(&changed)).unwrap();
        assert!(slot.held());
        assert_eq!(permits.available_permits(), 0);

        feedback::with_slot(slot.clone(), async {
            feedback::release_slot();
            assert!(!slot.held());
            assert_eq!(permits.available_permits(), 1);

            slot.acquire(&Cancellation::new()).await.unwrap();
            assert!(slot.held());
            assert_eq!(permits.available_permits(), 0);
        })
        .await;
    }

    #[tokio::test]
    async fn cancelled_capacity_reacquisition_does_not_fabricate_a_held_slot() {
        let permits = Arc::new(tokio::sync::Semaphore::new(1));
        let changed = Arc::new(AtomicWaker::new());
        let slot = Slot::new(Arc::clone(&permits), changed).unwrap();
        slot.release();
        let _other_owner = permits.clone().acquire_owned().await.unwrap();
        let cancellation = Cancellation::new();
        cancellation.cancel();

        assert_eq!(
            slot.acquire(&cancellation).await.unwrap_err().code,
            "CANCELLED"
        );
        assert!(!slot.held());
        assert_eq!(permits.available_permits(), 0);
    }

    #[test]
    fn presenting_an_approval_commits_its_actual_tool_step_and_fifo_locator() {
        let hub = Hub::default();
        let run = hub.start(metadata()).unwrap();
        let _receiver = register(&run, 501);
        let call = call();
        let question = approval_question(501);
        let (ack, mut acknowledged) = oneshot::channel();
        let change = Change::Present {
            call: call.clone(),
            id: 501,
            question: Box::new(question.clone()),
            ack,
        };
        let mut steps = vec![tool_step(&call)];
        let mut waits = Vec::new();

        let event = apply(change, &mut steps, &mut waits, &run)
            .unwrap()
            .expect("the saved approval is presented as an event");

        assert!(matches!(event, Event::Complete { message, .. } if message == question));
        assert_eq!(acknowledged.try_recv().unwrap().unwrap(), ());
        assert_eq!(steps[0].status, "waiting");
        assert_eq!(steps[0].call.as_ref(), Some(&call));
        assert_eq!(steps[0].waits[&501].kind, "approval");
        assert_eq!(waits.len(), 1);
        assert_eq!(waits[0].step, "tool-0");
        assert_eq!(
            run.0
                .order
                .lock()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            [501]
        );
    }

    #[test]
    fn answer_transition_rejects_non_fifo_approval_without_mutating_runtime_facts() {
        let hub = Hub::default();
        let run = hub.start(metadata()).unwrap();
        let _first = register(&run, 501);
        let _second = register(&run, 502);
        let call = call();
        let mut step = tool_step(&call);
        for id in [501, 502] {
            step.waits
                .insert(id, Wait::new("approval", approval_question(id), 1).unwrap());
        }
        let mut steps = vec![step];
        let mut waits = vec![crate::runtime::agent::wait::Wait {
            nodes: Vec::new(),
            step: "tool-0".into(),
        }];
        let (ack, mut acknowledged) = oneshot::channel();

        let result = apply(
            Change::Answered {
                call,
                id: 502,
                input: reply(502, true),
                ack,
            },
            &mut steps,
            &mut waits,
            &run,
        );

        assert_eq!(result.unwrap_err().code, "CONFLICT");
        assert_eq!(
            acknowledged.try_recv().unwrap().unwrap_err().code,
            "CONFLICT"
        );
        assert_eq!(steps[0].status, "running");
        assert_eq!(steps[0].waits[&502].status, "waiting");
        assert_eq!(waits.len(), 1);
        assert_eq!(
            run.0
                .order
                .lock()
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            [501, 502]
        );
    }

    #[test]
    fn answer_transition_consumes_only_the_matching_fifo_head_and_records_feedback() {
        let hub = Hub::default();
        let run = hub.start(metadata()).unwrap();
        let _receiver = register(&run, 501);
        let call = call();
        let mut step = tool_step(&call);
        step.status = "waiting".into();
        step.waits.insert(
            501,
            Wait::new("approval", approval_question(501), 1).unwrap(),
        );
        let mut steps = vec![step];
        let mut waits = vec![crate::runtime::agent::wait::Wait {
            nodes: Vec::new(),
            step: "tool-0".into(),
        }];
        let (ack, mut acknowledged) = oneshot::channel();

        assert!(
            apply(
                Change::Answered {
                    call,
                    id: 501,
                    input: reply(501, true),
                    ack,
                },
                &mut steps,
                &mut waits,
                &run,
            )
            .unwrap()
            .is_none()
        );

        assert_eq!(acknowledged.try_recv().unwrap().unwrap(), ());
        assert_eq!(steps[0].status, "running");
        assert_eq!(steps[0].waits[&501].status, "resolved");
        assert_eq!(steps[0].waits[&501].feedback.as_slice().len(), 1);
        assert!(waits.is_empty());
        assert!(run.0.order.lock().unwrap().is_empty());
    }
}
