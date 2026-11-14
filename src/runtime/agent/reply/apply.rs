//! Apply transient reply changes to canonical Steps and waits.

use super::*;

pub(crate) enum Change {
    PermissionStage {
        call: Call,
        state: Box<crate::runtime::agent::permission::State>,
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
    pub(crate) fn start(&self, metadata: Map<String, Value>) -> Result<Handle, Error> {
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
        let run = Handle(Arc::new(RunState {
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

pub(crate) fn apply(
    change: Change,
    steps: &mut [Step],
    waits: &mut Vec<wait::Wait>,
    owner: &Handle,
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
                        || record.status
                            != crate::runtime::agent::node::status::wait::Status::Waiting
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
                step.status = crate::runtime::agent::node::status::step::Status::Waiting;
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
                    .filter(|wait| {
                        wait.kind == "approval"
                            && wait.status
                                == crate::runtime::agent::node::status::wait::Status::Waiting
                    })
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
                step.status = crate::runtime::agent::node::status::step::Status::Running;
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
