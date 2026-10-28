//! Ordinary feedback handoff. Channels are transient; Steps own all facts.
use super::*;
use crate::Messages;
use std::future::Future;
use tokio::sync::mpsc;

pub(super) struct Delivery {
    pub count: usize,
    pub submitted: Vec<Value>,
    pub sender: Option<mpsc::UnboundedSender<Input>>,
}

pub(super) struct Saved {
    pub call: Call,
    pub id: i64,
    pub record: Wait,
    pub input: Option<Input>,
}

#[derive(Clone)]
struct Current {
    run: Option<Run>,
    call: Call,
    saved: Step,
    id_fn: Option<fn() -> i64>,
}

tokio::task_local! {
    static CURRENT: Current;
    static SLOT: Slot;
}

pub(crate) async fn with_slot<F: Future>(slot: Slot, future: F) -> F::Output {
    SLOT.scope(slot, future).await
}

pub(super) fn release_slot() {
    let _ = SLOT.try_with(Slot::release);
}

pub(super) async fn acquire_slot(cancellation: &Cancellation) -> Result<(), Error> {
    if let Ok(slot) = SLOT.try_with(Clone::clone) {
        slot.acquire(cancellation).await?;
    }
    Ok(())
}

/// Poll-local ownership, not a public ToolContext or serialized call frame.
pub(crate) async fn during<F: Future>(
    run: Option<Run>,
    call: Call,
    saved: Step,
    id_fn: Option<fn() -> i64>,
    future: F,
) -> F::Output {
    CURRENT
        .scope(
            Current {
                run,
                call,
                saved,
                id_fn,
            },
            future,
        )
        .await
}

pub(crate) async fn ask(
    question: Message,
    count: usize,
    cancellation: &Cancellation,
) -> Result<Messages, Error> {
    let record = Wait::new("input", question, count)?;
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "feedback wait cancelled"));
    }
    let current = CURRENT.try_with(Clone::clone).map_err(|_| {
        Error::new(
            "INVALID_STATE",
            "feedback requires a current Agent Tool execution",
        )
    })?;
    let run = current
        .run
        .as_ref()
        .ok_or_else(|| Error::new("INVALID_STATE", "feedback requires an active Agent run"))?;
    // Tasks hand progress to their plan owner; delegated Agents forward their
    // native scene through every containing task/child before FIFO advances.
    // Only scopes with their actual owner's progress handoff may enable it.
    let direct_task = !crate::runtime::agent::scope::active()
        && run.0.slot.is_some()
        && run
            .0
            .parent
            .as_ref()
            .is_some_and(|parent| parent.0.parent.is_none());
    let delegated = crate::runtime::agent::scope::feedback();
    if (crate::runtime::agent::scope::active() && !delegated)
        || (run.0.parent.is_some() && !direct_task && !delegated)
    {
        return Err(Error::new(
            "UNSUPPORTED",
            "ordinary feedback requires its owner's native progress handoff",
        ));
    }
    let saved = run
        .0
        .feedback
        .lock()
        .expect("saved feedback lock poisoned")
        .remove(&current.call.call_id);
    if let Some(saved) = saved {
        if saved.call != current.call || !same_question(&saved.record, &record) {
            return Err(Error::new(
                "CONFLICT",
                "saved feedback belongs to a different operation or question",
            ));
        }
        return collect(
            run,
            &current.call,
            saved.id,
            saved.record,
            saved.input,
            true,
            cancellation,
        )
        .await;
    }
    let previous = current
        .saved
        .waits
        .iter()
        .filter(|(_, wait)| wait.kind == "input")
        .collect::<Vec<_>>();
    if !previous.is_empty() {
        if current.saved.call.as_ref() != Some(&current.call) {
            return Err(Error::new(
                "CONFLICT",
                "saved feedback operation arguments changed",
            ));
        }
        let matching = previous
            .iter()
            .filter(|(_, wait)| same_question(wait, &record))
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(Error::new(
                "CONFLICT",
                "saved feedback question is changed or ambiguous",
            ));
        }
        let (_, previous) = *matching[0];
        previous.validate()?;
        if previous.status == "resolved" {
            return Ok(previous.feedback.clone());
        }
        return Err(Error::new(
            "WAITING",
            "saved feedback has no restored consumer",
        ));
    }
    let id = current
        .id_fn
        .ok_or_else(|| Error::new("INVALID_STATE", "feedback requires an id function"))?(
    );
    collect(run, &current.call, id, record, None, false, cancellation).await
}

fn same_question(left: &Wait, right: &Wait) -> bool {
    let mut left_meta = left.question.metadata.clone();
    let mut right_meta = right.question.metadata.clone();
    for key in ["chat_id", "reply_id"] {
        left_meta.remove(key);
        right_meta.remove(key);
    }
    left.kind == right.kind
        && left.count == right.count
        && left.question.role == right.question.role
        && left.question.content == right.question.content
        && left.question.values == right.question.values
        && left_meta == right_meta
}

async fn collect(
    run: &Run,
    call: &Call,
    id: i64,
    mut record: Wait,
    first: Option<Input>,
    restoring: bool,
    cancellation: &Cancellation,
) -> Result<Messages, Error> {
    let chat_id = run
        .0
        .metadata
        .get("chat_id")
        .and_then(Value::as_i64)
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "feedback requires chat_id"))?;
    record
        .question
        .metadata
        .insert("chat_id".into(), chat_id.into());
    record
        .question
        .metadata
        .insert("reply_id".into(), id.into());
    let (sender, mut receiver) = mpsc::unbounded_channel();
    let mut submitted = record
        .feedback
        .as_slice()
        .iter()
        .map(|message| serde_json::to_value(&message.content).expect("feedback serialization"))
        .collect::<Vec<_>>();
    if let Some(input) = &first {
        let signature = feedback_signature(input);
        if !submitted.contains(&signature) {
            submitted.push(signature);
        }
    }
    {
        let mut questions = run.0.questions.lock().expect("question lock poisoned");
        if run.closed() {
            return Err(Error::new("CANCELLED", "feedback owner has ended"));
        }
        if !restoring && questions.contains_key(&id) {
            return Err(Error::new(
                "CONFLICT",
                "id function reused an existing reply_id",
            ));
        }
        questions.insert(
            id,
            Question {
                active: false,
                owner: Arc::downgrade(&run.0),
                sender: None,
                submitted: None,
                feedback: Some(Delivery {
                    count: record.count,
                    submitted,
                    sender: Some(sender),
                }),
            },
        );
    }
    let (ack, presented) = oneshot::channel();
    run.push(Change::Collect {
        call: call.clone(),
        id,
        record: Box::new(record.clone()),
        present: first.is_none(),
        ack,
    });
    crate::runtime::cancellation::cancellable(
        async {
            presented
                .await
                .map_err(|_| Error::new("CANCELLED", "feedback owner ended"))?
        },
        cancellation,
    )
    .await?;
    release_slot();
    let mut first = first;
    loop {
        let input = if let Some(input) = first.take() {
            // Already reserved before the directory becomes active, so a live
            // reply cannot claim this restored submission's capacity.
            input
        } else {
            crate::runtime::cancellation::cancellable(
                async {
                    receiver
                        .recv()
                        .await
                        .ok_or_else(|| Error::new("CANCELLED", "feedback consumer ended"))
                },
                cancellation,
            )
            .await?
        };
        let message = crate::components::agent_input::user_message(input.clone());
        let (ack, consumed) = oneshot::channel();
        run.push(Change::Feedback {
            call: call.clone(),
            id,
            input,
            ack,
        });
        let complete = crate::runtime::cancellation::cancellable(
            async {
                consumed
                    .await
                    .map_err(|_| Error::new("CANCELLED", "feedback owner ended"))?
            },
            cancellation,
        )
        .await?;
        let local_complete = record.reply(message)?;
        if complete != local_complete {
            return Err(Error::new(
                "INVALID_STATE",
                "feedback progress differs from its Step",
            ));
        }
        if complete {
            run.continue_execution(cancellation).await?;
            acquire_slot(cancellation).await?;
            return Ok(record.feedback);
        }
    }
}

pub(super) fn submit(
    run: &Arc<Inner>,
    mut questions: std::sync::MutexGuard<'_, BTreeMap<i64, Question>>,
    id: i64,
    input: Input,
    signature: Value,
) -> Result<(), Error> {
    let delivery = questions
        .get_mut(&id)
        .and_then(|q| q.feedback.as_mut())
        .expect("feedback entry");
    if delivery.submitted.contains(&signature) {
        return Ok(());
    }
    if delivery.submitted.len() >= delivery.count {
        return Err(Error::new(
            "CONFLICT",
            "this question already received all required feedback",
        ));
    }
    if run
        .order
        .lock()
        .expect("question order lock poisoned")
        .front()
        != Some(&id)
    {
        return Err(Error::new(
            "CONFLICT",
            "reply must target the first waiting question",
        ));
    }
    let sender = delivery
        .sender
        .clone()
        .ok_or_else(|| Error::new("NOT_FOUND", "feedback consumer has ended"))?;
    delivery.submitted.push(signature);
    // Enqueue while holding the same short, synchronous submission lock.
    // Dropping it between reservation and send would allow concurrent replies
    // to overtake each other. Tokio's channel send never waits for consumption.
    if sender.send(input).is_err() {
        if let Some(question) = questions.get_mut(&id) {
            question.active = false;
            if let Some(delivery) = question.feedback.as_mut() {
                delivery.submitted.pop();
            }
        }
        return Err(Error::new("NOT_FOUND", "feedback consumer has ended"));
    }
    Ok(())
}

pub(super) fn apply(
    change: Change,
    steps: &mut [Step],
    waits: &mut Vec<wait::Wait>,
    owner: &Run,
) -> Result<Option<Event>, Error> {
    match change {
        Change::Collect {
            call,
            id,
            record,
            present,
            ack,
        } => {
            let result = (|| {
                record.validate()?;
                if record.kind != "input" || record.status != "waiting" {
                    return Err(Error::new(
                        "INVALID_STATE",
                        "invalid ordinary feedback wait",
                    ));
                }
                let step = crate::runtime::agent::tool_steps::current(steps, &call.call_id)
                    .ok_or_else(|| {
                        Error::new("INVALID_STATE", "feedback has no current operation")
                    })?;
                if step
                    .call
                    .as_ref()
                    .is_none_or(|saved| saved.name != call.name)
                    || (step.waits.contains_key(&id) && step.call.as_ref() != Some(&call))
                {
                    return Err(Error::new("CONFLICT", "feedback operation changed"));
                }
                if step
                    .waits
                    .get(&id)
                    .is_some_and(|saved| saved != record.as_ref())
                {
                    return Err(Error::new("CONFLICT", "feedback question changed in Step"));
                }
                let question = record.question.clone();
                step.call = Some(call);
                step.waits.entry(id).or_insert(*record);
                step.status = "waiting".into();
                let position = wait::Wait {
                    nodes: vec![],
                    step: step.id.clone(),
                };
                if !waits.contains(&position) {
                    waits.push(position);
                }
                let mut questions = owner.0.questions.lock().expect("question lock poisoned");
                questions.get_mut(&id).expect("registered feedback").active = true;
                let mut order = owner.0.order.lock().expect("question order lock poisoned");
                if !order.contains(&id) {
                    order.push_back(id);
                }
                Ok(present.then_some(Event::Complete {
                    message: question,
                    usage: None,
                    finish_reason: None,
                }))
            })();
            let _ = ack.send(result.as_ref().map(|_| ()).map_err(Clone::clone));
            result
        }
        Change::Feedback {
            call,
            id,
            input,
            ack,
        } => {
            let result = (|| {
                let step = crate::runtime::agent::tool_steps::current(steps, &call.call_id)
                    .filter(|step| step.call.as_ref() == Some(&call))
                    .ok_or_else(|| {
                        Error::new("INVALID_STATE", "feedback operation is no longer current")
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
                let record = step
                    .waits
                    .get_mut(&id)
                    .filter(|record| record.kind == "input")
                    .ok_or_else(|| {
                        Error::new("INVALID_STATE", "feedback question no longer exists")
                    })?;
                let complete = record.reply(crate::components::agent_input::user_message(input))?;
                if complete {
                    if !step.waits.values().any(|record| record.status == "waiting") {
                        step.status = "running".into();
                        waits.retain(|position| {
                            !position.nodes.is_empty() || position.step != step.id
                        });
                    }
                    if owner.0.parent.is_none() {
                        owner
                            .0
                            .order
                            .lock()
                            .expect("question order lock poisoned")
                            .pop_front();
                    }
                }
                Ok(complete)
            })();
            let _ = ack.send(result.clone());
            result.map(|_| None)
        }
        _ => unreachable!("ordinary feedback changes only"),
    }
}

impl Run {
    /// A planned task must first commit its resolved Step into the outer native
    /// scene. Only then may another task's feedback advance the shared FIFO.
    pub(crate) fn commit_feedback(&self, steps: &[Step]) {
        if self.0.parent.is_some() {
            return;
        }
        let mut order = self.0.order.lock().expect("question order lock poisoned");
        if order.front().is_some_and(|id| {
            steps.iter().any(|step| {
                step.waits
                    .get(id)
                    .is_some_and(|record| record.kind == "input" && record.status == "resolved")
            })
        }) {
            order.pop_front();
        }
    }

    pub(crate) fn commit_scene_feedback(&self, scene: &Value) -> Result<(), Error> {
        if self.0.parent.is_none() {
            self.commit_feedback(&super::restore::steps(scene)?);
        }
        Ok(())
    }
}
