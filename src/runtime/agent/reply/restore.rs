//! Rebuild transient approval consumers from actual saved Steps.
//! This is not another recovery store or a serialized authorization cache.
use super::*;

pub(crate) struct Restore {
    waits: Vec<wait::Wait>,
    reply_order: Vec<i64>,
    questions: BTreeMap<String, Question>,
    history: BTreeMap<i64, Option<Value>>,
    tasks: BTreeMap<wait::Node, Consumer>,
    feedback: BTreeMap<String, super::feedback::Saved>,
    feedback_history: BTreeMap<i64, super::feedback::Delivery>,
}

#[derive(Default)]
pub(super) struct Consumer {
    waits: Vec<wait::Wait>,
    questions: BTreeMap<String, Question>,
    children: BTreeMap<wait::Node, Consumer>,
    feedback: BTreeMap<String, super::feedback::Saved>,
}

// Follow native current containers only, never arbitrary business JSON or
// older executions. Child scenes are not copied into another recovery store.
pub(super) fn steps(scene: &Value) -> Result<Vec<Step>, Error> {
    wait::validate(scene)?;
    let mut steps: Vec<Step> = serde_json::from_value(
        scene
            .get("steps")
            .cloned()
            .unwrap_or_else(|| serde_json::json!([])),
    )
    .map_err(|_| Error::new("INVALID_STATE", "invalid saved Tool Steps"))?;
    node_steps(scene.get("nodes"), &mut steps)?;
    if let Some(attempt) = scene["execution"]["attempts"]
        .as_array()
        .and_then(|attempts| attempts.last())
    {
        node_steps(attempt.get("nodes"), &mut steps)?;
    }
    Ok(steps)
}

fn node_steps(nodes: Option<&Value>, result: &mut Vec<Step>) -> Result<(), Error> {
    for node in nodes
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|nodes| nodes.values())
    {
        if let Some(attempt) = node["execution"]["attempts"]
            .as_array()
            .and_then(|attempts| attempts.last())
        {
            if let Some(scene) = attempt.get("agent").filter(|scene| !scene.is_null()) {
                result.extend(steps(scene)?);
            } else {
                let current: Vec<Step> = serde_json::from_value(
                    attempt
                        .get("steps")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!([])),
                )
                .map_err(|_| Error::new("INVALID_STATE", "invalid saved task Tool Steps"))?;
                result.extend(current);
            }
            node_steps(attempt.get("nodes"), result)?;
        }
    }
    Ok(())
}

pub(super) struct Question {
    call: Call,
    id: i64,
    message: Message,
    input: Option<Input>,
}

fn same_feedback(steps: &[Step], input: &Input) -> Result<bool, Error> {
    let Some(id) = input.metadata.get("reply_id").and_then(Value::as_i64) else {
        return Ok(false);
    };
    let mut matching = steps.iter().filter_map(|step| step.waits.get(&id));
    let Some(record) = matching.next() else {
        return Ok(false);
    };
    if matching.next().is_some() {
        return Err(Error::new(
            "INVALID_STATE",
            "saved question identity is ambiguous",
        ));
    }
    if record.kind == "input" {
        record.validate()?;
        if record.question.metadata.get("chat_id") != input.metadata.get("chat_id")
            || record
                .question
                .metadata
                .get("reply_id")
                .and_then(Value::as_i64)
                != Some(id)
        {
            return Err(Error::new(
                "INVALID_STATE",
                "saved feedback question identity is invalid",
            ));
        }
        let message = crate::core::agent::input::user_message(input.clone());
        if record
            .feedback
            .as_slice()
            .iter()
            .any(|saved| saved.content == message.content)
        {
            return Ok(true);
        }
        if record.status == "resolved" {
            return Err(Error::new(
                "CONFLICT",
                "this question already received all required feedback",
            ));
        }
        return Ok(false);
    }
    if record.kind != "approval" || record.status != "resolved" {
        return Ok(false);
    }
    record.validate()?;
    if record.question.metadata.get("chat_id") != input.metadata.get("chat_id")
        || record
            .question
            .metadata
            .get("reply_id")
            .and_then(Value::as_i64)
            != Some(id)
        || record.feedback.as_slice().len() != 1
    {
        return Err(Error::new(
            "INVALID_STATE",
            "resolved approval has invalid saved feedback",
        ));
    }
    if record.feedback.as_slice()[0].content
        != crate::core::agent::input::user_message(input.clone()).content
    {
        return Err(Error::new(
            "CONFLICT",
            "this question already received a different reply",
        ));
    }
    approved(input)?;
    Ok(true)
}

pub(crate) fn duplicate(scene: &Value, input: &Input) -> Result<bool, Error> {
    if !input.metadata.contains_key("reply_id") {
        return Ok(false);
    }
    same_feedback(&steps(scene)?, input)
}

pub(crate) fn prepare(scene: &Value, input: &Input) -> Result<Option<Restore>, Error> {
    prepare_input(scene, Some(input))
}

/// A sibling restored by its Workflow owner has no submitted reply yet.
/// Recreate its own native consumer without fabricating external input.
pub(crate) fn pending(scene: &Value) -> Result<Option<Restore>, Error> {
    prepare_input(scene, None)
}

fn prepare_input(scene: &Value, input: Option<&Input>) -> Result<Option<Restore>, Error> {
    if !wait::validate(scene)? {
        return Ok(None);
    }
    let waits: Vec<wait::Wait> = serde_json::from_value(scene["waits"].clone())
        .map_err(|_| Error::new("INVALID_STATE", "invalid saved waits"))?;
    let id = match input.and_then(|input| input.metadata.get("reply_id")) {
        None if input.is_some() => {
            return Err(Error::new(
                "WAITING",
                "Agent is waiting for targeted external feedback",
            ));
        }
        Some(value) => Some(
            value
                .as_i64()
                .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "reply_id must be an i64"))?,
        ),
        None => None,
    };
    let all_steps = steps(scene)?;
    let mut history = BTreeMap::new();
    let mut feedback_history = BTreeMap::new();
    for step in &all_steps {
        for (id, record) in &step.waits {
            record.validate()?;
            if record.kind == "input" {
                if record.question.metadata.get("chat_id") != scene.get("chat_id")
                    || record
                        .question
                        .metadata
                        .get("reply_id")
                        .and_then(Value::as_i64)
                        != Some(*id)
                {
                    return Err(Error::new(
                        "INVALID_STATE",
                        "saved feedback question identity is invalid",
                    ));
                }
                feedback_history.insert(
                    *id,
                    super::feedback::Delivery {
                        count: record.count,
                        submitted: record
                            .feedback
                            .as_slice()
                            .iter()
                            .map(|message| {
                                serde_json::to_value(&message.content)
                                    .expect("feedback serialization")
                            })
                            .collect(),
                        sender: None,
                    },
                );
            }
            let submission = (record.kind == "approval"
                && record.status == "resolved"
                && record.feedback.as_slice().len() == 1
                && record.question.metadata.get("chat_id") == scene.get("chat_id")
                && record
                    .question
                    .metadata
                    .get("reply_id")
                    .and_then(Value::as_i64)
                    == Some(*id))
            .then(|| {
                serde_json::to_value(&record.feedback.as_slice()[0].content)
                    .expect("feedback serialization")
            });
            if history.insert(*id, submission).is_some() {
                return Err(Error::new(
                    "INVALID_STATE",
                    "saved question identity is ambiguous",
                ));
            }
        }
    }
    let mut questions = BTreeMap::new();
    let mut tasks = BTreeMap::<wait::Node, Consumer>::new();
    let duplicate = input
        .map(|input| same_feedback(&all_steps, input))
        .transpose()?
        .unwrap_or(false);
    let mut targeted = duplicate;
    let mut reply_order = Vec::new();
    let mut feedback = BTreeMap::new();
    let mut approval_input = false;
    for position in &waits {
        let step: Step = serde_json::from_value(
            wait::scene_step(scene, &position.nodes, &position.step)?.clone(),
        )
        .map_err(|_| Error::new("INVALID_STATE", "invalid saved waiting Step"))?;
        let unresolved = step
            .waits
            .iter()
            .filter(|(_, record)| record.status == "waiting")
            .collect::<Vec<_>>();
        if step.kind != "tool"
            || !matches!(step.stage.as_deref(), Some("before" | "call" | "after"))
            || unresolved.len() != 1
            || !matches!(unresolved[0].1.kind.as_str(), "approval" | "input")
        {
            return Err(Error::new(
                "WAITING",
                "saved wait requires its dedicated consumer",
            ));
        }
        let call = step
            .call
            .clone()
            .filter(|call| step.call_id.as_ref() == Some(&call.call_id))
            .ok_or_else(|| Error::new("INVALID_STATE", "saved approval has no actual Call"))?;
        let (&reply_id, record) = unresolved[0];
        if (record.kind == "approval" && step.stage.as_deref() == Some("after"))
            || (record.kind == "input" && step.stage.as_deref() == Some("before"))
        {
            return Err(Error::new(
                "INVALID_STATE",
                "saved feedback does not match Tool stage",
            ));
        }
        if record
            .question
            .metadata
            .get("reply_id")
            .and_then(Value::as_i64)
            != Some(reply_id)
            || record.question.metadata.get("chat_id") != scene.get("chat_id")
        {
            return Err(Error::new(
                "INVALID_STATE",
                "saved approval question identity is invalid",
            ));
        }
        reply_order.push(reply_id);
        let selected = Some(reply_id) == id;
        targeted |= selected;
        if record.kind == "input" {
            let mut destination = &mut feedback;
            let mut children = &mut tasks;
            // scene_step above has validated each current containment edge.
            // Rebuild consumers at those exact coordinates, never by Call ID
            // across sibling tasks or children.
            for (index, node) in position.nodes.iter().enumerate() {
                let consumer = children.entry(node.clone()).or_default();
                consumer.waits.push(wait::Wait {
                    nodes: position.nodes[index + 1..].to_vec(),
                    step: position.step.clone(),
                });
                destination = &mut consumer.feedback;
                children = &mut consumer.children;
            }
            if destination
                .insert(
                    call.call_id.clone(),
                    super::feedback::Saved {
                        call,
                        id: reply_id,
                        record: record.clone(),
                        input: input.filter(|_| selected).cloned(),
                    },
                )
                .is_some()
            {
                return Err(Error::new(
                    "INVALID_STATE",
                    "saved feedback operation is ambiguous",
                ));
            }
            continue;
        }
        approval_input |= selected;
        let mut owned_questions = &mut questions;
        let mut children = &mut tasks;
        for (index, node) in position.nodes.iter().enumerate() {
            let consumer = children.entry(node.clone()).or_default();
            consumer.waits.push(wait::Wait {
                nodes: position.nodes[index + 1..].to_vec(),
                step: position.step.clone(),
            });
            owned_questions = &mut consumer.questions;
            children = &mut consumer.children;
        }
        if owned_questions
            .insert(
                call.call_id.clone(),
                Question {
                    call,
                    id: reply_id,
                    message: record.question.clone(),
                    input: input.filter(|_| selected).cloned(),
                },
            )
            .is_some()
        {
            return Err(Error::new(
                "INVALID_STATE",
                "saved approval Call ownership is ambiguous",
            ));
        }
    }
    if input.is_some() && !targeted {
        return Err(Error::new(
            "NOT_FOUND",
            "reply does not identify a saved waiting question",
        ));
    }
    if input.is_some() && !duplicate && reply_order.first().is_some_and(|first| id != Some(*first))
    {
        return Err(Error::new(
            "CONFLICT",
            "replies must be submitted in the waiting question order",
        ));
    }
    if approval_input {
        approved(input.expect("selected approval has submitted input"))?;
    }
    Ok(Some(Restore {
        waits,
        reply_order,
        questions,
        history,
        tasks,
        feedback,
        feedback_history,
    }))
}

impl Run {
    pub(crate) fn restore_approvals(&self, restore: Restore) -> Vec<wait::Wait> {
        self.0.restoring.store(true, Ordering::Release);
        let mut directory = self.0.questions.lock().expect("question lock poisoned");
        // Resolved IDs are tombstones, not reusable grants. Retain their
        // reservation so a reset application generator cannot give an old
        // reply ID to a different operation after recovery.
        for (id, submitted) in restore.history {
            directory.insert(
                id,
                super::Question {
                    active: submitted.is_some(),
                    owner: Arc::downgrade(&self.0),
                    sender: None,
                    submitted,
                    feedback: None,
                },
            );
        }
        for (id, feedback) in restore.feedback_history {
            directory.insert(
                id,
                super::Question {
                    active: feedback.submitted.len() == feedback.count,
                    owner: Arc::downgrade(&self.0),
                    sender: None,
                    submitted: None,
                    feedback: Some(feedback),
                },
            );
        }
        *self
            .0
            .feedback
            .lock()
            .expect("saved feedback lock poisoned") = restore.feedback;
        let mut order = self.0.order.lock().expect("question order lock poisoned");
        if self.0.parent.is_none() {
            *order = restore.reply_order.into_iter().collect();
        } else {
            // An enclosing Workflow can already have restored the global
            // order. A local Agent must never replace it with its own subset.
            for id in restore.reply_order {
                if !order.contains(&id) {
                    order.push_back(id);
                }
            }
        }
        drop(order);
        *self
            .0
            .restored
            .lock()
            .expect("restored question lock poisoned") = restore.questions;
        *self
            .0
            .restored_tasks
            .lock()
            .expect("restored task lock poisoned") = restore.tasks;
        restore.waits
    }

    pub(crate) fn is_restoring(&self) -> bool {
        self.0.restoring.load(Ordering::Acquire)
    }

    /// Transfer only this saved execution's consumers to its actual Run.
    /// No new question, execution or persistence lifecycle is created here.
    pub(crate) fn restore_consumer(&self, node: &wait::Node, task: &Run) -> Vec<wait::Wait> {
        let Some(consumer) = self
            .0
            .restored_tasks
            .lock()
            .expect("restored task lock poisoned")
            .remove(node)
        else {
            return Vec::new();
        };
        *task
            .0
            .restored
            .lock()
            .expect("restored question lock poisoned") = consumer.questions;
        *task
            .0
            .restored_tasks
            .lock()
            .expect("restored consumer lock poisoned") = consumer.children;
        *task
            .0
            .feedback
            .lock()
            .expect("saved feedback lock poisoned") = consumer.feedback;
        task.0.restoring.store(true, Ordering::Release);
        consumer.waits
    }

    async fn resume_approval(
        &self,
        step: &Step,
        cancellation: &Cancellation,
    ) -> Result<Call, Error> {
        let saved = step
            .call
            .as_ref()
            .ok_or_else(|| Error::new("INVALID_STATE", "saved Call missing"))?;
        let question = self
            .0
            .restored
            .lock()
            .expect("restored question lock poisoned")
            .remove(&saved.call_id)
            .filter(|question| &question.call == saved)
            .ok_or_else(|| Error::new("WAITING", "saved approval has no restored consumer"))?;
        let (sender, receiver) = oneshot::channel();
        {
            let mut directory = self.0.questions.lock().expect("question lock poisoned");
            let entry = directory
                .get_mut(&question.id)
                .expect("reserved restored question");
            entry.owner = Arc::downgrade(&self.0);
            entry.active = question.input.is_some();
            entry.sender = question.input.is_none().then_some(sender);
            entry.submitted = question.input.as_ref().map(feedback_signature);
        }
        let input = if let Some(input) = question.input {
            input
        } else {
            let (ack, presented) = oneshot::channel();
            self.push(Change::Present {
                call: saved.clone(),
                id: question.id,
                question: Box::new(question.message),
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
            super::feedback::release_slot();
            crate::runtime::cancellation::cancellable(
                async {
                    receiver
                        .await
                        .map_err(|_| Error::new("CANCELLED", "question consumer ended"))
                },
                cancellation,
            )
            .await?
        };
        let decision = approved(&input)?;
        let (ack, consumed) = oneshot::channel();
        self.push(Change::Answered {
            call: saved.clone(),
            id: question.id,
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
        super::feedback::acquire_slot(cancellation).await?;
        if decision {
            Ok(saved.clone())
        } else {
            Err(Error::new("PERMISSION_DENIED", "operation denied by user"))
        }
    }
}

pub(crate) async fn consume(
    run: Option<&Run>,
    step: &Step,
    cancellation: &Cancellation,
) -> Result<Option<Call>, Error> {
    // A Tool can already have consumed its approval before asking for ordinary
    // input. Recover that exact operation's approval, not a fresh question
    // behind the still-pending input. The caller rechecks the current policy
    // and compares the actual Call, so Deny or changed arguments never inherit it.
    if step.stage.as_deref() == Some("call")
        && step
            .waits
            .values()
            .any(|wait| wait.kind == "input" && wait.status == "waiting")
    {
        let mut authorized = false;
        for record in step.waits.values().filter(|wait| wait.kind == "approval") {
            record.validate()?;
            if record.status != "resolved" {
                return Err(Error::new(
                    "INVALID_STATE",
                    "Tool input precedes its approval",
                ));
            }
            let feedback = &record.feedback.as_slice()[0];
            let input = Input {
                metadata: feedback.metadata.clone(),
                payload: feedback
                    .content
                    .iter()
                    .map(|part| crate::agent::Payload {
                        r#type: part.r#type.clone(),
                        data: Value::Object(part.data.clone()),
                        mime_type: None,
                    })
                    .collect(),
            };
            if !approved(&input)? {
                return Err(Error::new(
                    "PERMISSION_DENIED",
                    "saved operation was denied",
                ));
            }
            authorized = true;
        }
        if authorized {
            return step.call.clone().map(Some).ok_or_else(|| {
                Error::new("INVALID_STATE", "saved approved Tool has no actual Call")
            });
        }
    }
    if !matches!(step.stage.as_deref(), Some("before" | "call"))
        || !step
            .waits
            .values()
            .any(|wait| wait.status == "waiting" && wait.kind == "approval")
    {
        return Ok(None);
    }
    run.ok_or_else(|| Error::new("WAITING", "saved approval has no execution owner"))?
        .resume_approval(step, cancellation)
        .await
        .map(Some)
}
