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
        let message = crate::components::agent_input::user_message(input.clone());
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
        != crate::components::agent_input::user_message(input.clone()).content
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, agent::Payload, message::Role, runtime::agent::node};
    use serde_json::json;
    use std::collections::BTreeMap;

    const CHAT_ID: i64 = 41;
    const REPLY_ID: i64 = 701;

    fn question(reply_id: i64) -> Message {
        let mut message = Message::new(Role::Assistant);
        message.metadata.insert("chat_id".into(), CHAT_ID.into());
        message.metadata.insert("reply_id".into(), reply_id.into());
        message
            .values
            .insert("question".into(), json!("请批准操作"));
        message
    }

    fn feedback(approved: bool) -> Message {
        let mut message = Message::new(Role::User);
        message
            .content
            .push(crate::components::model::content::part::Part {
                r#type: "json".into(),
                data: json!({"value":{"approved":approved}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        message
    }

    fn approval_scene(reply_id: i64, resolved: Option<bool>) -> Value {
        let mut record = node::Wait::new("approval", question(reply_id), 1).unwrap();
        if let Some(approved) = resolved {
            record.reply(feedback(approved)).unwrap();
        }
        let call = Call {
            call_id: "call-1".into(),
            name: "publish".into(),
            arguments: serde_json::Map::new(),
        };
        let step = node::Step {
            id: "tool-1".into(),
            kind: "tool".into(),
            status: if resolved.is_some() {
                "running"
            } else {
                "waiting"
            }
            .into(),
            stage: Some("before".into()),
            call_id: Some(call.call_id.clone()),
            call: Some(call),
            waits: BTreeMap::from([(reply_id, record)]),
            ..Default::default()
        };
        json!({
            "type":"agent",
            "chat_id":CHAT_ID,
            "status":if resolved.is_some() { "running" } else { "waiting" },
            "steps":[step],
            "waits":if resolved.is_some() { json!([]) } else { json!([{"nodes":[],"step":"tool-1"}]) },
            "nodes":{}
        })
    }

    fn submitted(reply_id: i64, approved: bool) -> Input {
        Input {
            metadata: serde_json::Map::from_iter([
                ("chat_id".into(), CHAT_ID.into()),
                ("reply_id".into(), reply_id.into()),
            ]),
            payload: vec![Payload {
                r#type: "json".into(),
                data: json!({"value":{"approved":approved}}),
                mime_type: None,
            }],
        }
    }

    fn sibling_child_scene() -> Value {
        let mut nodes = BTreeMap::new();
        let mut waits = Vec::new();
        for (node_id, exec_id, reply_id) in [("left", 77, REPLY_ID), ("right", 78, REPLY_ID + 1)] {
            let record = node::Wait::new("approval", question(reply_id), 1).unwrap();
            let call = Call {
                call_id: "shared-call".into(),
                name: "publish".into(),
                arguments: serde_json::Map::new(),
            };
            let step = node::Step {
                id: "tool-1".into(),
                kind: "tool".into(),
                status: "waiting".into(),
                stage: Some("before".into()),
                call_id: Some(call.call_id.clone()),
                call: Some(call),
                waits: BTreeMap::from([(reply_id, record.clone())]),
                ..Default::default()
            };
            let child = crate::runtime::agent::State {
                chat_id: Some(CHAT_ID),
                r#type: "agent".into(),
                status: "waiting".into(),
                steps: vec![step],
                waits: vec![wait::Wait {
                    nodes: Vec::new(),
                    step: "tool-1".into(),
                }],
                ..Default::default()
            };
            let attempt = node::Attempt {
                idx: 0,
                status: node::Status::Waiting,
                created_time: None,
                finished_time: None,
                usage: None,
                finish_reason: None,
                steps: Vec::new(),
                messages: Default::default(),
                completion: None,
                nodes: BTreeMap::new(),
                agent: Some(serde_json::to_value(child).unwrap()),
                error: None,
            };
            nodes.insert(
                node_id.to_owned(),
                node::State {
                    status: node::Status::Waiting,
                    execution: Some(node::Execution {
                        exec_id,
                        attempts: vec![attempt],
                        messages: Default::default(),
                        usage: None,
                        error: None,
                    }),
                    ..Default::default()
                },
            );
            waits.push(wait::Wait {
                nodes: vec![wait::Node {
                    node_id: node_id.to_owned(),
                    exec_id,
                    idx: 0,
                }],
                step: "tool-1".into(),
            });
        }
        let root = crate::runtime::agent::State {
            chat_id: Some(CHAT_ID),
            r#type: "agent".into(),
            status: "waiting".into(),
            nodes,
            waits,
            ..Default::default()
        };
        serde_json::to_value(root).unwrap()
    }

    fn planned_sibling_child_scene() -> Value {
        let mut scene = sibling_child_scene();
        for (node_id, task_exec_id) in [("left", 88), ("right", 89)] {
            let node = &mut scene["nodes"][node_id];
            let attempt = &mut node["execution"]["attempts"][0];
            let child = std::mem::replace(&mut attempt["agent"], Value::Null);
            attempt["nodes"] = json!({"task-1": {
                "status":"waiting",
                "execution": {
                    "exec_id":task_exec_id,
                    "attempts":[{
                        "idx":0,
                        "status":"waiting",
                        "steps":[],
                        "nodes":{},
                        "agent":child
                    }]
                }
            }});
            scene["waits"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|wait| wait["nodes"][0]["node_id"] == node_id)
                .unwrap()["nodes"]
                .as_array_mut()
                .unwrap()
                .push(json!({"node_id":"task-1","exec_id":task_exec_id,"idx":0}));
        }
        scene
    }

    #[test]
    fn restore_parses_the_actual_wait_step_and_preserves_reply_identity() {
        let scene = approval_scene(REPLY_ID, None);
        let restore = prepare(&scene, &submitted(REPLY_ID, true))
            .unwrap()
            .expect("active approval is restored");

        assert_eq!(restore.reply_order, [REPLY_ID]);
        let question = restore.questions.get("call-1").unwrap();
        assert_eq!(question.id, REPLY_ID);
        assert_eq!(question.call.name, "publish");
        assert_eq!(
            question.input.as_ref().unwrap().metadata["reply_id"],
            REPLY_ID
        );
    }

    #[test]
    fn checkpoint_json_round_trip_restores_the_same_pending_approval() {
        let saved = approval_scene(REPLY_ID, None);
        let bytes = serde_json::to_vec(&saved).expect("serialize saved Runtime State");
        let restored: Value = serde_json::from_slice(&bytes).expect("deserialize Runtime State");
        let prepared = prepare(&restored, &submitted(REPLY_ID, true))
            .unwrap()
            .expect("saved pending approval can be rebuilt from persisted State");

        assert_eq!(prepared.reply_order, [REPLY_ID]);
        let question = prepared.questions.get("call-1").unwrap();
        assert_eq!(question.id, REPLY_ID);
        assert_eq!(question.call.name, "publish");
        assert!(question.call.arguments.is_empty());
    }

    #[test]
    fn restore_rejects_missing_or_unknown_reply_identity_without_dispatching() {
        let scene = approval_scene(REPLY_ID, None);
        let mut untargeted = submitted(REPLY_ID, true);
        untargeted.metadata.remove("reply_id");
        assert_eq!(prepare(&scene, &untargeted).err().unwrap().code, "WAITING");

        assert_eq!(
            prepare(&scene, &submitted(REPLY_ID + 1, true))
                .err()
                .unwrap()
                .code,
            "NOT_FOUND"
        );
    }

    #[test]
    fn restore_requires_the_saved_fifo_head_before_accepting_feedback() {
        let mut scene = approval_scene(REPLY_ID, None);
        let second_id = REPLY_ID + 1;
        let mut second = scene["steps"][0].clone();
        second["id"] = json!("tool-2");
        second["call_id"] = json!("call-2");
        second["call"]["call_id"] = json!("call-2");
        second["waits"] =
            json!({second_id.to_string(): second["waits"][REPLY_ID.to_string()].clone()});
        second["waits"][second_id.to_string()]["question"]["metadata"]["reply_id"] =
            json!(second_id);
        scene["steps"].as_array_mut().unwrap().push(second);
        scene["waits"]
            .as_array_mut()
            .unwrap()
            .push(json!({"nodes":[],"step":"tool-2"}));

        let error = prepare(&scene, &submitted(second_id, true)).err().unwrap();
        assert_eq!(error.code, "CONFLICT", "{}", error.message);
    }

    #[test]
    fn restored_approval_is_idempotent_only_for_the_same_saved_feedback() {
        let scene = approval_scene(REPLY_ID, Some(true));
        assert!(duplicate(&scene, &submitted(REPLY_ID, true)).unwrap());
        assert_eq!(
            duplicate(&scene, &submitted(REPLY_ID, false))
                .unwrap_err()
                .code,
            "CONFLICT"
        );
    }

    #[test]
    fn restore_does_not_turn_a_denial_into_an_approval() {
        let scene = approval_scene(REPLY_ID, None);
        let restore = prepare(&scene, &submitted(REPLY_ID, false))
            .unwrap()
            .expect("denial is still a valid reply to restore");

        assert_eq!(restore.reply_order, [REPLY_ID]);
        assert_eq!(
            restore
                .questions
                .get("call-1")
                .unwrap()
                .input
                .as_ref()
                .unwrap()
                .payload[0]
                .data["value"]["approved"],
            false
        );
    }

    #[tokio::test]
    async fn recovered_approval_cannot_authorize_a_changed_tool_call() {
        let scene = approval_scene(REPLY_ID, None);
        let restore = prepare(&scene, &submitted(REPLY_ID, true))
            .unwrap()
            .expect("pending approval is restorable");
        let hub = Hub::default();
        let run = hub
            .start(serde_json::from_value(json!({"chat_id":CHAT_ID})).unwrap())
            .unwrap();
        run.restore_approvals(restore);

        let mut step: Step = serde_json::from_value(scene["steps"][0].clone()).unwrap();
        step.call
            .as_mut()
            .unwrap()
            .arguments
            .insert("version".into(), json!(99));
        assert_eq!(
            run.resume_approval(&step, &crate::Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "WAITING",
            "a saved reply is bound to the exact serialized Call"
        );
    }

    #[tokio::test]
    async fn restored_reply_ids_remain_reserved_against_a_reused_generator() {
        fn reused_id() -> i64 {
            REPLY_ID
        }

        let mut scene = approval_scene(REPLY_ID, Some(true));
        let pending = approval_scene(REPLY_ID + 1, None);
        let mut pending_step = pending["steps"][0].clone();
        pending_step["id"] = json!("tool-2");
        pending_step["call_id"] = json!("call-2");
        pending_step["call"]["call_id"] = json!("call-2");
        scene["steps"].as_array_mut().unwrap().push(pending_step);
        scene["waits"] = pending["waits"].clone();
        scene["waits"][0]["step"] = json!("tool-2");
        let restore = prepare(&scene, &submitted(REPLY_ID, true))
            .unwrap()
            .expect("old reply is an idempotent duplicate while another question is pending");
        let hub = Hub::default();
        let run = hub
            .start(serde_json::from_value(json!({"chat_id":CHAT_ID})).unwrap())
            .unwrap();
        run.restore_approvals(restore);
        let step = Step {
            id: "new-tool".into(),
            kind: "tool".into(),
            status: "running".into(),
            call_id: Some("new-call".into()),
            stage: Some("before".into()),
            call: Some(Call {
                call_id: "new-call".into(),
                name: "publish".into(),
                arguments: Default::default(),
            }),
            ..Default::default()
        };

        assert_eq!(
            run.ask(
                step.call.as_ref().unwrap(),
                "approve",
                reused_id,
                &crate::Cancellation::new(),
            )
            .await
            .unwrap_err()
            .code,
            "CONFLICT"
        );
    }

    #[test]
    fn restore_routes_reused_call_ids_to_their_native_child_owners() {
        let scene = sibling_child_scene();
        let located = wait::scene_step(
            &scene,
            &[wait::Node {
                node_id: "left".into(),
                exec_id: 77,
                idx: 0,
            }],
            "tool-1",
        )
        .unwrap();
        assert!(std::ptr::eq(
            located,
            &scene["nodes"]["left"]["execution"]["attempts"][0]["agent"]["steps"][0]
        ));
        let located_right = wait::scene_step(
            &scene,
            &[wait::Node {
                node_id: "right".into(),
                exec_id: 78,
                idx: 0,
            }],
            "tool-1",
        )
        .unwrap();
        assert!(std::ptr::eq(
            located_right,
            &scene["nodes"]["right"]["execution"]["attempts"][0]["agent"]["steps"][0]
        ));
        let restore = prepare(&scene, &submitted(REPLY_ID, true))
            .unwrap()
            .expect("child approval is restored");
        let left = wait::Node {
            node_id: "left".into(),
            exec_id: 77,
            idx: 0,
        };
        let right = wait::Node {
            node_id: "right".into(),
            exec_id: 78,
            idx: 0,
        };

        assert_eq!(restore.reply_order, [REPLY_ID, REPLY_ID + 1]);
        let left_question = restore.tasks[&left].questions.get("shared-call").unwrap();
        let right_question = restore.tasks[&right].questions.get("shared-call").unwrap();
        assert_eq!(left_question.id, REPLY_ID);
        assert_eq!(right_question.id, REPLY_ID + 1);
        assert_eq!(
            left_question.input.as_ref().unwrap().metadata["reply_id"],
            REPLY_ID
        );
        assert!(right_question.input.is_none());
    }

    #[test]
    fn restore_keeps_planned_child_questions_under_both_native_coordinates() {
        let scene = planned_sibling_child_scene();
        let restore = prepare(&scene, &submitted(REPLY_ID, true))
            .unwrap()
            .expect("planned child approval is restored");
        let parent = wait::Node {
            node_id: "left".into(),
            exec_id: 77,
            idx: 0,
        };
        let task = wait::Node {
            node_id: "task-1".into(),
            exec_id: 88,
            idx: 0,
        };
        let child = &restore.tasks[&parent].children[&task];
        let question = child.questions.get("shared-call").unwrap();

        assert_eq!(question.id, REPLY_ID);
        assert_eq!(
            question.input.as_ref().unwrap().metadata["reply_id"],
            REPLY_ID
        );
        assert_eq!(restore.tasks[&parent].children.len(), 1);
    }
}
