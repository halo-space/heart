//! Root Agent execution facts owned by a SubAgent run.

use super::*;

/// Shared live root execution. Only its existing Execution is serialized;
/// locks and derived queues never become another recovery truth.
#[derive(Clone, Default)]
pub(crate) struct Root(Arc<Mutex<Option<crate::runtime::agent::node::Execution>>>);

impl Root {
    pub(crate) fn cancel_pending(&self) {
        if let Some(attempt) = self
            .0
            .lock()
            .expect("root execution lock poisoned")
            .as_mut()
            .and_then(|execution| execution.attempts.last_mut())
        {
            cancel_pending(&mut attempt.nodes);
        }
    }
    pub(crate) fn feedback_waiters(&self) -> usize {
        self.0
            .lock()
            .expect("root execution lock poisoned")
            .as_ref()
            .and_then(|execution| execution.attempts.last())
            .map(|attempt| {
                attempt
                    .nodes
                    .values()
                    .filter(|node| {
                        node.status == Status::Waiting
                            && node
                                .execution
                                .as_ref()
                                .and_then(|execution| execution.attempts.last())
                                .and_then(|attempt| attempt.agent.as_ref())
                                .is_some_and(crate::runtime::agent::wait::has_waiting_input)
                    })
                    .count()
            })
            .unwrap_or(0)
    }
    pub(crate) fn restore(snapshot: &Value) -> Result<Self, Error> {
        let execution: Option<crate::runtime::agent::node::Execution> = snapshot
            .get("execution")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|_| Error::new("INVALID_STATE", "invalid root Agent execution"))?
            .flatten();
        if execution
            .as_ref()
            .is_some_and(|execution| execution.attempts.is_empty())
        {
            return Err(Error::new("INVALID_STATE", "root execution has no Attempt"));
        }
        Ok(Self(Arc::new(Mutex::new(execution))))
    }

    pub(crate) fn prepare(
        &self,
        steps: &mut Vec<Step>,
        call: &Call,
        id_fn: Option<fn() -> i64>,
    ) -> Result<Option<Value>, Error> {
        let mut execution = self.0.lock().expect("root execution lock poisoned");
        if execution.is_none() {
            let id_fn = id_fn.ok_or_else(|| {
                Error::new(
                    "INVALID_STATE",
                    "SubAgent execution requires an id function",
                )
            })?;
            let mut state = State::default();
            state.begin_execution(id_fn())?;
            *execution = state.execution;
        }
        let attempt = execution
            .as_mut()
            .expect("created root execution")
            .attempts
            .last_mut()
            .expect("validated root Attempt");
        let scene = prepare(&mut attempt.nodes, steps, call, id_fn)?;
        attempt.status = Status::Running;
        Ok(scene)
    }

    pub(crate) fn complete(
        &self,
        steps: &mut [Step],
        call: &Call,
        result: &Result<Progress, Box<Failure>>,
        message: Option<Message>,
    ) -> Result<(), Error> {
        let mut execution = self.0.lock().expect("root execution lock poisoned");
        let attempt = execution
            .as_mut()
            .and_then(|execution| execution.attempts.last_mut())
            .ok_or_else(|| Error::new("INVALID_STATE", "root execution has no Attempt"))?;
        complete(&mut attempt.nodes, steps, call, result, message)
    }

    pub(crate) fn resume_calls(
        &self,
        steps: &[Step],
        messages: &mut crate::Messages,
    ) -> Result<Vec<(usize, Call)>, Error> {
        let execution = self.0.lock().expect("root execution lock poisoned");
        if let Some(attempt) = execution
            .as_ref()
            .and_then(|execution| execution.attempts.last())
        {
            resume_calls(&attempt.nodes, steps, messages)
        } else if steps.iter().any(|step| step.kind == "agent") {
            Err(Error::new(
                "INVALID_STATE",
                "child steps have no root execution",
            ))
        } else {
            Ok(Vec::new())
        }
    }

    pub(crate) fn pending(&self) -> bool {
        self.0
            .lock()
            .expect("root execution lock poisoned")
            .as_ref()
            .and_then(|execution| execution.attempts.last())
            .is_some_and(|attempt| {
                attempt
                    .nodes
                    .values()
                    .any(|node| matches!(node.status, Status::Running | Status::Waiting))
            })
    }

    pub(crate) fn message(&self, steps: &[Step], call: &Call) -> Option<Message> {
        let execution = self.0.lock().ok()?;
        message(&execution.as_ref()?.attempts.last()?.nodes, steps, call)
    }

    pub(crate) fn source(
        &self,
        steps: &[Step],
        call: &Call,
    ) -> Result<crate::runtime::agent::wait::Node, Error> {
        let execution = self.0.lock().expect("root execution lock poisoned");
        let nodes = &execution
            .as_ref()
            .and_then(|execution| execution.attempts.last())
            .ok_or_else(|| Error::new("INVALID_STATE", "child has no root Attempt"))?
            .nodes;
        source(nodes, steps, call)
    }

    pub(crate) fn observe(&self, observation: &Observation) -> Result<(), Error> {
        let mut execution = self.0.lock().expect("root execution lock poisoned");
        let nodes = &mut execution
            .as_mut()
            .and_then(|execution| execution.attempts.last_mut())
            .ok_or_else(|| Error::new("INVALID_STATE", "child has no root Attempt"))?
            .nodes;
        observe(nodes, observation)
    }

    pub(crate) fn sync_waits(
        &self,
        waits: &mut Vec<crate::runtime::agent::wait::Wait>,
        observation: &Observation,
    ) -> Result<(), Error> {
        let execution = self.0.lock().expect("root execution lock poisoned");
        let nodes = &execution
            .as_ref()
            .and_then(|execution| execution.attempts.last())
            .ok_or_else(|| Error::new("INVALID_STATE", "waiting child has no root Attempt"))?
            .nodes;
        sync_waits(nodes, waits, observation)
    }

    pub(crate) fn snapshot(
        &self,
        status: &str,
        usage: Option<Usage>,
        error: Option<Error>,
    ) -> Option<crate::runtime::agent::node::Execution> {
        let error = error.filter(|error| error.code != "WAITING");
        let mut execution = self.0.lock().expect("root execution lock poisoned");
        let execution = execution.as_mut()?;
        let attempt = execution
            .attempts
            .last_mut()
            .expect("validated root Attempt");
        let status = match status {
            "succeeded" => Status::Succeeded,
            "failed" => Status::Failed,
            "stopped" => Status::Cancelled,
            "waiting" => Status::Waiting,
            _ => Status::Running,
        };
        if matches!(
            status,
            Status::Succeeded | Status::Failed | Status::Cancelled
        ) && attempt.finished_time.is_none()
        {
            attempt.finished_time = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .and_then(|duration| i64::try_from(duration.as_millis()).ok())
                    .unwrap_or(0),
            );
        }
        attempt.status = status;
        attempt.usage = usage.clone();
        attempt.error = error.clone();
        execution.usage = usage;
        execution.error = error;
        Some(execution.clone())
    }
}

/// Stop only unfinished outer child executions; never rewrite a successful
/// child or invent inner Model facts for an uncooperative custom Agent.
pub(crate) fn cancel_pending(nodes: &mut BTreeMap<String, State>) {
    for node in nodes
        .values_mut()
        .filter(|node| matches!(node.status, Status::Running | Status::Waiting))
    {
        node.status = Status::Cancelled;
        node.set_attempt_result(Status::Cancelled, node.usage.clone(), None)
            .expect("active child has an actual Attempt");
    }
}

pub(crate) fn sync_waits(
    nodes: &BTreeMap<String, State>,
    waits: &mut Vec<crate::runtime::agent::wait::Wait>,
    observation: &Observation,
) -> Result<(), Error> {
    let Some(scene) = &observation.scene else {
        return Ok(());
    };
    let child = nodes
        .get(&observation.source.0)
        .and_then(|node| node.execution.as_ref())
        .filter(|execution| execution.exec_id == observation.source.1)
        .ok_or_else(|| Error::new("INVALID_STATE", "waiting child has no actual execution"))?;
    let attempt = child
        .attempts
        .last()
        .ok_or_else(|| Error::new("INVALID_STATE", "waiting child has no current Attempt"))?;
    let coordinate = crate::runtime::agent::wait::Node {
        node_id: observation.source.0.clone(),
        exec_id: child.exec_id,
        idx: attempt.idx,
    };
    let mut desired: Vec<crate::runtime::agent::wait::Wait> = scene
        .get("waits")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| Error::new("INVALID_STATE", "invalid child waiting locators"))?
        .unwrap_or_default();
    for wait in &mut desired {
        wait.nodes.insert(0, coordinate.clone());
    }
    waits.retain(|wait| wait.nodes.first() != Some(&coordinate) || desired.contains(wait));
    for wait in desired {
        if !waits.contains(&wait) {
            waits.push(wait);
        }
    }
    Ok(())
}

pub(crate) fn prepare(
    nodes: &mut BTreeMap<String, State>,
    steps: &mut Vec<Step>,
    call: &Call,
    id_fn: Option<fn() -> i64>,
) -> Result<Option<Value>, Error> {
    if let Some(step) = steps.iter().rev().find(|step| {
        step.kind == "agent"
            && step.call_id.as_deref() == Some(&call.call_id)
            && matches!(step.status.as_str(), "running" | "waiting")
    }) {
        if step.call.as_ref() != Some(call) {
            return Err(Error::new(
                "INVALID_STATE",
                "child step changed its selecting Call",
            ));
        }
        let [id] = step.to.as_slice() else {
            return Err(Error::new(
                "INVALID_STATE",
                "child step must identify one node",
            ));
        };
        let child = nodes
            .get_mut(id)
            .ok_or_else(|| Error::new("INVALID_STATE", "child node is missing"))?;
        child.status = Status::Running;
        let attempt = child.current_attempt_mut()?;
        attempt.status = Status::Running;
        return Ok(attempt.agent.clone());
    }
    let id_fn = id_fn.ok_or_else(|| {
        Error::new(
            "INVALID_STATE",
            "SubAgent execution requires an id function",
        )
    })?;
    let id = id_fn();
    let exec_id = id_fn();
    let node_id = id.to_string();
    if id == exec_id
        || nodes.contains_key(&node_id)
        || nodes.values().any(|node| {
            node.execution
                .as_ref()
                .is_some_and(|execution| execution.exec_id == exec_id)
        })
    {
        return Err(Error::new(
            "CONFLICT",
            "application generated duplicate child identities",
        ));
    }
    let mut node = State {
        status: Status::Running,
        ..State::default()
    };
    node.begin_execution(exec_id)?;
    nodes.insert(node_id.clone(), node);
    steps.push(Step {
        id: format!("agent_{}", steps.len()),
        kind: "agent".into(),
        status: "running".into(),
        call_id: Some(call.call_id.clone()),
        call: Some(call.clone()),
        to: vec![node_id],
        ..Step::default()
    });
    Ok(None)
}

pub(crate) fn complete(
    nodes: &mut BTreeMap<String, State>,
    steps: &mut [Step],
    call: &Call,
    result: &Result<Progress, Box<Failure>>,
    message: Option<Message>,
) -> Result<(), Error> {
    let step = steps
        .iter_mut()
        .rev()
        .find(|step| {
            step.kind == "agent"
                && step.call_id.as_deref() == Some(&call.call_id)
                && matches!(step.status.as_str(), "running" | "waiting")
        })
        .ok_or_else(|| Error::new("INVALID_STATE", "child step is missing"))?;
    let [id] = step.to.as_slice() else {
        return Err(Error::new(
            "INVALID_STATE",
            "child step must identify one node",
        ));
    };
    let node = nodes
        .get_mut(id)
        .ok_or_else(|| Error::new("INVALID_STATE", "child node is missing"))?;
    let (status, scene, delta, error) = match result {
        Ok(progress) => (
            if progress
                .scene
                .as_ref()
                .is_some_and(|scene| scene["status"] == "stopped")
                || progress.cancelled
            {
                Status::Cancelled
            } else {
                Status::Succeeded
            },
            progress.scene.clone(),
            progress.usage.clone(),
            None,
        ),
        Err(failure) => (
            match failure.error.code.as_str() {
                "WAITING" => Status::Waiting,
                "CANCELLED" => Status::Cancelled,
                _ => Status::Failed,
            },
            failure.scene.clone(),
            failure.usage.clone(),
            (failure.error.code != "WAITING").then(|| failure.error.clone()),
        ),
    };
    let usage = scene
        .as_ref()
        .and_then(|scene| scene.get("usage"))
        .map(|usage| serde_json::from_value::<Option<Usage>>(usage.clone()))
        .transpose()
        .map_err(|_| Error::new("INVALID_STATE", "child scene has invalid usage"))?
        .flatten()
        .or_else(|| {
            crate::model::token::merge(
                node.current_attempt_mut()
                    .ok()
                    .and_then(|attempt| attempt.usage.clone()),
                delta,
            )
        });
    node.status = status.clone();
    if let Some(scene) = scene {
        node.current_attempt_mut()?.agent = Some(scene);
    }
    if let Some(message) = message {
        let messages = crate::Messages::new(vec![message]);
        node.messages = Some(messages.clone());
        node.execution
            .as_mut()
            .expect("validated child execution")
            .messages = messages;
    }
    let finished_time = node.current_attempt_mut()?.finished_time;
    node.set_attempt_result(status.clone(), usage, error.clone())?;
    if finished_time.is_some() {
        node.current_attempt_mut()?.finished_time = finished_time;
    }
    step.status = match status {
        Status::Succeeded => "succeeded",
        Status::Waiting => "waiting",
        Status::Cancelled => "cancelled",
        _ => "failed",
    }
    .into();
    step.error = error;
    Ok(())
}

pub(crate) fn resume_calls(
    nodes: &BTreeMap<String, State>,
    steps: &[Step],
    messages: &mut crate::Messages,
) -> Result<Vec<(usize, Call)>, Error> {
    let batch = messages.0.iter().rposition(|message| {
        message.role == Role::Assistant
            && message
                .content
                .iter()
                .any(|part| part.r#type == "tool_call")
    });
    let mut calls = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for step in steps.iter().rev().filter(|step| step.kind == "agent") {
        let call = step
            .call
            .clone()
            .ok_or_else(|| Error::new("INVALID_STATE", "child step lost its Call"))?;
        if step.call_id.as_deref() != Some(&call.call_id) {
            return Err(Error::new(
                "INVALID_STATE",
                "child step has inconsistent Call identity",
            ));
        }
        // Call IDs belong to a selecting batch, not a global execution key.
        if !seen.insert(call.call_id.clone()) {
            continue;
        }
        let indexes = batch
            .and_then(|batch| messages.0.get(batch))
            .map(|message| {
                message
                    .content
                    .iter()
                    .filter(|part| part.r#type == "tool_call")
                    .enumerate()
                    .filter(|(_, part)| {
                        part.data.get("call_id").and_then(Value::as_str) == Some(&call.call_id)
                    })
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let [id] = step.to.as_slice() else {
            return Err(Error::new(
                "INVALID_STATE",
                "child step must identify one node",
            ));
        };
        let child = nodes
            .get(id)
            .ok_or_else(|| Error::new("INVALID_STATE", "child node is missing"))?;
        let execution = child
            .execution
            .as_ref()
            .ok_or_else(|| Error::new("INVALID_STATE", "child execution is missing"))?;
        if indexes.is_empty()
            && matches!(
                child.status,
                Status::Succeeded | Status::Failed | Status::Cancelled
            )
        {
            continue; // Historical children remain saved, but are not a current batch.
        }
        if indexes.len() != 1 {
            return Err(Error::new(
                "INVALID_STATE",
                "child step has no unique selecting model Call",
            ));
        }
        if matches!(
            child.status,
            Status::Succeeded | Status::Failed | Status::Cancelled
        ) && execution.messages.is_empty()
        {
            return Err(Error::new(
                "INVALID_STATE",
                "terminal child has no committed result",
            ));
        }
        calls.push((indexes[0], call));
    }
    if let Some(batch) = batch {
        let ids = calls
            .iter()
            .map(|(_, call)| call.call_id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let mut index = 0;
        messages.0.retain(|message| {
            let keep = index <= batch
                || !message
                    .metadata
                    .get("call_id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| ids.contains(id));
            index += 1;
            keep
        });
    }
    calls.sort_by_key(|(index, _)| *index);
    Ok(calls)
}

pub(crate) fn message(
    nodes: &BTreeMap<String, State>,
    steps: &[Step],
    call: &Call,
) -> Option<Message> {
    let step = steps
        .iter()
        .rev()
        .find(|step| step.kind == "agent" && step.call_id.as_deref() == Some(&call.call_id))?;
    let [id] = step.to.as_slice() else {
        return None;
    };
    let node = nodes.get(id)?;
    if matches!(
        node.status,
        Status::Succeeded | Status::Failed | Status::Cancelled
    ) {
        node.execution.as_ref()?.messages.0.first().cloned()
    } else {
        None
    }
}

#[derive(Debug)]
pub(crate) struct Progress {
    pub cancelled: bool,
    pub message: Message,
    /// The child Assistant message before Delegate converts it to a Function
    /// result. Handoff exposes this message as the outer final response.
    pub final_message: Message,
    pub usage: Option<Usage>,
    pub scene: Option<Value>,
}

pub(crate) fn source(
    nodes: &BTreeMap<String, State>,
    steps: &[Step],
    call: &Call,
) -> Result<crate::runtime::agent::wait::Node, Error> {
    let step = steps
        .iter()
        .rev()
        .find(|step| step.kind == "agent" && step.call_id.as_deref() == Some(&call.call_id))
        .ok_or_else(|| Error::new("INVALID_STATE", "child has no delegation Step"))?;
    let [id] = step.to.as_slice() else {
        return Err(Error::new(
            "INVALID_STATE",
            "child Step must identify one node",
        ));
    };
    let execution = nodes
        .get(id)
        .and_then(|node| node.execution.as_ref())
        .ok_or_else(|| Error::new("INVALID_STATE", "child has no execution"))?;
    let attempt = execution
        .attempts
        .last()
        .ok_or_else(|| Error::new("INVALID_STATE", "child has no current Attempt"))?;
    Ok(crate::runtime::agent::wait::Node {
        node_id: id.clone(),
        exec_id: execution.exec_id,
        idx: attempt.idx,
    })
}

/// Publish only the child's actual captured scene and new consumption facts.
/// An arbitrary internal Complete never proves task completion. A terminal
/// native scene does, and remains available if the parent closes immediately.
pub(crate) fn observe(
    nodes: &mut BTreeMap<String, State>,
    observation: &Observation,
) -> Result<(), Error> {
    if !observation.has_facts() {
        return Ok(());
    }
    let node = nodes
        .get_mut(&observation.source.0)
        .ok_or_else(|| Error::new("INVALID_STATE", "observed child is missing"))?;
    if node.execution.as_ref().map(|execution| execution.exec_id) != Some(observation.source.1) {
        return Err(Error::new(
            "INVALID_STATE",
            "observed child execution differs",
        ));
    }
    let usage = observation
        .scene
        .as_ref()
        .and_then(|scene| scene.get("usage"))
        .map(|usage| serde_json::from_value::<Option<Usage>>(usage.clone()))
        .transpose()
        .map_err(|_| Error::new("INVALID_STATE", "child scene has invalid usage"))?
        .flatten()
        .or_else(|| crate::model::token::merge(node.usage.clone(), observation.usage.clone()));
    let status = observation
        .scene
        .as_ref()
        .and_then(|scene| scene["status"].as_str())
        .and_then(|status| match status {
            "running" => Some(Status::Running),
            "succeeded" => Some(Status::Succeeded),
            "failed" => Some(Status::Failed),
            "stopped" => Some(Status::Cancelled),
            "waiting" => Some(Status::Waiting),
            _ => None,
        })
        .unwrap_or_else(|| node.status.clone());
    if let Some(scene) = &observation.scene {
        node.current_attempt_mut()?.agent = Some((**scene).clone());
    }
    if matches!(status, Status::Succeeded | Status::Cancelled)
        && let Some(message) = &observation.message
    {
        let messages = crate::Messages::new(vec![message.clone()]);
        node.messages = Some(messages.clone());
        node.execution
            .as_mut()
            .expect("validated child execution")
            .messages = messages;
    }
    node.status = status.clone();
    let error = observation
        .scene
        .as_ref()
        .and_then(|scene| scene.get("error"))
        .map(|error| serde_json::from_value::<Option<Error>>(error.clone()))
        .transpose()
        .map_err(|_| Error::new("INVALID_STATE", "child scene has invalid error"))?
        .flatten();
    let finished_time = node.current_attempt_mut()?.finished_time;
    node.set_attempt_result(status, usage, error)?;
    if finished_time.is_some() {
        node.current_attempt_mut()?.finished_time = finished_time;
    }
    Ok(())
}
