//! SubAgent assembly plus private child execution and usage settlement.
use crate::runtime::agent::node::{State, Status, Step};
use crate::{
    Cancellation, Error, Message,
    agent::Input,
    components::agent::Agent,
    event::Event,
    message::Role,
    model::token::{InputDetails, OutputDetails, Usage},
    tool::Call,
};
use futures_util::task::AtomicWaker;
use serde_json::Value;
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

type ErasedStream = Pin<Box<dyn futures_core::Stream<Item = Result<Event, Error>> + 'static>>;
type RunFuture = Pin<Box<dyn Future<Output = Result<ErasedStream, Error>> + 'static>>;

trait ErasedAgent: Send + Sync {
    fn run(&self, input: Input, cancellation: Cancellation) -> RunFuture;
}

struct TypedAgent<A>(Arc<A>);

impl<A> ErasedAgent for TypedAgent<A>
where
    A: Agent + 'static,
{
    fn run(&self, input: Input, cancellation: Cancellation) -> RunFuture {
        let agent = Arc::clone(&self.0);
        Box::pin(async move {
            // Parent run owns outward mode filtering for delegated events.
            let stream = agent.run(input, true, cancellation).await?;
            Ok(Box::pin(stream) as ErasedStream)
        })
    }
}

/// One explicitly registered Agent that another Agent may delegate to.
///
/// SubAgents are distinct from the internal Agent tool dispatcher: the
/// dispatcher contains Tool/MCP/Skill entries, while this retains a complete Agent
/// run boundary and its event stream.
#[derive(Clone)]
pub struct SubAgent {
    name: String,
    description: String,
    runner: Arc<dyn ErasedAgent>,
}

impl SubAgent {
    pub fn new<A>(name: impl Into<String>, agent: A) -> Result<Self, Error>
    where
        A: Agent + 'static,
    {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "subagent name must not be empty",
            ));
        }
        Ok(Self {
            name,
            description: String::new(),
            runner: Arc::new(TypedAgent(Arc::new(agent))),
        })
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description_text(&self) -> &str {
        &self.description
    }

    pub(crate) fn definition(&self) -> crate::components::tool::ToolDefinition {
        crate::components::tool::ToolDefinition {
            name: "agent_spawn".into(),
            description: "Delegate a child task to one of the registered SubAgents. The child input must be an agent::Input object.".into(),
            parameters: serde_json::from_value(serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "input": {"type": "object"}
                },
                "required": ["name", "input"],
                "additionalProperties": false
            }))
            .expect("static agent_spawn schema is an object"),
        }
    }

    pub(crate) fn has_name(&self, name: &str) -> bool {
        self.name == name
    }

    pub(crate) fn run(&self, input: Input, cancellation: Cancellation) -> RunFuture {
        self.runner.run(input, cancellation)
    }
}

impl std::fmt::Debug for SubAgent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SubAgent")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// Private, consumer-driven forwarding. Each running child can enqueue only
/// one event or native scene update, then waits for the parent to take it. No producer task, persisted
/// queue, replay protocol or second public stream is created.
#[derive(Clone, Default)]
pub(crate) struct Relay(Arc<RelayState>);

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
        }
    }
}

/// Cancellation is cooperative, not an asynchronous Drop. Acknowledge queued
/// facts first, then poll only immediately ready work. Fresh native scenes may
/// release another backpressured handoff; arbitrary non-cooperating streams
/// cannot keep the owner alive merely by emitting more text.
#[inline(never)]
pub(crate) fn cancel_ready<T>(
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

pub(crate) struct Observer {
    relay: Relay,
    chat_id: Option<i64>,
    source: (String, i64),
    interaction: Option<crate::runtime::agent::reply::Run>,
}

pub(crate) struct Observation {
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
        owner: Option<&crate::runtime::agent::reply::Run>,
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
        interaction: Option<crate::runtime::agent::reply::Run>,
    ) -> Self {
        self.interaction = interaction;
        self
    }
    async fn emit(
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

#[cfg(test)]
pub(crate) fn waiting_fixture(mut snapshot: Value, call: Call, id_fn: fn() -> i64) -> Value {
    let root = Root::default();
    let mut steps = Vec::new();
    root.prepare(&mut steps, &call, Some(id_fn)).unwrap();
    let mut selecting = Message::new(Role::Assistant);
    selecting.content.push(crate::model::content::part::Part {
        r#type: "tool_call".into(),
        data:
            serde_json::json!({"call_id":call.call_id,"name":call.name,"arguments":call.arguments})
                .as_object()
                .unwrap()
                .clone(),
    });
    let mut messages: crate::Messages =
        serde_json::from_value(snapshot["messages"].clone()).unwrap();
    messages.push(selecting);
    snapshot["messages"] = serde_json::to_value(messages).unwrap();
    snapshot["steps"] = serde_json::to_value(steps).unwrap();
    snapshot["execution"] = serde_json::to_value(root.snapshot("waiting", None, None)).unwrap();
    snapshot
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

#[derive(Debug)]
pub(crate) struct Failure {
    pub error: Error,
    pub scene: Option<Value>,
    pub usage: Option<Usage>,
}

impl Failure {
    pub(crate) fn new(error: Error) -> Box<Self> {
        Box::new(Self {
            error,
            scene: None,
            usage: None,
        })
    }
}

/// Shared run/session ownership is not a model-selectable child argument.
pub(crate) fn inherit_identity(
    input: &mut Input,
    parent: &serde_json::Map<String, Value>,
) -> Result<(), Error> {
    for key in ["chat_id", "session_id", "user_id", "tenant_id"] {
        if let Some(value) = parent.get(key) {
            if input
                .metadata
                .get(key)
                .is_some_and(|supplied| supplied != value)
            {
                return Err(Error::new(
                    "CONFLICT",
                    "SubAgent input belongs to a different run or session",
                ));
            }
            input.metadata.insert(key.to_owned(), value.clone());
        }
    }
    if let Some(value) = parent.get("agent_id") {
        input
            .metadata
            .entry("agent_id")
            .or_insert_with(|| value.clone());
    }
    Ok(())
}

fn additional_usage(
    scene: Option<&Value>,
    previous: Option<&Value>,
    observed: Option<Usage>,
) -> Result<Option<Usage>, Error> {
    // A cached terminal child does not capture a new scene or make another
    // model call. Its replayed Complete is delivery, not new consumption.
    if scene.is_none()
        && previous.is_some_and(|previous| {
            matches!(
                previous["status"].as_str(),
                Some("succeeded" | "failed" | "stopped")
            )
        })
    {
        return Ok(None);
    }
    let read = |scene: Option<&Value>| -> Result<Option<Usage>, Error> {
        scene
            .and_then(|scene| scene.get("usage"))
            .map(|usage| {
                serde_json::from_value(usage.clone())
                    .map_err(|_| Error::new("INVALID_STATE", "SubAgent scene has invalid usage"))
            })
            .transpose()
            .map(Option::flatten)
    };
    let Some(current) = read(scene)? else {
        return Ok(observed);
    };
    usage_since(Some(current), read(previous)?)
}

fn usage_since(current: Option<Usage>, previous: Option<Usage>) -> Result<Option<Usage>, Error> {
    if current == previous {
        return Ok(None);
    }
    let Some(current) = current else {
        return Ok(None);
    };
    let previous = previous.unwrap_or_default();
    let subtract = |current: Option<u64>, previous: Option<u64>| -> Result<Option<u64>, Error> {
        current
            .map(|current| {
                current.checked_sub(previous.unwrap_or(0)).ok_or_else(|| {
                    Error::new("INVALID_STATE", "SubAgent cumulative usage decreased")
                })
            })
            .transpose()
    };
    Ok(Some(Usage {
        input: subtract(current.input, previous.input)?,
        output: subtract(current.output, previous.output)?,
        total: subtract(current.total, previous.total)?,
        input_details: current
            .input_details
            .map(|v| {
                subtract(v.cached, previous.input_details.and_then(|v| v.cached))
                    .map(|cached| InputDetails { cached })
            })
            .transpose()?,
        output_details: current
            .output_details
            .map(|v| {
                subtract(v.think, previous.output_details.and_then(|v| v.think))
                    .map(|think| OutputDetails { think })
            })
            .transpose()?,
        details: Default::default(),
    }))
}

fn failure(
    error: Error,
    scope: &crate::runtime::agent::scope::Scope,
    previous: Option<&Value>,
    observed: Option<Usage>,
) -> Box<Failure> {
    let scene = scope.latest();
    match additional_usage(scene.as_ref(), previous, observed.clone()) {
        Ok(usage) => Box::new(Failure {
            error,
            scene,
            usage,
        }),
        Err(error) => Box::new(Failure {
            error,
            scene,
            usage: observed,
        }),
    }
}

pub(crate) async fn run(
    subagent: &SubAgent,
    input: Input,
    call: &Call,
    previous: Option<Value>,
    cancellation: Cancellation,
    id_fn: Option<fn() -> i64>,
    observer: Option<Observer>,
) -> Result<Progress, Box<Failure>> {
    // A descendant cannot enable progress handoff that its enclosing runtime
    // does not implement. Independent parents start the supported Agent path;
    // delegated Agent scopes inherit it, while Workflow scopes remain gated.
    let feedback = observer.is_some()
        && (!crate::runtime::agent::scope::active() || crate::runtime::agent::scope::feedback());
    let scope = crate::runtime::agent::scope::Scope::new(previous.clone())
        .with_id_fn(id_fn)
        .with_feedback(feedback)
        .with_interaction(
            observer
                .as_ref()
                .and_then(|observer| observer.interaction.clone()),
        );
    let mut stream =
        crate::runtime::agent::scope::run(scope.clone(), subagent.run(input, cancellation))
            .await
            .map_err(|error| failure(error, &scope, previous.as_ref(), None))?;
    let mut final_message = None;
    let mut observed = None;
    let mut cancelled = false;
    let mut published = None;
    let mut published_scene: Option<Arc<Value>> = None;
    enum Update {
        Event(Box<Result<Event, Error>>),
        Scene(Arc<Value>),
    }
    loop {
        let item = crate::runtime::agent::scope::run(
            scope.clone(),
            futures_util::future::poll_fn(|cx| match stream.as_mut().poll_next(cx) {
                Poll::Ready(item) => Poll::Ready(item.map(|event| Update::Event(Box::new(event)))),
                Poll::Pending => {
                    let latest = scope.latest_shared();
                    if observer.is_some()
                        && let Some(scene) = latest
                        && published_scene
                            .as_ref()
                            .is_none_or(|old| !Arc::ptr_eq(old, &scene))
                        && crate::runtime::agent::wait::has_input(&scene)
                    {
                        Poll::Ready(Some(Update::Scene(scene)))
                    } else {
                        Poll::Pending
                    }
                }
            }),
        )
        .await;
        let Some(item) = item else { break };
        let item = match item {
            Update::Scene(scene) => {
                let current = additional_usage(Some(&scene), previous.as_ref(), observed.clone())
                    .map_err(Failure::new)?;
                let usage =
                    usage_since(current.clone(), published.clone()).map_err(Failure::new)?;
                published = current;
                let feedback = feedback_changed(&scene, published_scene.as_deref());
                published_scene = Some(scene.clone());
                observer
                    .as_ref()
                    .expect("progress has an observer")
                    .emit(None, Some(scene), usage, None, feedback)
                    .await;
                continue;
            }
            Update::Event(item) => *item,
        };
        match item {
            Ok(event) => {
                cancelled |= matches!(&event, Event::Complete { finish_reason: Some(reason), .. } if reason == "cancelled");
                if let Event::Complete { message, usage, .. } = &event
                    && message.role == Role::Assistant
                {
                    observed = crate::model::token::merge(
                        observed,
                        usage.clone().or_else(|| message.usage.clone()),
                    );
                    final_message = Some(message.clone());
                }
                if let Some(observer) = &observer {
                    let scene = scope.latest_shared();
                    let current =
                        additional_usage(scene.as_deref(), previous.as_ref(), observed.clone())
                            .map_err(Failure::new)?;
                    let usage =
                        usage_since(current.clone(), published.clone()).map_err(Failure::new)?;
                    published = current;
                    let changed = match (&scene, &published_scene) {
                        (Some(current), Some(previous)) => !Arc::ptr_eq(current, previous),
                        (Some(_), None) => true,
                        _ => false,
                    };
                    let message = if changed
                        && scene.as_ref().is_some_and(|scene| {
                            matches!(scene["status"].as_str(), Some("succeeded" | "stopped"))
                        }) {
                        final_message
                            .clone()
                            .map(|message| result_message(message, call))
                    } else {
                        None
                    };
                    let feedback = changed
                        && scene.as_deref().is_some_and(|scene| {
                            feedback_changed(scene, published_scene.as_deref())
                        });
                    published_scene = scene.clone();
                    // Only real scene commits update the owner; text deltas
                    // do not copy the whole unchanged Agent State again.
                    observer
                        .emit(
                            Some(event),
                            scene.filter(|_| changed),
                            usage,
                            message,
                            feedback,
                        )
                        .await;
                }
            }
            Err(error) => {
                let mut failure = failure(error, &scope, previous.as_ref(), observed);
                failure.usage = usage_since(failure.usage, published).map_err(Failure::new)?;
                return Err(failure);
            }
        }
    }
    let mut message = match final_message {
        Some(message) => message,
        None => {
            let mut failure = failure(
                Error::new(
                    "INVALID_RESPONSE",
                    "SubAgent stream ended without Assistant Complete",
                ),
                &scope,
                previous.as_ref(),
                observed.clone(),
            );
            failure.usage = usage_since(failure.usage, published).map_err(Failure::new)?;
            return Err(failure);
        }
    };
    let scene = scope.latest();
    let usage = usage_since(
        additional_usage(scene.as_ref(), previous.as_ref(), observed).map_err(Failure::new)?,
        published,
    )
    .map_err(Failure::new)?;
    message = result_message(message, call);
    Ok(Progress {
        cancelled,
        message,
        usage,
        scene: scene.or(previous),
    })
}

fn feedback_changed(scene: &Value, previous: Option<&Value>) -> bool {
    let before = previous
        .map(crate::runtime::agent::wait::input_progress)
        .unwrap_or_default();
    crate::runtime::agent::wait::input_progress(scene)
        .iter()
        .any(|(id, count)| *count > before.get(id).copied().unwrap_or(0))
}

fn result_message(mut message: Message, call: &Call) -> Message {
    message.role = Role::Function;
    crate::runtime::observation::clear_source(&mut message);
    message
        .metadata
        .insert("call_id".into(), Value::String(call.call_id.clone()));
    message
        .metadata
        .insert("name".into(), Value::String(call.name.clone()));
    message
}
