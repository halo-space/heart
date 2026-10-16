//! Root Chat/ReAct live approval on the original run stream, without replay.
use futures_util::{StreamExt, poll, stream};
use halo_agents::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, SubAgent, toolkit::Toolkit},
    checkpoint::Checkpoint,
    core,
    event::Event,
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
    permission::{Behavior, Decision, Permission, Request as PermissionRequest},
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicI64, AtomicUsize, Ordering},
};

fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(2_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
fn input() -> Input {
    serde_json::from_value(
        json!({"metadata":{"chat_id":1001,"session_id":2001,"agent_id":3001},
    "payload":[{"type":"text","data":{"value":"weather"}}]}),
    )
    .unwrap()
}
fn answer(id: i64, decision: Value) -> Input {
    serde_json::from_value(json!({"metadata":{"chat_id":1001,"session_id":2001,"agent_id":3001,"reply_id":id},
        "payload":[{"type":"json","data":{"value":{"approved":decision,"arguments":{"city":"untrusted"}}}}]})).unwrap()
}

#[derive(Clone)]
struct Model {
    requests: Arc<Mutex<Vec<Request>>>,
    multiple: bool,
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let first = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request);
            requests.len() == 1
        };
        let mut message = Message::new(Role::Assistant);
        if first {
            for (id, city) in if self.multiple {
                vec![("c1", "Beijing"), ("c2", "Paris")]
            } else {
                vec![("c1", "Beijing")]
            } {
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: json!({"call_id":id,"name":"lookup","arguments":{"city":city}})
                        .as_object()
                        .unwrap()
                        .clone(),
                });
            }
        } else {
            message.values.insert("answer".into(), json!("done"));
        }
        Ok(stream::iter(vec![
            Ok(Event::Delta {
                message: Message::new(Role::Assistant),
            }),
            Ok(Event::Complete {
                message,
                usage: Some(Usage {
                    input: Some(10),
                    output: Some(3),
                    total: Some(13),
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
            }),
        ]))
    }
}

#[derive(Default)]
struct Log {
    before: AtomicUsize,
    calls: Mutex<Vec<Call>>,
    after: AtomicUsize,
    checks: AtomicUsize,
}
struct Lookup {
    definition: ToolDefinition,
    log: Arc<Log>,
    change: bool,
}
impl Lookup {
    fn new(log: Arc<Log>, change: bool) -> Self {
        Self { log, change, definition: ToolDefinition {
    name:"lookup".into(), description:"lookup weather".into(), parameters:json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}).as_object().unwrap().clone()
} }
    }
}
impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn before(&self, call: &mut Call, _: &Cancellation) -> Result<(), Error> {
        self.log.before.fetch_add(1, Ordering::SeqCst);
        if self.change {
            call.arguments.insert("city".into(), json!("Paris"));
        }
        Ok(())
    }
    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.log.calls.lock().unwrap().push(call.clone());
        Ok(Message::function(
            json!({"city":call.arguments["city"],"weather":"clear"}),
        ))
    }
    async fn after(&self, _: &Call, _: &mut Message, _: &Cancellation) -> Result<(), Error> {
        self.log.after.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
struct Ask(Arc<Log>);
impl Permission for Ask {
    async fn check(&self, _: &PermissionRequest, _: &Cancellation) -> Result<Decision, Error> {
        self.0.checks.fetch_add(1, Ordering::SeqCst);
        Ok(Decision {
            behavior: Behavior::Ask,
            reason: Some("Confirm operation".into()),
        })
    }
}

#[derive(Clone, Default)]
struct DelegatingModel(Arc<Mutex<Vec<Request>>>);
impl Chat for DelegatingModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut requests = self.0.lock().unwrap();
        let mut message = Message::new(Role::Assistant);
        if requests.is_empty() {
            for (index, name) in ["left", "right"].into_iter().enumerate() {
                message.content.push(Part {r#type:"tool_call".into(),data:json!({"call_id":format!("spawn_{index}"),"name":"agent_spawn","arguments":{
                    "name":name,"input":{"metadata":{"agent_id":3002+index},"payload":[{"type":"text","data":{"value":format!("child {name} query")}}]}
                }}).as_object().unwrap().clone()});
            }
        } else {
            message.values.insert("answer".into(), json!("parent done"));
        }
        requests.push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                input: Some(10),
                output: Some(3),
                total: Some(13),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}
struct ChildProbe {
    model: Model,
    log: Arc<Log>,
    checkpoint: Arc<Store>,
}
fn waiting_children(react: bool) -> (Vec<SubAgent>, Vec<ChildProbe>) {
    let mut children = Vec::new();
    let mut probes = Vec::new();
    for name in ["left", "right"] {
        let model = Model {
            requests: Default::default(),
            multiple: false,
        };
        let log = Arc::new(Log::default());
        let checkpoint = Arc::new(Store::default());
        let toolkit = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
        let child = if react {
            SubAgent::new(
                name,
                agent::react()
                    .model(model.clone())
                    .toolkit(toolkit)
                    .permission(Ask(log.clone()))
                    .checkpoint(checkpoint.clone())
                    .build()
                    .unwrap(),
            )
            .unwrap()
        } else {
            SubAgent::new(
                name,
                agent::chat()
                    .model(model.clone())
                    .toolkit(toolkit)
                    .permission(Ask(log.clone()))
                    .checkpoint(checkpoint.clone())
                    .build()
                    .unwrap(),
            )
            .unwrap()
        };
        children.push(child);
        probes.push(ChildProbe {
            model,
            log,
            checkpoint,
        });
    }
    (children, probes)
}
fn child_scene<'a>(state: &'a Value, node: &str) -> &'a Value {
    &state["execution"]["attempts"][0]["nodes"][node]["execution"]["attempts"][0]["agent"]
}
async fn child_reply<A: Agent>(
    agent: A,
    store: Arc<core::checkpoint::InMemory>,
    model: DelegatingModel,
    children: Vec<ChildProbe>,
    mode: bool,
    approve_first: bool,
) {
    let mut events = Box::pin(agent.run(input(), mode, Cancellation::new()).await.unwrap());
    let (a, question_a) = question(&mut events).await;
    let (b, question_b) = question(&mut events).await;
    let node_a = question_a.metadata["node_id"].as_str().unwrap().to_owned();
    let node_b = question_b.metadata["node_id"].as_str().unwrap().to_owned();
    assert_ne!(a, b);
    assert_ne!(node_a, node_b);
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(saved["status"], "waiting");
    assert_eq!(saved["waits"].as_array().unwrap().len(), 2);
    assert_eq!(saved["waits"][0]["nodes"][0]["node_id"], node_a);
    assert_eq!(saved["waits"][1]["nodes"][0]["node_id"], node_b);
    assert!(
        saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .all(|step| step.get("waits").is_none())
    );
    assert_eq!(saved["usage"]["total"], 39);
    for (node, id) in [(&node_a, a), (&node_b, b)] {
        let scene = child_scene(&saved, node);
        assert_eq!(scene["status"], "waiting");
        assert_eq!(scene["steps"][0]["call_id"], "c1");
        assert_eq!(
            scene["steps"][0]["waits"][id.to_string()]["status"],
            "waiting"
        );
        assert_eq!(scene["query"]["metadata"]["session_id"], 2001);
        assert_eq!(scene["waits"][0]["nodes"], json!([]));
    }
    let a_index = (child_scene(&saved, &node_a)["query"]["metadata"]["agent_id"]
        .as_u64()
        .unwrap()
        - 3002) as usize;
    let b_index = 1 - a_index;
    let restored = agent::chat()
        .model(Model {
            requests: Default::default(),
            multiple: false,
        })
        .checkpoint(store.clone())
        .build()
        .unwrap();
    assert!(
        matches!(restored.run(input(),mode,Cancellation::new()).await,Err(error) if error.code=="WAITING")
    );
    agent
        .reply(answer(b, json!(true)), Cancellation::new())
        .await
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let event = events.next().await.unwrap().unwrap();
            if !mode {
                assert!(!matches!(event, Event::Delta { .. }));
            }
            if let Event::Complete { message, .. } = event
                && message.metadata.get("node_id").and_then(Value::as_str) == Some(&node_b)
                && message.values.get("answer") == Some(&json!("done"))
            {
                break;
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(children[b_index].log.calls.lock().unwrap().len(), 1);
    assert!(children[a_index].log.calls.lock().unwrap().is_empty());
    assert_eq!(
        model.0.lock().unwrap().len(),
        1,
        "parent inference must wait for its other child"
    );
    agent
        .reply(answer(b, json!(true)), Cancellation::new())
        .await
        .unwrap();
    assert_eq!(
        agent
            .reply(answer(b, json!(false)), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    agent
        .reply(answer(a, json!(approve_first)), Cancellation::new())
        .await
        .unwrap();
    finish(&mut events).await;
    assert_eq!(
        children[a_index].log.calls.lock().unwrap().len(),
        usize::from(approve_first)
    );
    assert_eq!(model.0.lock().unwrap().len(), 2);
    let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(final_state["status"], "succeeded");
    assert_eq!(final_state["waits"], json!([]));
    assert_eq!(final_state["usage"]["total"], 78);
    for (node, id) in [(&node_a, a), (&node_b, b)] {
        let child = &final_state["execution"]["attempts"][0]["nodes"][node];
        assert_eq!(child["status"], "succeeded");
        assert_eq!(child["execution"]["attempts"].as_array().unwrap().len(), 1);
        assert_eq!(
            child["execution"]["exec_id"],
            saved["execution"]["attempts"][0]["nodes"][node]["execution"]["exec_id"]
        );
        let scene = child_scene(&final_state, node);
        let record = &scene["steps"][0]["waits"][id.to_string()];
        assert_eq!(record["status"], "resolved");
        assert_eq!(record["feedback"].as_array().unwrap().len(), 1);
        assert_eq!(scene["usage"]["total"], 26);
        assert_eq!(scene["waits"], json!([]));
    }
    for child in &children {
        assert_eq!(child.checkpoint.loads.load(Ordering::SeqCst), 0);
        assert_eq!(child.checkpoint.saves.load(Ordering::SeqCst), 0);
        assert!(child.model.requests.lock().unwrap().iter().all(|request| {
            let value = serde_json::to_string(request).unwrap();
            !value.contains("reply_id") && !value.contains("untrusted")
        }));
    }
    assert!(model.0.lock().unwrap().iter().all(|request| {
        let value = serde_json::to_string(request).unwrap();
        !value.contains("reply_id") && !value.contains("untrusted")
    }));
}
#[tokio::test]
async fn parent_reply_targets_actual_child_steps_on_the_original_stream() {
    for parent_react in [false, true] {
        for child_react in [false, true] {
            for mode in [false, true] {
                for approve in [false, true] {
                    let (children, probes) = waiting_children(child_react);
                    let store = Arc::new(core::checkpoint::InMemory::new());
                    let model = DelegatingModel::default();
                    if parent_react {
                        child_reply(
                            agent::react()
                                .model(model.clone())
                                .subagents(children)
                                .checkpoint(store.clone())
                                .with_id_fn(next_id)
                                .build()
                                .unwrap(),
                            store,
                            model,
                            probes,
                            mode,
                            approve,
                        )
                        .await;
                    } else {
                        child_reply(
                            agent::chat()
                                .model(model.clone())
                                .subagents(children)
                                .checkpoint(store.clone())
                                .with_id_fn(next_id)
                                .build()
                                .unwrap(),
                            store,
                            model,
                            probes,
                            mode,
                            approve,
                        )
                        .await;
                    }
                }
            }
        }
    }
}

async fn cancelled_children<A: Agent>(
    agent: A,
    store: Arc<core::checkpoint::InMemory>,
    probes: Vec<ChildProbe>,
    drop_stream: bool,
) {
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(input(), false, cancellation.clone())
            .await
            .unwrap(),
    );
    let (a, _) = question(&mut events).await;
    let (b, _) = question(&mut events).await;
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    if drop_stream {
        drop(events);
        assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
        assert!(
            matches!(agent.run(input(),false,Cancellation::new()).await,Err(error) if error.code=="WAITING")
        );
    } else {
        cancellation.cancel();
        let output = finish(&mut events).await;
        assert!(
            matches!(output.last(),Some(Event::Complete {finish_reason:Some(reason),..}) if reason=="cancelled")
        );
        let stopped = store.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(stopped["status"], "stopped");
        assert_eq!(stopped["waits"], json!([]));
        assert_eq!(stopped["usage"]["total"], 39);
        let cached = agent
            .run(input(), false, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(
            matches!(cached.as_slice(),[Ok(Event::Complete{finish_reason:Some(reason),..})] if reason=="cancelled")
        );
    }
    for id in [a, b] {
        assert_eq!(
            agent
                .reply(answer(id, json!(true)), Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
    }
    for child in probes {
        assert!(child.log.calls.lock().unwrap().is_empty());
        assert_eq!(child.model.requests.lock().unwrap().len(), 1);
    }
}
#[tokio::test]
async fn parent_cancellation_or_drop_releases_child_consumers_without_tool_execution() {
    for parent_react in [false, true] {
        for child_react in [false, true] {
            for drop_stream in [false, true] {
                let (children, probes) = waiting_children(child_react);
                let store = Arc::new(core::checkpoint::InMemory::new());
                if parent_react {
                    cancelled_children(
                        agent::react()
                            .model(DelegatingModel::default())
                            .subagents(children)
                            .checkpoint(store.clone())
                            .with_id_fn(next_id)
                            .build()
                            .unwrap(),
                        store,
                        probes,
                        drop_stream,
                    )
                    .await;
                } else {
                    cancelled_children(
                        agent::chat()
                            .model(DelegatingModel::default())
                            .subagents(children)
                            .checkpoint(store.clone())
                            .with_id_fn(next_id)
                            .build()
                            .unwrap(),
                        store,
                        probes,
                        drop_stream,
                    )
                    .await;
                }
            }
        }
    }
}

#[derive(Default)]
struct ChildQuestionStore(core::checkpoint::InMemory);
impl Checkpoint for ChildQuestionStore {
    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        self.0.load(id, cancellation).await
    }
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        if state["waits"]
            .as_array()
            .is_some_and(|waits| !waits.is_empty())
        {
            return Err(Error::new("INTERNAL", "child question save failed"));
        }
        self.0.save(id, state, cancellation).await
    }
    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        self.0.delete(id, cancellation).await
    }
}
async fn child_save_failure<A: Agent>(
    agent: A,
    store: Arc<ChildQuestionStore>,
    probes: Vec<ChildProbe>,
) {
    let output = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        agent
            .run(input(), false, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    })
    .await
    .unwrap();
    assert!(
        matches!(output.last(),Some(Err(error)) if error.message=="child question save failed")
    );
    assert!(output.iter().all(|event|!matches!(event,Ok(Event::Complete{message,..}) if message.metadata.contains_key("reply_id"))));
    let state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(state["status"], "failed");
    assert_eq!(state["waits"], json!([]));
    let calls: usize = probes
        .iter()
        .map(|probe| probe.model.requests.lock().unwrap().len())
        .sum();
    assert_eq!(
        state["usage"]["total"],
        13 * (1 + calls),
        "already returned sibling consumption cannot disappear at save failure"
    );
    let cached = agent
        .run(input(), false, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(cached.as_slice(),[Err(error)] if error.message=="child question save failed")
    );
    for child in probes {
        assert!(child.log.calls.lock().unwrap().is_empty());
    }
}
#[tokio::test]
async fn unsaved_child_question_is_not_presented_and_failed_parent_remains_readable() {
    for parent_react in [false, true] {
        for child_react in [false, true] {
            let (children, probes) = waiting_children(child_react);
            let store = Arc::new(ChildQuestionStore::default());
            if parent_react {
                child_save_failure(
                    agent::react()
                        .model(DelegatingModel::default())
                        .subagents(children)
                        .checkpoint(store.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                    store,
                    probes,
                )
                .await;
            } else {
                child_save_failure(
                    agent::chat()
                        .model(DelegatingModel::default())
                        .subagents(children)
                        .checkpoint(store.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                    store,
                    probes,
                )
                .await;
            }
        }
    }
}

struct DelayedChildStore {
    inner: core::checkpoint::InMemory,
    gate: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    blocked: AtomicUsize,
}
impl Checkpoint for DelayedChildStore {
    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        self.inner.load(id, cancellation).await
    }
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        let gate = if state["waits"]
            .as_array()
            .is_some_and(|waits| !waits.is_empty())
        {
            self.gate.lock().unwrap().take()
        } else {
            None
        };
        if let Some(gate) = gate {
            self.blocked.fetch_add(1, Ordering::SeqCst);
            gate.await.unwrap();
        }
        self.inner.save(id, state, cancellation).await
    }
    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, cancellation).await
    }
}
async fn cancel_child_save<A: Agent>(
    agent: A,
    store: Arc<DelayedChildStore>,
    release: tokio::sync::oneshot::Sender<()>,
    probes: Vec<ChildProbe>,
) {
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(input(), false, cancellation.clone())
            .await
            .unwrap(),
    );
    let mut output = Vec::new();
    loop {
        match poll!(events.next()) {
            std::task::Poll::Ready(Some(event)) => output.push(event.unwrap()),
            std::task::Poll::Pending => break,
            _ => panic!("expected child question persistence to wait"),
        }
    }
    assert_eq!(store.blocked.load(Ordering::SeqCst), 1);
    cancellation.cancel();
    release.send(()).unwrap();
    output.extend(finish(&mut events).await);
    assert!(output.iter().all(|event| !matches!(event,Event::Complete{message,..} if message.metadata.contains_key("reply_id"))));
    let stopped = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(stopped["status"], "stopped");
    assert_eq!(stopped["waits"], json!([]));
    for child in probes {
        assert!(child.log.calls.lock().unwrap().is_empty());
    }
}
#[tokio::test]
async fn cancelling_outer_child_question_save_never_presents_an_actionable_question() {
    for parent_react in [false, true] {
        for child_react in [false, true] {
            let (children, probes) = waiting_children(child_react);
            let (release, gate) = tokio::sync::oneshot::channel();
            let store = Arc::new(DelayedChildStore {
                inner: core::checkpoint::InMemory::new(),
                gate: Mutex::new(Some(gate)),
                blocked: AtomicUsize::new(0),
            });
            if parent_react {
                cancel_child_save(
                    agent::react()
                        .model(DelegatingModel::default())
                        .subagents(children)
                        .checkpoint(store.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                    store,
                    release,
                    probes,
                )
                .await;
            } else {
                cancel_child_save(
                    agent::chat()
                        .model(DelegatingModel::default())
                        .subagents(children)
                        .checkpoint(store.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                    store,
                    release,
                    probes,
                )
                .await;
            }
        }
    }
}

async fn child_plan_reply<A: Agent>(
    agent: A,
    store: Arc<core::checkpoint::InMemory>,
    left: Arc<Log>,
    right: Arc<Log>,
    mode: bool,
) {
    let mut events = Box::pin(agent.run(input(), mode, Cancellation::new()).await.unwrap());
    let mut ids = Vec::new();
    for _ in 0..3 {
        ids.push(question(&mut events).await.0);
    }
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    let waits = saved["waits"].as_array().unwrap();
    assert_eq!(waits.len(), 3);
    assert_eq!(
        waits
            .iter()
            .filter(|wait| wait["nodes"].as_array().unwrap().len() == 2)
            .count(),
        2
    );
    assert_eq!(saved["usage"]["total"], 78);
    let restored = agent::chat()
        .model(Model {
            requests: Default::default(),
            multiple: false,
        })
        .checkpoint(store.clone())
        .build()
        .unwrap();
    assert!(
        matches!(restored.run(input(),mode,Cancellation::new()).await,Err(error) if error.code=="WAITING")
    );
    assert!(left.calls.lock().unwrap().is_empty());
    assert!(right.calls.lock().unwrap().is_empty());
    for id in ids {
        agent
            .reply(answer(id, json!(true)), Cancellation::new())
            .await
            .unwrap();
    }
    finish(&mut events).await;
    let state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(state["status"], "succeeded");
    assert_eq!(state["waits"], json!([]));
    assert_eq!(state["usage"]["total"], 156);
    assert_eq!(left.calls.lock().unwrap().len(), 2);
    assert_eq!(right.calls.lock().unwrap().len(), 1);
    for wait in waits {
        let path = wait["nodes"].as_array().unwrap();
        let node = path[0]["node_id"].as_str().unwrap();
        let scene = child_scene(&state, node);
        if path.len() == 2 {
            let task = path[1]["node_id"].as_str().unwrap();
            let execution = &scene["nodes"][task]["execution"];
            assert_eq!(execution["exec_id"], path[1]["exec_id"]);
            assert_eq!(execution["attempts"].as_array().unwrap().len(), 1);
            assert!(
                execution["attempts"][0]["steps"][0]["waits"]
                    .as_object()
                    .unwrap()
                    .values()
                    .all(|record| record["status"] == "resolved")
            );
        }
    }
}
#[tokio::test]
async fn child_owned_plan_waits_keep_their_full_native_path_through_parent_reply() {
    for parent_react in [false, true] {
        for mode in [false, true] {
            let left = Arc::new(Log::default());
            let right = Arc::new(Log::default());
            let left_child = SubAgent::new(
                "left",
                agent::react()
                    .model(PlannedModel::default())
                    .toolkit(Toolkit::new().tools([Lookup::new(left.clone(), false)]))
                    .permission(Ask(left.clone()))
                    .build()
                    .unwrap(),
            )
            .unwrap();
            let right_child = SubAgent::new(
                "right",
                agent::chat()
                    .model(Model {
                        requests: Default::default(),
                        multiple: false,
                    })
                    .toolkit(Toolkit::new().tools([Lookup::new(right.clone(), false)]))
                    .permission(Ask(right.clone()))
                    .build()
                    .unwrap(),
            )
            .unwrap();
            let children = vec![left_child, right_child];
            let store = Arc::new(core::checkpoint::InMemory::new());
            if parent_react {
                child_plan_reply(
                    agent::react()
                        .model(DelegatingModel::default())
                        .subagents(children)
                        .checkpoint(store.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                    store,
                    left,
                    right,
                    mode,
                )
                .await;
            } else {
                child_plan_reply(
                    agent::chat()
                        .model(DelegatingModel::default())
                        .subagents(children)
                        .checkpoint(store.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                    store,
                    left,
                    right,
                    mode,
                )
                .await;
            }
        }
    }
}

#[derive(Clone, Default)]
struct PlannedModel(Arc<Mutex<Vec<Request>>>);

struct PlannedDelegatingModel {
    model: PlannedModel,
    gate: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    started: Arc<AtomicUsize>,
}
impl PlannedDelegatingModel {
    fn new(model: PlannedModel) -> Self {
        Self {
            model,
            gate: Mutex::new(None),
            started: Default::default(),
        }
    }
}
impl Chat for PlannedDelegatingModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let node = task_id(&request).map(str::to_owned);
        let gate = if node.as_deref() == Some("b") {
            self.gate.lock().unwrap().take()
        } else {
            None
        };
        if let Some(gate) = gate {
            self.started.fetch_add(1, Ordering::SeqCst);
            gate.await.unwrap();
        }
        let mut events = self.model.stream(request).await?.collect::<Vec<_>>().await;
        for event in &mut events {
            if let Ok(Event::Complete { message, .. }) = event {
                for part in &mut message.content {
                    if part.r#type == "tool_call" && part.data.get("name") == Some(&json!("lookup"))
                    {
                        let name = if node.as_deref() == Some("a") {
                            "left"
                        } else {
                            "right"
                        };
                        part.data = json!({"call_id":"same_spawn","name":"agent_spawn","arguments":{
                            "name":name,"input":{"metadata":{"agent_id":if name=="left" {3002} else {3003}},"payload":[{"type":"text","data":{"value":format!("task {name}")}}]}
                        }}).as_object().unwrap().clone();
                    }
                }
            }
        }
        Ok(stream::iter(events))
    }
}

fn task_child_scene<'a>(state: &'a Value, task: &str, child: &str) -> &'a Value {
    &state["nodes"][task]["execution"]["attempts"][0]["nodes"][child]["execution"]["attempts"][0]["agent"]
}

#[tokio::test]
async fn planned_delegation_replies_keep_task_and_child_ownership_and_release_capacity() {
    for child_react in [false, true] {
        for mode in [false, true] {
            for capacity in [1, 2] {
                for approve_first in [false, true] {
                    let model = PlannedModel::default();
                    let (children, probes) = waiting_children(child_react);
                    let store = Arc::new(core::checkpoint::InMemory::new());
                    let agent = agent::react()
                        .model(PlannedDelegatingModel::new(model.clone()))
                        .subagents(children)
                        .with_max_concurrency(capacity)
                        .with_id_fn(next_id)
                        .checkpoint(store.clone())
                        .build()
                        .unwrap();
                    let mut events =
                        Box::pin(agent.run(input(), mode, Cancellation::new()).await.unwrap());
                    let (first, _) = question(&mut events).await;
                    let (second, _) = question(&mut events).await;
                    let waiting = store.load(1001, &Cancellation::new()).await.unwrap();
                    let waits = waiting["waits"].as_array().unwrap();
                    assert_eq!(waits.len(), 2);
                    assert_eq!(waiting["usage"]["total"], 78);
                    let mut ownership = Vec::new();
                    for wait in waits {
                        let path = wait["nodes"].as_array().unwrap();
                        assert_eq!(path.len(), 2);
                        let task = path[0]["node_id"].as_str().unwrap();
                        let child = path[1]["node_id"].as_str().unwrap();
                        let scene = task_child_scene(&waiting, task, child);
                        let record = &scene["steps"][0]["waits"];
                        let id = record
                            .as_object()
                            .unwrap()
                            .keys()
                            .next()
                            .unwrap()
                            .parse::<i64>()
                            .unwrap();
                        assert_eq!(scene["waits"][0]["nodes"], json!([]));
                        assert_eq!(scene["steps"][0]["call_id"], "c1");
                        assert!(
                            waiting["nodes"][task]["execution"]["attempts"][0]["steps"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .all(|step| step.get("waits").is_none())
                        );
                        ownership.push((
                            task.to_owned(),
                            child.to_owned(),
                            id,
                            path[0]["exec_id"].clone(),
                            path[1]["exec_id"].clone(),
                        ));
                    }
                    let fresh = agent::react()
                        .model(PlannedDelegatingModel::new(PlannedModel::default()))
                        .checkpoint(store.clone())
                        .build()
                        .unwrap();
                    assert!(
                        matches!(fresh.run(input(),mode,Cancellation::new()).await,Err(error) if error.code=="WAITING")
                    );
                    agent
                        .reply(answer(second, json!(true)), Cancellation::new())
                        .await
                        .unwrap();
                    let second_task = &ownership.iter().find(|item| item.2 == second).unwrap().0;
                    tokio::time::timeout(std::time::Duration::from_secs(2), async {
                        loop {
                            let event = events.next().await.unwrap().unwrap();
                            if !mode {
                                assert!(!matches!(event, Event::Delta { .. }));
                            }
                            if let Event::Complete { message, .. } = event
                                && message.values.get("result") == Some(&json!(second_task))
                            {
                                break;
                            }
                        }
                    })
                    .await
                    .unwrap();
                    assert_eq!(
                        probes
                            .iter()
                            .map(|probe| probe.log.calls.lock().unwrap().len())
                            .sum::<usize>(),
                        1
                    );
                    assert!(
                        model
                            .0
                            .lock()
                            .unwrap()
                            .iter()
                            .all(|request| task_id(request) != Some("c"))
                    );
                    agent
                        .reply(answer(second, json!(true)), Cancellation::new())
                        .await
                        .unwrap();
                    assert_eq!(
                        agent
                            .reply(answer(second, json!(false)), Cancellation::new())
                            .await
                            .unwrap_err()
                            .code,
                        "CONFLICT"
                    );
                    agent
                        .reply(answer(first, json!(approve_first)), Cancellation::new())
                        .await
                        .unwrap();
                    let output = finish(&mut events).await;
                    if !mode {
                        assert!(
                            output
                                .iter()
                                .all(|event| !matches!(event, Event::Delta { .. }))
                        );
                    }
                    let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
                    assert_eq!(final_state["status"], "succeeded");
                    assert_eq!(final_state["usage"]["total"], 156);
                    assert_eq!(final_state["waits"], json!([]));
                    for (task, child, id, task_exec, child_exec) in ownership {
                        let execution = &final_state["nodes"][&task]["execution"];
                        assert_eq!(execution["exec_id"], task_exec);
                        assert_eq!(execution["attempts"].as_array().unwrap().len(), 1);
                        assert_eq!(
                            execution["attempts"][0]["nodes"][&child]["execution"]["exec_id"],
                            child_exec
                        );
                        let scene = task_child_scene(&final_state, &task, &child);
                        let record = &scene["steps"][0]["waits"][id.to_string()];
                        assert_eq!(record["status"], "resolved");
                        assert_eq!(record["feedback"].as_array().unwrap().len(), 1);
                        assert_eq!(scene["usage"]["total"], 26);
                    }
                    assert_eq!(
                        probes
                            .iter()
                            .map(|probe| probe.log.calls.lock().unwrap().len())
                            .sum::<usize>(),
                        1 + usize::from(approve_first)
                    );
                    for probe in &probes {
                        assert_eq!(probe.checkpoint.loads.load(Ordering::SeqCst), 0);
                        assert_eq!(probe.checkpoint.saves.load(Ordering::SeqCst), 0);
                        for request in probe.model.requests.lock().unwrap().iter() {
                            let data = serde_json::to_string(request).unwrap();
                            assert!(!data.contains("reply_id"));
                            assert!(!data.contains("untrusted"));
                        }
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn planned_child_reply_reacquires_parent_capacity_before_executing() {
    for child_react in [false, true] {
        let model = PlannedModel::default();
        let started = Arc::new(AtomicUsize::new(0));
        let (release, gate) = tokio::sync::oneshot::channel();
        let (children, probes) = waiting_children(child_react);
        let agent = agent::react()
            .model(PlannedDelegatingModel {
                model: model.clone(),
                gate: Mutex::new(Some(gate)),
                started: started.clone(),
            })
            .subagents(children)
            .with_max_concurrency(1)
            .with_id_fn(next_id)
            .build()
            .unwrap();
        let mut events = Box::pin(
            agent
                .run(input(), false, Cancellation::new())
                .await
                .unwrap(),
        );
        let (first, _) = question(&mut events).await;
        assert!(poll!(events.next()).is_pending());
        assert_eq!(
            started.load(Ordering::SeqCst),
            1,
            "child wait must release the enclosing plan task slot"
        );
        agent
            .reply(answer(first, json!(true)), Cancellation::new())
            .await
            .unwrap();
        assert!(poll!(events.next()).is_pending());
        assert!(
            probes
                .iter()
                .all(|probe| probe.log.calls.lock().unwrap().is_empty()),
            "reply cannot bypass B's occupied task slot"
        );
        release.send(()).unwrap();
        let (second, _) = question(&mut events).await;
        agent
            .reply(answer(second, json!(true)), Cancellation::new())
            .await
            .unwrap();
        finish(&mut events).await;
        assert!(
            probes
                .iter()
                .all(|probe| probe.log.calls.lock().unwrap().len() == 1)
        );
    }
}

#[tokio::test]
async fn planned_children_cancel_or_drop_without_authorizing_saved_questions() {
    for child_react in [false, true] {
        for drop_stream in [false, true] {
            let model = PlannedModel::default();
            let (children, probes) = waiting_children(child_react);
            let store = Arc::new(core::checkpoint::InMemory::new());
            let agent = agent::react()
                .model(PlannedDelegatingModel::new(model.clone()))
                .subagents(children)
                .with_max_concurrency(1)
                .with_id_fn(next_id)
                .checkpoint(store.clone())
                .build()
                .unwrap();
            let cancellation = Cancellation::new();
            let mut events = Box::pin(
                agent
                    .run(input(), false, cancellation.clone())
                    .await
                    .unwrap(),
            );
            let (first, _) = question(&mut events).await;
            let (second, _) = question(&mut events).await;
            let saved = store.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(saved["usage"]["total"], 78);
            if drop_stream {
                drop(events);
                assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
                assert!(
                    matches!(agent.run(input(),false,Cancellation::new()).await,Err(error) if error.code=="WAITING")
                );
            } else {
                cancellation.cancel();
                let output = finish(&mut events).await;
                assert!(output.iter().any(|event|matches!(event,Event::Complete{finish_reason:Some(reason),..} if reason=="cancelled")));
                let stopped = store.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(stopped["status"], "stopped");
                assert_eq!(stopped["usage"]["total"], 78);
                assert_eq!(stopped["waits"], json!([]));
                for wait in saved["waits"].as_array().unwrap() {
                    let path = &wait["nodes"];
                    let task = path[0]["node_id"].as_str().unwrap();
                    let child = path[1]["node_id"].as_str().unwrap();
                    assert_eq!(stopped["nodes"][task]["status"], "cancelled");
                    assert_eq!(
                        task_child_scene(&stopped, task, child)["steps"][0]["waits"],
                        task_child_scene(&saved, task, child)["steps"][0]["waits"]
                    );
                }
            }
            for id in [first, second] {
                assert_eq!(
                    agent
                        .reply(answer(id, json!(true)), Cancellation::new())
                        .await
                        .unwrap_err()
                        .code,
                    "NOT_FOUND"
                );
            }
            assert!(
                probes
                    .iter()
                    .all(|probe| probe.log.calls.lock().unwrap().is_empty())
            );
            assert_eq!(model.0.lock().unwrap().len(), 4);
        }
    }
}

#[tokio::test]
async fn planned_child_own_plan_keeps_all_three_native_wait_coordinates() {
    for mode in [false, true] {
        let left = Arc::new(Log::default());
        let right = Arc::new(Log::default());
        let left_child = SubAgent::new(
            "left",
            agent::react()
                .model(PlannedModel::default())
                .toolkit(Toolkit::new().tools([Lookup::new(left.clone(), false)]))
                .permission(Ask(left.clone()))
                .with_max_concurrency(1)
                .build()
                .unwrap(),
        )
        .unwrap();
        let right_child = SubAgent::new(
            "right",
            agent::chat()
                .model(Model {
                    requests: Default::default(),
                    multiple: false,
                })
                .toolkit(Toolkit::new().tools([Lookup::new(right.clone(), false)]))
                .permission(Ask(right.clone()))
                .build()
                .unwrap(),
        )
        .unwrap();
        let store = Arc::new(core::checkpoint::InMemory::new());
        let agent = agent::react()
            .model(PlannedDelegatingModel::new(PlannedModel::default()))
            .subagents(vec![left_child, right_child])
            .with_max_concurrency(1)
            .with_id_fn(next_id)
            .with_max_iters(40)
            .checkpoint(store.clone())
            .build()
            .unwrap();
        let mut events = Box::pin(agent.run(input(), mode, Cancellation::new()).await.unwrap());
        let mut ids = Vec::new();
        for _ in 0..3 {
            ids.push(question(&mut events).await.0);
        }
        let saved = store.load(1001, &Cancellation::new()).await.unwrap();
        let waits = saved["waits"].as_array().unwrap();
        assert_eq!(waits.len(), 3);
        assert_eq!(
            waits
                .iter()
                .filter(|wait| wait["nodes"].as_array().unwrap().len() == 3)
                .count(),
            2
        );
        assert_eq!(saved["usage"]["total"], 117);
        let restored = agent::chat()
            .model(Model {
                requests: Default::default(),
                multiple: false,
            })
            .checkpoint(store.clone())
            .build()
            .unwrap();
        assert!(
            matches!(restored.run(input(),mode,Cancellation::new()).await,Err(error) if error.code=="WAITING")
        );
        for id in ids.into_iter().rev() {
            agent
                .reply(answer(id, json!(true)), Cancellation::new())
                .await
                .unwrap();
        }
        let output = finish(&mut events).await;
        if !mode {
            assert!(
                output
                    .iter()
                    .all(|event| !matches!(event, Event::Delta { .. }))
            );
        }
        assert_eq!(left.calls.lock().unwrap().len(), 2);
        assert_eq!(right.calls.lock().unwrap().len(), 1);
        let state = store.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(state["status"], "succeeded");
        assert_eq!(state["usage"]["total"], 234);
        assert_eq!(state["waits"], json!([]));
        for wait in waits {
            let path = wait["nodes"].as_array().unwrap();
            let task = path[0]["node_id"].as_str().unwrap();
            let child = path[1]["node_id"].as_str().unwrap();
            let scene = task_child_scene(&state, task, child);
            if path.len() == 3 {
                let inner = path[2]["node_id"].as_str().unwrap();
                let execution = &scene["nodes"][inner]["execution"];
                assert_eq!(execution["exec_id"], path[2]["exec_id"]);
                assert_eq!(execution["attempts"].as_array().unwrap().len(), 1);
                assert!(
                    execution["attempts"][0]["steps"][0]["waits"]
                        .as_object()
                        .unwrap()
                        .values()
                        .all(|record| record["status"] == "resolved")
                );
            }
        }
    }
}

#[tokio::test]
async fn planned_child_question_save_failure_never_presents_or_executes_it() {
    for child_react in [false, true] {
        let model = PlannedModel::default();
        let (children, probes) = waiting_children(child_react);
        let store = Arc::new(ChildQuestionStore::default());
        let agent = agent::react()
            .model(PlannedDelegatingModel::new(model.clone()))
            .subagents(children)
            .with_max_concurrency(1)
            .with_id_fn(next_id)
            .checkpoint(store.clone())
            .build()
            .unwrap();
        let output = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            agent
                .run(input(), false, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        })
        .await
        .unwrap();
        assert!(
            matches!(output.last(),Some(Err(error)) if error.message=="child question save failed")
        );
        assert!(output.iter().all(|event|!matches!(event,Ok(Event::Complete{message,..}) if message.metadata.contains_key("reply_id"))));
        let state = store.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(state["status"], "failed");
        assert_eq!(state["waits"], json!([]));
        let calls = model.0.lock().unwrap().len()
            + probes
                .iter()
                .map(|probe| probe.model.requests.lock().unwrap().len())
                .sum::<usize>();
        assert_eq!(state["usage"]["total"], 13 * calls);
        assert!(
            probes
                .iter()
                .all(|probe| probe.log.calls.lock().unwrap().is_empty())
        );
        let cached = agent
            .run(input(), false, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(
            matches!(cached.as_slice(),[Err(error)] if error.message=="child question save failed")
        );
    }
}

#[tokio::test]
async fn cancelling_planned_child_question_save_does_not_publish_the_question() {
    for child_react in [false, true] {
        let (children, probes) = waiting_children(child_react);
        let (release, gate) = tokio::sync::oneshot::channel();
        let store = Arc::new(DelayedChildStore {
            inner: Default::default(),
            gate: Mutex::new(Some(gate)),
            blocked: AtomicUsize::new(0),
        });
        let agent = agent::react()
            .model(PlannedDelegatingModel::new(PlannedModel::default()))
            .subagents(children)
            .with_max_concurrency(1)
            .with_id_fn(next_id)
            .checkpoint(store.clone())
            .build()
            .unwrap();
        let cancellation = Cancellation::new();
        let mut events = Box::pin(
            agent
                .run(input(), false, cancellation.clone())
                .await
                .unwrap(),
        );
        let mut reached_save = false;
        for _ in 0..20 {
            match poll!(events.next()) {
                std::task::Poll::Pending => {
                    if store.blocked.load(Ordering::SeqCst) > 0 {
                        reached_save = true;
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                std::task::Poll::Ready(Some(Ok(Event::Complete { message, .. }))) => {
                    assert!(!message.metadata.contains_key("reply_id"))
                }
                std::task::Poll::Ready(Some(Ok(_))) => {}
                other => panic!("ended before pending question save: {other:?}"),
            }
        }
        assert!(reached_save);
        cancellation.cancel();
        release.send(()).unwrap();
        let output = finish(&mut events).await;
        assert!(output.iter().all(|event|!matches!(event,Event::Complete{message,..} if message.metadata.contains_key("reply_id"))));
        let state = store.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(state["status"], "stopped");
        assert_eq!(state["waits"], json!([]));
        assert!(
            probes
                .iter()
                .all(|probe| probe.log.calls.lock().unwrap().is_empty())
        );
    }
}

struct BusyTool {
    definition: ToolDefinition,
    gate: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    started: Arc<AtomicUsize>,
}
impl Tool for BusyTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        let gate = self.gate.lock().unwrap().take().unwrap();
        self.started.fetch_add(1, Ordering::SeqCst);
        gate.await.unwrap();
        Ok(Message::function(json!({"done":true})))
    }
}
struct BusyDelegatingModel(PlannedDelegatingModel);
impl Chat for BusyDelegatingModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let task = task_id(&request).map(str::to_owned);
        let mut events = self.0.stream(request).await?.collect::<Vec<_>>().await;
        for event in &mut events {
            if let Ok(Event::Complete { message, .. }) = event {
                if message.values.contains_key("nodes") {
                    message
                        .values
                        .get_mut("nodes")
                        .unwrap()
                        .as_object_mut()
                        .unwrap()
                        .insert(
                            "d".into(),
                            json!({"name":"extra","objective":"independent third task"}),
                        );
                    message
                        .values
                        .get_mut("edges")
                        .unwrap()
                        .as_array_mut()
                        .unwrap()
                        .push(json!({"from":"d","to":"c"}));
                }
                if task.as_deref() == Some("a")
                    && message
                        .content
                        .iter()
                        .any(|part| part.data.get("name") == Some(&json!("agent_spawn")))
                {
                    message.content.push(Part {
                        r#type: "tool_call".into(),
                        data: json!({"call_id":"busy_call","name":"busy","arguments":{}})
                            .as_object()
                            .unwrap()
                            .clone(),
                    });
                }
            }
        }
        Ok(stream::iter(events))
    }
}

#[tokio::test]
async fn child_wait_does_not_release_a_task_still_running_another_tool() {
    for child_react in [false, true] {
        let model = PlannedModel::default();
        let b_started = Arc::new(AtomicUsize::new(0));
        let busy_started = Arc::new(AtomicUsize::new(0));
        let (_b_release, b_gate) = tokio::sync::oneshot::channel();
        let (_busy_release, busy_gate) = tokio::sync::oneshot::channel();
        let (children, probes) = waiting_children(child_react);
        let busy = BusyTool {
            definition: ToolDefinition {
                name: "busy".into(),
                description: "retained active work".into(),
                parameters: json!({"type":"object"}).as_object().unwrap().clone(),
            },
            gate: Mutex::new(Some(busy_gate)),
            started: busy_started.clone(),
        };
        let agent = agent::react()
            .model(BusyDelegatingModel(PlannedDelegatingModel {
                model: model.clone(),
                gate: Mutex::new(Some(b_gate)),
                started: b_started.clone(),
            }))
            .toolkit(Toolkit::new().tools([busy]))
            .subagents(children)
            .with_max_concurrency(2)
            .with_id_fn(next_id)
            .build()
            .unwrap();
        let cancellation = Cancellation::new();
        let mut events = Box::pin(
            agent
                .run(input(), false, cancellation.clone())
                .await
                .unwrap(),
        );
        let (id, _) = question(&mut events).await;
        assert!(poll!(events.next()).is_pending());
        assert_eq!(b_started.load(Ordering::SeqCst), 1);
        assert_eq!(busy_started.load(Ordering::SeqCst), 1);
        assert!(
            model
                .0
                .lock()
                .unwrap()
                .iter()
                .all(|request| task_id(request) != Some("d")),
            "A still runs busy and B still runs its model; a child approval wait must not admit D"
        );
        cancellation.cancel();
        finish(&mut events).await;
        assert!(
            probes
                .iter()
                .all(|probe| probe.log.calls.lock().unwrap().is_empty())
        );
        assert_eq!(
            agent
                .reply(answer(id, json!(true)), Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
    }
}

fn task_id(request: &Request) -> Option<&str> {
    request
        .messages
        .as_slice()
        .iter()
        .find_map(|message| message.values.get("plan_node").and_then(Value::as_str))
}
impl Chat for PlannedModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        let mut requests = self.0.lock().unwrap();
        if let Some(node) = task_id(&request) {
            if node == "c"
                || requests
                    .iter()
                    .any(|previous| task_id(previous) == Some(node))
            {
                message.values.insert("result".into(), json!(node));
            } else {
                // Deliberately reuse the Tool Call ID in distinct tasks.
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: json!({"call_id":"same_call","name":"lookup","arguments":{"city":node}})
                        .as_object()
                        .unwrap()
                        .clone(),
                });
            }
        } else if requests.is_empty() {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"plan_1","name":"plan","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        } else if request
            .messages
            .as_slice()
            .iter()
            .any(|message| message.values.contains_key("plan_complete"))
        {
            message.values.insert("answer".into(), json!("all done"));
        } else {
            message.values = json!({"version":1,"nodes":{
                "a":{"name":"lookup","objective":"look up a"},
                "b":{"name":"lookup","objective":"look up b"},
                "c":{"name":"summarize","objective":"combine both"}
            },"edges":[{"from":"a","to":"c"},{"from":"b","to":"c"}]})
            .as_object()
            .unwrap()
            .clone();
        }
        requests.push(request);
        Ok(stream::iter(vec![
            Ok(Event::Delta {
                message: Message::new(Role::Assistant),
            }),
            Ok(Event::Complete {
                message,
                usage: Some(Usage {
                    input: Some(10),
                    output: Some(3),
                    total: Some(13),
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
            }),
        ]))
    }
}

#[tokio::test]
async fn planned_questions_keep_native_owners_and_continue_only_the_answered_task() {
    for (mode, approve_first) in [false, true]
        .into_iter()
        .flat_map(|mode| [false, true].map(|approve| (mode, approve)))
    {
        let log = Arc::new(Log::default());
        let model = PlannedModel::default();
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let agent = agent::react()
            .model(model.clone())
            .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
            .permission(Ask(log.clone()))
            .with_max_concurrency(1)
            .with_id_fn(next_id)
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap();
        let mut events = Box::pin(agent.run(input(), mode, Cancellation::new()).await.unwrap());
        let (first, first_question) = question(&mut events).await;
        let (second, second_question) = question(&mut events).await;
        assert_ne!(first, second);
        let a = first_question.metadata["node_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let b = second_question.metadata["node_id"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_ne!(a, b);
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(saved["status"], "waiting");
        assert_eq!(saved["waits"].as_array().unwrap().len(), 2);
        assert_eq!(saved["usage"]["total"], 52);
        assert_eq!(saved["waits"][0]["nodes"][0]["node_id"], a);
        assert_eq!(saved["waits"][1]["nodes"][0]["node_id"], b);
        assert!(saved["steps"].as_array().is_none_or(Vec::is_empty));
        let identities = [&a, &b].map(|node| saved["nodes"][node]["execution"]["exec_id"].clone());
        for (node, id) in [(&a, first), (&b, second)] {
            let attempt = &saved["nodes"][node]["execution"]["attempts"][0];
            assert_eq!(saved["nodes"][node]["status"], "waiting");
            assert_eq!(attempt["status"], "waiting");
            assert_eq!(attempt["steps"][0]["call_id"], "same_call");
            assert_eq!(
                attempt["steps"][0]["waits"][id.to_string()]["status"],
                "waiting"
            );
            assert!(
                !saved["nodes"][node]["messages"]
                    .to_string()
                    .contains("Confirm operation")
            );
            assert!(
                !attempt["messages"]
                    .to_string()
                    .contains("Confirm operation")
            );
        }
        assert!(log.calls.lock().unwrap().is_empty());
        // A new instance exercises restore-time native coverage validation,
        // not the original live Hub. This scope still refuses process resume.
        let restored = agent::react()
            .model(PlannedModel::default())
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap();
        assert!(
            matches!(restored.run(input(),mode,Cancellation::new()).await,Err(error) if error.code=="WAITING")
        );
        assert_eq!(
            checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
            saved
        );

        agent
            .reply(answer(second, json!(true)), Cancellation::new())
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let event = events.next().await.unwrap().unwrap();
                if !mode {
                    assert!(!matches!(event, Event::Delta { .. }));
                }
                if let Event::Complete { message, .. } = event
                    && message.metadata.get("node_id").and_then(Value::as_str) == Some(&b)
                    && message.values.get("result").and_then(Value::as_str) == Some(&b)
                {
                    break;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(log.calls.lock().unwrap().len(), 1);
        assert_eq!(log.calls.lock().unwrap()[0].arguments["city"], b);
        assert!(
            model
                .0
                .lock()
                .unwrap()
                .iter()
                .all(|request| task_id(request) != Some("c"))
        );
        let one_left = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(one_left["waits"].as_array().unwrap().len(), 1);
        assert_eq!(one_left["waits"][0]["nodes"][0]["node_id"], a);
        assert_eq!(one_left["nodes"][&b]["status"], "succeeded");
        // Completed branch history accepts the same late retry, never A.
        agent
            .reply(answer(second, json!(true)), Cancellation::new())
            .await
            .unwrap();
        assert_eq!(
            agent
                .reply(answer(second, json!(false)), Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "CONFLICT"
        );
        assert!(poll!(events.next()).is_pending());
        agent
            .reply(answer(first, json!(approve_first)), Cancellation::new())
            .await
            .unwrap();
        let mut tail = Vec::new();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while let Some(event) = events.next().await {
                tail.push(event.unwrap());
            }
        })
        .await
        .unwrap();
        if !mode {
            assert!(
                tail.iter()
                    .all(|event| !matches!(event, Event::Delta { .. }))
            );
        }
        assert_eq!(
            log.calls.lock().unwrap().len(),
            1 + usize::from(approve_first)
        );
        assert_eq!(
            log.before.load(Ordering::SeqCst),
            1 + usize::from(approve_first)
        );
        let final_state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(final_state["status"], "succeeded");
        assert_eq!(final_state["waits"], json!([]));
        assert_eq!(final_state["usage"]["total"], 104);
        for ((node, id), exec) in [(&a, first), (&b, second)].into_iter().zip(identities) {
            let execution = &final_state["nodes"][node]["execution"];
            assert_eq!(execution["exec_id"], exec);
            assert_eq!(execution["attempts"].as_array().unwrap().len(), 1);
            assert_eq!(
                execution["attempts"][0]["steps"][0]["waits"][id.to_string()]["status"],
                "resolved"
            );
        }
        assert!(
            model
                .0
                .lock()
                .unwrap()
                .iter()
                .all(|request| !serde_json::to_string(request).unwrap().contains("reply_id"))
        );
    }
}

#[tokio::test]
async fn planned_waits_release_consumers_on_cancel_or_drop_without_fabricated_feedback() {
    for drop_stream in [false, true] {
        let log = Arc::new(Log::default());
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let model = PlannedModel::default();
        let agent = agent::react()
            .model(model.clone())
            .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
            .permission(Ask(log.clone()))
            .checkpoint(checkpoint.clone())
            .with_id_fn(next_id)
            .build()
            .unwrap();
        let cancellation = Cancellation::new();
        let mut events = Box::pin(
            agent
                .run(input(), false, cancellation.clone())
                .await
                .unwrap(),
        );
        let (first, _) = question(&mut events).await;
        let (second, _) = question(&mut events).await;
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        if drop_stream {
            drop(events);
            assert_eq!(
                checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
                saved
            );
            assert!(
                matches!(agent.run(input(),false,Cancellation::new()).await,Err(error) if error.code=="WAITING")
            );
        } else {
            cancellation.cancel();
            let output = finish(&mut events).await;
            assert!(output.iter().any(|event| matches!(event,Event::Complete{finish_reason:Some(reason),..} if reason=="cancelled")));
            let stopped = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(stopped["status"], "stopped");
            assert_eq!(stopped["waits"], json!([]));
            assert_eq!(stopped["usage"]["total"], 52);
            for node in ["a", "b"] {
                assert_eq!(stopped["nodes"][node]["status"], "cancelled");
                let step = &stopped["nodes"][node]["execution"]["attempts"][0]["steps"][0];
                assert_eq!(step["status"], "cancelled");
                assert!(
                    step["waits"]
                        .as_object()
                        .unwrap()
                        .values()
                        .all(|record| record["status"] == "waiting"
                            && record["feedback"].as_array().unwrap().is_empty())
                );
            }
            let cached = agent
                .run(input(), false, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert!(
                matches!(cached.as_slice(),[Ok(Event::Complete{finish_reason:Some(reason),..})] if reason=="cancelled")
            );
        }
        for id in [first, second] {
            assert_eq!(
                agent
                    .reply(answer(id, json!(true)), Cancellation::new())
                    .await
                    .unwrap_err()
                    .code,
                "NOT_FOUND"
            );
        }
        assert!(log.calls.lock().unwrap().is_empty());
        assert_eq!(model.0.lock().unwrap().len(), 4);
    }
}

struct GatedPlannedModel {
    model: PlannedModel,
    gate: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    started: Arc<AtomicUsize>,
}

struct MultiPlannedModel(PlannedModel);
impl Chat for MultiPlannedModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let a = task_id(&request) == Some("a");
        let mut events = self.0.stream(request).await?.collect::<Vec<_>>().await;
        if a {
            for event in &mut events {
                if let Ok(Event::Complete { message, .. }) = event
                    && message
                        .content
                        .iter()
                        .any(|part| part.r#type == "tool_call")
                {
                    message.content.push(Part {r#type:"tool_call".into(),data:json!({"call_id":"queued_call","name":"lookup","arguments":{"city":"a2"}}).as_object().unwrap().clone()});
                }
            }
        }
        Ok(stream::iter(events))
    }
}

#[tokio::test]
async fn queued_tool_calls_do_not_keep_an_externally_waiting_task_admitted() {
    let log = Arc::new(Log::default());
    let agent = agent::react()
        .model(MultiPlannedModel(PlannedModel::default()))
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .with_max_concurrency(1)
        .with_id_fn(next_id)
        .build()
        .unwrap();
    let mut events = Box::pin(
        agent
            .run(input(), false, Cancellation::new())
            .await
            .unwrap(),
    );
    let (a, first) = question(&mut events).await;
    let (b, second) = question(&mut events).await;
    assert_eq!(first.metadata["node_id"], "a");
    assert_eq!(
        second.metadata["node_id"], "b",
        "A's queued, unstarted Tool is not active work"
    );
    agent
        .reply(answer(b, json!(true)), Cancellation::new())
        .await
        .unwrap();
    loop {
        let event = tokio::time::timeout(std::time::Duration::from_secs(2), events.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if let Event::Complete { message, .. } = event
            && message.values.get("result") == Some(&json!("b"))
        {
            break;
        }
    }
    agent
        .reply(answer(a, json!(true)), Cancellation::new())
        .await
        .unwrap();
    let (queued, third) = question(&mut events).await;
    assert_eq!(third.metadata["node_id"], "a");
    assert_ne!(queued, a);
    assert_eq!(log.calls.lock().unwrap().len(), 2);
    agent
        .reply(answer(queued, json!(true)), Cancellation::new())
        .await
        .unwrap();
    finish(&mut events).await;
    assert_eq!(log.calls.lock().unwrap().len(), 3);
}
impl Chat for GatedPlannedModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let gate = if task_id(&request) == Some("b") {
            self.gate.lock().unwrap().take()
        } else {
            None
        };
        if let Some(gate) = gate {
            self.started.fetch_add(1, Ordering::SeqCst);
            gate.await.unwrap();
        }
        self.model.stream(request).await
    }
}

#[tokio::test]
async fn answered_task_reacquires_capacity_before_tool_execution() {
    let log = Arc::new(Log::default());
    let model = PlannedModel::default();
    let started = Arc::new(AtomicUsize::new(0));
    let (release, gate) = tokio::sync::oneshot::channel();
    let agent = agent::react()
        .model(GatedPlannedModel {
            model: model.clone(),
            gate: Mutex::new(Some(gate)),
            started: started.clone(),
        })
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .with_max_concurrency(1)
        .with_id_fn(next_id)
        .build()
        .unwrap();
    let mut events = Box::pin(
        agent
            .run(input(), false, Cancellation::new())
            .await
            .unwrap(),
    );
    let (a, question_a) = question(&mut events).await;
    assert_eq!(question_a.metadata["node_id"], "a");
    assert!(poll!(events.next()).is_pending());
    assert_eq!(
        started.load(Ordering::SeqCst),
        1,
        "A's human wait must release capacity for B's model"
    );
    agent
        .reply(answer(a, json!(true)), Cancellation::new())
        .await
        .unwrap();
    assert!(poll!(events.next()).is_pending());
    assert!(
        log.calls.lock().unwrap().is_empty(),
        "approval cannot execute A while B occupies the only node slot"
    );
    release.send(()).unwrap();
    let (b, question_b) = question(&mut events).await;
    assert_eq!(question_b.metadata["node_id"], "b");
    // B now waits. A may finish, but C still needs B's committed result.
    loop {
        let event = tokio::time::timeout(std::time::Duration::from_secs(2), events.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if let Event::Complete { message, .. } = event
            && message.values.get("result") == Some(&json!("a"))
        {
            break;
        }
    }
    assert_eq!(log.calls.lock().unwrap().len(), 1);
    assert!(
        model
            .0
            .lock()
            .unwrap()
            .iter()
            .all(|request| task_id(request) != Some("c"))
    );
    agent
        .reply(answer(b, json!(true)), Cancellation::new())
        .await
        .unwrap();
    finish(&mut events).await;
    assert_eq!(log.calls.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn planned_approval_save_failure_does_not_leave_actionable_orphan_waits() {
    let log = Arc::new(Log::default());
    let store = Arc::new(FailingStore::default());
    let agent = agent::react()
        .model(PlannedModel::default())
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .checkpoint(store.clone())
        .with_id_fn(next_id)
        .build()
        .unwrap();
    let mut events = Box::pin(
        agent
            .run(input(), false, Cancellation::new())
            .await
            .unwrap(),
    );
    let output = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        let mut out = Vec::new();
        while let Some(event) = events.next().await {
            out.push(event);
        }
        out
    })
    .await
    .unwrap();
    assert!(matches!(output.last(),Some(Err(error)) if error.message=="waiting save failed"));
    assert!(output.iter().all(|event| !matches!(event,Ok(Event::Complete{message,..}) if message.metadata.contains_key("reply_id"))));
    let failed = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["waits"], json!([]));
    let cached = agent
        .run(input(), false, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(cached.as_slice(),[Err(error)] if error.message=="waiting save failed"));
    assert!(log.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn restored_plan_save_failure_does_not_rebill_previous_task_usage() {
    // Build the native selected-call scene with real execution, then retain
    // the stage immediately before asking. This is NOT restoring an approved
    // question: native human waits still fail closed on process restoration.
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let original = agent::react()
        .model(PlannedModel::default())
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .checkpoint(checkpoint.clone())
        .with_id_fn(next_id)
        .build()
        .unwrap();
    let mut events = Box::pin(
        original
            .run(input(), false, Cancellation::new())
            .await
            .unwrap(),
    );
    question(&mut events).await;
    question(&mut events).await;
    let mut scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    drop(events);
    scene["status"] = json!("running");
    scene["waits"] = json!([]);
    for node in ["a", "b"] {
        scene["nodes"][node]["status"] = json!("running");
        let attempt = &mut scene["nodes"][node]["execution"]["attempts"][0];
        attempt["status"] = json!("running");
        attempt["steps"][0]["status"] = json!("running");
        attempt["steps"][0].as_object_mut().unwrap().remove("waits");
    }
    assert_eq!(scene["usage"]["total"], 52);
    let store = Arc::new(FailingStore::default());
    store.save(1001, scene, &Cancellation::new()).await.unwrap();
    let model = PlannedModel::default();
    let agent = agent::react()
        .model(model.clone())
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .checkpoint(store.clone())
        .with_id_fn(next_id)
        .build()
        .unwrap();
    let output = agent
        .run(input(), false, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(output.last(),Some(Err(error)) if error.message=="waiting save failed"));
    let failed = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        failed["usage"]["total"], 52,
        "saved task usage was already settled before restoration"
    );
    assert!(model.0.lock().unwrap().is_empty());
    assert!(log.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn cancellation_during_planned_question_save_never_presents_that_question() {
    let log = Arc::new(Log::default());
    let (release, gate) = tokio::sync::oneshot::channel();
    let store = Arc::new(DelayedStore {
        inner: core::checkpoint::InMemory::new(),
        blocked: AtomicUsize::new(0),
        gate: Mutex::new(Some(gate)),
    });
    let agent = agent::react()
        .model(PlannedModel::default())
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .checkpoint(store.clone())
        .with_id_fn(next_id)
        .build()
        .unwrap();
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(input(), false, cancellation.clone())
            .await
            .unwrap(),
    );
    let mut prefix = Vec::new();
    loop {
        match poll!(events.next()) {
            std::task::Poll::Ready(Some(event)) => prefix.push(event.unwrap()),
            std::task::Poll::Pending => break,
            _ => panic!("expected pending question save"),
        }
    }
    assert_eq!(store.blocked.load(Ordering::SeqCst), 1);
    cancellation.cancel();
    release.send(()).unwrap();
    prefix.extend(finish(&mut events).await);
    assert!(prefix.iter().all(|event| !matches!(event,Event::Complete{message,..} if message.metadata.contains_key("reply_id"))));
    assert!(log.calls.lock().unwrap().is_empty());
    let stopped = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(stopped["status"], "stopped");
    assert_eq!(stopped["waits"], json!([]));
}

async fn question<S: futures_core::Stream<Item = Result<Event, Error>> + Unpin>(
    events: &mut S,
) -> (i64, Message) {
    loop {
        let event = tokio::time::timeout(std::time::Duration::from_secs(2), events.next())
            .await
            .expect("question should arrive")
            .unwrap()
            .unwrap();
        if let Event::Complete { message, .. } = event
            && let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64)
        {
            return (id, message);
        }
    }
}
async fn finish<S: futures_core::Stream<Item = Result<Event, Error>> + Unpin>(
    events: &mut S,
) -> Vec<Event> {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        let mut output = Vec::new();
        while let Some(event) = events.next().await {
            output.push(event.unwrap());
        }
        output
    })
    .await
    .expect("original run should continue")
}

async fn exercise<A: Agent>(
    agent: A,
    checkpoint: Arc<core::checkpoint::InMemory>,
    model: Model,
    log: Arc<Log>,
    mode: bool,
    approve: bool,
) {
    let mut events = Box::pin(agent.run(input(), mode, Cancellation::new()).await.unwrap());
    let (id, question) = question(&mut events).await;
    assert_eq!(question.metadata["chat_id"], 1001);
    assert_eq!(model.requests.lock().unwrap().len(), 1);
    assert_eq!(log.before.load(Ordering::SeqCst), 0);
    assert!(log.calls.lock().unwrap().is_empty());
    let waiting = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(waiting["status"], "waiting");
    assert_eq!(waiting["waits"].as_array().unwrap().len(), 1);
    assert_eq!(
        waiting["steps"][0]["waits"][id.to_string()]["status"],
        "waiting"
    );
    let _: halo_agents::runtime::agent::State = serde_json::from_value(waiting.clone()).unwrap();
    let mut invalid = answer(id, json!(true));
    invalid.metadata.insert("session_id".into(), json!(2002));
    assert_eq!(
        agent
            .reply(invalid, Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    assert_eq!(
        agent
            .reply(answer(id + 100_000, json!(true)), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert_eq!(
        agent
            .reply(answer(id, json!("true")), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        waiting
    );
    agent
        .reply(answer(id, json!(approve)), Cancellation::new())
        .await
        .unwrap();
    agent
        .reply(answer(id, json!(approve)), Cancellation::new())
        .await
        .unwrap();
    assert_eq!(
        agent
            .reply(answer(id, json!("changed")), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    assert_eq!(
        agent
            .reply(answer(id, json!(!approve)), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    // Submission did not execute work or require polling a second stream.
    assert!(log.calls.lock().unwrap().is_empty());
    let output = finish(&mut events).await;
    if !mode {
        assert!(
            output
                .iter()
                .all(|event| matches!(event, Event::Complete { .. }))
        );
    }
    assert_eq!(log.calls.lock().unwrap().len(), usize::from(approve));
    assert_eq!(log.before.load(Ordering::SeqCst), usize::from(approve));
    assert_eq!(log.after.load(Ordering::SeqCst), usize::from(approve));
    if approve {
        assert_eq!(log.calls.lock().unwrap()[0].arguments["city"], "Beijing");
    }
    assert_eq!(
        log.checks.load(Ordering::SeqCst),
        1,
        "approved call must not re-enter an unchanged ask policy"
    );
    {
        let requests = model.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[1].messages.as_slice().iter().all(|message| {
            !message.metadata.contains_key("reply_id")
                && message.content.iter().all(|part| {
                    part.data
                        .get("value")
                        .is_none_or(|value| value.get("approved").is_none())
                })
        }));
        if !approve {
            assert!(requests[1].messages.as_slice().iter().any(|message| {
                message
                    .values
                    .get("error")
                    .is_some_and(|error| error["code"] == "PERMISSION_DENIED")
            }));
        }
    }
    let final_state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(final_state["status"], "succeeded");
    assert_eq!(final_state["waits"], json!([]));
    assert_eq!(final_state["usage"]["total"], 26);
    assert_eq!(
        final_state["steps"][0]["waits"][id.to_string()]["status"],
        "resolved"
    );
    assert_eq!(
        final_state["steps"][0]["waits"][id.to_string()]["feedback"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        agent
            .reply(answer(id, json!(approve)), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
}

#[tokio::test]
async fn root_chat_and_react_consume_approval_on_the_original_stream() {
    for react in [false, true] {
        for mode in [false, true] {
            for approve in [false, true] {
                let log = Arc::new(Log::default());
                let model = Model {
                    requests: Default::default(),
                    multiple: false,
                };
                let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                let toolkit = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
                if react {
                    exercise(
                        agent::react()
                            .model(model.clone())
                            .toolkit(toolkit)
                            .permission(Ask(log.clone()))
                            .checkpoint(checkpoint.clone())
                            .with_id_fn(next_id)
                            .build()
                            .unwrap(),
                        checkpoint,
                        model,
                        log,
                        mode,
                        approve,
                    )
                    .await;
                } else {
                    exercise(
                        agent::chat()
                            .model(model.clone())
                            .toolkit(toolkit)
                            .permission(Ask(log.clone()))
                            .checkpoint(checkpoint.clone())
                            .with_id_fn(next_id)
                            .build()
                            .unwrap(),
                        checkpoint,
                        model,
                        log,
                        mode,
                        approve,
                    )
                    .await;
                }
            }
        }
    }
}

async fn changed<A: Agent>(
    agent: A,
    checkpoint: Arc<core::checkpoint::InMemory>,
    model: Model,
    log: Arc<Log>,
) {
    let mut events = Box::pin(
        agent
            .run(input(), false, Cancellation::new())
            .await
            .unwrap(),
    );
    let (first, _) = question(&mut events).await;
    agent
        .reply(answer(first, json!(true)), Cancellation::new())
        .await
        .unwrap();
    let (second, question) = question(&mut events).await;
    assert_ne!(first, second);
    assert!(
        question.content[0].data["value"]
            .as_str()
            .unwrap()
            .contains("Paris")
    );
    agent
        .reply(answer(first, json!(true)), Cancellation::new())
        .await
        .unwrap();
    assert!(
        poll!(events.next()).is_pending(),
        "old duplicate must not release the new question"
    );
    assert!(log.calls.lock().unwrap().is_empty());
    let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(state["steps"][0]["call"]["arguments"]["city"], "Paris");
    assert_eq!(state["steps"][0]["waits"].as_object().unwrap().len(), 2);
    assert_eq!(
        state["steps"][0]["waits"][first.to_string()]["status"],
        "resolved"
    );
    agent
        .reply(answer(second, json!(false)), Cancellation::new())
        .await
        .unwrap();
    finish(&mut events).await;
    assert!(log.calls.lock().unwrap().is_empty());
    assert_eq!(log.before.load(Ordering::SeqCst), 1);
    assert_eq!(model.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn before_changed_arguments_need_their_own_question_without_reusing_approval() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let model = Model {
            requests: Default::default(),
            multiple: false,
        };
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let toolkit = Toolkit::new().tools([Lookup::new(log.clone(), true)]);
        if react {
            changed(
                agent::react()
                    .model(model.clone())
                    .toolkit(toolkit)
                    .permission(Ask(log.clone()))
                    .checkpoint(checkpoint.clone())
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                checkpoint,
                model,
                log,
            )
            .await;
        } else {
            changed(
                agent::chat()
                    .model(model.clone())
                    .toolkit(toolkit)
                    .permission(Ask(log.clone()))
                    .checkpoint(checkpoint.clone())
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                checkpoint,
                model,
                log,
            )
            .await;
        }
    }
}

async fn parallel<A: Agent>(agent: A, model: Model, log: Arc<Log>) {
    let mut events = Box::pin(agent.run(input(), true, Cancellation::new()).await.unwrap());
    let (first, _) = question(&mut events).await;
    let (second, _) = question(&mut events).await;
    assert_ne!(first, second);
    agent
        .reply(answer(second, json!(true)), Cancellation::new())
        .await
        .unwrap();
    // Poll the original execution once; Chat may retain the batch's output
    // until its sibling resolves, while ReAct may emit the completed Tool.
    let _ = poll!(events.next());
    assert_eq!(log.calls.lock().unwrap().len(), 1);
    assert_eq!(model.requests.lock().unwrap().len(), 1);
    agent
        .reply(answer(first, json!(false)), Cancellation::new())
        .await
        .unwrap();
    finish(&mut events).await;
    assert_eq!(
        log.calls.lock().unwrap().len(),
        1,
        "one approval cannot release both operations"
    );
    assert_eq!(model.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn concurrent_questions_accept_the_targeted_answer_without_broadcasting() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let model = Model {
            requests: Default::default(),
            multiple: true,
        };
        let toolkit = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
        if react {
            parallel(
                agent::react()
                    .model(model.clone())
                    .toolkit(toolkit)
                    .permission(Ask(log.clone()))
                    .with_max_concurrency(2)
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                model,
                log,
            )
            .await;
        } else {
            parallel(
                agent::chat()
                    .model(model.clone())
                    .toolkit(toolkit)
                    .permission(Ask(log.clone()))
                    .with_max_concurrency(2)
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                model,
                log,
            )
            .await;
        }
    }
}

async fn cancelled<A: Agent>(
    agent: A,
    checkpoint: Arc<core::checkpoint::InMemory>,
    log: Arc<Log>,
    drop_stream: bool,
) {
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(input(), false, cancellation.clone())
            .await
            .unwrap(),
    );
    let (id, _) = question(&mut events).await;
    if drop_stream {
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        drop(events);
        assert_eq!(
            agent
                .reply(answer(id, json!(true)), Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
        let error = match agent.run(input(), true, Cancellation::new()).await {
            Ok(_) => panic!("saved waiting facts must not be replayed as a new run"),
            Err(error) => error,
        };
        assert_eq!(
            error.code, "WAITING",
            "dropped run must not retain a live routing handle"
        );
        assert_eq!(
            checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
            saved
        );
    } else {
        cancellation.cancel();
        let output = finish(&mut events).await;
        assert!(
            matches!(output.as_slice(),[Event::Complete {finish_reason:Some(reason),..}] if reason == "cancelled")
        );
        assert_eq!(
            agent
                .reply(answer(id, json!(true)), Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(saved["status"], "stopped");
        assert_eq!(
            saved["usage"]["total"], 13,
            "waiting/cancellation must not rebill the completed model call"
        );
        assert_eq!(saved["waits"], json!([]));
        assert_eq!(saved["steps"][0]["status"], "cancelled");
        assert_eq!(
            saved["steps"][0]["waits"][id.to_string()]["status"],
            "waiting",
            "cancellation is not approval"
        );
        let cached = agent
            .run(input(), false, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(cached.iter().all(Result::is_ok));
    }
    assert!(log.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn cancellation_and_stream_drop_release_live_handoffs_without_running_tools() {
    for react in [false, true] {
        for drop_stream in [false, true] {
            let log = Arc::new(Log::default());
            let model = Model {
                requests: Default::default(),
                multiple: false,
            };
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let toolkit = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
            if react {
                cancelled(
                    agent::react()
                        .model(model)
                        .toolkit(toolkit)
                        .permission(Ask(log.clone()))
                        .checkpoint(checkpoint.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                    checkpoint,
                    log,
                    drop_stream,
                )
                .await;
            } else {
                cancelled(
                    agent::chat()
                        .model(model)
                        .toolkit(toolkit)
                        .permission(Ask(log.clone()))
                        .checkpoint(checkpoint.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                    checkpoint,
                    log,
                    drop_stream,
                )
                .await;
            }
        }
    }
}

#[derive(Default)]
struct Store {
    inner: core::checkpoint::InMemory,
    loads: AtomicUsize,
    saves: AtomicUsize,
}
impl Checkpoint for Store {
    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.inner.load(id, cancellation).await
    }
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        self.saves.fetch_add(1, Ordering::SeqCst);
        self.inner.save(id, state, cancellation).await
    }
    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, cancellation).await
    }
}
async fn no_reload<A: Agent>(agent: A, store: Arc<Store>) {
    let mut events = Box::pin(agent.run(input(), true, Cancellation::new()).await.unwrap());
    let (id, _) = question(&mut events).await;
    let reads = store.loads.load(Ordering::SeqCst);
    let writes = store.saves.load(Ordering::SeqCst);
    let error = match agent.run(input(), true, Cancellation::new()).await {
        Ok(_) => panic!("duplicate live run"),
        Err(error) => error,
    };
    assert_eq!(error.code, "CONFLICT");
    let cancelled = Cancellation::new();
    cancelled.cancel();
    assert_eq!(
        agent
            .reply(answer(id, json!(true)), cancelled)
            .await
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    agent
        .reply(answer(id, json!(true)), Cancellation::new())
        .await
        .unwrap();
    assert_eq!(store.loads.load(Ordering::SeqCst), reads);
    assert_eq!(
        store.saves.load(Ordering::SeqCst),
        writes,
        "submission is not another persistence trigger"
    );
    finish(&mut events).await;
}

#[tokio::test]
async fn live_submission_and_duplicate_run_do_not_reload_checkpoint() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let model = Model {
            requests: Default::default(),
            multiple: false,
        };
        let store = Arc::new(Store::default());
        let toolkit = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
        if react {
            no_reload(
                agent::react()
                    .model(model)
                    .toolkit(toolkit)
                    .permission(Ask(log))
                    .checkpoint(store.clone())
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                store,
            )
            .await;
        } else {
            no_reload(
                agent::chat()
                    .model(model)
                    .toolkit(toolkit)
                    .permission(Ask(log))
                    .checkpoint(store.clone())
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                store,
            )
            .await;
        }
    }
}

struct DelayedStore {
    inner: core::checkpoint::InMemory,
    blocked: AtomicUsize,
    gate: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}
impl Checkpoint for DelayedStore {
    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        self.inner.load(id, cancellation).await
    }
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        let gate = if state["status"] == "waiting" {
            self.gate.lock().unwrap().take()
        } else {
            None
        };
        if let Some(gate) = gate {
            self.blocked.fetch_add(1, Ordering::SeqCst);
            gate.await.unwrap();
        }
        self.inner.save(id, state, cancellation).await
    }
    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, cancellation).await
    }
}
async fn cancel_before_presentation<A: Agent>(
    agent: A,
    store: Arc<DelayedStore>,
    release: tokio::sync::oneshot::Sender<()>,
    log: Arc<Log>,
) {
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(input(), false, cancellation.clone())
            .await
            .unwrap(),
    );
    assert!(matches!(
        events.next().await,
        Some(Ok(Event::Complete { usage: Some(_), .. }))
    ));
    assert!(poll!(events.next()).is_pending());
    assert_eq!(store.blocked.load(Ordering::SeqCst), 1);
    cancellation.cancel();
    release.send(()).unwrap();
    let output = finish(&mut events).await;
    assert!(
        matches!(output.as_slice(),[Event::Complete{message,finish_reason:Some(reason),..}]
        if reason == "cancelled" && !message.metadata.contains_key("reply_id"))
    );
    assert!(log.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn cancellation_during_question_save_never_presents_a_cancelled_question() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let model = Model {
            requests: Default::default(),
            multiple: false,
        };
        let (release, gate) = tokio::sync::oneshot::channel();
        let store = Arc::new(DelayedStore {
            inner: core::checkpoint::InMemory::new(),
            blocked: AtomicUsize::new(0),
            gate: Mutex::new(Some(gate)),
        });
        let toolkit = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
        if react {
            cancel_before_presentation(
                agent::react()
                    .model(model)
                    .toolkit(toolkit)
                    .permission(Ask(log.clone()))
                    .checkpoint(store.clone())
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                store,
                release,
                log,
            )
            .await;
        } else {
            cancel_before_presentation(
                agent::chat()
                    .model(model)
                    .toolkit(toolkit)
                    .permission(Ask(log.clone()))
                    .checkpoint(store.clone())
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                store,
                release,
                log,
            )
            .await;
        }
    }
}

#[derive(Default)]
struct FailingStore(core::checkpoint::InMemory);
impl Checkpoint for FailingStore {
    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        self.0.load(id, cancellation).await
    }
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        if state["status"] == "waiting" {
            return Err(Error::new("INTERNAL", "waiting save failed"));
        }
        self.0.save(id, state, cancellation).await
    }
    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        self.0.delete(id, cancellation).await
    }
}
async fn failed_save<A: Agent>(agent: A, store: Arc<FailingStore>, log: Arc<Log>) {
    let events = agent
        .run(input(), false, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(events.as_slice(),[Ok(Event::Complete{..}),Err(error)] if error.message == "waiting save failed")
    );
    assert!(events.iter().all(|event| !matches!(event,Ok(Event::Complete{message,..}) if message.metadata.contains_key("reply_id"))));
    let state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(state["status"], "failed");
    assert_eq!(state["waits"], json!([]));
    let cached = agent
        .run(input(), false, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(cached.as_slice(),[Err(error)] if error.message == "waiting save failed"));
    assert!(log.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn question_save_failure_is_not_presented_and_terminal_error_is_still_recoverable() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let model = Model {
            requests: Default::default(),
            multiple: false,
        };
        let store = Arc::new(FailingStore::default());
        let toolkit = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
        if react {
            failed_save(
                agent::react()
                    .model(model)
                    .toolkit(toolkit)
                    .permission(Ask(log.clone()))
                    .checkpoint(store.clone())
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                store,
                log,
            )
            .await;
        } else {
            failed_save(
                agent::chat()
                    .model(model)
                    .toolkit(toolkit)
                    .permission(Ask(log.clone()))
                    .checkpoint(store.clone())
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                store,
                log,
            )
            .await;
        }
    }
}
