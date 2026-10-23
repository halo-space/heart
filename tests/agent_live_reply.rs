//! Root Chat/ReAct live approval on the original run stream, without replay.
use futures_util::{StreamExt, poll, stream};
use halo::{
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
        let capabilities = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
        let child = if react {
            SubAgent::new(
                name,
                agent::react()
                    .model(model.clone())
                    .toolkit(capabilities)
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
                    .toolkit(capabilities)
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
        assert_eq!(
            scene["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["call_id"],
            "c1"
        );
        assert_eq!(
            scene["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["waits"][id.to_string()]["status"],
            "waiting"
        );
        assert_eq!(scene["query"]["metadata"]["session_id"], 2001);
        assert_eq!(scene["waits"][0]["nodes"], json!([]));
    }
    let a_index = (child_scene(&saved, &node_a)["query"]["metadata"]["agent_id"]
        .as_u64()
        .unwrap()
        - 3002) as usize;
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
    let mut prefix = reply_in_fifo(&agent, &mut events, answer(a, json!(approve_first))).await;
    prefix.extend(reply_in_fifo(&agent, &mut events, answer(b, json!(true))).await);
    prefix.extend(finish(&mut events).await);
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
        let record = &scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][id.to_string()];
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
    let mut output = Vec::new();
    for id in ids {
        output.extend(reply_in_fifo(&agent, &mut events, answer(id, json!(true))).await);
    }
    output.extend(finish(&mut events).await);
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
                execution["attempts"][0]["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|step| step["kind"] == "tool")
                    .expect("actual Tool Step")["waits"]
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
                        let record = &scene["steps"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|step| step["kind"] == "tool")
                            .expect("actual Tool Step")["waits"];
                        let id = record
                            .as_object()
                            .unwrap()
                            .keys()
                            .next()
                            .unwrap()
                            .parse::<i64>()
                            .unwrap();
                        assert_eq!(scene["waits"][0]["nodes"], json!([]));
                        assert_eq!(
                            scene["steps"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .find(|step| step["kind"] == "tool")
                                .expect("actual Tool Step")["call_id"],
                            "c1"
                        );
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
                    let mut output =
                        reply_in_fifo(&agent, &mut events, answer(first, json!(approve_first)))
                            .await;
                    output.extend(
                        reply_in_fifo(&agent, &mut events, answer(second, json!(true))).await,
                    );
                    assert_eq!(
                        agent
                            .reply(answer(second, json!(false)), Cancellation::new())
                            .await
                            .unwrap_err()
                            .code,
                        "CONFLICT"
                    );
                    output.extend(finish(&mut events).await);
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
                        let record = &scene["steps"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|step| step["kind"] == "tool")
                            .expect("actual Tool Step")["waits"][id.to_string()];
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
                        task_child_scene(&stopped, task, child)["steps"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|step| step["kind"] == "tool")
                            .expect("actual Tool Step")["waits"],
                        task_child_scene(&saved, task, child)["steps"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|step| step["kind"] == "tool")
                            .expect("actual Tool Step")["waits"]
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
        let mut output = Vec::new();
        for id in ids {
            output.extend(reply_in_fifo(&agent, &mut events, answer(id, json!(true))).await);
        }
        output.extend(finish(&mut events).await);
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
                    execution["attempts"][0]["steps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|step| step["kind"] == "tool")
                        .expect("actual Tool Step")["waits"]
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
        assert!(
            saved["steps"]
                .as_array()
                .unwrap()
                .iter()
                .all(|step| step["kind"] == "model"),
            "task Tool Steps stay in their actual Attempt"
        );
        let identities = [&a, &b].map(|node| saved["nodes"][node]["execution"]["exec_id"].clone());
        for (node, id) in [(&a, first), (&b, second)] {
            let attempt = &saved["nodes"][node]["execution"]["attempts"][0];
            assert_eq!(saved["nodes"][node]["status"], "waiting");
            assert_eq!(attempt["status"], "waiting");
            assert_eq!(
                attempt["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|step| step["kind"] == "tool")
                    .expect("actual Tool Step")["call_id"],
                "same_call"
            );
            assert_eq!(
                attempt["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|step| step["kind"] == "tool")
                    .expect("actual Tool Step")["waits"][id.to_string()]["status"],
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

        let mut prefix =
            reply_in_fifo(&agent, &mut events, answer(first, json!(approve_first))).await;
        prefix.extend(reply_in_fifo(&agent, &mut events, answer(second, json!(true))).await);
        // Repeating the same submitted response is idempotent; changing it conflicts.
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
                execution["attempts"][0]["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|step| step["kind"] == "tool")
                    .expect("actual Tool Step")["waits"][id.to_string()]["status"],
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
                let step = &stopped["nodes"][node]["execution"]["attempts"][0]["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|step| step["kind"] == "tool")
                    .expect("actual Tool Step");
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
    assert_eq!(
        agent
            .reply(answer(b, json!(true)), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    let mut prefix = reply_in_fifo(&agent, &mut events, answer(a, json!(true))).await;
    let (queued, third) = question(&mut events).await;
    assert_eq!(third.metadata["node_id"], "a");
    assert_ne!(queued, a);
    prefix.extend(reply_in_fifo(&agent, &mut events, answer(b, json!(true))).await);
    prefix.extend(reply_in_fifo(&agent, &mut events, answer(queued, json!(true))).await);
    prefix.extend(finish(&mut events).await);
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
        attempt["steps"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["status"] = json!("running");
        attempt["steps"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")
            .as_object_mut()
            .unwrap()
            .remove("waits");
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

async fn reply_in_fifo<A: Agent>(
    agent: &A,
    events: &mut std::pin::Pin<Box<A::Stream>>,
    input: Input,
) -> Vec<Event> {
    let mut observed = Vec::new();
    loop {
        tokio::select! {
            biased;
            event = events.next() => match event {
                Some(Ok(event)) => observed.push(event),
                Some(Err(error)) => panic!("Agent run failed while advancing FIFO: {error:?}"),
                None => panic!("Agent run ended before the requested reply became the FIFO head"),
            },
            result = agent.reply(input.clone(), Cancellation::new()) => match result {
                Ok(()) => return observed,
                Err(error) if error.code == "CONFLICT" => {},
                Err(error) => panic!("unexpected FIFO reply error: {error:?}"),
            }
        }
    }
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
        waiting["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][id.to_string()]["status"],
        "waiting"
    );
    let _: halo::runtime::agent::State = serde_json::from_value(waiting.clone()).unwrap();
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
        final_state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][id.to_string()]["status"],
        "resolved"
    );
    assert_eq!(
        final_state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][id.to_string()]["feedback"]
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
                let capabilities = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
                if react {
                    exercise(
                        agent::react()
                            .model(model.clone())
                            .toolkit(capabilities)
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
                            .toolkit(capabilities)
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
    assert_eq!(
        state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["call"]["arguments"]["city"],
        "Paris"
    );
    assert_eq!(
        state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"]
            .as_object()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][first.to_string()]["status"],
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
        let capabilities = Toolkit::new().tools([Lookup::new(log.clone(), true)]);
        if react {
            changed(
                agent::react()
                    .model(model.clone())
                    .toolkit(capabilities)
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
                    .toolkit(capabilities)
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
    assert_eq!(
        agent
            .reply(answer(second, json!(true)), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT",
        "reply_id identifies B but cannot skip the FIFO head A"
    );
    assert!(log.calls.lock().unwrap().is_empty());
    agent
        .reply(answer(first, json!(false)), Cancellation::new())
        .await
        .unwrap();
    loop {
        tokio::select! {
            biased;
            event = events.next() => {
                event.expect("the original Agent stream remains active").unwrap();
            }
            result = agent.reply(answer(second, json!(true)), Cancellation::new()) => {
                match result {
                    Ok(()) => break,
                    Err(error) if error.code == "CONFLICT" => {}
                    Err(error) => panic!("unexpected FIFO reply error: {error:?}"),
                }
            }
        }
    }
    finish(&mut events).await;
    assert_eq!(
        log.calls.lock().unwrap().len(),
        1,
        "one approval cannot release both operations"
    );
    assert_eq!(model.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn concurrent_question_replies_are_consumed_in_fifo_order() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let model = Model {
            requests: Default::default(),
            multiple: true,
        };
        let capabilities = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
        if react {
            parallel(
                agent::react()
                    .model(model.clone())
                    .toolkit(capabilities)
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
                    .toolkit(capabilities)
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
        assert_eq!(
            saved["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["status"],
            "cancelled"
        );
        assert_eq!(
            saved["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["waits"][id.to_string()]["status"],
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
            let capabilities = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
            if react {
                cancelled(
                    agent::react()
                        .model(model)
                        .toolkit(capabilities)
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
                        .toolkit(capabilities)
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

fn no_new_restore_id() -> i64 {
    panic!("saved approval recovery must not allocate a new execution/question ID")
}

struct RecoveredChild {
    model: RecoveryOnlyModel,
    log: Arc<Log>,
    checkpoint: Arc<Store>,
}

fn recovered_children(react: bool) -> (Vec<SubAgent>, Vec<RecoveredChild>) {
    let mut children = Vec::new();
    let mut probes = Vec::new();
    for name in ["left", "right"] {
        let model = RecoveryOnlyModel::default();
        let log = Arc::new(Log::default());
        let checkpoint = Arc::new(Store::default());
        let capabilities = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
        let child = if react {
            SubAgent::new(
                name,
                agent::react()
                    .model(model.clone())
                    .toolkit(capabilities)
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
                    .toolkit(capabilities)
                    .permission(Ask(log.clone()))
                    .checkpoint(checkpoint.clone())
                    .build()
                    .unwrap(),
            )
            .unwrap()
        };
        children.push(child);
        probes.push(RecoveredChild {
            model,
            log,
            checkpoint,
        });
    }
    (children, probes)
}

// Read only the native containment named by an actual saved wait. This helper
// also reads the same Step after its question has become resolved.
fn located_step<'a>(scene: &'a Value, locator: &Value) -> &'a Value {
    let mut owner = scene;
    for coordinate in locator["nodes"].as_array().unwrap() {
        let id = coordinate["node_id"].as_str().unwrap();
        let node = owner["nodes"]
            .get(id)
            .or_else(|| owner["execution"]["attempts"].as_array()?.last()?["nodes"].get(id))
            .unwrap();
        assert_eq!(node["execution"]["exec_id"], coordinate["exec_id"]);
        owner = node["execution"]["attempts"]
            .as_array()
            .unwrap()
            .last()
            .unwrap();
        assert_eq!(owner["idx"], coordinate["idx"]);
        if !owner["agent"].is_null() {
            owner = &owner["agent"];
        }
    }
    owner["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["id"] == locator["step"])
        .unwrap()
}

fn located_reply_id(scene: &Value, index: usize) -> i64 {
    located_step(scene, &scene["waits"][index])["waits"]
        .as_object()
        .unwrap()
        .iter()
        .find(|(_, wait)| wait["status"] == "waiting")
        .unwrap()
        .0
        .parse()
        .unwrap()
}

async fn save_child_questions<A: Agent>(
    agent: A,
    store: Arc<core::checkpoint::InMemory>,
    mode: bool,
) -> Value {
    let mut events = Box::pin(agent.run(input(), mode, Cancellation::new()).await.unwrap());
    question(&mut events).await;
    question(&mut events).await;
    store.load(1001, &Cancellation::new()).await.unwrap()
}

async fn recover_direct_children<A: Agent>(
    agent: A,
    store: Arc<core::checkpoint::InMemory>,
    saved: Value,
    model: RecoveryOnlyModel,
    probes: Vec<RecoveredChild>,
    mode: bool,
    approve: bool,
) {
    let first = located_reply_id(&saved, 0);
    let second = located_reply_id(&saved, 1);
    let mut events = Box::pin(
        agent
            .run(answer(first, json!(true)), mode, Cancellation::new())
            .await
            .unwrap(),
    );
    assert_eq!(question(&mut events).await.0, second);
    agent
        .reply(answer(second, json!(approve)), Cancellation::new())
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
    assert_eq!(final_state["query"], saved["query"]);
    assert_eq!(final_state["usage"]["total"], 78);
    assert_eq!(
        final_state["execution"]["exec_id"],
        saved["execution"]["exec_id"]
    );
    assert_eq!(
        final_state["execution"]["attempts"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        model.0.lock().unwrap().len(),
        1,
        "parent selection must not replay"
    );
    for (index, reply) in [first, second].into_iter().enumerate() {
        let locator = &saved["waits"][index];
        let step = located_step(&final_state, locator);
        assert_eq!(step["waits"][reply.to_string()]["status"], "resolved");
        assert_eq!(
            step["waits"][reply.to_string()]["feedback"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(step["call"], located_step(&saved, locator)["call"]);
    }
    assert_eq!(
        probes
            .iter()
            .map(|probe| probe.log.calls.lock().unwrap().len())
            .sum::<usize>(),
        1 + usize::from(approve)
    );
    for probe in &probes {
        assert_eq!(
            probe.model.0.lock().unwrap().len(),
            1,
            "child Tool selection must not replay"
        );
        assert_eq!(probe.checkpoint.loads.load(Ordering::SeqCst), 0);
        assert_eq!(probe.checkpoint.saves.load(Ordering::SeqCst), 0);
        assert!(probe.model.0.lock().unwrap().iter().all(|request| {
            let request = serde_json::to_string(request).unwrap();
            !request.contains("reply_id") && !request.contains("untrusted")
        }));
    }
    let mut cached = Box::pin(
        agent
            .run(answer(second, json!(approve)), mode, Cancellation::new())
            .await
            .unwrap(),
    );
    finish(&mut cached).await;
    assert_eq!(model.0.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn saved_direct_child_approvals_restore_actual_children_without_redelegating() {
    for parent_react in [false, true] {
        for child_react in [false, true] {
            for mode in [false, true] {
                for approve in [false, true] {
                    let store = Arc::new(core::checkpoint::InMemory::new());
                    let (children, _) = waiting_children(child_react);
                    let saved = if parent_react {
                        save_child_questions(
                            agent::react()
                                .model(DelegatingModel::default())
                                .subagents(children)
                                .checkpoint(store.clone())
                                .with_max_concurrency(2)
                                .with_id_fn(next_id)
                                .build()
                                .unwrap(),
                            store.clone(),
                            mode,
                        )
                        .await
                    } else {
                        save_child_questions(
                            agent::chat()
                                .model(DelegatingModel::default())
                                .subagents(children)
                                .checkpoint(store.clone())
                                .with_max_concurrency(2)
                                .with_id_fn(next_id)
                                .build()
                                .unwrap(),
                            store.clone(),
                            mode,
                        )
                        .await
                    };
                    let (children, probes) = recovered_children(child_react);
                    let model = RecoveryOnlyModel::default();
                    if parent_react {
                        recover_direct_children(
                            agent::react()
                                .model(model.clone())
                                .subagents(children)
                                .checkpoint(store.clone())
                                .with_max_concurrency(2)
                                .with_id_fn(no_new_restore_id)
                                .build()
                                .unwrap(),
                            store,
                            saved,
                            model,
                            probes,
                            mode,
                            approve,
                        )
                        .await;
                    } else {
                        recover_direct_children(
                            agent::chat()
                                .model(model.clone())
                                .subagents(children)
                                .checkpoint(store.clone())
                                .with_max_concurrency(2)
                                .with_id_fn(no_new_restore_id)
                                .build()
                                .unwrap(),
                            store,
                            saved,
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

#[tokio::test]
async fn saved_planned_child_approvals_restore_through_task_and_child_coordinates() {
    for child_react in [false, true] {
        for mode in [false, true] {
            for capacity in [1, 2] {
                for approve in [false, true] {
                    let store = Arc::new(core::checkpoint::InMemory::new());
                    let (children, _) = waiting_children(child_react);
                    let saved = save_child_questions(
                        agent::react()
                            .model(PlannedDelegatingModel::new(PlannedModel::default()))
                            .subagents(children)
                            .checkpoint(store.clone())
                            .with_max_concurrency(capacity)
                            .with_id_fn(next_id)
                            .build()
                            .unwrap(),
                        store.clone(),
                        mode,
                    )
                    .await;
                    assert_eq!(saved["usage"]["total"], 78);
                    assert!(
                        saved["waits"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .all(|wait| wait["nodes"].as_array().unwrap().len() == 2)
                    );
                    let first = located_reply_id(&saved, 0);
                    let second = located_reply_id(&saved, 1);
                    let (children, probes) = recovered_children(child_react);
                    let model = RestoredPlanModel::default();
                    let restored = agent::react()
                        .model(model.clone())
                        .subagents(children)
                        .checkpoint(store.clone())
                        .with_max_concurrency(capacity)
                        .with_id_fn(next_id)
                        .build()
                        .unwrap();
                    let mut events = Box::pin(
                        restored
                            .run(answer(first, json!(true)), mode, Cancellation::new())
                            .await
                            .unwrap(),
                    );
                    assert_eq!(question(&mut events).await.0, second);
                    restored
                        .reply(answer(second, json!(approve)), Cancellation::new())
                        .await
                        .unwrap();
                    // The saved FIFO head is resolved first; the next child
                    // remains waiting until its own turn.
                    let second_task = saved["waits"][1]["nodes"][0]["node_id"].as_str().unwrap();
                    let completion =
                        tokio::time::timeout(std::time::Duration::from_secs(2), async {
                            loop {
                                if let Event::Complete { message, .. } =
                                    events.next().await.unwrap().unwrap()
                                    && message.values.get("result") == Some(&json!(second_task))
                                {
                                    break;
                                }
                            }
                        })
                        .await;
                    assert!(
                        completion.is_ok(),
                        "planned child timeout: child_react={child_react}, mode={mode}, capacity={capacity}, approve={approve}, task_requests={:?}, child_requests={:?}, calls={:?}",
                        model
                            .0
                            .lock()
                            .unwrap()
                            .iter()
                            .map(task_id)
                            .collect::<Vec<_>>(),
                        probes
                            .iter()
                            .map(|probe| probe.model.0.lock().unwrap().len())
                            .collect::<Vec<_>>(),
                        probes
                            .iter()
                            .map(|probe| probe.log.calls.lock().unwrap().len())
                            .collect::<Vec<_>>()
                    );
                    assert!(
                        model
                            .0
                            .lock()
                            .unwrap()
                            .iter()
                            .all(|request| task_id(request) != Some("c"))
                    );
                    let completed_calls = probes
                        .iter()
                        .map(|probe| probe.log.calls.lock().unwrap().len())
                        .sum::<usize>();
                    assert!((1..=2).contains(&completed_calls));
                    assert_eq!(
                        restored
                            .reply(answer(second, json!(!approve)), Cancellation::new())
                            .await
                            .unwrap_err()
                            .code,
                        "CONFLICT"
                    );
                    let events = finish(&mut events).await;
                    if !mode {
                        assert!(
                            events
                                .iter()
                                .all(|event| !matches!(event, Event::Delta { .. }))
                        );
                    }
                    let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
                    assert_eq!(final_state["status"], "succeeded");
                    assert_eq!(final_state["query"], saved["query"]);
                    assert_eq!(final_state["usage"]["total"], 156);
                    assert_eq!(model.0.lock().unwrap().len(), 4);
                    for (index, id) in [first, second].into_iter().enumerate() {
                        let locator = &saved["waits"][index];
                        let task = locator["nodes"][0]["node_id"].as_str().unwrap();
                        let old = &saved["nodes"][task]["execution"];
                        let new = &final_state["nodes"][task]["execution"];
                        assert_eq!(new["exec_id"], old["exec_id"]);
                        assert_eq!(new["attempts"].as_array().unwrap().len(), 1);
                        assert_eq!(
                            new["attempts"][0]["created_time"],
                            old["attempts"][0]["created_time"]
                        );
                        assert_eq!(
                            located_step(&final_state, locator)["waits"][id.to_string()]["status"],
                            "resolved"
                        );
                    }
                    for probe in probes {
                        assert_eq!(probe.model.0.lock().unwrap().len(), 1);
                        assert_eq!(probe.checkpoint.loads.load(Ordering::SeqCst), 0);
                        assert_eq!(probe.checkpoint.saves.load(Ordering::SeqCst), 0);
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn saved_child_plan_approvals_restore_all_three_native_coordinates() {
    for mode in [false, true] {
        let store = Arc::new(core::checkpoint::InMemory::new());
        let left = Arc::new(Log::default());
        let right = Arc::new(Log::default());
        let left_child = SubAgent::new(
            "left",
            agent::react()
                .model(PlannedModel::default())
                .toolkit(Toolkit::new().tools([Lookup::new(left, false)]))
                .permission(Ask(Default::default()))
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
                .toolkit(Toolkit::new().tools([Lookup::new(right, false)]))
                .permission(Ask(Default::default()))
                .build()
                .unwrap(),
        )
        .unwrap();
        let original = agent::react()
            .model(PlannedDelegatingModel::new(PlannedModel::default()))
            .subagents(vec![left_child, right_child])
            .with_max_concurrency(1)
            .with_id_fn(next_id)
            .with_max_iters(40)
            .checkpoint(store.clone())
            .build()
            .unwrap();
        let mut events = Box::pin(
            original
                .run(input(), mode, Cancellation::new())
                .await
                .unwrap(),
        );
        for _ in 0..3 {
            question(&mut events).await;
        }
        let saved = store.load(1001, &Cancellation::new()).await.unwrap();
        drop(events);
        drop(original);
        let ids = (0..3)
            .map(|index| located_reply_id(&saved, index))
            .collect::<Vec<_>>();
        let log = Arc::new(Log::default());
        let inner_model = RestoredPlanModel::default();
        let inner_store = Arc::new(Store::default());
        let left = SubAgent::new(
            "left",
            agent::react()
                .model(inner_model.clone())
                .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                .permission(Ask(log.clone()))
                .checkpoint(inner_store.clone())
                .with_max_concurrency(1)
                .build()
                .unwrap(),
        )
        .unwrap();
        let right_model = RecoveryOnlyModel::default();
        let right = SubAgent::new(
            "right",
            agent::chat()
                .model(right_model.clone())
                .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                .permission(Ask(log.clone()))
                .checkpoint(inner_store.clone())
                .build()
                .unwrap(),
        )
        .unwrap();
        let model = RestoredPlanModel::default();
        let restored = agent::react()
            .model(model.clone())
            .subagents(vec![left, right])
            .with_max_concurrency(1)
            .with_id_fn(next_id)
            .with_max_iters(40)
            .checkpoint(store.clone())
            .build()
            .unwrap();
        let selected = ids[0];
        let mut events = Box::pin(
            restored
                .run(answer(selected, json!(false)), mode, Cancellation::new())
                .await
                .unwrap(),
        );
        let mut pending = Vec::new();
        for _ in 0..2 {
            pending.push(question(&mut events).await.0);
        }
        assert_eq!(pending, ids[1..]);
        let mut output = Vec::new();
        for id in pending {
            output.extend(reply_in_fifo(&restored, &mut events, answer(id, json!(true))).await);
        }
        output.extend(finish(&mut events).await);
        let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(final_state["status"], "succeeded");
        assert_eq!(final_state["usage"]["total"], 234);
        assert_eq!(final_state["waits"], json!([]));
        assert_eq!(log.calls.lock().unwrap().len(), 2);
        assert_eq!(inner_model.0.lock().unwrap().len(), 4);
        assert_eq!(right_model.0.lock().unwrap().len(), 1);
        assert_eq!(model.0.lock().unwrap().len(), 4);
        assert_eq!(inner_store.loads.load(Ordering::SeqCst), 0);
        assert_eq!(inner_store.saves.load(Ordering::SeqCst), 0);
        for (index, id) in ids.into_iter().enumerate() {
            assert_eq!(
                located_step(&final_state, &saved["waits"][index])["waits"][id.to_string()]["status"],
                "resolved"
            );
        }
    }
}

async fn saved_nested_fixture(planned: bool, child_react: bool, mode: bool) -> Value {
    let store = Arc::new(core::checkpoint::InMemory::new());
    let (children, _) = waiting_children(child_react);
    if planned {
        save_child_questions(
            agent::react()
                .model(PlannedDelegatingModel::new(PlannedModel::default()))
                .subagents(children)
                .checkpoint(store.clone())
                .with_max_concurrency(1)
                .with_id_fn(next_id)
                .build()
                .unwrap(),
            store,
            mode,
        )
        .await
    } else {
        save_child_questions(
            agent::react()
                .model(DelegatingModel::default())
                .subagents(children)
                .checkpoint(store.clone())
                .with_id_fn(next_id)
                .build()
                .unwrap(),
            store,
            mode,
        )
        .await
    }
}

#[tokio::test]
async fn saved_child_approval_rejects_invalid_feedback_and_stale_inner_coordinates_before_dispatch()
{
    for planned in [false, true] {
        let saved = saved_nested_fixture(planned, false, false).await;
        let id = located_reply_id(&saved, 0);
        let store = Arc::new(Store::default());
        store
            .inner
            .save(1001, saved.clone(), &Cancellation::new())
            .await
            .unwrap();
        let model = RecoveryOnlyModel::default();
        let (children, probes) = recovered_children(false);
        let agent = agent::react()
            .model(model.clone())
            .subagents(children)
            .checkpoint(store.clone())
            .with_id_fn(no_new_restore_id)
            .build()
            .unwrap();
        let mut wrong = answer(id, json!(true));
        wrong.metadata.insert("agent_id".into(), json!(999));
        for (input, code) in [
            (input(), "WAITING"),
            (answer(id + 999, json!(true)), "NOT_FOUND"),
            (answer(id, json!("yes")), "INVALID_ARGUMENTS"),
            (wrong, "CONFLICT"),
        ] {
            assert!(
                matches!(agent.run(input, false, Cancellation::new()).await, Err(error) if error.code == code)
            );
            assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
        }
        let depth = saved["waits"][0]["nodes"].as_array().unwrap().len();
        for field in ["node_id", "exec_id", "idx"] {
            let mut corrupt = saved.clone();
            corrupt["waits"][0]["nodes"][depth - 1][field] = if field == "node_id" {
                json!("missing")
            } else {
                json!(999)
            };
            store
                .inner
                .save(1001, corrupt.clone(), &Cancellation::new())
                .await
                .unwrap();
            assert!(
                matches!(agent.run(answer(id, json!(true)), false, Cancellation::new()).await, Err(error) if error.code == "INVALID_STATE")
            );
            assert_eq!(
                store.load(1001, &Cancellation::new()).await.unwrap(),
                corrupt
            );
        }
        assert_eq!(store.saves.load(Ordering::SeqCst), 0);
        assert!(model.0.lock().unwrap().is_empty());
        for probe in probes {
            assert!(probe.model.0.lock().unwrap().is_empty());
            assert!(probe.log.calls.lock().unwrap().is_empty());
            assert_eq!(probe.log.checks.load(Ordering::SeqCst), 0);
            assert_eq!(probe.checkpoint.loads.load(Ordering::SeqCst), 0);
        }
    }
}

async fn finish_addressed_child<A: Agent>(
    agent: A,
    saved: &Value,
    store: Arc<core::checkpoint::InMemory>,
) -> Value {
    let first = located_reply_id(saved, 0);
    let path = saved["waits"][0]["nodes"].as_array().unwrap();
    let child = path.last().unwrap()["node_id"].as_str().unwrap();
    let mut events = Box::pin(
        agent
            .run(answer(first, json!(true)), false, Cancellation::new())
            .await
            .unwrap(),
    );
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if let Event::Complete {
                message,
                usage: Some(_),
                finish_reason: Some(reason),
            } = events.next().await.unwrap().unwrap()
                && message.metadata.get("node_id") == Some(&json!(child))
                && reason == "stop"
            {
                break;
            }
        }
    })
    .await
    .unwrap();
    while let std::task::Poll::Ready(Some(event)) = poll!(events.next()) {
        event.unwrap();
    }
    let partial = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(partial["waits"].as_array().unwrap().len(), 1);
    assert_eq!(
        located_step(&partial, &saved["waits"][0])["waits"][first.to_string()]["status"],
        "resolved"
    );
    partial
}

async fn finish_remaining_child<A: Agent>(
    agent: A,
    saved: &Value,
    partial: &Value,
    store: Arc<core::checkpoint::InMemory>,
) {
    let first = located_reply_id(saved, 0);
    let second = located_reply_id(saved, 1);
    assert!(
        matches!(agent.run(answer(first, json!(false)), false, Cancellation::new()).await, Err(error) if error.code == "CONFLICT")
    );
    assert_eq!(
        store.load(1001, &Cancellation::new()).await.unwrap(),
        *partial
    );
    let mut events = Box::pin(
        agent
            .run(answer(second, json!(true)), false, Cancellation::new())
            .await
            .unwrap(),
    );
    finish(&mut events).await;
    let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(final_state["status"], "succeeded");
    assert_eq!(
        located_step(&final_state, &saved["waits"][0]),
        located_step(partial, &saved["waits"][0])
    );
    assert_eq!(final_state["query"], saved["query"]);
    assert_eq!(
        final_state["usage"]["total"],
        if saved.get("graph").is_some_and(|graph| !graph.is_null()) {
            156
        } else {
            78
        }
    );
}

#[tokio::test]
async fn a_second_child_recovery_reuses_successful_siblings_and_retains_duplicate_history() {
    for planned in [false, true] {
        let saved = saved_nested_fixture(planned, false, false).await;
        let store = Arc::new(core::checkpoint::InMemory::new());
        store
            .save(1001, saved.clone(), &Cancellation::new())
            .await
            .unwrap();
        let (children, original) = recovered_children(false);
        let partial = if planned {
            finish_addressed_child(
                agent::react()
                    .model(RestoredPlanModel::default())
                    .subagents(children)
                    .checkpoint(store.clone())
                    .with_max_concurrency(1)
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                &saved,
                store.clone(),
            )
            .await
        } else {
            finish_addressed_child(
                agent::react()
                    .model(RecoveryOnlyModel::default())
                    .subagents(children)
                    .checkpoint(store.clone())
                    .with_id_fn(no_new_restore_id)
                    .build()
                    .unwrap(),
                &saved,
                store.clone(),
            )
            .await
        };
        assert_eq!(
            original
                .iter()
                .map(|probe| probe.log.calls.lock().unwrap().len())
                .sum::<usize>(),
            1
        );
        let (children, probes) = recovered_children(false);
        if planned {
            finish_remaining_child(
                agent::react()
                    .model(RestoredPlanModel::default())
                    .subagents(children)
                    .checkpoint(store.clone())
                    .with_max_concurrency(1)
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                &saved,
                &partial,
                store.clone(),
            )
            .await;
        } else {
            finish_remaining_child(
                agent::react()
                    .model(RecoveryOnlyModel::default())
                    .subagents(children)
                    .checkpoint(store.clone())
                    .with_id_fn(no_new_restore_id)
                    .build()
                    .unwrap(),
                &saved,
                &partial,
                store.clone(),
            )
            .await;
        }
        assert_eq!(
            probes
                .iter()
                .map(|probe| probe.log.calls.lock().unwrap().len())
                .sum::<usize>(),
            1,
            "the remaining child task executes once after its FIFO turn"
        );
        assert_eq!(
            probes
                .iter()
                .map(|probe| probe.model.0.lock().unwrap().len())
                .sum::<usize>(),
            1,
            "successful child must not run again"
        );
    }
}

#[tokio::test]
async fn restored_child_approval_does_not_execute_when_outer_save_fails_or_is_cancelled() {
    let saved = saved_nested_fixture(false, false, false).await;
    let id = located_reply_id(&saved, 0);
    let store = Arc::new(FailingStore::default());
    store
        .0
        .save(1001, saved.clone(), &Cancellation::new())
        .await
        .unwrap();
    let (children, probes) = recovered_children(false);
    let model = RecoveryOnlyModel::default();
    let agent = agent::react()
        .model(model.clone())
        .subagents(children)
        .checkpoint(store.clone())
        .with_id_fn(no_new_restore_id)
        .build()
        .unwrap();
    let events = agent
        .run(answer(id, json!(true)), false, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Err(error) if error.message == "waiting save failed"))
    );
    assert!(events.iter().all(|event| !matches!(event, Ok(Event::Complete { message, .. }) if message.metadata.contains_key("reply_id"))));
    assert!(model.0.lock().unwrap().is_empty());
    assert!(
        probes
            .iter()
            .all(|probe| probe.log.calls.lock().unwrap().is_empty()
                && probe.model.0.lock().unwrap().is_empty())
    );
    assert_eq!(
        store.load(1001, &Cancellation::new()).await.unwrap()["usage"]["total"],
        39
    );
    let (release, gate) = tokio::sync::oneshot::channel();
    let store = Arc::new(DelayedStore {
        inner: core::checkpoint::InMemory::new(),
        blocked: AtomicUsize::new(0),
        gate: Mutex::new(Some(gate)),
    });
    store
        .inner
        .save(1001, saved, &Cancellation::new())
        .await
        .unwrap();
    let (children, probes) = recovered_children(false);
    let agent = agent::react()
        .model(model)
        .subagents(children)
        .checkpoint(store.clone())
        .with_id_fn(no_new_restore_id)
        .build()
        .unwrap();
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(answer(id, json!(true)), false, cancellation.clone())
            .await
            .unwrap(),
    );
    assert!(poll!(events.next()).is_pending());
    assert_eq!(store.blocked.load(Ordering::SeqCst), 1);
    cancellation.cancel();
    release.send(()).unwrap();
    let events = finish(&mut events).await;
    assert!(events.iter().all(|event| !matches!(event, Event::Complete { message, .. } if message.metadata.contains_key("reply_id"))));
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

async fn child_worker_finish<A: Agent>(
    agent: A,
    saved: Value,
    store: Arc<core::checkpoint::InMemory>,
    probes: Vec<RecoveredChild>,
    mode: bool,
    approve: bool,
) {
    let first = located_reply_id(&saved, 0);
    let second = located_reply_id(&saved, 1);
    let mut events = Box::pin(
        agent
            .run(answer(first, json!(true)), mode, Cancellation::new())
            .await
            .unwrap(),
    );
    assert_eq!(question(&mut events).await.0, second);
    agent
        .reply(answer(second, json!(approve)), Cancellation::new())
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
    let state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(state["status"], "succeeded");
    assert_eq!(state["query"], saved["query"]);
    assert_eq!(state["waits"], json!([]));
    assert_eq!(
        state["usage"]["total"],
        if saved.get("graph").is_some_and(|graph| !graph.is_null()) {
            156
        } else {
            78
        }
    );
    for (index, id) in [first, second].into_iter().enumerate() {
        let locator = &saved["waits"][index];
        assert_eq!(
            located_step(&state, locator)["call"],
            located_step(&saved, locator)["call"]
        );
        assert_eq!(
            located_step(&state, locator)["waits"][id.to_string()]["status"],
            "resolved"
        );
    }
    assert_eq!(
        probes
            .iter()
            .map(|probe| probe.log.calls.lock().unwrap().len())
            .sum::<usize>(),
        1 + usize::from(approve)
    );
    for probe in probes {
        assert_eq!(probe.model.0.lock().unwrap().len(), 1);
        assert_eq!(probe.checkpoint.loads.load(Ordering::SeqCst), 0);
        assert_eq!(probe.checkpoint.saves.load(Ordering::SeqCst), 0);
        assert!(probe.model.0.lock().unwrap().iter().all(|request| {
            let json = serde_json::to_string(request).unwrap();
            !json.contains("reply_id") && !json.contains("untrusted")
        }));
    }
}

#[tokio::test]
#[ignore = "explicitly invoked by child_approvals_recover_in_fresh_processes_from_outer_checkpoint_only"]
async fn child_approval_recovery_worker() {
    let path = std::env::var("HEART_CHILD_APPROVAL_FILE").unwrap();
    let planned = std::env::var("HEART_CHILD_APPROVAL_PLANNED").unwrap() == "true";
    let child_react = std::env::var("HEART_CHILD_APPROVAL_REACT").unwrap() == "true";
    let mode = std::env::var("HEART_CHILD_APPROVAL_MODE").unwrap() == "true";
    if std::env::var("HEART_CHILD_APPROVAL_PHASE").unwrap() == "save" {
        let saved = saved_nested_fixture(planned, child_react, mode).await;
        std::fs::write(path, serde_json::to_vec(&saved).unwrap()).unwrap();
        return;
    }
    let saved: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let approve = std::env::var("HEART_CHILD_APPROVAL_DECISION").unwrap() == "true";
    let store = Arc::new(core::checkpoint::InMemory::new());
    store
        .save(1001, saved.clone(), &Cancellation::new())
        .await
        .unwrap();
    let (children, probes) = recovered_children(child_react);
    if planned {
        let model = RestoredPlanModel::default();
        child_worker_finish(
            agent::react()
                .model(model.clone())
                .subagents(children)
                .checkpoint(store.clone())
                .with_max_concurrency(1)
                .with_id_fn(restored_plan_id)
                .build()
                .unwrap(),
            saved,
            store,
            probes,
            mode,
            approve,
        )
        .await;
        assert_eq!(model.0.lock().unwrap().len(), 4);
    } else {
        let model = RecoveryOnlyModel::default();
        child_worker_finish(
            agent::react()
                .model(model.clone())
                .subagents(children)
                .checkpoint(store.clone())
                .with_id_fn(no_new_restore_id)
                .build()
                .unwrap(),
            saved,
            store,
            probes,
            mode,
            approve,
        )
        .await;
        assert_eq!(model.0.lock().unwrap().len(), 1);
    }
}

#[test]
fn child_approvals_recover_in_fresh_processes_from_outer_checkpoint_only() {
    for planned in [false, true] {
        for child_react in [false, true] {
            for mode in [false, true] {
                for approve in [false, true] {
                    let path = std::env::temp_dir().join(format!(
                        "heart-child-approval-{}-{planned}-{child_react}-{mode}-{approve}.json",
                        std::process::id()
                    ));
                    for phase in ["save", "recover"] {
                        let output = std::process::Command::new(std::env::current_exe().unwrap())
                            .args([
                                "--exact",
                                "child_approval_recovery_worker",
                                "--ignored",
                                "--nocapture",
                            ])
                            .env("HEART_CHILD_APPROVAL_FILE", &path)
                            .env("HEART_CHILD_APPROVAL_PLANNED", planned.to_string())
                            .env("HEART_CHILD_APPROVAL_REACT", child_react.to_string())
                            .env("HEART_CHILD_APPROVAL_MODE", mode.to_string())
                            .env("HEART_CHILD_APPROVAL_DECISION", approve.to_string())
                            .env("HEART_CHILD_APPROVAL_PHASE", phase)
                            .output()
                            .unwrap();
                        assert!(
                            output.status.success(),
                            "{phase} worker failed: {}{}",
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        );
                    }
                    std::fs::remove_file(path).unwrap();
                }
            }
        }
    }
}

#[derive(Clone, Default)]
struct RestoredPlanModel(Arc<Mutex<Vec<Request>>>);

// Produce a real waiting snapshot rather than hand-writing a second recovery
// schema. Only the returned bytes cross to each new consumer in these tests.
async fn saved_plan_questions(mode: bool) -> Value {
    let store = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let agent = agent::react()
        .model(PlannedModel::default())
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .checkpoint(store.clone())
        .with_max_concurrency(1)
        .with_id_fn(next_id)
        .build()
        .unwrap();
    let mut events = Box::pin(agent.run(input(), mode, Cancellation::new()).await.unwrap());
    question(&mut events).await;
    question(&mut events).await;
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    assert!(log.calls.lock().unwrap().is_empty());
    saved
}

fn plan_reply_id(scene: &Value, index: usize) -> i64 {
    let locator = &scene["waits"][index];
    let node = locator["nodes"][0]["node_id"].as_str().unwrap();
    let id = locator["step"].as_str().unwrap();
    let step = scene["nodes"][node]["execution"]["attempts"][0]["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["id"] == id)
        .unwrap();
    step["waits"]
        .as_object()
        .unwrap()
        .iter()
        .find(|(_, record)| record["status"] == "waiting")
        .unwrap()
        .0
        .parse()
        .unwrap()
}

#[tokio::test]
async fn saved_planned_approval_rejects_bad_input_and_stale_native_targets_without_side_effects() {
    let saved = saved_plan_questions(false).await;
    let id = plan_reply_id(&saved, 0);
    let store = Arc::new(Store::default());
    store
        .inner
        .save(1001, saved.clone(), &Cancellation::new())
        .await
        .unwrap();
    let log = Arc::new(Log::default());
    let model = RestoredPlanModel::default();
    let agent = agent::react()
        .model(model.clone())
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .checkpoint(store.clone())
        .with_id_fn(next_id)
        .build()
        .unwrap();
    let mut wrong_owner = answer(id, json!(true));
    wrong_owner.metadata.insert("session_id".into(), json!(99));
    for (reply, code) in [
        (input(), "WAITING"),
        (answer(id + 9999, json!(true)), "NOT_FOUND"),
        (answer(id, json!("yes")), "INVALID_ARGUMENTS"),
        (wrong_owner, "CONFLICT"),
    ] {
        assert!(
            matches!(agent.run(reply, false, Cancellation::new()).await, Err(error) if error.code == code)
        );
        assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
    }
    for field in ["exec_id", "idx", "node_id"] {
        let mut bad = saved.clone();
        bad["waits"][1]["nodes"][0][field] = if field == "node_id" {
            json!("missing")
        } else {
            json!(99999)
        };
        store
            .inner
            .save(1001, bad.clone(), &Cancellation::new())
            .await
            .unwrap();
        assert!(
            matches!(agent.run(answer(id, json!(true)), false, Cancellation::new()).await, Err(error) if error.code == "INVALID_STATE")
        );
        assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), bad);
    }
    assert_eq!(store.saves.load(Ordering::SeqCst), 0);
    assert!(model.0.lock().unwrap().is_empty());
    assert_eq!(log.checks.load(Ordering::SeqCst), 0);
    assert_eq!(log.before.load(Ordering::SeqCst), 0);
    assert!(log.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn saved_planned_approval_reuses_a_successful_sibling_on_a_second_recovery() {
    for mode in [false, true] {
        let saved = saved_plan_questions(mode).await;
        let first = plan_reply_id(&saved, 0);
        let second = plan_reply_id(&saved, 1);
        let a = saved["waits"][0]["nodes"][0]["node_id"].as_str().unwrap();
        let b = saved["waits"][1]["nodes"][0]["node_id"].as_str().unwrap();
        let store = Arc::new(core::checkpoint::InMemory::new());
        store
            .save(1001, saved.clone(), &Cancellation::new())
            .await
            .unwrap();
        let first_log = Arc::new(Log::default());
        let first_model = RestoredPlanModel::default();
        let original = agent::react()
            .model(first_model.clone())
            .toolkit(Toolkit::new().tools([Lookup::new(first_log.clone(), false)]))
            .permission(Ask(first_log.clone()))
            .checkpoint(store.clone())
            .with_max_concurrency(1)
            .with_id_fn(next_id)
            .build()
            .unwrap();
        assert!(matches!(
            original
                .run(answer(second, json!(true)), mode, Cancellation::new())
                .await,
            Err(error) if error.code == "CONFLICT"
        ));
        assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
        let mut events = Box::pin(
            original
                .run(answer(first, json!(true)), mode, Cancellation::new())
                .await
                .unwrap(),
        );
        assert_eq!(question(&mut events).await.0, second);
        // Commit A's terminal native facts and the remaining B wait at the
        // existing waiting boundary before dropping the stream.
        assert!(poll!(events.next()).is_pending());
        let partial = store.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(partial["nodes"][a]["status"], "succeeded");
        drop(events);
        drop(original);
        assert_eq!(first_log.calls.lock().unwrap().len(), 1);
        let log = Arc::new(Log::default());
        let model = RestoredPlanModel::default();
        let restored = agent::react()
            .model(model.clone())
            .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
            .permission(Ask(log.clone()))
            .checkpoint(store.clone())
            .with_max_concurrency(1)
            .with_id_fn(next_id)
            .build()
            .unwrap();
        let mut events = Box::pin(
            restored
                .run(answer(second, json!(true)), mode, Cancellation::new())
                .await
                .unwrap(),
        );
        finish(&mut events).await;
        let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(final_state["nodes"][a], partial["nodes"][a]);
        assert_eq!(final_state["nodes"][b]["status"], "succeeded");
        assert_eq!(
            final_state["nodes"][a]["execution"]["exec_id"],
            saved["nodes"][a]["execution"]["exec_id"]
        );
        assert_eq!(final_state["usage"]["total"], 104);
        assert_eq!(model.0.lock().unwrap().len(), 3);
        assert_eq!(log.calls.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn restored_planned_questions_reject_non_head_feedback_and_preserve_saved_fifo_order() {
    let mut saved = saved_plan_questions(false).await;
    saved["waits"].as_array_mut().unwrap().reverse();
    let first = plan_reply_id(&saved, 0);
    let second = plan_reply_id(&saved, 1);
    let store = Arc::new(core::checkpoint::InMemory::new());
    store
        .save(1001, saved.clone(), &Cancellation::new())
        .await
        .unwrap();
    let log = Arc::new(Log::default());
    let agent = agent::react()
        .model(RestoredPlanModel::default())
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .checkpoint(store.clone())
        .with_max_concurrency(1)
        .with_id_fn(next_id)
        .build()
        .unwrap();
    assert!(matches!(
        agent.run(answer(second, json!(true)), false, Cancellation::new()).await,
        Err(error) if error.code == "CONFLICT"
    ));
    assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
    let mut events = Box::pin(
        agent
            .run(answer(first, json!(true)), false, Cancellation::new())
            .await
            .unwrap(),
    );
    assert_eq!(question(&mut events).await.0, second);
    let waiting = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(waiting["waits"].as_array().unwrap().len(), 1);
    assert_eq!(plan_reply_id(&waiting, 0), second);
    agent
        .reply(answer(second, json!(false)), Cancellation::new())
        .await
        .unwrap();
    finish(&mut events).await;
    assert_eq!(log.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn restored_planned_approval_save_failure_and_cancellation_do_not_authorize_tools() {
    let saved = saved_plan_questions(false).await;
    let id = plan_reply_id(&saved, 0);
    let failed = Arc::new(FailingStore::default());
    failed
        .0
        .save(1001, saved.clone(), &Cancellation::new())
        .await
        .unwrap();
    let log = Arc::new(Log::default());
    let model = RestoredPlanModel::default();
    let agent = agent::react()
        .model(model.clone())
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .checkpoint(failed.clone())
        .with_max_concurrency(1)
        .with_id_fn(next_id)
        .build()
        .unwrap();
    let events = agent
        .run(answer(id, json!(true)), false, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Err(error) if error.message == "waiting save failed"))
    );
    assert!(events.iter().all(|event| !matches!(event, Ok(Event::Complete { message, .. }) if message.metadata.contains_key("reply_id"))));
    // Task A was approved and completed before saving the next waiting
    // question failed. That failure must not authorize task B's Tool.
    assert_eq!(model.0.lock().unwrap().len(), 1);
    assert_eq!(log.calls.lock().unwrap().len(), 1);
    let terminal = failed.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(terminal["waits"], json!([]));
    assert_eq!(terminal["usage"]["total"], 65);
    assert_eq!(
        agent
            .reply(answer(id, json!(true)), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );

    let (release, gate) = tokio::sync::oneshot::channel();
    let store = Arc::new(DelayedStore {
        inner: core::checkpoint::InMemory::new(),
        blocked: AtomicUsize::new(0),
        gate: Mutex::new(Some(gate)),
    });
    store
        .inner
        .save(1001, saved, &Cancellation::new())
        .await
        .unwrap();
    let agent = agent::react()
        .model(model.clone())
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .checkpoint(store.clone())
        .with_max_concurrency(1)
        .with_id_fn(next_id)
        .build()
        .unwrap();
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(answer(id, json!(true)), false, cancellation.clone())
            .await
            .unwrap(),
    );
    assert!(matches!(
        poll!(events.next()),
        std::task::Poll::Ready(Some(Ok(Event::Complete { message, .. })))
            if message.role == Role::Function
    ));
    assert!(poll!(events.next()).is_pending());
    assert_eq!(store.blocked.load(Ordering::SeqCst), 1);
    let model_calls_before_cancel = model.0.lock().unwrap().len();
    let tool_calls_before_cancel = log.calls.lock().unwrap().len();
    cancellation.cancel();
    release.send(()).unwrap();
    let events = finish(&mut events).await;
    assert!(events.iter().all(|event| !matches!(event, Event::Complete { message, .. } if message.metadata.contains_key("reply_id"))));
    assert_eq!(model.0.lock().unwrap().len(), model_calls_before_cancel);
    assert_eq!(log.calls.lock().unwrap().len(), tool_calls_before_cancel);
    assert_eq!(
        agent
            .reply(answer(id, json!(true)), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
}

fn restored_plan_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(9_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}

#[tokio::test]
#[ignore = "invoked in fresh OS processes by planned_approval_recovers_in_a_new_os_process_using_only_checkpoint_bytes"]
async fn planned_approval_recovery_worker() {
    let path = std::env::var("HEART_PLAN_APPROVAL_FILE").unwrap();
    let mode = std::env::var("HEART_PLAN_APPROVAL_MODE").unwrap() == "true";
    if std::env::var("HEART_PLAN_APPROVAL_PHASE").unwrap() == "save" {
        let saved = saved_plan_questions(mode).await;
        std::fs::write(path, serde_json::to_vec(&saved).unwrap()).unwrap();
        return;
    }
    let saved: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let approve = std::env::var("HEART_PLAN_APPROVAL_DECISION").unwrap() == "true";
    let first = plan_reply_id(&saved, 0);
    let second = plan_reply_id(&saved, 1);
    let store = Arc::new(core::checkpoint::InMemory::new());
    store
        .save(1001, saved.clone(), &Cancellation::new())
        .await
        .unwrap();
    let log = Arc::new(Log::default());
    let model = RestoredPlanModel::default();
    let agent = agent::react()
        .model(model.clone())
        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
        .permission(Ask(log.clone()))
        .checkpoint(store.clone())
        .with_max_concurrency(1)
        .with_id_fn(restored_plan_id)
        .build()
        .unwrap();
    let mut events = Box::pin(
        agent
            .run(answer(first, json!(false)), mode, Cancellation::new())
            .await
            .unwrap(),
    );
    assert_eq!(question(&mut events).await.0, second);
    agent
        .reply(answer(second, json!(approve)), Cancellation::new())
        .await
        .unwrap();
    let events = finish(&mut events).await;
    if !mode {
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, Event::Delta { .. }))
        );
    }
    let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(final_state["status"], "succeeded");
    assert_eq!(final_state["usage"]["total"], 104);
    assert_eq!(final_state["query"], saved["query"]);
    assert_eq!(model.0.lock().unwrap().len(), 4);
    assert_eq!(log.calls.lock().unwrap().len(), usize::from(approve));
    for locator in saved["waits"].as_array().unwrap() {
        let node = locator["nodes"][0]["node_id"].as_str().unwrap();
        let old = &saved["nodes"][node]["execution"];
        let new = &final_state["nodes"][node]["execution"];
        assert_eq!(new["exec_id"], old["exec_id"]);
        assert_eq!(new["attempts"].as_array().unwrap().len(), 1);
        assert_eq!(
            new["attempts"][0]["created_time"],
            old["attempts"][0]["created_time"]
        );
        assert_eq!(
            new["attempts"][0]["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["waits"]
                .as_object()
                .unwrap()
                .len(),
            1
        );
    }
    assert_eq!(final_state["nodes"]["c"]["execution"]["exec_id"], 9_000_000);
}

#[test]
fn planned_approval_recovers_in_a_new_os_process_using_only_checkpoint_bytes() {
    for mode in [false, true] {
        for approve in [false, true] {
            let path = std::env::temp_dir().join(format!(
                "heart-plan-approval-{}-{mode}-{approve}.json",
                std::process::id()
            ));
            for phase in ["save", "recover"] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "planned_approval_recovery_worker",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env("HEART_PLAN_APPROVAL_FILE", &path)
                    .env("HEART_PLAN_APPROVAL_MODE", mode.to_string())
                    .env("HEART_PLAN_APPROVAL_DECISION", approve.to_string())
                    .env("HEART_PLAN_APPROVAL_PHASE", phase)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{phase} worker failed: {}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            std::fs::remove_file(path).unwrap();
        }
    }
}
impl Chat for RestoredPlanModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        // A fresh model instance must never select the saved calls or plan
        // again. Task a/b may reason only after their actual Tool has settled.
        let mut message = Message::new(Role::Assistant);
        if let Some(node) = task_id(&request) {
            if node != "c" {
                assert!(
                    request.messages.as_slice().iter().any(|message| {
                        message.role == Role::Tool || message.role == Role::Function
                    }),
                    "restored task {node} lost its saved Tool continuation"
                );
            }
            message.values.insert("result".into(), json!(node));
        } else {
            assert!(
                request
                    .messages
                    .as_slice()
                    .iter()
                    .any(|message| { message.values.contains_key("plan_complete") }),
                "root planning/selection must not replay"
            );
            message.values.insert("answer".into(), json!("all done"));
        }
        assert!(
            !serde_json::to_string(&request)
                .unwrap()
                .contains("reply_id")
        );
        assert!(
            !serde_json::to_string(&request)
                .unwrap()
                .contains("untrusted")
        );
        self.0.lock().unwrap().push(request);
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
async fn saved_planned_approvals_route_same_call_ids_to_their_actual_tasks() {
    for mode in [false, true] {
        for approve in [false, true] {
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let original_log = Arc::new(Log::default());
            let original = agent::react()
                .model(PlannedModel::default())
                .toolkit(Toolkit::new().tools([Lookup::new(original_log.clone(), false)]))
                .permission(Ask(original_log.clone()))
                .checkpoint(checkpoint.clone())
                .with_max_concurrency(1)
                .with_id_fn(next_id)
                .build()
                .unwrap();
            let mut events = Box::pin(
                original
                    .run(input(), mode, Cancellation::new())
                    .await
                    .unwrap(),
            );
            let (first, first_message) = question(&mut events).await;
            let (second, second_message) = question(&mut events).await;
            let a = first_message.metadata["node_id"]
                .as_str()
                .unwrap()
                .to_owned();
            let b = second_message.metadata["node_id"]
                .as_str()
                .unwrap()
                .to_owned();
            let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            drop(events);
            drop(original);
            assert!(original_log.calls.lock().unwrap().is_empty());

            let log = Arc::new(Log::default());
            let model = RestoredPlanModel::default();
            let restored = agent::react()
                .model(model.clone())
                .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                .permission(Ask(log.clone()))
                .checkpoint(checkpoint.clone())
                .with_max_concurrency(1)
                .with_id_fn(next_id)
                .build()
                .unwrap();
            let mut events = Box::pin(
                restored
                    .run(answer(first, json!(false)), mode, Cancellation::new())
                    .await
                    .unwrap(),
            );
            let (again, _) = question(&mut events).await;
            assert_eq!(again, second, "next saved question retains its ID");
            restored
                .reply(answer(second, json!(approve)), Cancellation::new())
                .await
                .unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                loop {
                    let event = events.next().await.unwrap().unwrap();
                    if !mode {
                        assert!(!matches!(event, Event::Delta { .. }));
                    }
                    if let Event::Complete { message, .. } = event
                        && message.metadata["node_id"] == b
                        && message.values.get("result") == Some(&json!(b))
                    {
                        break;
                    }
                }
            })
            .await
            .unwrap();
            assert_eq!(log.calls.lock().unwrap().len(), usize::from(approve));
            if approve {
                assert_eq!(log.calls.lock().unwrap()[0].arguments["city"], b);
            }
            assert!(
                model
                    .0
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|request| task_id(request) != Some("c"))
            );
            let partial = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(partial["waits"], json!([]));
            assert_eq!(partial["nodes"][&b]["status"], "succeeded");
            assert_eq!(
                restored
                    .reply(answer(second, json!(!approve)), Cancellation::new())
                    .await
                    .unwrap_err()
                    .code,
                "CONFLICT"
            );
            let tail = finish(&mut events).await;
            if !mode {
                assert!(
                    tail.iter()
                        .all(|event| !matches!(event, Event::Delta { .. }))
                );
            }
            let final_state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(final_state["status"], "succeeded");
            assert_eq!(final_state["waits"], json!([]));
            assert_eq!(final_state["query"], saved["query"]);
            assert_eq!(final_state["usage"]["total"], 104);
            for (node, id) in [(&a, first), (&b, second)] {
                let old = &saved["nodes"][node]["execution"];
                let new = &final_state["nodes"][node]["execution"];
                assert_eq!(new["exec_id"], old["exec_id"]);
                assert_eq!(new["created_time"], old["created_time"]);
                assert_eq!(new["attempts"].as_array().unwrap().len(), 1);
                assert_eq!(
                    new["attempts"][0]["steps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|step| step["kind"] == "tool")
                        .expect("actual Tool Step")["waits"][id.to_string()]["status"],
                    "resolved"
                );
            }
            assert_eq!(log.calls.lock().unwrap().len(), usize::from(approve));
            assert_eq!(model.0.lock().unwrap().len(), 4);
            // Saved task history, not just root history, also protects a
            // repeated terminal run from replaying the plan/tools.
            let mut terminal = Box::pin(
                restored
                    .run(answer(second, json!(approve)), mode, Cancellation::new())
                    .await
                    .unwrap(),
            );
            finish(&mut terminal).await;
            assert_eq!(model.0.lock().unwrap().len(), 4);
        }
    }
}

static HISTORICAL_REPLY_ID: AtomicI64 = AtomicI64::new(0);
fn reused_historical_reply_id() -> i64 {
    HISTORICAL_REPLY_ID.load(Ordering::SeqCst)
}

#[derive(Clone, Default)]
struct SequentialModel(Arc<Mutex<Vec<Request>>>);
impl Chat for SequentialModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut requests = self.0.lock().unwrap();
        let index = requests.len();
        requests.push(request);
        let mut message = Message::new(Role::Assistant);
        if index < 2 {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({
                    "call_id":format!("sequential_{index}"), "name":"lookup",
                    "arguments":{"city": if index == 0 { "London" } else { "Beijing" }}
                })
                .as_object()
                .unwrap()
                .clone(),
            });
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

async fn preserve_old_question_ids<A: Agent>(
    create: impl Fn(bool) -> A,
    store: Arc<Store>,
    log: Arc<Log>,
) {
    let original = create(false);
    let mut events = Box::pin(
        original
            .run(input(), true, Cancellation::new())
            .await
            .unwrap(),
    );
    let (old, _) = question(&mut events).await;
    original
        .reply(answer(old, json!(true)), Cancellation::new())
        .await
        .unwrap();
    let (pending, _) = question(&mut events).await;
    assert_ne!(old, pending);
    assert_eq!(log.calls.lock().unwrap().len(), 1);
    drop(events);
    drop(original);
    HISTORICAL_REPLY_ID.store(old, Ordering::SeqCst);
    let restored = create(true);
    let mut events = Box::pin(
        restored
            .run(answer(pending, json!(true)), true, Cancellation::new())
            .await
            .unwrap(),
    );
    let output = finish(&mut events).await;
    assert!(output.iter().all(|event| !matches!(event, Event::Complete { message, .. } if message.metadata.contains_key("reply_id"))));
    assert_eq!(
        log.calls.lock().unwrap().len(),
        1,
        "the changed operation cannot execute under an old reply ID"
    );
    let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        final_state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][old.to_string()]["status"],
        "resolved"
    );
    let pending_step = final_state["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["kind"] == "tool" && step["waits"].get(pending.to_string()).is_some())
        .unwrap();
    assert_eq!(
        pending_step["waits"][pending.to_string()]["status"],
        "resolved"
    );
    assert_eq!(pending_step["error"]["code"], "CONFLICT");
}

#[tokio::test]
async fn recovery_reserves_historical_question_ids_without_reusing_their_approval() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let model = SequentialModel::default();
        let store = Arc::new(Store::default());
        if react {
            preserve_old_question_ids(
                |restored| {
                    agent::react()
                        .model(model.clone())
                        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), restored)]))
                        .permission(Ask(log.clone()))
                        .checkpoint(store.clone())
                        .with_id_fn(if restored {
                            reused_historical_reply_id
                        } else {
                            next_id
                        })
                        .build()
                        .unwrap()
                },
                store.clone(),
                log.clone(),
            )
            .await;
        } else {
            preserve_old_question_ids(
                |restored| {
                    agent::chat()
                        .model(model.clone())
                        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), restored)]))
                        .permission(Ask(log.clone()))
                        .checkpoint(store.clone())
                        .with_id_fn(if restored {
                            reused_historical_reply_id
                        } else {
                            next_id
                        })
                        .build()
                        .unwrap()
                },
                store.clone(),
                log.clone(),
            )
            .await;
        }
    }
}

async fn recover_late_duplicate<A: Agent>(
    create: impl Fn(bool) -> A,
    store: Arc<Store>,
    log: Arc<Log>,
    model: SequentialModel,
) {
    let original = create(false);
    let mut events = Box::pin(
        original
            .run(input(), true, Cancellation::new())
            .await
            .unwrap(),
    );
    let (old, _) = question(&mut events).await;
    original
        .reply(answer(old, json!(true)), Cancellation::new())
        .await
        .unwrap();
    let (pending, _) = question(&mut events).await;
    drop(events);
    drop(original);
    let restored = create(true);
    let mut events = Box::pin(
        restored
            .run(answer(old, json!(true)), true, Cancellation::new())
            .await
            .unwrap(),
    );
    let (again, _) = question(&mut events).await;
    assert_eq!(again, pending);
    assert_eq!(
        log.calls.lock().unwrap().len(),
        1,
        "late duplicate never reruns the completed operation"
    );
    restored
        .reply(answer(old, json!(true)), Cancellation::new())
        .await
        .unwrap();
    assert_eq!(
        restored
            .reply(answer(old, json!(false)), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    restored
        .reply(answer(pending, json!(true)), Cancellation::new())
        .await
        .unwrap();
    finish(&mut events).await;
    drop(events);
    drop(restored);
    assert_eq!(log.calls.lock().unwrap().len(), 2);
    assert_eq!(model.0.lock().unwrap().len(), 3);
    let terminal = store.load(1001, &Cancellation::new()).await.unwrap();
    let writes = store.saves.load(Ordering::SeqCst);
    let terminal_agent = create(true);
    let mut replay = Box::pin(
        terminal_agent
            .run(answer(old, json!(true)), true, Cancellation::new())
            .await
            .unwrap(),
    );
    let replay = finish(&mut replay).await;
    assert_eq!(replay.len(), 1);
    assert_eq!(log.calls.lock().unwrap().len(), 2);
    assert_eq!(model.0.lock().unwrap().len(), 3);
    assert_eq!(store.saves.load(Ordering::SeqCst), writes);
    let error = match terminal_agent
        .run(answer(old, json!(false)), true, Cancellation::new())
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("conflicting duplicate accepted"),
    };
    assert_eq!(error.code, "CONFLICT");
    assert_eq!(
        store.load(1001, &Cancellation::new()).await.unwrap(),
        terminal
    );
}

#[tokio::test]
async fn late_saved_replies_are_idempotent_without_approving_the_next_question_or_reexecuting_terminal_runs()
 {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let model = SequentialModel::default();
        let store = Arc::new(Store::default());
        if react {
            recover_late_duplicate(
                |restored| {
                    agent::react()
                        .model(model.clone())
                        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                        .permission(Ask(log.clone()))
                        .checkpoint(store.clone())
                        .with_id_fn(if restored { no_new_restore_id } else { next_id })
                        .build()
                        .unwrap()
                },
                store.clone(),
                log.clone(),
                model.clone(),
            )
            .await;
        } else {
            recover_late_duplicate(
                |restored| {
                    agent::chat()
                        .model(model.clone())
                        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                        .permission(Ask(log.clone()))
                        .checkpoint(store.clone())
                        .with_id_fn(if restored { no_new_restore_id } else { next_id })
                        .build()
                        .unwrap()
                },
                store.clone(),
                log.clone(),
                model.clone(),
            )
            .await;
        }
    }
}

async fn recover_two_root_approvals<A: Agent>(
    create: impl Fn(bool) -> A,
    store: Arc<Store>,
    model: Model,
    log: Arc<Log>,
    mode: bool,
) {
    let original = create(false);
    let mut events = Box::pin(
        original
            .run(input(), mode, Cancellation::new())
            .await
            .unwrap(),
    );
    let (a, _) = question(&mut events).await;
    let (b, _) = question(&mut events).await;
    assert_ne!(a, b);
    drop(events);
    drop(original);
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    let restored = create(true);
    assert!(matches!(
        restored.run(answer(b, json!(true)), mode, Cancellation::new()).await,
        Err(error) if error.code == "CONFLICT"
    ));
    assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
    let mut events = Box::pin(
        restored
            .run(answer(a, json!(false)), mode, Cancellation::new())
            .await
            .unwrap(),
    );
    let (again, _) = question(&mut events).await;
    assert_eq!(again, b, "the next question retains its saved ID");
    loop {
        match poll!(events.next()) {
            std::task::Poll::Ready(Some(event)) => {
                event.unwrap();
            }
            std::task::Poll::Pending => break,
            _ => panic!("the other saved approval must remain pending"),
        }
    }
    assert_eq!(model.requests.lock().unwrap().len(), 1);
    assert!(log.calls.lock().unwrap().is_empty());
    restored
        .reply(answer(b, json!(true)), Cancellation::new())
        .await
        .unwrap();
    assert_eq!(
        restored
            .reply(answer(b, json!(false)), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    finish(&mut events).await;
    assert_eq!(log.calls.lock().unwrap().len(), 1);
    assert_eq!(model.requests.lock().unwrap().len(), 2);
    let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(final_state["usage"]["total"], 26);
    assert_eq!(final_state["query"], saved["query"]);
    assert_eq!(final_state["waits"], json!([]));
    for id in [a, b] {
        let original = saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool" && step["waits"].get(id.to_string()).is_some())
            .unwrap();
        let step = final_state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["id"] == original["id"])
            .unwrap();
        assert_eq!(step["id"], original["id"]);
        assert_eq!(step["waits"][id.to_string()]["status"], "resolved");
        assert_eq!(
            step["waits"][id.to_string()]["feedback"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }
}

#[tokio::test]
async fn saved_root_approvals_resume_only_the_addressed_call_and_keep_other_consumers_live() {
    for react in [false, true] {
        for mode in [false, true] {
            let log = Arc::new(Log::default());
            let model = Model {
                requests: Default::default(),
                multiple: true,
            };
            let store = Arc::new(Store::default());
            if react {
                recover_two_root_approvals(
                    |restored| {
                        agent::react()
                            .model(model.clone())
                            .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                            .permission(Ask(log.clone()))
                            .checkpoint(store.clone())
                            .with_id_fn(if restored { no_new_restore_id } else { next_id })
                            .build()
                            .unwrap()
                    },
                    store.clone(),
                    model.clone(),
                    log.clone(),
                    mode,
                )
                .await;
            } else {
                recover_two_root_approvals(
                    |restored| {
                        agent::chat()
                            .model(model.clone())
                            .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                            .permission(Ask(log.clone()))
                            .checkpoint(store.clone())
                            .with_id_fn(if restored { no_new_restore_id } else { next_id })
                            .build()
                            .unwrap()
                    },
                    store.clone(),
                    model.clone(),
                    log.clone(),
                    mode,
                )
                .await;
            }
        }
    }
}

async fn reject_invalid_saved_approval<A: Agent>(
    create: impl Fn(bool) -> A,
    store: Arc<Store>,
    model: Model,
    log: Arc<Log>,
) {
    let original = create(false);
    let mut events = Box::pin(
        original
            .run(input(), true, Cancellation::new())
            .await
            .unwrap(),
    );
    let (id, _) = question(&mut events).await;
    drop(events);
    drop(original);
    let before = store.load(1001, &Cancellation::new()).await.unwrap();
    let writes = store.saves.load(Ordering::SeqCst);
    let restored = create(true);
    let mut wrong_owner = answer(id, json!(true));
    wrong_owner
        .metadata
        .insert("session_id".into(), json!(9999));
    for (reply, expected) in [
        (answer(id + 10_000, json!(true)), "NOT_FOUND"),
        (answer(id, json!("yes")), "INVALID_ARGUMENTS"),
        (wrong_owner, "CONFLICT"),
        (input(), "WAITING"),
    ] {
        let error = match restored.run(reply, true, Cancellation::new()).await {
            Err(error) => error,
            Ok(_) => panic!("invalid saved feedback accepted"),
        };
        assert_eq!(error.code, expected);
        assert_eq!(
            store.load(1001, &Cancellation::new()).await.unwrap(),
            before
        );
        assert_eq!(store.saves.load(Ordering::SeqCst), writes);
        assert_eq!(model.requests.lock().unwrap().len(), 1);
        assert_eq!(log.before.load(Ordering::SeqCst), 0);
        assert!(log.calls.lock().unwrap().is_empty());
    }
    let mut output = Box::pin(
        restored
            .run(answer(id, json!(true)), true, Cancellation::new())
            .await
            .unwrap(),
    );
    finish(&mut output).await;
    assert_eq!(log.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn invalid_saved_approval_preserves_checkpoint_and_never_invokes_model_hooks_or_tools() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let model = Model {
            requests: Default::default(),
            multiple: false,
        };
        let store = Arc::new(Store::default());
        if react {
            reject_invalid_saved_approval(
                |restored| {
                    agent::react()
                        .model(model.clone())
                        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                        .permission(Ask(log.clone()))
                        .checkpoint(store.clone())
                        .with_id_fn(if restored { no_new_restore_id } else { next_id })
                        .build()
                        .unwrap()
                },
                store.clone(),
                model.clone(),
                log.clone(),
            )
            .await;
        } else {
            reject_invalid_saved_approval(
                |restored| {
                    agent::chat()
                        .model(model.clone())
                        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                        .permission(Ask(log.clone()))
                        .checkpoint(store.clone())
                        .with_id_fn(if restored { no_new_restore_id } else { next_id })
                        .build()
                        .unwrap()
                },
                store.clone(),
                model.clone(),
                log.clone(),
            )
            .await;
        }
    }
}

async fn changed_recovered_call<A: Agent>(
    create: impl Fn(bool) -> A,
    store: Arc<Store>,
    log: Arc<Log>,
) {
    let original = create(false);
    let mut events = Box::pin(
        original
            .run(input(), true, Cancellation::new())
            .await
            .unwrap(),
    );
    let (old, _) = question(&mut events).await;
    drop(events);
    drop(original);
    let restored = create(true);
    let mut events = Box::pin(
        restored
            .run(answer(old, json!(true)), true, Cancellation::new())
            .await
            .unwrap(),
    );
    let (new, message) = question(&mut events).await;
    assert_ne!(new, old);
    assert!(serde_json::to_string(&message).unwrap().contains("Paris"));
    assert!(log.calls.lock().unwrap().is_empty());
    let state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["call"]["arguments"]["city"],
        "Paris"
    );
    assert_eq!(
        state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][old.to_string()]["status"],
        "resolved"
    );
    assert_eq!(
        state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][new.to_string()]["status"],
        "waiting"
    );
    restored
        .reply(answer(old, json!(true)), Cancellation::new())
        .await
        .unwrap();
    assert!(log.calls.lock().unwrap().is_empty());
    restored
        .reply(answer(new, json!(false)), Cancellation::new())
        .await
        .unwrap();
    finish(&mut events).await;
    assert!(log.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_changed_operation_after_recovery_cannot_reuse_the_old_approval() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let model = Model {
            requests: Default::default(),
            multiple: false,
        };
        let store = Arc::new(Store::default());
        if react {
            changed_recovered_call(
                |restored| {
                    agent::react()
                        .model(model.clone())
                        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), restored)]))
                        .permission(Ask(log.clone()))
                        .checkpoint(store.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap()
                },
                store.clone(),
                log.clone(),
            )
            .await;
        } else {
            changed_recovered_call(
                |restored| {
                    agent::chat()
                        .model(model.clone())
                        .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), restored)]))
                        .permission(Ask(log.clone()))
                        .checkpoint(store.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap()
                },
                store.clone(),
                log.clone(),
            )
            .await;
        }
    }
}

#[derive(Clone, Default)]
struct RecoveryOnlyModel(Arc<Mutex<Vec<Request>>>);
impl Chat for RecoveryOnlyModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.0.lock().unwrap().push(request);
        let mut message = Message::new(Role::Assistant);
        message.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"restored result"})
                .as_object()
                .unwrap()
                .clone(),
        });
        Ok(stream::iter(
            [Ok(Event::Complete {
                message,
                usage: Some(Usage {
                    input: Some(10),
                    output: Some(3),
                    total: Some(13),
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
            })]
            .into_iter()
            .collect::<Vec<_>>(),
        ))
    }
}

async fn write_root_wait<A: Agent>(
    agent: A,
    store: Arc<Store>,
    path: &std::path::Path,
    mode: bool,
) {
    let mut events = Box::pin(agent.run(input(), mode, Cancellation::new()).await.unwrap());
    question(&mut events).await;
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    std::fs::write(path, serde_json::to_vec(&saved).unwrap()).unwrap();
}

async fn finish_root_worker<A: Agent>(
    agent: A,
    store: Arc<Store>,
    model: RecoveryOnlyModel,
    log: Arc<Log>,
    mode: bool,
    allow: bool,
) {
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    let id: i64 = saved["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["kind"] == "tool")
        .expect("actual Tool Step")["waits"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let mut events = Box::pin(
        agent
            .run(answer(id, json!(allow)), mode, Cancellation::new())
            .await
            .unwrap(),
    );
    finish(&mut events).await;
    assert_eq!(log.calls.lock().unwrap().len(), usize::from(allow));
    assert_eq!(
        model.0.lock().unwrap().len(),
        1,
        "only the post-tool model call runs in the new process"
    );
    let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(final_state["status"], "succeeded");
    assert_eq!(final_state["usage"]["total"], 26);
    assert_eq!(final_state["query"], saved["query"]);
    assert_eq!(
        final_state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["id"],
        saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["id"]
    );
    assert_eq!(
        final_state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][id.to_string()]["status"],
        "resolved"
    );
    let context = serde_json::to_string(&model.0.lock().unwrap()[0].messages).unwrap();
    assert!(!context.contains("untrusted"));
    assert!(!context.contains("reply_id"));
}

#[tokio::test]
#[ignore = "invoked by root_approval_recovers_in_a_new_os_process_using_only_checkpoint_bytes"]
async fn root_approval_recovery_worker() {
    let path = std::path::PathBuf::from(std::env::var_os("HEART_ROOT_APPROVAL_FILE").unwrap());
    let react = std::env::var("HEART_ROOT_APPROVAL_KIND").unwrap() == "react";
    let mode = std::env::var("HEART_ROOT_APPROVAL_MODE").unwrap() == "true";
    let allow = std::env::var("HEART_ROOT_APPROVAL_ALLOW").unwrap() == "true";
    let restoring = std::env::var("HEART_ROOT_APPROVAL_PHASE").unwrap() == "restore";
    let store = Arc::new(Store::default());
    let log = Arc::new(Log::default());
    if restoring {
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        store.save(1001, saved, &Cancellation::new()).await.unwrap();
        let model = RecoveryOnlyModel::default();
        if react {
            finish_root_worker(
                agent::react()
                    .model(model.clone())
                    .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                    .permission(Ask(log.clone()))
                    .checkpoint(store.clone())
                    .with_id_fn(no_new_restore_id)
                    .build()
                    .unwrap(),
                store.clone(),
                model,
                log,
                mode,
                allow,
            )
            .await;
        } else {
            finish_root_worker(
                agent::chat()
                    .model(model.clone())
                    .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                    .permission(Ask(log.clone()))
                    .checkpoint(store.clone())
                    .with_id_fn(no_new_restore_id)
                    .build()
                    .unwrap(),
                store.clone(),
                model,
                log,
                mode,
                allow,
            )
            .await;
        }
    } else {
        let model = Model {
            requests: Default::default(),
            multiple: false,
        };
        if react {
            write_root_wait(
                agent::react()
                    .model(model)
                    .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                    .permission(Ask(log))
                    .checkpoint(store.clone())
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                store,
                &path,
                mode,
            )
            .await;
        } else {
            write_root_wait(
                agent::chat()
                    .model(model)
                    .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                    .permission(Ask(log))
                    .checkpoint(store.clone())
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
                store,
                &path,
                mode,
            )
            .await;
        }
    }
}

#[test]
fn root_approval_recovers_in_a_new_os_process_using_only_checkpoint_bytes() {
    let path = std::env::temp_dir().join(format!(
        "heart-root-approval-{}-{}.json",
        std::process::id(),
        next_id()
    ));
    for kind in ["chat", "react"] {
        for mode in ["false", "true"] {
            for allow in ["false", "true"] {
                for phase in ["save", "restore"] {
                    let output = std::process::Command::new(std::env::current_exe().unwrap())
                        .args([
                            "--exact",
                            "root_approval_recovery_worker",
                            "--ignored",
                            "--nocapture",
                        ])
                        .env("HEART_ROOT_APPROVAL_FILE", &path)
                        .env("HEART_ROOT_APPROVAL_KIND", kind)
                        .env("HEART_ROOT_APPROVAL_MODE", mode)
                        .env("HEART_ROOT_APPROVAL_ALLOW", allow)
                        .env("HEART_ROOT_APPROVAL_PHASE", phase)
                        .output()
                        .unwrap();
                    assert!(
                        output.status.success(),
                        "{kind}/{mode}/{allow}/{phase}: {} {}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
            }
        }
    }
    std::fs::remove_file(&path).unwrap();
}

async fn recover_root_approval<A: Agent>(
    create: impl Fn(bool) -> A,
    store: Arc<Store>,
    model: Model,
    log: Arc<Log>,
    mode: bool,
    allow: bool,
) {
    let original = create(false);
    let mut events = Box::pin(
        original
            .run(input(), mode, Cancellation::new())
            .await
            .unwrap(),
    );
    let (id, _) = question(&mut events).await;
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(saved["stream"], mode);
    assert!(log.calls.lock().unwrap().is_empty());
    drop(events);
    drop(original);
    let restored = create(true);
    let mut output = Box::pin(
        restored
            // The Checkpoint owns the mode for this continued run; the new
            // invocation's argument must not switch delivery halfway through.
            .run(answer(id, json!(allow)), !mode, Cancellation::new())
            .await
            .unwrap(),
    );
    let output = finish(&mut output).await;
    assert_eq!(
        model.requests.lock().unwrap().len(),
        2,
        "the selecting model call is not replayed"
    );
    assert_eq!(log.calls.lock().unwrap().len(), usize::from(allow));
    if allow {
        assert_eq!(log.calls.lock().unwrap()[0].arguments["city"], "Beijing");
    }
    for event in &output {
        assert!(!matches!(event, Event::Delta { .. }) || mode);
        if let Event::Complete { message, .. } = event {
            assert!(
                !message.metadata.contains_key("reply_id"),
                "consuming an answer does not re-present the question"
            );
        }
    }
    let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(final_state["status"], "succeeded");
    assert_eq!(final_state["stream"], mode);
    assert_eq!(final_state["usage"]["total"], 26);
    assert_eq!(final_state["query"], saved["query"]);
    assert_eq!(
        final_state["execution"]["exec_id"],
        saved["execution"]["exec_id"]
    );
    assert_eq!(
        final_state["execution"].is_null(),
        saved["execution"].is_null()
    );
    assert_eq!(
        final_state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["id"],
        saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["id"]
    );
    assert_eq!(
        final_state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["call_id"],
        saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["call_id"]
    );
    assert_eq!(final_state["waits"], json!([]));
    assert_eq!(
        final_state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][id.to_string()]["status"],
        "resolved"
    );
    let feedback = &final_state["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["kind"] == "tool")
        .expect("actual Tool Step")["waits"][id.to_string()]["feedback"];
    assert_eq!(feedback.as_array().unwrap().len(), 1);
    let requests = model.requests.lock().unwrap();
    let context = serde_json::to_value(&requests.last().unwrap().messages)
        .unwrap()
        .to_string();
    assert!(!context.contains("untrusted"));
    assert!(!context.contains("reply_id"));
}

#[tokio::test]
async fn saved_root_approval_resumes_on_a_fresh_agent_without_replaying_the_selecting_model() {
    for react in [false, true] {
        for mode in [false, true] {
            for allow in [false, true] {
                let log = Arc::new(Log::default());
                let model = Model {
                    requests: Default::default(),
                    multiple: false,
                };
                let store = Arc::new(Store::default());
                if react {
                    recover_root_approval(
                        |restored| {
                            agent::react()
                                .model(model.clone())
                                .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                                .permission(Ask(log.clone()))
                                .checkpoint(store.clone())
                                .with_id_fn(if restored { no_new_restore_id } else { next_id })
                                .build()
                                .unwrap()
                        },
                        store.clone(),
                        model.clone(),
                        log.clone(),
                        mode,
                        allow,
                    )
                    .await;
                } else {
                    recover_root_approval(
                        |restored| {
                            agent::chat()
                                .model(model.clone())
                                .toolkit(Toolkit::new().tools([Lookup::new(log.clone(), false)]))
                                .permission(Ask(log.clone()))
                                .checkpoint(store.clone())
                                .with_id_fn(if restored { no_new_restore_id } else { next_id })
                                .build()
                                .unwrap()
                        },
                        store.clone(),
                        model.clone(),
                        log.clone(),
                        mode,
                        allow,
                    )
                    .await;
                }
            }
        }
    }
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
        let capabilities = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
        if react {
            no_reload(
                agent::react()
                    .model(model)
                    .toolkit(capabilities)
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
                    .toolkit(capabilities)
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
        let capabilities = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
        if react {
            cancel_before_presentation(
                agent::react()
                    .model(model)
                    .toolkit(capabilities)
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
                    .toolkit(capabilities)
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
        let capabilities = Toolkit::new().tools([Lookup::new(log.clone(), false)]);
        if react {
            failed_save(
                agent::react()
                    .model(model)
                    .toolkit(capabilities)
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
                    .toolkit(capabilities)
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
