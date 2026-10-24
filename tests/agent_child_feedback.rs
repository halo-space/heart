//! Direct delegated feedback retains native child scenes, never parent Steps.
use futures_util::{StreamExt, poll, stream};
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::{self, Agent, Input, SubAgent, },
    checkpoint::Checkpoint,
    core,
    event::Event,
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
    runtime::agent::wait,
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicI64, Ordering},
    },
};
fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(21_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
fn no_new_id() -> i64 {
    panic!("saved child/execution/question IDs must be retained")
}
fn input() -> Input {
    serde_json::from_value(json!({"metadata":{"chat_id":1001,"session_id":2001,"agent_id":3001},"payload":[{"type":"text","data":{"value":"complete both children"}}]})).unwrap()
}
fn answer(id: i64, value: &str) -> Input {
    let mut i = input();
    i.metadata.insert("reply_id".into(), id.into());
    i.payload[0].data = json!({"value":value});
    i
}
#[derive(Clone)]
struct Model {
    parent: bool,
    requests: Arc<Mutex<Vec<Request>>>,
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        if self.parent {
            if request
                .messages
                .as_slice()
                .iter()
                .any(|m| m.metadata.get("name").and_then(Value::as_str) == Some("agent_spawn"))
            {
                message.values.insert("answer".into(), json!("done"));
            } else {
                for (id, item) in [("left_spawn", "left"), ("right_spawn", "right")] {
                    message.content.push(Part{r#type:"tool_call".into(),data:json!({"call_id":id,"name":"agent_spawn","arguments":{"name":"worker","input":{"payload":[{"type":"text","data":{"value":item}}]}}}).as_object().unwrap().clone()});
                }
            }
        } else {
            let item = request
                .messages
                .as_slice()
                .iter()
                .filter(|m| m.role == Role::User)
                .flat_map(|m| &m.content)
                .find_map(|p| p.data.get("value").and_then(Value::as_str))
                .unwrap()
                .to_owned();
            if request
                .messages
                .as_slice()
                .iter()
                .any(|m| m.values.get("item").and_then(Value::as_str) == Some(item.as_str()))
            {
                message.values.insert("answer".into(), json!(item));
            } else {
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: json!({"call_id":"same_call","name":"collect","arguments":{"item":item}})
                        .as_object()
                        .unwrap()
                        .clone(),
                });
            }
        }
        self.requests.lock().unwrap().push(request);
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
    received: Mutex<BTreeMap<String, Messages>>,
}
struct Collect {
    definition: ToolDefinition,
    log: Arc<Log>,
    count: usize,
}
impl Tool for Collect {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let item = call.arguments["item"].as_str().unwrap().to_owned();
        let mut question = Message::new(Role::Assistant);
        question.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"请补充反馈"}).as_object().unwrap().clone(),
        });
        let messages = wait::ask(question, self.count, cancellation).await?;
        self.log
            .received
            .lock()
            .unwrap()
            .insert(item.clone(), messages);
        Ok(Message::function(json!({"item":item})))
    }
}
fn child(
    react: bool,
    log: Arc<Log>,
    requests: Arc<Mutex<Vec<Request>>>,
    checkpoint: Arc<core::checkpoint::InMemory>,
    count: usize,
) -> SubAgent {
    let tool=Collect{definition:ToolDefinition{name:"collect".into(),description:"collect feedback".into(),parameters:json!({"type":"object","properties":{"item":{"type":"string"}},"required":["item"],"additionalProperties":false}).as_object().unwrap().clone()},log,count};
    let model = Model {
        parent: false,
        requests,
    };
    if react {
        SubAgent::new(
            "worker",
            agent::react()
                .model(model)
                .tools(tool)
                .checkpoint(checkpoint)
                .build()
                .unwrap(),
        )
        .unwrap()
    } else {
        SubAgent::new(
            "worker",
            agent::chat()
                .model(model)
                .tools(tool)
                .checkpoint(checkpoint)
                .build()
                .unwrap(),
        )
        .unwrap()
    }
}
async fn question<S: futures_core::Stream<Item = Result<Event, Error>>>(
    events: &mut Pin<Box<S>>,
) -> (i64, String) {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            if let Event::Complete { message, .. } = events.next().await.expect("question").unwrap()
                && let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64)
            {
                return (id, message.metadata["node_id"].as_str().unwrap().to_owned());
            }
        }
    })
    .await
    .expect("both direct children must be able to wait with capacity one")
}
fn node<'a>(scene: &'a Value, id: &str) -> &'a Value {
    &scene["execution"]["attempts"][0]["nodes"][id]
}
fn child_scene<'a>(scene: &'a Value, id: &str) -> &'a Value {
    &node(scene, id)["execution"]["attempts"][0]["agent"]
}
fn record<'a>(scene: &'a Value, node_id: &str, id: i64) -> &'a Value {
    &child_scene(scene, node_id)["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["kind"] == "tool")
        .expect("actual Tool Step")["waits"][id.to_string()]
}
async fn finish<S: futures_core::Stream<Item = Result<Event, Error>>>(
    events: &mut Pin<Box<S>>,
) -> Vec<Event> {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        let mut tail = vec![];
        while let Some(event) = events.next().await {
            tail.push(event.unwrap());
        }
        tail
    })
    .await
    .expect("parent must finish after its children")
}
async fn verify<A: Agent>(
    agent: A,
    checkpoint: Arc<core::checkpoint::InMemory>,
    own: Arc<core::checkpoint::InMemory>,
    log: Arc<Log>,
    requests: Arc<Mutex<Vec<Request>>>,
    count: usize,
    mode: bool,
) {
    let mut events = Box::pin(agent.run(input(), mode, Cancellation::new()).await.unwrap());
    let (a, first) = question(&mut events).await;
    let (b, second) = question(&mut events).await;
    let original = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_ne!(a, b);
    assert_ne!(first, second);
    assert_eq!(original["waits"][0]["nodes"][0]["node_id"], first);
    assert_eq!(original["waits"][1]["nodes"][0]["node_id"], second);
    assert_eq!(
        original["steps"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|s| s.get("waits").is_some())
            .count(),
        0
    );
    assert_eq!(
        agent
            .reply(answer(b, "out of order"), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    for idx in 0..count {
        agent
            .reply(answer(a, &format!("first-{idx}")), Cancellation::new())
            .await
            .unwrap();
        agent
            .reply(answer(a, &format!("first-{idx}")), Cancellation::new())
            .await
            .unwrap();
        if idx + 1 < count {
            assert!(poll!(events.next()).is_pending());
            let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(
                record(&saved, &first, a)["feedback"]
                    .as_array()
                    .unwrap()
                    .len(),
                idx + 1
            );
            assert_eq!(record(&saved, &second, b)["feedback"], json!([]));
            assert_eq!(saved["query"], original["query"]);
            assert_eq!(saved["usage"], original["usage"]);
            assert!(log.received.lock().unwrap().is_empty());
        }
    }
    let mut prefix = vec![];
    tokio::time::timeout(std::time::Duration::from_secs(3),async{loop{
        let event=events.next().await.unwrap().unwrap();
        let done=matches!(&event,Event::Complete{message,..} if message.metadata.get("node_id").and_then(Value::as_str)==Some(first.as_str()) && message.values.contains_key("answer"));
        prefix.push(event);if done{break;}
    }}).await.unwrap();
    assert_eq!(log.received.lock().unwrap().len(), 1);
    assert_eq!(
        requests.lock().unwrap().len(),
        1,
        "parent must not infer before both children finish"
    );
    agent
        .reply(answer(a, "first-0"), Cancellation::new())
        .await
        .unwrap();
    for idx in 0..count {
        agent
            .reply(answer(b, &format!("second-{idx}")), Cancellation::new())
            .await
            .unwrap();
    }
    prefix.extend(finish(&mut events).await);
    assert!(!prefix.iter().any(
        |e| matches!(e,Event::Complete{message,..} if message.metadata.contains_key("reply_id"))
    ));
    if !mode {
        assert!(!prefix.iter().any(|e| matches!(e, Event::Delta { .. })));
    }
    let final_state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(final_state["status"], "succeeded");
    assert_eq!(final_state["usage"]["total"], 78);
    for (n, id) in [(&first, a), (&second, b)] {
        assert_eq!(
            node(&final_state, n)["execution"]["exec_id"],
            node(&original, n)["execution"]["exec_id"]
        );
        assert_eq!(
            record(&final_state, n, id)["feedback"]
                .as_array()
                .unwrap()
                .len(),
            count
        );
        assert_eq!(record(&final_state, n, id)["status"], "resolved");
    }
    assert_eq!(
        own.load(1001, &Cancellation::new()).await.unwrap_err().code,
        "NOT_FOUND"
    );
}
#[tokio::test]
async fn direct_children_collect_native_feedback_on_the_parent_stream_without_receipts() {
    for react_child in [false, true] {
        for mode in [false, true] {
            for capacity in [1, 2] {
                for count in [1, 3] {
                    macro_rules! check {
                        ($constructor:expr) => {{
                            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                            let own = Arc::new(core::checkpoint::InMemory::new());
                            let log = Arc::new(Log::default());
                            let requests = Arc::new(Mutex::new(vec![]));
                            let agent = $constructor
                                .model(Model {
                                    parent: true,
                                    requests: requests.clone(),
                                })
                                .subagents([child(
                                    react_child,
                                    log.clone(),
                                    Default::default(),
                                    own.clone(),
                                    count,
                                )])
                                .checkpoint(checkpoint.clone())
                                .with_id_fn(next_id)
                                .with_max_concurrency(capacity)
                                .build()
                                .unwrap();
                            verify(agent, checkpoint, own, log, requests, count, mode).await;
                        }};
                    }
                    check!(agent::chat());
                    check!(agent::react());
                }
            }
        }
    }
}

async fn save<A: Agent>(
    agent: &A,
    checkpoint: &core::checkpoint::InMemory,
    mode: bool,
) -> (i64, String, i64, String, Value) {
    let mut events = Box::pin(agent.run(input(), mode, Cancellation::new()).await.unwrap());
    let (a, first) = question(&mut events).await;
    let (b, second) = question(&mut events).await;
    agent
        .reply(answer(a, "first"), Cancellation::new())
        .await
        .unwrap();
    assert!(poll!(events.next()).is_pending());
    let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        record(&state, &first, a)["feedback"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    drop(events);
    (a, first, b, second, state)
}
struct Saved {
    a: i64,
    first: String,
    b: i64,
    second: String,
    state: Value,
}
async fn restore<A: Agent>(
    agent: A,
    checkpoint: Arc<core::checkpoint::InMemory>,
    saved: Saved,
    requests: Arc<Mutex<Vec<Request>>>,
    child_requests: Arc<Mutex<Vec<Request>>>,
    mode: bool,
) {
    let mut events = Box::pin(
        agent
            .run(answer(saved.a, "second"), !mode, Cancellation::new())
            .await
            .unwrap(),
    );
    let mut prefix = vec![];
    for _ in 0..8 {
        match poll!(events.next()) {
            std::task::Poll::Ready(Some(Ok(event))) => prefix.push(event),
            std::task::Poll::Pending => break,
            other => panic!("unexpected {other:?}"),
        }
    }
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["stream"], mode);
    assert_eq!(scene["query"], saved.state["query"]);
    assert_eq!(scene["usage"], saved.state["usage"]);
    assert_eq!(
        record(&scene, &saved.first, saved.a)["feedback"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        record(&scene, &saved.second, saved.b)["feedback"],
        json!([])
    );
    assert!(requests.lock().unwrap().is_empty());
    assert!(child_requests.lock().unwrap().is_empty());
    agent
        .reply(answer(saved.a, "first"), Cancellation::new())
        .await
        .unwrap();
    assert!(poll!(events.next()).is_pending());
    agent
        .reply(answer(saved.a, "third"), Cancellation::new())
        .await
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(3),async{loop{let event=events.next().await.unwrap().unwrap();let done=matches!(&event,Event::Complete{message,..} if message.metadata.get("node_id").and_then(Value::as_str)==Some(saved.first.as_str()) && message.values.contains_key("answer"));prefix.push(event);if done{break;}}}).await.unwrap();
    for text in ["one", "two", "three"] {
        agent
            .reply(answer(saved.b, text), Cancellation::new())
            .await
            .unwrap();
    }
    prefix.extend(finish(&mut events).await);
    if !mode {
        assert!(!prefix.iter().any(|e| matches!(e, Event::Delta { .. })));
    }
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["status"], "succeeded");
    assert_eq!(scene["usage"]["total"], 78);
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert_eq!(child_requests.lock().unwrap().len(), 2);
    for (n, id) in [(&saved.first, saved.a), (&saved.second, saved.b)] {
        assert_eq!(
            record(&scene, n, id)["question"],
            record(&saved.state, n, id)["question"]
        );
        assert_eq!(
            node(&scene, n)["execution"]["exec_id"],
            node(&saved.state, n)["execution"]["exec_id"]
        );
    }
}
#[tokio::test]
async fn fresh_parent_restores_direct_child_partial_feedback_without_redelegating() {
    for react_child in [false, true] {
        for mode in [false, true] {
            macro_rules! check {
                ($constructor:expr) => {{
                    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                    let own = Arc::new(core::checkpoint::InMemory::new());
                    let old = $constructor
                        .model(Model {
                            parent: true,
                            requests: Default::default(),
                        })
                        .subagents([child(
                            react_child,
                            Default::default(),
                            Default::default(),
                            own.clone(),
                            3,
                        )])
                        .checkpoint(checkpoint.clone())
                        .with_id_fn(next_id)
                        .with_max_concurrency(1)
                        .build()
                        .unwrap();
                    let (a, first, b, second, state) = save(&old, &checkpoint, mode).await;
                    let requests = Arc::new(Mutex::new(vec![]));
                    let child_requests = Arc::new(Mutex::new(vec![]));
                    let agent = $constructor
                        .model(Model {
                            parent: true,
                            requests: requests.clone(),
                        })
                        .subagents([child(
                            react_child,
                            Default::default(),
                            child_requests.clone(),
                            own,
                            3,
                        )])
                        .checkpoint(checkpoint.clone())
                        .with_id_fn(no_new_id)
                        .with_max_concurrency(1)
                        .build()
                        .unwrap();
                    restore(
                        agent,
                        checkpoint,
                        Saved {
                            a,
                            first,
                            b,
                            second,
                            state,
                        },
                        requests,
                        child_requests,
                        mode,
                    )
                    .await;
                }};
            }
            check!(agent::chat());
            check!(agent::react());
        }
    }
}

async fn cancel<A: Agent>(agent: A, checkpoint: Arc<core::checkpoint::InMemory>, log: Arc<Log>) {
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(input(), false, cancellation.clone())
            .await
            .unwrap(),
    );
    let (a, first) = question(&mut events).await;
    let (b, _) = question(&mut events).await;
    agent
        .reply(answer(a, "first"), Cancellation::new())
        .await
        .unwrap();
    assert!(poll!(events.next()).is_pending());
    let before = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    cancellation.cancel();
    let tail = finish(&mut events).await;
    assert!(tail.iter().any(
        |e| matches!(e,Event::Complete{finish_reason:Some(reason),..} if reason=="cancelled")
    ));
    let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(saved["status"], "stopped");
    assert_eq!(saved["waits"], json!([]));
    assert_eq!(record(&saved, &first, a), record(&before, &first, a));
    assert_eq!(saved["usage"], before["usage"]);
    assert!(log.received.lock().unwrap().is_empty());
    for id in [a, b] {
        assert_eq!(
            agent
                .reply(answer(id, "late"), Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
    }
}
#[tokio::test]
async fn parent_cancellation_retains_partial_child_facts_without_completing_tools() {
    for react_child in [false, true] {
        macro_rules! check {
            ($constructor:expr) => {{
                let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                let log = Arc::new(Log::default());
                let agent = $constructor
                    .model(Model {
                        parent: true,
                        requests: Default::default(),
                    })
                    .subagents([child(
                        react_child,
                        log.clone(),
                        Default::default(),
                        Default::default(),
                        3,
                    )])
                    .checkpoint(checkpoint.clone())
                    .with_id_fn(next_id)
                    .with_max_concurrency(1)
                    .build()
                    .unwrap();
                cancel(agent, checkpoint, log).await;
            }};
        }
        check!(agent::chat());
        check!(agent::react());
    }
}
#[derive(Clone)]
struct FailPartial(Arc<core::checkpoint::InMemory>);
impl Checkpoint for FailPartial {
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        let partial = state["execution"]["attempts"][0]["nodes"]
            .as_object()
            .is_some_and(|nodes| {
                nodes.values().any(|node| {
                    node["execution"]["attempts"][0]["agent"]["steps"]
                        .as_array()
                        .is_some_and(|steps| {
                            steps.iter().any(|step| {
                                step["waits"].as_object().is_some_and(|waits| {
                                    waits.values().any(|record| {
                                        record["kind"] == "input"
                                            && record["feedback"]
                                                .as_array()
                                                .is_some_and(|messages| messages.len() == 1)
                                    })
                                })
                            })
                        })
                })
            });
        if state["status"] == "waiting" && partial {
            return Err(Error::new(
                "SAVE_FAILED",
                "outer partial feedback save failed",
            ));
        }
        self.0.save(id, state, cancellation).await
    }
    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        self.0.load(id, cancellation).await
    }
    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        self.0.delete(id, cancellation).await
    }
}
async fn save_fails<A: Agent>(
    agent: A,
    checkpoint: Arc<core::checkpoint::InMemory>,
    log: Arc<Log>,
) {
    let mut events = Box::pin(
        agent
            .run(input(), false, Cancellation::new())
            .await
            .unwrap(),
    );
    let (a, first) = question(&mut events).await;
    let (b, _) = question(&mut events).await;
    agent
        .reply(answer(a, "first"), Cancellation::new())
        .await
        .unwrap();
    let error = tokio::time::timeout(std::time::Duration::from_secs(3), events.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert_eq!(error.code, "SAVE_FAILED");
    assert!(events.next().await.is_none());
    let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(state["status"], "failed");
    assert_eq!(state["waits"], json!([]));
    assert_eq!(
        record(&state, &first, a)["feedback"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(record(&state, &first, a)["status"], "waiting");
    assert!(log.received.lock().unwrap().is_empty());
    for id in [a, b] {
        assert_eq!(
            agent
                .reply(answer(id, "late"), Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
    }
}
#[tokio::test]
async fn outer_partial_save_failure_never_completes_child_or_leaves_an_actionable_receiver() {
    for react_child in [false, true] {
        macro_rules! check {
            ($constructor:expr) => {{
                let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                let log = Arc::new(Log::default());
                let agent = $constructor
                    .model(Model {
                        parent: true,
                        requests: Default::default(),
                    })
                    .subagents([child(
                        react_child,
                        log.clone(),
                        Default::default(),
                        Default::default(),
                        3,
                    )])
                    .checkpoint(FailPartial(checkpoint.clone()))
                    .with_id_fn(next_id)
                    .with_max_concurrency(1)
                    .build()
                    .unwrap();
                save_fails(agent, checkpoint, log).await;
            }};
        }
        check!(agent::chat());
        check!(agent::react());
    }
}

#[tokio::test]
async fn stale_child_coordinates_and_wrong_feedback_leave_the_outer_scene_unchanged() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let old = agent::chat()
        .model(Model {
            parent: true,
            requests: Default::default(),
        })
        .subagents([child(
            false,
            Default::default(),
            Default::default(),
            Default::default(),
            3,
        )])
        .checkpoint(checkpoint.clone())
        .with_id_fn(next_id)
        .with_max_concurrency(1)
        .build()
        .unwrap();
    let (a, _, b, _, scene) = save(&old, &checkpoint, false).await;
    for case in ["owner", "fifo", "stale"] {
        let mut state = scene.clone();
        let mut response = answer(if case == "fifo" { b } else { a }, "next");
        if case == "owner" {
            response.metadata.insert("session_id".into(), json!(7777));
        }
        if case == "stale" {
            state["waits"][0]["nodes"][0]["exec_id"] = json!(-1);
        }
        checkpoint
            .save(1001, state.clone(), &Cancellation::new())
            .await
            .unwrap();
        let requests = Arc::new(Mutex::new(vec![]));
        let child_requests = Arc::new(Mutex::new(vec![]));
        let agent = agent::chat()
            .model(Model {
                parent: true,
                requests: requests.clone(),
            })
            .subagents([child(
                false,
                Default::default(),
                child_requests.clone(),
                Default::default(),
                3,
            )])
            .checkpoint(checkpoint.clone())
            .with_id_fn(no_new_id)
            .build()
            .unwrap();
        let error = match agent.run(response, false, Cancellation::new()).await {
            Ok(_) => panic!("invalid feedback accepted"),
            Err(error) => error,
        };
        assert_eq!(
            error.code,
            if case == "stale" {
                "INVALID_STATE"
            } else {
                "CONFLICT"
            }
        );
        assert!(requests.lock().unwrap().is_empty());
        assert!(child_requests.lock().unwrap().is_empty());
        assert_eq!(
            checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
            state
        );
    }
}

#[tokio::test]
#[ignore = "explicitly invoked by direct_child_feedback_recovers_in_fresh_processes"]
async fn child_feedback_process_worker() {
    let path = std::env::var("HEART_CHILD_FEEDBACK_STATE").unwrap();
    let phase = std::env::var("HEART_CHILD_FEEDBACK_PHASE").unwrap();
    let mode = std::env::var("HEART_CHILD_FEEDBACK_MODE").unwrap() == "true";
    let child_react = std::env::var("HEART_CHILD_FEEDBACK_CHILD").unwrap() == "react";
    macro_rules! run {
        ($constructor:expr) => {{
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            if phase == "save" {
                let agent = $constructor
                    .model(Model {
                        parent: true,
                        requests: Default::default(),
                    })
                    .subagents([child(
                        child_react,
                        Default::default(),
                        Default::default(),
                        Default::default(),
                        3,
                    )])
                    .checkpoint(checkpoint.clone())
                    .with_id_fn(next_id)
                    .with_max_concurrency(1)
                    .build()
                    .unwrap();
                let (_, _, _, _, scene) = save(&agent, &checkpoint, mode).await;
                std::fs::write(&path, serde_json::to_vec(&scene).unwrap()).unwrap();
            } else {
                let state: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                let pairs = state["waits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|position| {
                        let n = position["nodes"][0]["node_id"].as_str().unwrap().to_owned();
                        let id = child_scene(&state, &n)["steps"]
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
                            .parse::<i64>()
                            .unwrap();
                        (id, n)
                    })
                    .collect::<Vec<_>>();
                checkpoint
                    .save(1001, state.clone(), &Cancellation::new())
                    .await
                    .unwrap();
                let requests = Arc::new(Mutex::new(vec![]));
                let child_requests = Arc::new(Mutex::new(vec![]));
                let agent = $constructor
                    .model(Model {
                        parent: true,
                        requests: requests.clone(),
                    })
                    .subagents([child(
                        child_react,
                        Default::default(),
                        child_requests.clone(),
                        Default::default(),
                        3,
                    )])
                    .checkpoint(checkpoint.clone())
                    .with_id_fn(no_new_id)
                    .with_max_concurrency(1)
                    .build()
                    .unwrap();
                restore(
                    agent,
                    checkpoint,
                    Saved {
                        a: pairs[0].0,
                        first: pairs[0].1.clone(),
                        b: pairs[1].0,
                        second: pairs[1].1.clone(),
                        state,
                    },
                    requests,
                    child_requests,
                    mode,
                )
                .await;
            }
        }};
    }
    if std::env::var("HEART_CHILD_FEEDBACK_PARENT").unwrap() == "react" {
        run!(agent::react());
    } else {
        run!(agent::chat());
    }
}
#[test]
fn direct_child_feedback_recovers_in_fresh_processes() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "heart-child-feedback-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    for parent in ["chat", "react"] {
        for child in ["chat", "react"] {
            for mode in ["false", "true"] {
                let path = directory.join(format!("{parent}-{child}-{mode}.json"));
                for phase in ["save", "restore"] {
                    let output = std::process::Command::new(std::env::current_exe().unwrap())
                        .args([
                            "--ignored",
                            "--exact",
                            "child_feedback_process_worker",
                            "--nocapture",
                        ])
                        .env("HEART_CHILD_FEEDBACK_STATE", &path)
                        .env("HEART_CHILD_FEEDBACK_PHASE", phase)
                        .env("HEART_CHILD_FEEDBACK_PARENT", parent)
                        .env("HEART_CHILD_FEEDBACK_CHILD", child)
                        .env("HEART_CHILD_FEEDBACK_MODE", mode)
                        .output()
                        .unwrap();
                    assert!(
                        output.status.success(),
                        "{parent}/{child}/{mode}/{phase}: {}\n{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                std::fs::remove_file(path).unwrap();
            }
        }
    }
    std::fs::remove_dir(directory).unwrap();
}
