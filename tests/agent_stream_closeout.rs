//! Cancelling and draining is a semantic closeout; dropping the owning stream
//! is not an asynchronous checkpoint writer or a successful Complete.
use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
};

use futures_core::Stream;
use futures_util::{StreamExt, poll, stream};
use halo::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, SubAgent},
    checkpoint::Checkpoint,
    core,
    event::Event,
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
};
use serde_json::{Value, json};

#[derive(Default)]
struct Store {
    backend: core::checkpoint::InMemory,
    loads: AtomicUsize,
    saves: AtomicUsize,
}
impl Checkpoint for Store {
    async fn save(&self, id: i64, scene: Value, c: &Cancellation) -> Result<(), Error> {
        self.saves.fetch_add(1, Ordering::SeqCst);
        self.backend.save(id, scene, c).await
    }
    async fn load(&self, id: i64, c: &Cancellation) -> Result<Value, Error> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.backend.load(id, c).await
    }
    async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
        self.backend.delete(id, c).await
    }
}
#[derive(Default)]
struct Log {
    parent: AtomicUsize,
    model: AtomicUsize,
    dropped: AtomicUsize,
}
fn text(value: &str) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":value}).as_object().unwrap().clone(),
    });
    message
}
fn partial() -> Message {
    let mut message = text("received partial");
    message.usage = Some(Usage {
        input: Some(30),
        output: Some(10),
        total: Some(40),
        ..Default::default()
    });
    message
}
struct Pending {
    first: bool,
    log: Arc<Log>,
}
impl Stream for Pending {
    type Item = Result<Event, Error>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.first {
            self.first = false;
            Poll::Ready(Some(Ok(Event::Delta { message: partial() })))
        } else {
            Poll::Pending
        }
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        self.log.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
struct Model(Arc<Log>);
impl Chat for Model {
    type Stream = Pending;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        self.0.model.fetch_add(1, Ordering::SeqCst);
        Ok(Pending {
            first: true,
            log: self.0.clone(),
        })
    }
}
struct Parent(Arc<Log>);
impl Chat for Parent {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        assert_eq!(
            self.0.parent.fetch_add(1, Ordering::SeqCst),
            0,
            "cancelled child cannot cause another parent inference"
        );
        let mut message = Message::new(Role::Assistant);
        message.content.push(Part {
            r#type: "tool_call".into(),
            data: json!({
                "name":"agent_spawn","call_id":"delegate","arguments":{"name":"child","input":{
                    "payload":[{"type":"text","data":{"value":"child query"}}]
                }}
            })
            .as_object()
            .unwrap()
            .clone(),
        });
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                total: Some(2),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}
fn ids() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(96_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
fn input() -> Input {
    serde_json::from_value(json!({"metadata":{"chat_id":1001,"session_id":2001},"payload":[{"type":"text","data":{"value":"question"}}]})).unwrap()
}
#[allow(clippy::too_many_arguments)]
async fn start(
    parent_react: bool,
    child_react: Option<bool>,
    mode: bool,
    store: Arc<Store>,
    child_store: Arc<Store>,
    log: Arc<Log>,
    cancellation: Cancellation,
) -> Pin<Box<dyn Stream<Item = Result<Event, Error>>>> {
    if let Some(child_react) = child_react {
        let child = if child_react {
            SubAgent::new(
                "child",
                agent::react()
                    .model(Model(log.clone()))
                    .checkpoint(child_store)
                    .build()
                    .unwrap(),
            )
            .unwrap()
        } else {
            SubAgent::new(
                "child",
                agent::chat()
                    .model(Model(log.clone()))
                    .checkpoint(child_store)
                    .build()
                    .unwrap(),
            )
            .unwrap()
        };
        if parent_react {
            Box::pin(
                agent::react()
                    .model(Parent(log))
                    .subagents(vec![child])
                    .with_id_fn(ids)
                    .checkpoint(store)
                    .build()
                    .unwrap()
                    .run(input(), mode, cancellation)
                    .await
                    .unwrap(),
            )
        } else {
            Box::pin(
                agent::chat()
                    .model(Parent(log))
                    .subagents(vec![child])
                    .with_id_fn(ids)
                    .checkpoint(store)
                    .build()
                    .unwrap()
                    .run(input(), mode, cancellation)
                    .await
                    .unwrap(),
            )
        }
    } else if parent_react {
        Box::pin(
            agent::react()
                .model(Model(log))
                .checkpoint(store)
                .build()
                .unwrap()
                .run(input(), mode, cancellation)
                .await
                .unwrap(),
        )
    } else {
        Box::pin(
            agent::chat()
                .model(Model(log))
                .checkpoint(store)
                .build()
                .unwrap()
                .run(input(), mode, cancellation)
                .await
                .unwrap(),
        )
    }
}
async fn until_partial<S: Stream<Item = Result<Event, Error>> + ?Sized>(events: &mut Pin<Box<S>>) {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            if let Event::Delta { message } = events.next().await.expect("live stream").unwrap() {
                assert_eq!(message.content, partial().content);
                return;
            }
        }
    })
    .await
    .unwrap();
}
async fn until_pending<S: Stream<Item = Result<Event, Error>> + ?Sized>(
    events: &mut Pin<Box<S>>,
    log: &Log,
) {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            match poll!(events.next()) {
                Poll::Ready(Some(Ok(Event::Complete { .. }))) => {}
                Poll::Pending if log.model.load(Ordering::SeqCst) == 1 => break,
                Poll::Pending => tokio::task::yield_now().await,
                other => panic!("{other:?}"),
            }
        }
    })
    .await
    .unwrap();
}
fn check_attempt(scene: &Value) {
    assert_eq!(scene["status"], "stopped");
    let step = scene["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["kind"] == "model")
        .unwrap();
    assert!(step["message"].is_null());
    assert_eq!(step["attempts"][0]["status"], "cancelled");
    assert_eq!(
        step["attempts"][0]["message"]["content"],
        serde_json::to_value(partial().content).unwrap()
    );
    assert_eq!(step["attempts"][0]["usage"]["total"], 40);
    assert_eq!(scene["usage"]["total"], 40);
}

#[tokio::test]
async fn dropping_the_stream_releases_work_without_writing_or_fabricating_success() {
    for react in [false, true] {
        for child in [None, Some(false), Some(true)] {
            let store = Arc::new(Store::default());
            let child_store = Arc::new(Store::default());
            let log = Arc::new(Log::default());
            let mut events = start(
                react,
                child,
                true,
                store.clone(),
                child_store.clone(),
                log.clone(),
                Cancellation::new(),
            )
            .await;
            until_partial(&mut events).await;
            let saved = store.backend.load(1001, &Cancellation::new()).await.ok();
            let writes = store.saves.load(Ordering::SeqCst);
            drop(events);
            assert_eq!(log.dropped.load(Ordering::SeqCst), 1);
            tokio::task::yield_now().await;
            assert_eq!(store.saves.load(Ordering::SeqCst), writes);
            assert_eq!(
                store.backend.load(1001, &Cancellation::new()).await.ok(),
                saved
            );
            assert_eq!(log.model.load(Ordering::SeqCst), 1);
            assert_eq!(child_store.loads.load(Ordering::SeqCst), 0);
            assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
        }
    }
}

#[tokio::test]
async fn cancelling_and_draining_independent_models_keeps_partial_and_consumption() {
    for react in [false, true] {
        let store = Arc::new(Store::default());
        let log = Arc::new(Log::default());
        let cancellation = Cancellation::new();
        let mut events = start(
            react,
            None,
            true,
            store.clone(),
            Arc::new(Store::default()),
            log.clone(),
            cancellation.clone(),
        )
        .await;
        until_partial(&mut events).await;
        cancellation.cancel();
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            events.collect::<Vec<_>>(),
        )
        .await
        .unwrap();
        assert!(
            matches!(output.last(),Some(Ok(Event::Complete { finish_reason:Some(reason),.. })) if reason == "cancelled"),
            "{output:?}"
        );
        let saved = store
            .backend
            .load(1001, &Cancellation::new())
            .await
            .unwrap();
        assert_eq!(saved["status"], "stopped");
        check_attempt(&saved);
        assert_eq!(log.dropped.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn cancelling_parent_keeps_actual_child_partial_and_consumption() {
    for parent_react in [false, true] {
        for child_react in [false, true] {
            for mode in [false, true] {
                let store = Arc::new(Store::default());
                let child_store = Arc::new(Store::default());
                let log = Arc::new(Log::default());
                let cancellation = Cancellation::new();
                let mut events = start(
                    parent_react,
                    Some(child_react),
                    mode,
                    store.clone(),
                    child_store.clone(),
                    log.clone(),
                    cancellation.clone(),
                )
                .await;
                if mode {
                    until_partial(&mut events).await;
                } else {
                    until_pending(&mut events, &log).await;
                }
                cancellation.cancel();
                let output = tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    events.collect::<Vec<_>>(),
                )
                .await
                .unwrap();
                assert!(
                    matches!(output.last(),Some(Ok(Event::Complete { finish_reason:Some(reason),.. })) if reason == "cancelled"),
                    "{output:?}"
                );
                let saved = store
                    .backend
                    .load(1001, &Cancellation::new())
                    .await
                    .unwrap();
                assert_eq!(saved["status"], "stopped");
                let node = saved["execution"]["attempts"][0]["nodes"]
                    .as_object()
                    .unwrap()
                    .values()
                    .next()
                    .unwrap();
                assert_eq!(
                    node["status"], "cancelled",
                    "parent={parent_react}, child={child_react}, mode={mode}: {saved}"
                );
                let native = &node["execution"]["attempts"][0]["agent"];
                check_attempt(native);
                assert_eq!(saved["usage"]["total"], 42);
                assert_eq!(log.parent.load(Ordering::SeqCst), 1);
                assert_eq!(log.model.load(Ordering::SeqCst), 1);
                assert_eq!(log.dropped.load(Ordering::SeqCst), 1);
                assert_eq!(child_store.loads.load(Ordering::SeqCst), 0);
                assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
                assert_eq!(output.iter().filter(|event| matches!(event,Ok(Event::Complete { finish_reason:Some(reason),..}) if reason == "cancelled")).count(), 2);
                if !mode {
                    assert!(!output.iter().any(|e| matches!(e, Ok(Event::Delta { .. }))));
                }
            }
        }
    }
}

struct Opaque(Arc<Log>, bool);
struct OpaqueStream {
    log: Arc<Log>,
    first: bool,
    chatty: bool,
}
impl Stream for OpaqueStream {
    type Item = Result<Event, Error>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.first || self.chatty {
            self.first = false;
            Poll::Ready(Some(Ok(Event::Delta {
                message: text("opaque output"),
            })))
        } else {
            Poll::Pending
        }
    }
}
impl Drop for OpaqueStream {
    fn drop(&mut self) {
        self.log.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
impl Agent for Opaque {
    type Stream = OpaqueStream;
    async fn run(&self, _: Input, _: bool, _: Cancellation) -> Result<Self::Stream, Error> {
        self.0.model.fetch_add(1, Ordering::SeqCst);
        Ok(OpaqueStream {
            log: self.0.clone(),
            first: true,
            chatty: self.1,
        })
    }
    async fn reply(&self, _: Input, _: Cancellation) -> Result<(), Error> {
        Err(Error::new("NOT_FOUND", "no questions"))
    }
}
#[tokio::test]
async fn cancellation_does_not_wait_for_uncooperative_custom_children_or_invent_their_scene() {
    for react in [false, true] {
        for chatty in [false, true] {
            let store = Arc::new(Store::default());
            let log = Arc::new(Log::default());
            let c = Cancellation::new();
            let child = SubAgent::new("child", Opaque(log.clone(), chatty)).unwrap();
            let mut events: Pin<Box<dyn Stream<Item = Result<Event, Error>>>> = if react {
                Box::pin(
                    agent::react()
                        .model(Parent(log.clone()))
                        .subagents(vec![child])
                        .with_id_fn(ids)
                        .checkpoint(store.clone())
                        .build()
                        .unwrap()
                        .run(input(), true, c.clone())
                        .await
                        .unwrap(),
                )
            } else {
                Box::pin(
                    agent::chat()
                        .model(Parent(log.clone()))
                        .subagents(vec![child])
                        .with_id_fn(ids)
                        .checkpoint(store.clone())
                        .build()
                        .unwrap()
                        .run(input(), true, c.clone())
                        .await
                        .unwrap(),
                )
            };
            loop {
                if matches!(events.next().await.unwrap().unwrap(), Event::Delta { .. }) {
                    break;
                }
            }
            c.cancel();
            let tail = tokio::time::timeout(
                std::time::Duration::from_secs(3),
                events.collect::<Vec<_>>(),
            )
            .await
            .unwrap();
            assert!(
                matches!(tail.last(),Some(Ok(Event::Complete { finish_reason:Some(reason),.. })) if reason=="cancelled"),
                "{tail:?}"
            );
            assert!(
                !tail
                    .iter()
                    .any(|event| matches!(event, Ok(Event::Delta { .. })))
            );
            let saved = store
                .backend
                .load(1001, &Cancellation::new())
                .await
                .unwrap();
            let node = saved["execution"]["attempts"][0]["nodes"]
                .as_object()
                .unwrap()
                .values()
                .next()
                .unwrap();
            assert_eq!(node["status"], "cancelled");
            assert!(node["execution"]["attempts"][0]["agent"].is_null());
            assert_eq!(saved["usage"]["total"], 2);
            assert_eq!(log.model.load(Ordering::SeqCst), 1);
            assert_eq!(log.dropped.load(Ordering::SeqCst), 1);
        }
    }
}

struct PlannedParent(Arc<Log>);
impl Chat for PlannedParent {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let task = request
            .messages
            .as_slice()
            .iter()
            .rev()
            .find_map(|message| message.values.get("plan_node").and_then(Value::as_str));
        if task == Some("a") {
            return Parent(self.0.clone()).stream(request).await;
        }
        let (message, tokens) = if task == Some("b") {
            (text("sibling result"), 5)
        } else if request.response_format.is_some() {
            let mut message = Message::new(Role::Assistant);
            message.values = json!({"version":1,"nodes":{
                "a":{"name":"research","objective":"delegate work"},
                "b":{"name":"research","objective":"independent work"}
            },"edges":[]})
            .as_object()
            .unwrap()
            .clone();
            (message, 3)
        } else {
            assert!(
                !request
                    .messages
                    .as_slice()
                    .iter()
                    .any(|m| m.values.contains_key("plan_complete")),
                "cancelled task must not start final inference"
            );
            let mut message = Message::new(Role::Assistant);
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"name":"plan","call_id":"plan","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
            (message, 2)
        };
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                total: Some(tokens),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}
#[tokio::test]
async fn cancelling_planned_delegation_keeps_child_facts_on_current_task_attempt() {
    for child_react in [false, true] {
        for mode in [false, true] {
            for capacity in [1, 2] {
                let store = Arc::new(Store::default());
                let child_store = Arc::new(Store::default());
                let log = Arc::new(Log::default());
                let c = Cancellation::new();
                let child = if child_react {
                    SubAgent::new(
                        "child",
                        agent::react()
                            .model(Model(log.clone()))
                            .checkpoint(child_store.clone())
                            .build()
                            .unwrap(),
                    )
                    .unwrap()
                } else {
                    SubAgent::new(
                        "child",
                        agent::chat()
                            .model(Model(log.clone()))
                            .checkpoint(child_store.clone())
                            .build()
                            .unwrap(),
                    )
                    .unwrap()
                };
                let agent = agent::react()
                    .model(PlannedParent(log.clone()))
                    .subagents(vec![child])
                    .checkpoint(store.clone())
                    .with_id_fn(ids)
                    .with_max_iters(6)
                    .with_max_concurrency(capacity)
                    .build()
                    .unwrap();
                let mut events: Pin<Box<dyn Stream<Item = Result<Event, Error>>>> =
                    Box::pin(agent.run(input(), mode, c.clone()).await.unwrap());
                if mode {
                    until_partial(&mut events).await;
                } else {
                    until_pending(&mut events, &log).await;
                }
                c.cancel();
                let tail = tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    events.collect::<Vec<_>>(),
                )
                .await
                .unwrap();
                assert!(
                    matches!(tail.last(),Some(Ok(Event::Complete { finish_reason:Some(reason),.. })) if reason=="cancelled"),
                    "{tail:?}"
                );
                let saved = store
                    .backend
                    .load(1001, &Cancellation::new())
                    .await
                    .unwrap();
                assert_eq!(saved["status"], "stopped");
                let task = &saved["nodes"]["a"];
                assert_eq!(task["status"], "cancelled");
                assert_eq!(task["execution"]["attempts"].as_array().unwrap().len(), 1);
                let current = &task["execution"]["attempts"][0];
                let node = current["nodes"]
                    .as_object()
                    .unwrap()
                    .values()
                    .next()
                    .unwrap();
                assert_eq!(node["status"], "cancelled");
                check_attempt(&node["execution"]["attempts"][0]["agent"]);
                assert_eq!(task["usage"]["total"], 42);
                let total = saved["nodes"]
                    .as_object()
                    .unwrap()
                    .values()
                    .map(|node| node["usage"]["total"].as_u64().unwrap_or(0))
                    .sum::<u64>();
                assert_eq!(saved["usage"]["total"], 5 + total);
                assert_eq!(log.parent.load(Ordering::SeqCst), 1);
                assert_eq!(log.model.load(Ordering::SeqCst), 1);
                assert_eq!(log.dropped.load(Ordering::SeqCst), 1);
                assert_eq!(child_store.loads.load(Ordering::SeqCst), 0);
                assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
            }
        }
    }
}

fn workflow(
    react: bool,
    child_react: Option<bool>,
    depth: usize,
    log: Arc<Log>,
    child_store: Arc<Store>,
) -> halo::runtime::workflow::CompiledWorkflow {
    use halo::runtime::workflow::WorkflowBuilder;
    let mut builder = WorkflowBuilder::new("cancel-owner");
    if let Some(child_react) = child_react {
        let child = if child_react {
            SubAgent::new(
                "child",
                agent::react()
                    .model(Model(log.clone()))
                    .checkpoint(child_store.clone())
                    .build()
                    .unwrap(),
            )
            .unwrap()
        } else {
            SubAgent::new(
                "child",
                agent::chat()
                    .model(Model(log.clone()))
                    .checkpoint(child_store.clone())
                    .build()
                    .unwrap(),
            )
            .unwrap()
        };
        if react {
            builder
                .add_agent_node(
                    "agent",
                    agent::react()
                        .model(Parent(log))
                        .subagents(vec![child])
                        .with_id_fn(ids)
                        .checkpoint(child_store)
                        .build()
                        .unwrap(),
                )
                .unwrap();
        } else {
            builder
                .add_agent_node(
                    "agent",
                    agent::chat()
                        .model(Parent(log))
                        .subagents(vec![child])
                        .with_id_fn(ids)
                        .checkpoint(child_store)
                        .build()
                        .unwrap(),
                )
                .unwrap();
        }
    } else if react {
        builder
            .add_agent_node(
                "agent",
                agent::react()
                    .model(Model(log))
                    .checkpoint(child_store)
                    .build()
                    .unwrap(),
            )
            .unwrap();
    } else {
        builder
            .add_agent_node(
                "agent",
                agent::chat()
                    .model(Model(log))
                    .checkpoint(child_store)
                    .build()
                    .unwrap(),
            )
            .unwrap();
    }
    finish_workflow(builder, depth)
}

fn finish_workflow(
    mut builder: halo::runtime::workflow::WorkflowBuilder,
    depth: usize,
) -> halo::runtime::workflow::CompiledWorkflow {
    use halo::runtime::workflow::{
        WorkflowBuilder,
        graph::{END, START},
    };
    builder
        .add_node("sibling", halo::node::Echo)
        .unwrap();
    builder.add_node("merge", halo::node::Echo).unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge(START, "sibling").unwrap();
    builder.add_join(["agent", "sibling"], "merge").unwrap();
    builder.add_edge("merge", END).unwrap();
    let mut graph = builder.compile().unwrap();
    for index in 0..depth {
        let mut parent = WorkflowBuilder::new(format!("cancel-container-{index}"));
        parent.add_graph_node("child", &graph).unwrap();
        parent.add_edge(START, "child").unwrap();
        parent.add_edge("child", END).unwrap();
        graph = parent.compile().unwrap();
    }
    graph
}

fn workflow_input() -> halo::Messages {
    let mut message = Message::new(Role::User);
    message.metadata = input().metadata;
    halo::Messages::new([message])
}

fn workflow_attempt<'a>(scene: &'a Value, node: &str) -> &'a Value {
    &scene["nodes"][node]["executions"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()["attempts"][0]
}

#[tokio::test]
async fn workflow_cancellation_closes_actual_nested_agent_before_checkpoint() {
    use halo::runtime::workflow::Runtime;
    for react in [false, true] {
        for child in [None, Some(false), Some(true)] {
            for depth in [0, 2] {
                let log = Arc::new(Log::default());
                let store = Arc::new(Store::default());
                let child_store = Arc::new(Store::default());
                let graph = workflow(react, child, depth, log.clone(), child_store.clone());
                let runtime = Runtime::new(1001, graph.clone())
                    .unwrap()
                    .with_checkpoint(store.clone());
                let c = Cancellation::new();
                let mut output = Box::pin(
                    runtime
                        .run(workflow_input(), Arc::new(ids), c.clone())
                        .await
                        .unwrap(),
                );
                until_partial(&mut output).await;
                until_pending(&mut output, &log).await;
                assert_eq!(store.saves.load(Ordering::SeqCst), 0);
                c.cancel();
                let tail = tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    output.collect::<Vec<_>>(),
                )
                .await
                .unwrap();
                assert!(
                    matches!(tail.last(), Some(Err(error)) if error.code == "CANCELLED"),
                    "{tail:?}"
                );
                let saved = runtime.snapshot().unwrap();
                assert_eq!(saved["status"], "cancelled");
                assert_eq!(
                    saved["usage"]["total"],
                    if child.is_some() { 42 } else { 40 }
                );
                assert_eq!(store.saves.load(Ordering::SeqCst), 1);
                assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
                let mut leaf = &saved;
                for _ in 0..depth {
                    assert_eq!(workflow_attempt(leaf, "child")["status"], "cancelled");
                    leaf = &workflow_attempt(leaf, "child")["graph"];
                    assert_eq!(leaf["status"], "cancelled");
                }
                assert!(
                    leaf["nodes"]["merge"]["executions"]
                        .as_object()
                        .unwrap()
                        .is_empty()
                );
                let attempt = workflow_attempt(leaf, "agent");
                assert_eq!(attempt["status"], "cancelled");
                assert_eq!(leaf["nodes"]["sibling"]["status"], "succeeded");
                let native = &attempt["agent"];
                if child.is_some() {
                    assert_eq!(native["status"], "stopped");
                    let node = native["execution"]["attempts"][0]["nodes"]
                        .as_object()
                        .unwrap()
                        .values()
                        .next()
                        .unwrap();
                    assert_eq!(node["status"], "cancelled");
                    check_attempt(&node["execution"]["attempts"][0]["agent"]);
                } else {
                    check_attempt(native);
                }
                assert_eq!(log.model.load(Ordering::SeqCst), 1);
                assert_eq!(
                    log.parent.load(Ordering::SeqCst),
                    usize::from(child.is_some())
                );
                assert_eq!(log.dropped.load(Ordering::SeqCst), 1);
                assert_eq!(child_store.loads.load(Ordering::SeqCst), 0);
                assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
                let expected = if child.is_some() { 2 } else { 1 };
                assert_eq!(
                    tail.iter()
                        .filter(|event| matches!(event, Ok(Event::Complete {
                    finish_reason: Some(reason), ..}) if reason == "cancelled"))
                        .count(),
                    expected
                );

                // Terminal reads retain the same complete scene and perform
                // no model work, ID allocation or second checkpoint write.
                let restored = Runtime::from_snapshot(
                    1001,
                    graph,
                    serde_json::from_str(&saved.to_string()).unwrap(),
                )
                .unwrap()
                .with_checkpoint(store.clone());
                let events = restored
                    .run(
                        workflow_input(),
                        Arc::new(|| panic!("terminal ID")),
                        Cancellation::new(),
                    )
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                assert!(matches!(events.last(), Some(Err(error)) if error.code == "CANCELLED"));
                assert_eq!(
                    events
                        .iter()
                        .filter(|event| matches!(event, Ok(Event::Complete {
                        finish_reason: Some(reason), ..}) if reason == "cancelled"))
                        .count(),
                    expected,
                );
                assert_eq!(restored.snapshot().unwrap(), saved);
                assert_eq!(store.saves.load(Ordering::SeqCst), 1);
                assert_eq!(log.model.load(Ordering::SeqCst), 1);
            }
        }
    }
}

#[tokio::test]
async fn dropping_workflow_releases_nested_agents_without_checkpoint_closeout() {
    use halo::runtime::workflow::Runtime;
    for react in [false, true] {
        for child in [None, Some(false), Some(true)] {
            let log = Arc::new(Log::default());
            let store = Arc::new(Store::default());
            let child_store = Arc::new(Store::default());
            let runtime = Runtime::new(
                1001,
                workflow(react, child, 2, log.clone(), child_store.clone()),
            )
            .unwrap()
            .with_checkpoint(store.clone());
            let mut events = Box::pin(
                runtime
                    .run(workflow_input(), Arc::new(ids), Cancellation::new())
                    .await
                    .unwrap(),
            );
            until_partial(&mut events).await;
            drop(events);
            tokio::task::yield_now().await;
            assert_eq!(log.dropped.load(Ordering::SeqCst), 1);
            assert_eq!(log.model.load(Ordering::SeqCst), 1);
            assert_eq!(store.saves.load(Ordering::SeqCst), 0);
            assert!(
                store
                    .backend
                    .load(1001, &Cancellation::new())
                    .await
                    .is_err()
            );
            assert_eq!(child_store.loads.load(Ordering::SeqCst), 0);
            assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
            assert_ne!(runtime.snapshot().unwrap()["status"], "succeeded");
        }
    }
}

struct PlannedChild(Arc<Log>);

struct OpaqueCancelled;
impl Agent for OpaqueCancelled {
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>>>>;
    async fn run(&self, _: Input, _: bool, _: Cancellation) -> Result<Self::Stream, Error> {
        Ok(Box::pin(
            stream::iter(vec![Ok(Event::Complete {
                message: text("custom cancelled"),
                usage: None,
                finish_reason: Some("cancelled".into()),
            })])
            .chain(stream::pending()),
        ))
    }
    async fn reply(&self, _: Input, _: Cancellation) -> Result<(), Error> {
        Err(Error::new("NOT_FOUND", "no questions"))
    }
}

#[tokio::test]
async fn opaque_agents_cancelled_complete_still_closes_without_waiting_for_eof() {
    use halo::runtime::workflow::{
        Runtime, WorkflowBuilder,
        graph::{END, START},
    };
    let mut builder = WorkflowBuilder::new("custom-cancelled");
    builder.add_agent_node("agent", OpaqueCancelled).unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let store = Arc::new(Store::default());
    let runtime = Runtime::new(1001, builder.compile().unwrap())
        .unwrap()
        .with_checkpoint(store.clone());
    let events = runtime
        .run(workflow_input(), Arc::new(ids), Cancellation::new())
        .await
        .unwrap();
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        events.collect::<Vec<_>>(),
    )
    .await
    .unwrap();
    assert!(matches!(output.last(), Some(Err(error)) if error.code == "CANCELLED"));
    let saved = runtime.snapshot().unwrap();
    assert_eq!(saved["status"], "cancelled");
    assert!(workflow_attempt(&saved, "agent")["agent"].is_null());
    assert_eq!(store.saves.load(Ordering::SeqCst), 1);
    assert_eq!(
        output
            .iter()
            .filter(|event| matches!(event,
        Ok(Event::Complete { finish_reason: Some(reason), ..}) if reason == "cancelled"))
            .count(),
        1
    );
}

impl Chat for PlannedChild {
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let task = request
            .messages
            .as_slice()
            .iter()
            .rev()
            .find_map(|message| message.values.get("plan_node").and_then(Value::as_str));
        if task == Some("a") {
            return Ok(Box::pin(Model(self.0.clone()).stream(request).await?));
        }
        Ok(Box::pin(
            PlannedParent(self.0.clone()).stream(request).await?,
        ))
    }
}

#[tokio::test]
async fn cancellation_reaches_a_childs_own_plan_and_retains_each_native_owner() {
    for parent in 0..3 {
        for mode in [false, true] {
            for capacity in [1, 2] {
                let log = Arc::new(Log::default());
                let store = Arc::new(Store::default());
                let child_store = Arc::new(Store::default());
                let child = SubAgent::new(
                    "child",
                    agent::react()
                        .model(PlannedChild(log.clone()))
                        .with_id_fn(ids)
                        .with_max_iters(6)
                        .with_max_concurrency(capacity)
                        .checkpoint(child_store.clone())
                        .build()
                        .unwrap(),
                )
                .unwrap();
                let c = Cancellation::new();
                let mut events: Pin<Box<dyn Stream<Item = Result<Event, Error>>>> = match parent {
                    0 => Box::pin(
                        agent::chat()
                            .model(Parent(log.clone()))
                            .subagents(vec![child])
                            .checkpoint(store.clone())
                            .with_id_fn(ids)
                            .build()
                            .unwrap()
                            .run(input(), mode, c.clone())
                            .await
                            .unwrap(),
                    ),
                    1 => Box::pin(
                        agent::react()
                            .model(Parent(log.clone()))
                            .subagents(vec![child])
                            .checkpoint(store.clone())
                            .with_id_fn(ids)
                            .build()
                            .unwrap()
                            .run(input(), mode, c.clone())
                            .await
                            .unwrap(),
                    ),
                    _ => Box::pin(
                        agent::react()
                            .model(PlannedParent(log.clone()))
                            .subagents(vec![child])
                            .checkpoint(store.clone())
                            .with_id_fn(ids)
                            .with_max_iters(6)
                            .with_max_concurrency(capacity)
                            .build()
                            .unwrap()
                            .run(input(), mode, c.clone())
                            .await
                            .unwrap(),
                    ),
                };
                if mode {
                    until_partial(&mut events).await;
                }
                until_pending(&mut events, &log).await;
                c.cancel();
                let tail = tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    events.collect::<Vec<_>>(),
                )
                .await
                .unwrap();
                assert!(
                    matches!(tail.last(), Some(Ok(Event::Complete {
                    finish_reason: Some(reason), ..})) if reason == "cancelled"),
                    "{tail:?}"
                );
                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(saved["status"], "stopped");
                let owner = if parent == 2 {
                    let node = &saved["nodes"]["a"];
                    assert_eq!(node["status"], "cancelled");
                    &node["execution"]["attempts"][0]
                } else {
                    &saved["execution"]["attempts"][0]
                };
                let child = owner["nodes"].as_object().unwrap().values().next().unwrap();
                assert_eq!(child["status"], "cancelled");
                let native = &child["execution"]["attempts"][0]["agent"];
                assert_eq!(native["status"], "stopped");
                let task = &native["nodes"]["a"];
                assert_eq!(task["status"], "cancelled");
                assert_eq!(task["execution"]["attempts"].as_array().unwrap().len(), 1);
                let model = &task["execution"]["attempts"][0]["steps"][0];
                assert_eq!(model["attempts"][0]["status"], "cancelled");
                assert_eq!(
                    model["attempts"][0]["message"]["content"],
                    serde_json::to_value(partial().content).unwrap()
                );
                assert_eq!(model["attempts"][0]["usage"]["total"], 40);
                let task_total = native["nodes"]
                    .as_object()
                    .unwrap()
                    .values()
                    .map(|node| node["usage"]["total"].as_u64().unwrap_or(0))
                    .sum::<u64>();
                assert_eq!(native["usage"]["total"], 5 + task_total);
                let child_total = native["usage"]["total"].as_u64().unwrap();
                if parent < 2 {
                    assert_eq!(saved["usage"]["total"], child_total + 2);
                } else {
                    assert_eq!(saved["nodes"]["a"]["usage"]["total"], child_total + 2);
                    let parent_total = saved["nodes"]
                        .as_object()
                        .unwrap()
                        .values()
                        .map(|node| node["usage"]["total"].as_u64().unwrap_or(0))
                        .sum::<u64>();
                    assert_eq!(saved["usage"]["total"], 5 + parent_total);
                }
                assert_eq!(log.parent.load(Ordering::SeqCst), 1);
                assert_eq!(log.model.load(Ordering::SeqCst), 1);
                assert_eq!(log.dropped.load(Ordering::SeqCst), 1);
                assert_eq!(child_store.loads.load(Ordering::SeqCst), 0);
                assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PlannedEmbedding {
    Plan,
    PlannedDelegation(bool),
    DelegatedPlan(bool),
    BothPlans,
}

fn planned_workflow(
    kind: PlannedEmbedding,
    capacity: usize,
    depth: usize,
    log: Arc<Log>,
    store: Arc<Store>,
) -> halo::runtime::workflow::CompiledWorkflow {
    use halo::runtime::workflow::WorkflowBuilder;
    let mut builder = WorkflowBuilder::new("planned-cancel-owner");
    if matches!(kind, PlannedEmbedding::Plan) {
        builder
            .add_agent_node(
                "agent",
                agent::react()
                    .model(PlannedChild(log))
                    .with_id_fn(ids)
                    .with_max_iters(6)
                    .with_max_concurrency(capacity)
                    .checkpoint(store)
                    .build()
                    .unwrap(),
            )
            .unwrap();
    } else {
        let child = match kind {
            PlannedEmbedding::PlannedDelegation(false) => SubAgent::new(
                "child",
                agent::chat()
                    .model(Model(log.clone()))
                    .checkpoint(store.clone())
                    .build()
                    .unwrap(),
            )
            .unwrap(),
            PlannedEmbedding::PlannedDelegation(true) => SubAgent::new(
                "child",
                agent::react()
                    .model(Model(log.clone()))
                    .checkpoint(store.clone())
                    .build()
                    .unwrap(),
            )
            .unwrap(),
            _ => SubAgent::new(
                "child",
                agent::react()
                    .model(PlannedChild(log.clone()))
                    .checkpoint(store.clone())
                    .with_id_fn(ids)
                    .with_max_iters(6)
                    .with_max_concurrency(capacity)
                    .build()
                    .unwrap(),
            )
            .unwrap(),
        };
        match kind {
            PlannedEmbedding::DelegatedPlan(false) => {
                builder
                    .add_agent_node(
                        "agent",
                        agent::chat()
                            .model(Parent(log))
                            .subagents(vec![child])
                            .with_id_fn(ids)
                            .checkpoint(store)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();
            }
            PlannedEmbedding::DelegatedPlan(true) => {
                builder
                    .add_agent_node(
                        "agent",
                        agent::react()
                            .model(Parent(log))
                            .subagents(vec![child])
                            .with_id_fn(ids)
                            .checkpoint(store)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();
            }
            _ => {
                builder
                    .add_agent_node(
                        "agent",
                        agent::react()
                            .model(PlannedParent(log))
                            .subagents(vec![child])
                            .with_id_fn(ids)
                            .checkpoint(store)
                            .with_max_iters(6)
                            .with_max_concurrency(capacity)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();
            }
        }
    }
    finish_workflow(builder, depth)
}

fn cancelled_task(scene: &Value) -> &Value {
    assert_eq!(
        scene["status"],
        "stopped",
        "native tasks: {:?}",
        scene["nodes"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, node)| (id, node["status"].as_str()))
            .collect::<Vec<_>>()
    );
    let node = &scene["nodes"]["a"];
    assert_eq!(node["status"], "cancelled");
    assert!(node["execution"]["exec_id"].as_i64().unwrap() > 0);
    assert_eq!(node["execution"]["attempts"].as_array().unwrap().len(), 1);
    let total = scene["nodes"]
        .as_object()
        .unwrap()
        .values()
        .map(|node| node["usage"]["total"].as_u64().unwrap_or(0))
        .sum::<u64>();
    assert_eq!(scene["usage"]["total"], 5 + total);
    &node["execution"]["attempts"][0]
}

fn check_embedded_plan(scene: &Value, kind: PlannedEmbedding) {
    assert_eq!(scene["status"], "stopped");
    let owner = match kind {
        PlannedEmbedding::Plan
        | PlannedEmbedding::PlannedDelegation(_)
        | PlannedEmbedding::BothPlans => cancelled_task(scene),
        PlannedEmbedding::DelegatedPlan(_) => &scene["execution"]["attempts"][0],
    };
    if matches!(kind, PlannedEmbedding::Plan) {
        let step = &owner["steps"][0];
        assert_eq!(step["attempts"][0]["usage"]["total"], 40);
        assert_eq!(step["attempts"][0]["status"], "cancelled");
        assert_eq!(
            step["attempts"][0]["message"]["content"],
            serde_json::to_value(partial().content).unwrap()
        );
        return;
    }
    let child = owner["nodes"].as_object().unwrap().values().next().unwrap();
    assert_eq!(child["status"], "cancelled");
    let native = &child["execution"]["attempts"][0]["agent"];
    if matches!(kind, PlannedEmbedding::PlannedDelegation(_)) {
        check_attempt(native);
    } else {
        let task = cancelled_task(native);
        let step = &task["steps"][0];
        assert_eq!(step["attempts"][0]["usage"]["total"], 40);
        assert_eq!(step["attempts"][0]["status"], "cancelled");
        assert_eq!(
            step["attempts"][0]["message"]["content"],
            serde_json::to_value(partial().content).unwrap()
        );
    }
    let child_total = native["usage"]["total"].as_u64().unwrap();
    match kind {
        PlannedEmbedding::DelegatedPlan(_) => assert_eq!(scene["usage"]["total"], 2 + child_total),
        _ => assert_eq!(scene["nodes"]["a"]["usage"]["total"], 2 + child_total),
    }
}

#[tokio::test]
async fn workflow_plans_and_child_plans_close_in_their_actual_nested_owners() {
    use halo::runtime::workflow::Runtime;
    let kinds = [
        PlannedEmbedding::Plan,
        PlannedEmbedding::PlannedDelegation(false),
        PlannedEmbedding::PlannedDelegation(true),
        PlannedEmbedding::DelegatedPlan(false),
        PlannedEmbedding::DelegatedPlan(true),
        PlannedEmbedding::BothPlans,
    ];
    for kind in kinds {
        for depth in [0, 2] {
            for capacity in [1, 2] {
                let log = Arc::new(Log::default());
                let store = Arc::new(Store::default());
                let child_store = Arc::new(Store::default());
                let graph =
                    planned_workflow(kind, capacity, depth, log.clone(), child_store.clone());
                let runtime = Runtime::new(1001, graph.clone())
                    .unwrap()
                    .with_checkpoint(store.clone());
                let c = Cancellation::new();
                let mut events = Box::pin(
                    runtime
                        .run(workflow_input(), Arc::new(ids), c.clone())
                        .await
                        .unwrap(),
                );
                until_partial(&mut events).await;
                until_pending(&mut events, &log).await;
                assert_eq!(store.saves.load(Ordering::SeqCst), 0);
                c.cancel();
                let output = tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    events.collect::<Vec<_>>(),
                )
                .await
                .expect("nested cancel must close");
                assert!(
                    matches!(output.last(), Some(Err(error)) if error.code == "CANCELLED"),
                    "{kind:?}/{capacity}/{depth}: {output:?}"
                );
                let saved = runtime.snapshot().unwrap();
                assert_eq!(saved["status"], "cancelled");
                let mut leaf = &saved;
                for _ in 0..depth {
                    leaf = &workflow_attempt(leaf, "child")["graph"];
                }
                let native = &workflow_attempt(leaf, "agent")["agent"];
                check_embedded_plan(native, kind);
                assert_eq!(saved["usage"], native["usage"]);
                assert_eq!(leaf["nodes"]["sibling"]["status"], "succeeded");
                assert!(
                    leaf["nodes"]["merge"]["executions"]
                        .as_object()
                        .unwrap()
                        .is_empty()
                );
                assert_eq!(store.saves.load(Ordering::SeqCst), 1);
                assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
                assert_eq!(child_store.loads.load(Ordering::SeqCst), 0);
                assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
                assert_eq!(log.model.load(Ordering::SeqCst), 1);
                assert_eq!(log.dropped.load(Ordering::SeqCst), 1);
                let count = output
                    .iter()
                    .filter(|event| {
                        matches!(event, Ok(Event::Complete {
                    finish_reason: Some(reason), ..}) if reason == "cancelled")
                    })
                    .count();
                assert!(count > 0);
                let restored = Runtime::from_snapshot(
                    1001,
                    graph,
                    serde_json::from_str(&saved.to_string()).unwrap(),
                )
                .unwrap()
                .with_checkpoint(store.clone());
                let output = restored
                    .run(
                        workflow_input(),
                        Arc::new(|| panic!("terminal ID allocation")),
                        Cancellation::new(),
                    )
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                assert!(matches!(output.last(), Some(Err(error)) if error.code == "CANCELLED"));
                assert_eq!(
                    output
                        .iter()
                        .filter(|event| matches!(event, Ok(Event::Complete {
                    finish_reason: Some(reason), ..}) if reason == "cancelled"))
                        .count(),
                    count
                );
                assert_eq!(restored.snapshot().unwrap(), saved);
                assert_eq!(store.saves.load(Ordering::SeqCst), 1);
                assert_eq!(log.model.load(Ordering::SeqCst), 1);
            }
        }
    }
}

struct FailingCloseout {
    inner: Store,
    fail: AtomicBool,
    after_write: bool,
    attempts: AtomicUsize,
}
fn storage_error() -> Error {
    Error::new("WRITE_FAILED", "checkpoint closeout failed").with_details("backend failure")
}
impl Checkpoint for FailingCloseout {
    async fn save(&self, id: i64, scene: Value, c: &Cancellation) -> Result<(), Error> {
        assert!(
            !c.is_cancelled(),
            "closeout uses its own cancellation scope"
        );
        self.attempts.fetch_add(1, Ordering::SeqCst);
        let fail = self.fail.swap(false, Ordering::SeqCst);
        if !fail || self.after_write {
            self.inner.save(id, scene, c).await?;
        }
        if fail { Err(storage_error()) } else { Ok(()) }
    }
    async fn load(&self, id: i64, c: &Cancellation) -> Result<Value, Error> {
        self.inner.load(id, c).await
    }
    async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, c).await
    }
}

#[tokio::test]
async fn failed_outer_checkpoint_keeps_cancelled_scene_and_never_reexecutes_work() {
    use halo::runtime::workflow::Runtime;
    for after_write in [false, true] {
        for depth in [0, 2] {
            let log = Arc::new(Log::default());
            let child_store = Arc::new(Store::default());
            let store = Arc::new(FailingCloseout {
                inner: Store::default(),
                fail: AtomicBool::new(true),
                after_write,
                attempts: AtomicUsize::new(0),
            });
            let graph = planned_workflow(
                PlannedEmbedding::BothPlans,
                2,
                depth,
                log.clone(),
                child_store.clone(),
            );
            let runtime = Runtime::new(1001, graph.clone())
                .unwrap()
                .with_checkpoint(store.clone());
            let previous = runtime.snapshot().unwrap();
            store
                .inner
                .backend
                .save(1001, previous.clone(), &Cancellation::new())
                .await
                .unwrap();
            let c = Cancellation::new();
            let mut events = Box::pin(
                runtime
                    .run(workflow_input(), Arc::new(ids), c.clone())
                    .await
                    .unwrap(),
            );
            until_partial(&mut events).await;
            until_pending(&mut events, &log).await;
            c.cancel();
            let output = tokio::time::timeout(
                std::time::Duration::from_secs(3),
                events.collect::<Vec<_>>(),
            )
            .await
            .unwrap();
            assert_eq!(output.last(), Some(&Err(storage_error())));
            assert!(
                output
                    .iter()
                    .any(|event| matches!(event, Ok(Event::Complete {
                finish_reason: Some(reason), ..}) if reason == "cancelled"))
            );
            let saved = runtime.snapshot().unwrap();
            assert_eq!(saved["status"], "cancelled");
            let mut leaf = &saved;
            for _ in 0..depth {
                leaf = &workflow_attempt(leaf, "child")["graph"];
            }
            check_embedded_plan(
                &workflow_attempt(leaf, "agent")["agent"],
                PlannedEmbedding::BothPlans,
            );
            assert_eq!(store.attempts.load(Ordering::SeqCst), 1);
            let persisted = store.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(
                persisted,
                if after_write { saved.clone() } else { previous }
            );
            // The caller explicitly retries storage using the existing API,
            // rather than re-running the cancelled model/Tool invocation.
            runtime
                .save_checkpoint(store.as_ref(), &Cancellation::new())
                .await
                .unwrap();
            assert_eq!(store.attempts.load(Ordering::SeqCst), 2);
            assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
            let restored = Runtime::from_snapshot(
                1001,
                graph,
                serde_json::from_str(&saved.to_string()).unwrap(),
            )
            .unwrap()
            .with_checkpoint(store.clone());
            let output = restored
                .run(
                    workflow_input(),
                    Arc::new(|| panic!("terminal ID")),
                    Cancellation::new(),
                )
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert!(matches!(output.last(), Some(Err(error)) if error.code == "CANCELLED"));
            assert_eq!(restored.snapshot().unwrap(), saved);
            assert_eq!(store.attempts.load(Ordering::SeqCst), 2);
            assert_eq!(log.parent.load(Ordering::SeqCst), 1);
            assert_eq!(log.model.load(Ordering::SeqCst), 1);
            assert_eq!(log.dropped.load(Ordering::SeqCst), 1);
            assert_eq!(child_store.loads.load(Ordering::SeqCst), 0);
            assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
        }
    }
}

#[derive(Default)]
struct DelayedCloseout {
    inner: Store,
    entered: AtomicBool,
    release: tokio::sync::Notify,
}
impl Checkpoint for DelayedCloseout {
    async fn save(&self, id: i64, scene: Value, c: &Cancellation) -> Result<(), Error> {
        assert!(!c.is_cancelled());
        self.entered.store(true, Ordering::SeqCst);
        self.release.notified().await;
        self.inner.save(id, scene, c).await
    }
    async fn load(&self, id: i64, c: &Cancellation) -> Result<Value, Error> {
        self.inner.load(id, c).await
    }
    async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, c).await
    }
}

#[tokio::test]
async fn pending_outer_closeout_is_owned_by_the_stream_without_hidden_writers() {
    use halo::runtime::workflow::Runtime;
    for drop_while_saving in [false, true] {
        let store = Arc::new(DelayedCloseout::default());
        let child_store = Arc::new(Store::default());
        let log = Arc::new(Log::default());
        let runtime = Runtime::new(
            1001,
            planned_workflow(
                PlannedEmbedding::BothPlans,
                2,
                2,
                log.clone(),
                child_store.clone(),
            ),
        )
        .unwrap()
        .with_checkpoint(store.clone());
        let c = Cancellation::new();
        let mut events = Box::pin(
            runtime
                .run(workflow_input(), Arc::new(ids), c.clone())
                .await
                .unwrap(),
        );
        until_partial(&mut events).await;
        until_pending(&mut events, &log).await;
        c.cancel();
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            loop {
                match poll!(events.next()) {
                    Poll::Ready(Some(Ok(Event::Complete { .. }))) => {}
                    Poll::Pending if store.entered.load(Ordering::SeqCst) => break,
                    Poll::Pending => tokio::task::yield_now().await,
                    other => panic!("{other:?}"),
                }
            }
        })
        .await
        .unwrap();
        assert!(
            runtime.snapshot().is_err(),
            "live stream still owns the scene"
        );
        assert_eq!(store.inner.saves.load(Ordering::SeqCst), 0);
        if drop_while_saving {
            drop(events);
            store.release.notify_one();
            tokio::task::yield_now().await;
            assert!(store.load(1001, &Cancellation::new()).await.is_err());
            assert_eq!(store.inner.saves.load(Ordering::SeqCst), 0);
        } else {
            store.release.notify_one();
            let output = tokio::time::timeout(
                std::time::Duration::from_secs(3),
                events.collect::<Vec<_>>(),
            )
            .await
            .unwrap();
            assert!(matches!(output.last(), Some(Err(error)) if error.code == "CANCELLED"));
            assert_eq!(store.inner.saves.load(Ordering::SeqCst), 1);
            assert_eq!(
                store.load(1001, &Cancellation::new()).await.unwrap(),
                runtime.snapshot().unwrap()
            );
        }
        let saved = runtime.snapshot().unwrap();
        assert_eq!(saved["status"], "cancelled");
        let leaf = &workflow_attempt(&workflow_attempt(&saved, "child")["graph"], "child")["graph"];
        check_embedded_plan(
            &workflow_attempt(leaf, "agent")["agent"],
            PlannedEmbedding::BothPlans,
        );
        assert_eq!(log.model.load(Ordering::SeqCst), 1);
        assert_eq!(log.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(child_store.loads.load(Ordering::SeqCst), 0);
        assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
    }
}
