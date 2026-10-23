//! Ordinary feedback follows actual planned/delegated containment.
use futures_util::{StreamExt, poll, stream};
use halo::{
    Cancellation, Error, Message, Messages,
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
    static NEXT: AtomicI64 = AtomicI64::new(41_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
fn no_new_id() -> i64 {
    panic!("restoration must retain existing task/child/question identities")
}
fn input() -> Input {
    serde_json::from_value(json!({"metadata":{"chat_id":1001,"session_id":2001,"agent_id":3001},"payload":[{"type":"text","data":{"value":"original query"}}]})).unwrap()
}
fn answer(id: i64, value: &str) -> Input {
    let mut input = input();
    input.metadata.insert("reply_id".into(), id.into());
    input.payload[0].data = json!({"value":value});
    input
}
fn call(message: &mut Message, name: &str, id: &str, arguments: Value) {
    message.content.push(Part {
        r#type: "tool_call".into(),
        data: json!({"call_id":id,"name":name,"arguments":arguments})
            .as_object()
            .unwrap()
            .clone(),
    });
}
#[derive(Clone)]
struct Model {
    plan: bool,
    parent: bool,
    requests: Arc<Mutex<Vec<Request>>>,
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        let task = request
            .messages
            .as_slice()
            .iter()
            .find_map(|m| m.values.get("plan_node").and_then(Value::as_str));
        let complete = request
            .messages
            .as_slice()
            .iter()
            .any(|m| m.values.contains_key("plan_complete"));
        let selected = request.messages.as_slice().iter().any(|m| {
            m.metadata.get("name").and_then(Value::as_str)
                == Some(if self.parent {
                    "agent_spawn"
                } else {
                    "collect"
                })
        });
        if self.plan && task.is_none() && !complete {
            if !request.messages.as_slice().iter().any(|message| {
                message.role == Role::System
                    && message.content.iter().any(|part| {
                        part.data
                            .get("value")
                            .and_then(Value::as_str)
                            .is_some_and(|text| {
                                text.starts_with("Return only a complete JSON Agent plan")
                            })
                    })
            }) {
                call(&mut message, "plan", "planning", json!({}));
            } else {
                message.values = json!({"version":1,"nodes":{
                    "a":{"name":"first","objective":"perform first task"},
                    "b":{"name":"second","objective":"perform second task"}
                },"edges":[]})
                .as_object()
                .unwrap()
                .clone();
            }
        } else if complete || selected {
            message.values.insert("answer".into(), json!("done"));
        } else if self.parent {
            for item in task
                .map(|task| vec![task])
                .unwrap_or_else(|| vec!["a", "b"])
            {
                call(
                    &mut message,
                    "agent_spawn",
                    item,
                    json!({"name":"worker","input":{"payload":[{"type":"text","data":{"value":item}}]}}),
                );
            }
        } else {
            let item = request
                .messages
                .as_slice()
                .iter()
                .filter(|m| m.role == Role::User)
                .flat_map(|m| &m.content)
                .find_map(|p| p.data.get("value").and_then(Value::as_str))
                .unwrap();
            call(
                &mut message,
                "collect",
                "same_call",
                json!({"item":format!("{item}:{}",task.unwrap_or("direct"))}),
            );
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
struct Log(Mutex<BTreeMap<String, Messages>>);
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
        let mut question = Message::new(Role::Assistant);
        question.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"请补充反馈"}).as_object().unwrap().clone(),
        });
        let feedback = wait::ask(question, self.count, cancellation).await?;
        let item = call.arguments["item"].as_str().unwrap();
        assert!(
            self.log
                .0
                .lock()
                .unwrap()
                .insert(item.into(), feedback)
                .is_none(),
            "successful sibling must not repeat"
        );
        Ok(Message::function(json!({"item":item})))
    }
}
fn children(
    plan: bool,
    react: bool,
    count: usize,
    log: Arc<Log>,
    requests: Arc<Mutex<Vec<Request>>>,
    own: Arc<core::checkpoint::InMemory>,
) -> Vec<SubAgent> {
    let model = Model {
        plan,
        parent: false,
        requests,
    };
    let tool=Collect{log,count,definition:ToolDefinition{name:"collect".into(),description:"collect input".into(),parameters:json!({"type":"object","properties":{"item":{"type":"string"}},"required":["item"],"additionalProperties":false}).as_object().unwrap().clone()}};
    vec![if react {
        SubAgent::new(
            "worker",
            agent::react()
                .model(model)
                .toolkit(Toolkit::new().tools(tool))
                .checkpoint(own)
                .with_max_concurrency(1)
                .build()
                .unwrap(),
        )
        .unwrap()
    } else {
        SubAgent::new(
            "worker",
            agent::chat()
                .model(model)
                .toolkit(Toolkit::new().tools(tool))
                .checkpoint(own)
                .build()
                .unwrap(),
        )
        .unwrap()
    }]
}
async fn question<S: futures_core::Stream<Item = Result<Event, Error>>>(
    events: &mut Pin<Box<S>>,
) -> i64 {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            if let Event::Complete { message, .. } = events.next().await.expect("question").unwrap()
                && let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64)
            {
                return id;
            }
        }
    })
    .await
    .expect("nested questions must release real admission")
}
async fn finish<S: futures_core::Stream<Item = Result<Event, Error>>>(
    events: &mut Pin<Box<S>>,
) -> Vec<Event> {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        let mut output = vec![];
        while let Some(event) = events.next().await {
            output.push(event.unwrap());
        }
        output
    })
    .await
    .expect("nested feedback must continue")
}

async fn tool_result<S: futures_core::Stream<Item = Result<Event, Error>>>(
    events: &mut Pin<Box<S>>,
) -> Event {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            let event = events.next().await.expect("Tool result").unwrap();
            if matches!(&event, Event::Complete { message, .. }
                if message.values.contains_key("item"))
            {
                return event;
            }
        }
    })
    .await
    .expect("completed feedback must resume its exact Tool")
}
fn native_step<'a>(scene: &'a Value, position: &Value) -> &'a Value {
    let mut owner = scene;
    for coordinate in position["nodes"].as_array().unwrap() {
        let key = coordinate["node_id"].as_str().unwrap();
        let node = owner
            .get("nodes")
            .and_then(|nodes| nodes.get(key))
            .or_else(|| {
                owner["execution"]["attempts"]
                    .as_array()
                    .and_then(|attempts| attempts.last())
                    .and_then(|attempt| attempt["nodes"].get(key))
            })
            .unwrap();
        assert_eq!(node["execution"]["exec_id"], coordinate["exec_id"]);
        let attempt = node["execution"]["attempts"]
            .as_array()
            .unwrap()
            .last()
            .unwrap();
        assert_eq!(attempt["idx"], coordinate["idx"]);
        owner = attempt
            .get("agent")
            .filter(|scene| scene.is_object())
            .unwrap_or(attempt);
    }
    owner["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["id"] == position["step"])
        .unwrap()
}
fn record<'a>(scene: &'a Value, position: &Value, id: i64) -> &'a Value {
    &native_step(scene, position)["waits"][id.to_string()]
}

async fn settled<S: futures_core::Stream<Item = Result<Event, Error>>>(
    events: &mut Pin<Box<S>>,
    store: &core::checkpoint::InMemory,
    position: &Value,
    id: i64,
    count: usize,
    output: &mut Vec<Event>,
) -> Value {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            match poll!(events.next()) {
                std::task::Poll::Ready(Some(Ok(event))) => output.push(event),
                std::task::Poll::Ready(Some(Err(error))) => panic!("{error:?}"),
                _ => {}
            }
            let scene = store.load(1001, &Cancellation::new()).await.unwrap();
            if record(&scene, position, id)["feedback"]
                .as_array()
                .unwrap()
                .len()
                == count
            {
                return scene;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("all containing native scenes must commit partial feedback")
}
async fn collect<A: Agent>(
    agent: &A,
    events: &mut Pin<Box<stream::Fuse<A::Stream>>>,
    store: &core::checkpoint::InMemory,
    ids: &[i64],
    original: &Value,
    options: (usize, bool),
    log: &Log,
) {
    let (count, mode) = options;
    let mut output = vec![];
    for (index, id) in ids.iter().enumerate() {
        if index + 1 < ids.len() {
            assert_eq!(
                agent
                    .reply(answer(ids[index + 1], "early"), Cancellation::new())
                    .await
                    .unwrap_err()
                    .code,
                "CONFLICT"
            );
        }
        for part in 0..count {
            agent
                .reply(
                    answer(*id, &format!("feedback-{index}-{part}")),
                    Cancellation::new(),
                )
                .await
                .unwrap_or_else(|error| {
                    panic!("part {part} question {index} ids {ids:?}: {error:?}")
                });
            agent
                .reply(
                    answer(*id, &format!("feedback-{index}-{part}")),
                    Cancellation::new(),
                )
                .await
                .unwrap();
            if part + 1 == count {
                output.push(tool_result(events).await);
            }
            loop {
                match poll!(events.next()) {
                    std::task::Poll::Ready(Some(Ok(event))) => output.push(event),
                    std::task::Poll::Ready(Some(Err(error))) => panic!("{error:?}"),
                    std::task::Poll::Pending | std::task::Poll::Ready(None) => break,
                }
            }
            let saved = settled(
                events,
                store,
                &original["waits"][index],
                *id,
                part + 1,
                &mut output,
            )
            .await;
            let r = record(&saved, &original["waits"][index], *id);
            assert_eq!(r["feedback"].as_array().unwrap().len(), part + 1);
            assert_eq!(
                r["status"],
                if part + 1 < count {
                    "waiting"
                } else {
                    "resolved"
                }
            );
        }
    }
    output.extend(finish(events).await);
    if !mode {
        assert!(
            !output
                .iter()
                .any(|event| matches!(event, Event::Delta { .. }))
        );
    }
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(saved["status"], "succeeded");
    assert_eq!(saved["waits"], json!([]));
    assert_eq!(saved["query"], original["query"]);
    assert_eq!(log.0.lock().unwrap().len(), ids.len());
    for (index, id) in ids.iter().enumerate() {
        assert_eq!(
            record(&saved, &original["waits"][index], *id)["status"],
            "resolved"
        );
    }
}

#[tokio::test]
async fn planned_delegations_and_child_plans_collect_in_native_fifo() {
    for (parent_plan, child_plan, child_react) in [
        (true, false, false),
        (true, false, true),
        (false, true, true),
        (true, true, true),
    ] {
        for mode in [false, true] {
            for capacity in [1, 2] {
                for count in [1, 3] {
                    let store = Arc::new(core::checkpoint::InMemory::new());
                    let own = Arc::new(core::checkpoint::InMemory::new());
                    let log = Arc::new(Log::default());
                    let parent_requests = Arc::new(Mutex::new(vec![]));
                    let child_requests = Arc::new(Mutex::new(vec![]));
                    let agent = agent::react()
                        .model(Model {
                            plan: parent_plan,
                            parent: true,
                            requests: parent_requests.clone(),
                        })
                        .subagents(children(
                            child_plan,
                            child_react,
                            count,
                            log.clone(),
                            child_requests.clone(),
                            own.clone(),
                        ))
                        .checkpoint(store.clone())
                        .with_id_fn(next_id)
                        .with_max_concurrency(capacity)
                        .with_max_iters(40)
                        .build()
                        .unwrap();
                    let mut events = Box::pin(
                        agent
                            .run(input(), mode, Cancellation::new())
                            .await
                            .unwrap()
                            .fuse(),
                    );
                    let mut ids = vec![];
                    for _ in 0..if child_plan { 4 } else { 2 } {
                        ids.push(question(&mut events).await);
                    }
                    let original = store.load(1001, &Cancellation::new()).await.unwrap();
                    assert_eq!(original["waits"].as_array().unwrap().len(), ids.len());
                    assert!(
                        original["waits"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .all(|wait| wait["nodes"].as_array().unwrap().len()
                                == 1 + usize::from(parent_plan) + usize::from(child_plan))
                    );
                    collect(
                        &agent,
                        &mut events,
                        &store,
                        &ids,
                        &original,
                        (count, mode),
                        &log,
                    )
                    .await;
                    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                    assert_eq!(
                        saved["usage"]["total"].as_u64(),
                        Some(
                            13 * (parent_requests.lock().unwrap().len()
                                + child_requests.lock().unwrap().len())
                                as u64
                        )
                    );
                    assert_eq!(
                        own.load(1001, &Cancellation::new()).await.unwrap_err().code,
                        "NOT_FOUND"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn nested_partial_feedback_restores_only_from_outer_checkpoint() {
    for (parent_plan, child_plan) in [(true, false), (false, true), (true, true)] {
        for mode in [false, true] {
            let store = Arc::new(core::checkpoint::InMemory::new());
            let own = Arc::new(core::checkpoint::InMemory::new());
            let log = Arc::new(Log::default());
            let first = agent::react()
                .model(Model {
                    plan: parent_plan,
                    parent: true,
                    requests: Default::default(),
                })
                .subagents(children(
                    child_plan,
                    true,
                    3,
                    log.clone(),
                    Default::default(),
                    own.clone(),
                ))
                .checkpoint(store.clone())
                .with_id_fn(next_id)
                .with_max_concurrency(1)
                .with_max_iters(40)
                .build()
                .unwrap();
            let mut events = Box::pin(
                first
                    .run(input(), mode, Cancellation::new())
                    .await
                    .unwrap()
                    .fuse(),
            );
            let mut ids = vec![];
            for _ in 0..if child_plan { 4 } else { 2 } {
                ids.push(question(&mut events).await);
            }
            first
                .reply(answer(ids[0], "first-part"), Cancellation::new())
                .await
                .unwrap();
            assert!(poll!(events.next()).is_pending());
            let saved = store.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(
                record(&saved, &saved["waits"][0], ids[0])["feedback"]
                    .as_array()
                    .unwrap()
                    .len(),
                1
            );
            drop(events);
            drop(first);
            let requests = Arc::new(Mutex::new(vec![]));
            let child_requests = Arc::new(Mutex::new(vec![]));
            let fresh = agent::react()
                .model(Model {
                    plan: parent_plan,
                    parent: true,
                    requests: requests.clone(),
                })
                .subagents(children(
                    child_plan,
                    true,
                    3,
                    log.clone(),
                    child_requests.clone(),
                    own.clone(),
                ))
                .checkpoint(store.clone())
                .with_id_fn(no_new_id)
                .with_max_concurrency(1)
                .with_max_iters(40)
                .build()
                .unwrap();
            let mut events = Box::pin(
                fresh
                    .run(answer(ids[0], "second-part"), !mode, Cancellation::new())
                    .await
                    .unwrap()
                    .fuse(),
            );
            // Drive restored native progress, without rerunning selectors.
            loop {
                match poll!(events.next()) {
                    std::task::Poll::Ready(Some(Ok(_))) => {}
                    std::task::Poll::Pending => break,
                    other => panic!("unexpected restore: {other:?}"),
                }
            }
            assert!(requests.lock().unwrap().is_empty());
            assert!(child_requests.lock().unwrap().is_empty());
            let latest = store.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(
                record(&latest, &saved["waits"][0], ids[0])["feedback"]
                    .as_array()
                    .unwrap()
                    .len(),
                2
            );
            for (index, id) in ids.iter().enumerate() {
                for n in if index == 0 { 2 } else { 0 }..3 {
                    fresh
                        .reply(
                            answer(*id, &format!("restored-{index}-{n}")),
                            Cancellation::new(),
                        )
                        .await
                        .unwrap();
                    if n == 2 {
                        tool_result(&mut events).await;
                    }
                    loop {
                        match poll!(events.next()) {
                            std::task::Poll::Ready(Some(Ok(_))) => {}
                            std::task::Poll::Pending | std::task::Poll::Ready(None) => break,
                            other => panic!("unexpected early terminal: {other:?}"),
                        }
                    }
                }
            }
            let _ = finish(&mut events).await;
            let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(final_state["status"], "succeeded");
            assert_eq!(
                final_state["usage"]["total"].as_u64(),
                Some(
                    saved["usage"]["total"].as_u64().unwrap()
                        + 13 * (requests.lock().unwrap().len()
                            + child_requests.lock().unwrap().len())
                            as u64
                )
            );
            assert_eq!(log.0.lock().unwrap().len(), ids.len());
            assert_eq!(
                own.load(1001, &Cancellation::new()).await.unwrap_err().code,
                "NOT_FOUND"
            );
        }
    }
}

#[tokio::test]
async fn chat_parent_forwards_ordinary_feedback_from_child_plan_tasks() {
    for mode in [false, true] {
        for capacity in [1, 2] {
            let store = Arc::new(core::checkpoint::InMemory::new());
            let own = Arc::new(core::checkpoint::InMemory::new());
            let log = Arc::new(Log::default());
            let agent = agent::chat()
                .model(Model {
                    parent: true,
                    plan: false,
                    requests: Default::default(),
                })
                .subagents(children(
                    true,
                    true,
                    3,
                    log.clone(),
                    Default::default(),
                    own.clone(),
                ))
                .checkpoint(store.clone())
                .with_id_fn(next_id)
                .with_max_concurrency(capacity)
                .build()
                .unwrap();
            let mut events = Box::pin(
                agent
                    .run(input(), mode, Cancellation::new())
                    .await
                    .unwrap()
                    .fuse(),
            );
            let mut ids = vec![];
            for _ in 0..4 {
                ids.push(question(&mut events).await);
            }
            let original = store.load(1001, &Cancellation::new()).await.unwrap();
            assert!(
                original["waits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|position| position["nodes"].as_array().unwrap().len() == 2)
            );
            collect(
                &agent,
                &mut events,
                &store,
                &ids,
                &original,
                (3, mode),
                &log,
            )
            .await;
            assert_eq!(
                own.load(1001, &Cancellation::new()).await.unwrap_err().code,
                "NOT_FOUND"
            );
        }
    }
}

struct BlockFinal {
    model: Model,
    entered: Arc<std::sync::atomic::AtomicBool>,
}
impl Chat for BlockFinal {
    type Stream = <Model as Chat>::Stream;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        if request.messages.as_slice().iter().any(|message| {
            message.metadata.get("name").and_then(Value::as_str) == Some("agent_spawn")
        }) {
            self.entered.store(true, Ordering::SeqCst);
            return std::future::pending().await;
        }
        self.model.stream(request).await
    }
}

#[tokio::test]
async fn last_child_feedback_saves_running_not_an_empty_waiting_scene() {
    let store = Arc::new(core::checkpoint::InMemory::new());
    let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let agent = agent::chat()
        .model(BlockFinal {
            model: Model {
                parent: true,
                plan: false,
                requests: Default::default(),
            },
            entered: entered.clone(),
        })
        .subagents(children(
            true,
            true,
            1,
            Default::default(),
            Default::default(),
            Default::default(),
        ))
        .checkpoint(store.clone())
        .with_id_fn(next_id)
        .with_max_concurrency(1)
        .build()
        .unwrap();
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(input(), true, cancellation.clone())
            .await
            .unwrap()
            .fuse(),
    );
    let mut ids = vec![];
    for _ in 0..4 {
        ids.push(question(&mut events).await);
    }
    let original = store.load(1001, &Cancellation::new()).await.unwrap();
    for (index, id) in ids.iter().enumerate() {
        agent
            .reply(answer(*id, "feedback"), Cancellation::new())
            .await
            .unwrap();
        tool_result(&mut events).await;
        settled(
            &mut events,
            &store,
            &original["waits"][index],
            *id,
            1,
            &mut vec![],
        )
        .await;
    }
    loop {
        match poll!(events.next()) {
            std::task::Poll::Ready(Some(Ok(_))) => {}
            std::task::Poll::Pending => break,
            other => panic!("unexpected final result {other:?}"),
        }
    }
    assert!(entered.load(Ordering::SeqCst));
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(saved["waits"], json!([]));
    assert_eq!(saved["status"], "running");
    cancellation.cancel();
    let _ = finish(&mut events).await;
}

#[derive(Clone)]
struct FailPartial(Arc<core::checkpoint::InMemory>);
impl Checkpoint for FailPartial {
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        if state["status"] == "waiting"
            && state["waits"].as_array().is_some_and(|positions| {
                positions.iter().any(|position| {
                    native_step(&state, position)["waits"]
                        .as_object()
                        .unwrap()
                        .values()
                        .any(|record| {
                            record["kind"] == "input"
                                && record["feedback"]
                                    .as_array()
                                    .is_some_and(|feedback| !feedback.is_empty())
                        })
                })
            })
        {
            return Err(Error::new(
                "SAVE_FAILED",
                "outer nested partial save failed",
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

#[tokio::test]
async fn nested_cancel_and_save_failure_close_consumers_without_completing_tools() {
    for cancel in [false, true] {
        for mode in [false, true] {
            let store = Arc::new(core::checkpoint::InMemory::new());
            let log = Arc::new(Log::default());
            let agent = agent::react()
                .model(Model {
                    parent: true,
                    plan: true,
                    requests: Default::default(),
                })
                .subagents(children(
                    true,
                    true,
                    3,
                    log.clone(),
                    Default::default(),
                    Default::default(),
                ))
                .checkpoint(FailPartial(store.clone()))
                .with_id_fn(next_id)
                .with_max_concurrency(1)
                .with_max_iters(40)
                .build()
                .unwrap();
            let cancellation = Cancellation::new();
            let mut events = Box::pin(
                agent
                    .run(input(), mode, cancellation.clone())
                    .await
                    .unwrap()
                    .fuse(),
            );
            let mut ids = vec![];
            for _ in 0..4 {
                ids.push(question(&mut events).await);
            }
            if cancel {
                cancellation.cancel();
                let output = finish(&mut events).await;
                assert!(output.iter().any(|event|matches!(event,Event::Complete{finish_reason:Some(reason),..} if reason=="cancelled")));
            } else {
                agent
                    .reply(answer(ids[0], "partial"), Cancellation::new())
                    .await
                    .unwrap();
                let error = tokio::time::timeout(std::time::Duration::from_secs(3), events.next())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap_err();
                assert_eq!(error.code, "SAVE_FAILED");
                assert!(events.next().await.is_none());
            }
            let scene = store.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(scene["status"], if cancel { "stopped" } else { "failed" });
            assert_eq!(scene["waits"], json!([]));
            assert!(log.0.lock().unwrap().is_empty());
            for id in ids {
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
    }
}

#[tokio::test]
async fn nested_restoration_rejects_stale_or_misdirected_input_before_calls_and_writes() {
    let store = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let first = agent::react()
        .model(Model {
            parent: true,
            plan: true,
            requests: Default::default(),
        })
        .subagents(children(
            true,
            true,
            3,
            log.clone(),
            Default::default(),
            Default::default(),
        ))
        .checkpoint(store.clone())
        .with_id_fn(next_id)
        .with_max_concurrency(1)
        .with_max_iters(40)
        .build()
        .unwrap();
    let mut events = Box::pin(
        first
            .run(input(), false, Cancellation::new())
            .await
            .unwrap()
            .fuse(),
    );
    let mut ids = vec![];
    for _ in 0..4 {
        ids.push(question(&mut events).await);
    }
    let original = store.load(1001, &Cancellation::new()).await.unwrap();
    drop(events);
    drop(first);
    for case in ["exec", "attempt", "step", "session", "order"] {
        let mut saved = original.clone();
        let mut response = answer(ids[0], "feedback");
        match case {
            "exec" => saved["waits"][0]["nodes"][2]["exec_id"] = json!(-1),
            "attempt" => saved["waits"][0]["nodes"][1]["idx"] = json!(99),
            "step" => saved["waits"][0]["step"] = json!("missing"),
            "session" => {
                response.metadata.insert("session_id".into(), json!(2002));
            }
            "order" => response = answer(ids[1], "wrong head"),
            _ => unreachable!(),
        }
        store
            .save(1001, saved.clone(), &Cancellation::new())
            .await
            .unwrap();
        let requests = Arc::new(Mutex::new(vec![]));
        let child_requests = Arc::new(Mutex::new(vec![]));
        let agent = agent::react()
            .model(Model {
                parent: true,
                plan: true,
                requests: requests.clone(),
            })
            .subagents(children(
                true,
                true,
                3,
                log.clone(),
                child_requests.clone(),
                Default::default(),
            ))
            .checkpoint(store.clone())
            .with_id_fn(no_new_id)
            .build()
            .unwrap();
        let error = match agent.run(response, false, Cancellation::new()).await {
            Err(error) => error,
            Ok(_) => panic!("invalid {case} accepted"),
        };
        assert_eq!(
            error.code,
            if matches!(case, "session" | "order") {
                "CONFLICT"
            } else {
                "INVALID_STATE"
            }
        );
        assert!(requests.lock().unwrap().is_empty());
        assert!(child_requests.lock().unwrap().is_empty());
        assert!(log.0.lock().unwrap().is_empty());
        assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
    }
}

#[tokio::test]
#[ignore = "explicitly invoked by nested_feedback_restores_in_fresh_processes"]
async fn nested_feedback_process_worker() {
    let path = std::env::var("HEART_NESTED_FEEDBACK_FILE").unwrap();
    let save = std::env::var("HEART_NESTED_FEEDBACK_PHASE").unwrap() == "save";
    let parent_plan = std::env::var("HEART_NESTED_FEEDBACK_PARENT_PLAN").unwrap() == "true";
    let child_plan = std::env::var("HEART_NESTED_FEEDBACK_CHILD_PLAN").unwrap() == "true";
    let mode = std::env::var("HEART_NESTED_FEEDBACK_MODE").unwrap() == "true";
    let store = Arc::new(core::checkpoint::InMemory::new());
    let own = Arc::new(core::checkpoint::InMemory::new());
    let previous = if save {
        None
    } else {
        let state: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        store
            .save(1001, state.clone(), &Cancellation::new())
            .await
            .unwrap();
        Some(state)
    };
    let log = Arc::new(Log::default());
    let requests = Arc::new(Mutex::new(vec![]));
    let child_requests = Arc::new(Mutex::new(vec![]));
    let agent = agent::react()
        .model(Model {
            parent: true,
            plan: parent_plan,
            requests: requests.clone(),
        })
        .subagents(children(
            child_plan,
            true,
            3,
            log.clone(),
            child_requests.clone(),
            own.clone(),
        ))
        .checkpoint(store.clone())
        .with_id_fn(if save { next_id } else { no_new_id })
        .with_max_concurrency(1)
        .with_max_iters(40)
        .build()
        .unwrap();
    if let Some(previous) = previous {
        let ids = previous["waits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|position| {
                native_step(&previous, position)["waits"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .find(|(_, record)| record["kind"] == "input" && record["status"] == "waiting")
                    .unwrap()
                    .0
                    .parse::<i64>()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut events = Box::pin(
            agent
                .run(answer(ids[0], "second"), !mode, Cancellation::new())
                .await
                .unwrap()
                .fuse(),
        );
        let mut output = vec![];
        settled(
            &mut events,
            &store,
            &previous["waits"][0],
            ids[0],
            2,
            &mut output,
        )
        .await;
        assert!(requests.lock().unwrap().is_empty());
        assert!(child_requests.lock().unwrap().is_empty());
        for (index, id) in ids.iter().enumerate() {
            for n in if index == 0 { 2 } else { 0 }..3 {
                agent
                    .reply(
                        answer(*id, &format!("process-{index}-{n}")),
                        Cancellation::new(),
                    )
                    .await
                    .unwrap();
                if n == 2 {
                    output.push(tool_result(&mut events).await);
                }
                settled(
                    &mut events,
                    &store,
                    &previous["waits"][index],
                    *id,
                    n + 1,
                    &mut output,
                )
                .await;
            }
        }
        output.extend(finish(&mut events).await);
        if !mode {
            assert!(
                !output
                    .iter()
                    .any(|event| matches!(event, Event::Delta { .. }))
            );
        }
        let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(final_state["status"], "succeeded");
        assert_eq!(final_state["stream"], mode);
        assert_eq!(final_state["query"], previous["query"]);
        assert_eq!(
            final_state["usage"]["total"].as_u64(),
            Some(
                previous["usage"]["total"].as_u64().unwrap()
                    + 13 * (requests.lock().unwrap().len() + child_requests.lock().unwrap().len())
                        as u64
            )
        );
        assert_eq!(log.0.lock().unwrap().len(), ids.len());
        for (index, id) in ids.iter().enumerate() {
            assert_eq!(
                record(&final_state, &previous["waits"][index], *id)["status"],
                "resolved"
            );
        }
        assert_eq!(
            own.load(1001, &Cancellation::new()).await.unwrap_err().code,
            "NOT_FOUND"
        );
        std::fs::write(path, serde_json::to_vec(&final_state).unwrap()).unwrap();
    } else {
        let mut events = Box::pin(
            agent
                .run(input(), mode, Cancellation::new())
                .await
                .unwrap()
                .fuse(),
        );
        let mut ids = vec![];
        for _ in 0..if child_plan { 4 } else { 2 } {
            ids.push(question(&mut events).await);
        }
        let original = store.load(1001, &Cancellation::new()).await.unwrap();
        agent
            .reply(answer(ids[0], "first"), Cancellation::new())
            .await
            .unwrap();
        let saved = settled(
            &mut events,
            &store,
            &original["waits"][0],
            ids[0],
            1,
            &mut vec![],
        )
        .await;
        std::fs::write(path, serde_json::to_vec(&saved).unwrap()).unwrap();
    }
}

#[test]
fn nested_feedback_restores_in_fresh_processes() {
    for (parent_plan, child_plan) in [(true, false), (false, true), (true, true)] {
        for mode in [false, true] {
            let path = std::env::temp_dir().join(format!(
                "heart-nested-feedback-{}-{}.json",
                std::process::id(),
                next_id()
            ));
            for phase in ["save", "restore"] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "nested_feedback_process_worker",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env("HEART_NESTED_FEEDBACK_FILE", &path)
                    .env("HEART_NESTED_FEEDBACK_PHASE", phase)
                    .env("HEART_NESTED_FEEDBACK_PARENT_PLAN", parent_plan.to_string())
                    .env("HEART_NESTED_FEEDBACK_CHILD_PLAN", child_plan.to_string())
                    .env("HEART_NESTED_FEEDBACK_MODE", mode.to_string())
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{phase}: {}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            let scene: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            assert_eq!(scene["status"], "succeeded");
            std::fs::remove_file(path).unwrap();
        }
    }
}

#[tokio::test]
async fn workflow_now_owns_the_verified_nested_feedback_handoff() {
    use halo::runtime::workflow::{
        Runtime, WorkflowBuilder,
        graph::{END, START},
    };
    for child_plan in [false, true] {
        let log = Arc::new(Log::default());
        let requests = Arc::new(Mutex::new(vec![]));
        let embedded = agent::react()
            .model(Model {
                parent: true,
                plan: false,
                requests: Default::default(),
            })
            .subagents(children(
                child_plan,
                true,
                1,
                log.clone(),
                requests.clone(),
                Default::default(),
            ))
            .with_id_fn(next_id)
            .with_max_iters(40)
            .build()
            .unwrap();
        let mut builder = WorkflowBuilder::new("ordinary-feedback-owned-by-workflow");
        builder.add_agent_node("agent", embedded).unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = Runtime::new(1001, builder.compile().unwrap()).unwrap();
        let initial = Messages::new([Message::function(json!({"query":"original query"}))]);
        let mut events = Box::pin(
            workflow
                .run(initial, Arc::new(next_id), Cancellation::new())
                .await
                .unwrap()
                .fuse(),
        );
        let mut ids = vec![];
        for _ in 0..if child_plan { 4 } else { 2 } {
            ids.push(question(&mut events).await);
        }
        for id in ids {
            let mut response = Message::new(Role::User);
            response.metadata = json!({"chat_id":1001,"reply_id":id})
                .as_object()
                .unwrap()
                .clone();
            response.content.push(Part {
                r#type: "text".into(),
                data: json!({"value":"feedback"}).as_object().unwrap().clone(),
            });
            workflow
                .reply(Messages::new([response]), Cancellation::new())
                .await
                .unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                loop {
                    if let Event::Complete {
                        message,
                        finish_reason,
                        ..
                    } = events.next().await.unwrap().unwrap()
                        && finish_reason.as_deref() == Some("tool")
                        && message.metadata.get("name").and_then(Value::as_str) == Some("collect")
                    {
                        break;
                    }
                }
            })
            .await
            .unwrap();
        }
        let _ = finish(&mut events).await;
        assert_eq!(log.0.lock().unwrap().len(), if child_plan { 4 } else { 2 });
        assert!(!requests.lock().unwrap().iter().any(|request| {
            serde_json::to_string(&request.messages)
                .unwrap()
                .contains("UNSUPPORTED")
        }));
    }
}
