//! Workflow owns embedded ordinary feedback and the only saved scene.
use futures_util::{StreamExt, poll, stream};
use halo_agents::{
    Cancellation, Error, Message, Messages,
    agent::{self, SubAgent, toolkit::Toolkit},
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
    runtime::{
        agent::wait,
        workflow::{
            CompiledWorkflow, Runtime, WorkflowBuilder,
            graph::{END, START},
        },
    },
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
    },
};
fn next_id() -> i64 {
    static ID: AtomicI64 = AtomicI64::new(61_000_000);
    ID.fetch_add(1, Ordering::SeqCst)
}
fn no_new_id() -> i64 {
    panic!("existing embedded identities must survive recovery")
}

struct Tracked<A> {
    agent: A,
    done: Arc<AtomicBool>,
}
struct TrackedStream<S> {
    stream: Pin<Box<S>>,
    done: Arc<AtomicBool>,
}
impl<S: futures_core::Stream> futures_core::Stream for TrackedStream<S> {
    type Item = S::Item;
    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let result = this.stream.as_mut().poll_next(cx);
        if matches!(result, std::task::Poll::Ready(None)) {
            this.done.store(true, Ordering::SeqCst);
        }
        result
    }
}
impl<A: agent::Agent> agent::Agent for Tracked<A> {
    type Stream = TrackedStream<A::Stream>;
    async fn run(
        &self,
        input: agent::Input,
        stream: bool,
        cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        Ok(TrackedStream {
            stream: Box::pin(self.agent.run(input, stream, cancellation).await?),
            done: self.done.clone(),
        })
    }
    async fn reply(&self, input: agent::Input, cancellation: Cancellation) -> Result<(), Error> {
        self.agent.reply(input, cancellation).await
    }
}

fn repeated_graph(
    log: Arc<Log>,
    requests: Arc<Mutex<Vec<Request>>>,
    id_fn: fn() -> i64,
    done: Arc<AtomicBool>,
) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("repeated-feedback");
    for label in ["a", "b"] {
        let agent = agent::react()
            .model(Model {
                label: label.into(),
                plan: false,
                delegate: false,
                requests: requests.clone(),
            })
            .toolkit(toolkit(log.clone(), 3))
            .with_id_fn(id_fn)
            .build()
            .unwrap();
        builder
            .add_agent_node(
                label,
                Tracked {
                    agent,
                    done: if label == "b" {
                        done.clone()
                    } else {
                        Default::default()
                    },
                },
            )
            .unwrap();
        builder.add_edge(START, label).unwrap();
    }
    builder
        .add_node("merge", move |_: Messages, _: Cancellation| {
            let log = log.clone();
            async move {
                log.merged.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Error>(Messages::default())
            }
        })
        .unwrap();
    builder.add_join(["a", "b"], "merge").unwrap();
    builder.add_edge("merge", END).unwrap();
    let mut graph = builder.compile().unwrap();
    for i in 0..2 {
        let mut parent = WorkflowBuilder::new(format!("repeated-wrapper-{i}"));
        parent.add_graph_node("inside", &graph).unwrap();
        parent.add_edge(START, "inside").unwrap();
        parent.add_edge("inside", END).unwrap();
        graph = parent.compile().unwrap();
    }
    graph
}

fn inner_graph(mut state: &Value) -> &Value {
    while let Some(node) = state["nodes"].get("inside") {
        state = &node["executions"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap()["attempts"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()["graph"];
    }
    state
}

#[tokio::test]
async fn repeated_interruptions_keep_successful_siblings_and_only_resume_remaining_feedback() {
    let store = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let done = Arc::new(AtomicBool::new(false));
    let first = Runtime::new(
        1001,
        repeated_graph(log.clone(), Default::default(), next_id, done.clone()),
    )
    .unwrap()
    .with_checkpoint(store.clone())
    .with_max_concurrency(2)
    .unwrap();
    let mut events = Box::pin(
        first
            .run(initial(), Arc::new(next_id), Cancellation::new())
            .await
            .unwrap()
            .fuse(),
    );
    let b = question(&mut events).await;
    let a = question(&mut events).await;
    assert_eq!(
        question_ids(&store.load(1001, &Cancellation::new()).await.unwrap()),
        [b, a]
    );
    for part in 0..3 {
        first
            .reply(answer(b, &format!("b-{part}")), Cancellation::new())
            .await
            .unwrap();
        settled(&mut events, &store, b, part + 1).await;
    }
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while !done.load(Ordering::SeqCst) {
            if let std::task::Poll::Ready(Some(result)) = poll!(events.next()) {
                result.unwrap();
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    first
        .reply(answer(a, "a-one"), Cancellation::new())
        .await
        .unwrap();
    let saved = settled(&mut events, &store, a, 1).await;
    let successful = inner_graph(&saved)["nodes"]["b"].clone();
    let execution = successful["executions"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap();
    assert_eq!(execution["status"], "succeeded");
    assert_eq!(log.received.lock().unwrap().len(), 1);
    drop(events);
    drop(first);
    let requests = Arc::new(Mutex::new(vec![]));
    let second = Runtime::from_snapshot(
        1001,
        repeated_graph(log.clone(), requests.clone(), no_new_id, done.clone()),
        saved.clone(),
    )
    .unwrap()
    .with_checkpoint(store.clone())
    .with_max_concurrency(1)
    .unwrap();
    second
        .reply(answer(b, "b-2"), Cancellation::new())
        .await
        .unwrap();
    second
        .reply(answer(a, "a-two"), Cancellation::new())
        .await
        .unwrap();
    let mut events = Box::pin(
        second
            .run(initial(), Arc::new(next_id), Cancellation::new())
            .await
            .unwrap()
            .fuse(),
    );
    let saved_again = settled(&mut events, &store, a, 2).await;
    assert_eq!(inner_graph(&saved_again)["nodes"]["b"], successful);
    assert_eq!(saved_again["usage"], saved["usage"]);
    assert!(requests.lock().unwrap().is_empty());
    assert_eq!(log.received.lock().unwrap().len(), 1);
    drop(events);
    drop(second);
    let third = Runtime::from_snapshot(
        1001,
        repeated_graph(log.clone(), requests.clone(), no_new_id, done),
        saved_again,
    )
    .unwrap()
    .with_checkpoint(store.clone())
    .with_max_concurrency(1)
    .unwrap();
    third
        .reply(answer(a, "a-three"), Cancellation::new())
        .await
        .unwrap();
    let mut events = Box::pin(
        third
            .run(initial(), Arc::new(next_id), Cancellation::new())
            .await
            .unwrap()
            .fuse(),
    );
    let output = finish(&mut events).await;
    assert!(output.iter().all(Result::is_ok), "{output:?}");
    let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(final_state["status"], "succeeded");
    assert_eq!(inner_graph(&final_state)["nodes"]["b"], successful);
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert_eq!(log.received.lock().unwrap().len(), 2);
    assert_eq!(log.merged.load(Ordering::SeqCst), 1);
    assert_eq!(
        final_state["usage"]["total"].as_u64(),
        Some(saved["usage"]["total"].as_u64().unwrap() + 13)
    );
}
fn initial() -> Messages {
    let mut message = Message::new(Role::User);
    message.metadata = json!({"chat_id":1001,"session_id":2001,"user_id":3001,"agent_id":4001})
        .as_object()
        .unwrap()
        .clone();
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"original query"})
            .as_object()
            .unwrap()
            .clone(),
    });
    Messages::new([message])
}
fn answer(id: i64, text: &str) -> Messages {
    let mut message = initial().0.remove(0);
    message.metadata.insert("reply_id".into(), id.into());
    message.content[0].data = json!({"value":text}).as_object().unwrap().clone();
    Messages::new([message])
}
#[derive(Clone)]
struct Model {
    label: String,
    plan: bool,
    delegate: bool,
    requests: Arc<Mutex<Vec<Request>>>,
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        // With two slots, b asks before a despite a sorting first. Recovery
        // must use persisted arrival order, not the Workflow's node map.
        if self.label == "a" {
            tokio::task::yield_now().await;
        }
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
                == Some(if self.delegate {
                    "agent_spawn"
                } else {
                    "collect"
                })
                || (m.role == Role::Tool
                    && m.values.get("name").and_then(Value::as_str) == Some("collect"))
        });
        if self.plan && task.is_none() && !complete {
            let planning = request.messages.as_slice().iter().any(|m| {
                m.role == Role::System
                    && m.content.iter().any(|p| {
                        p.data
                            .get("value")
                            .and_then(Value::as_str)
                            .is_some_and(|t| {
                                t.starts_with("Return only a complete JSON Agent plan")
                            })
                    })
            });
            if planning {
                message.values=json!({"version":1,"nodes":{"x":{"name":"first","objective":"first task"},"y":{"name":"second","objective":"second task"}},"edges":[]}).as_object().unwrap().clone();
            } else {
                message.content.push(tool_call("plan", json!({})));
            }
        } else if complete || selected {
            message.values.insert("answer".into(), json!(&self.label));
        } else if self.delegate {
            message.content.push(tool_call("agent_spawn",json!({"name":"worker","input":{"payload":[{"type":"text","data":{"value":"child task"}}]}})));
        } else {
            message.content.push(tool_call(
                "collect",
                json!({"item":format!("{}:{}",self.label,task.unwrap_or("direct"))}),
            ));
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
fn tool_call(name: &str, arguments: Value) -> Part {
    Part {
        r#type: "tool_call".into(),
        data: json!({"call_id":"same_call","name":name,"arguments":arguments})
            .as_object()
            .unwrap()
            .clone(),
    }
}
#[derive(Default)]
struct Log {
    received: Mutex<BTreeMap<String, Messages>>,
    merged: AtomicUsize,
}
struct Collect {
    definition: ToolDefinition,
    count: usize,
    log: Arc<Log>,
}
impl Tool for Collect {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let mut question = Message::new(Role::Assistant);
        question.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"请提供反馈"}).as_object().unwrap().clone(),
        });
        let feedback = wait::ask(question, self.count, cancellation).await?;
        let item = call.arguments["item"].as_str().unwrap();
        assert!(
            self.log
                .received
                .lock()
                .unwrap()
                .insert(item.into(), feedback)
                .is_none(),
            "successful item replayed"
        );
        Ok(Message::function(json!({"item":item})))
    }
}
fn toolkit(log: Arc<Log>, count: usize) -> Toolkit {
    Toolkit::new().tool(Collect{log,count,definition:ToolDefinition{name:"collect".into(),description:"collect feedback".into(),parameters:json!({"type":"object","properties":{"item":{"type":"string"}},"required":["item"],"additionalProperties":false}).as_object().unwrap().clone()}})
}
#[derive(Clone, Copy, Debug)]
enum Kind {
    Chat,
    React,
    Plan,
    ChildPlan,
}
#[derive(Clone, Copy)]
struct Case {
    kind: Kind,
    depth: usize,
    branches: usize,
    count: usize,
    ids: fn() -> i64,
}
fn compiled(
    case: Case,
    log: Arc<Log>,
    requests: Arc<Mutex<Vec<Request>>>,
    own: Arc<core::checkpoint::InMemory>,
) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("ordinary-workflow");
    for label in ["a", "b"].into_iter().take(case.branches) {
        let model = Model {
            label: label.into(),
            plan: matches!(case.kind, Kind::Plan),
            delegate: matches!(case.kind, Kind::ChildPlan),
            requests: requests.clone(),
        };
        if matches!(case.kind, Kind::Chat) {
            builder
                .add_agent_node(
                    label,
                    agent::chat()
                        .model(model)
                        .toolkit(toolkit(log.clone(), case.count))
                        .checkpoint(own.clone())
                        .with_id_fn(case.ids)
                        .with_max_concurrency(1)
                        .build()
                        .unwrap(),
                )
                .unwrap();
        } else {
            let mut children = vec![];
            if matches!(case.kind, Kind::ChildPlan) {
                children.push(
                    SubAgent::new(
                        "worker",
                        agent::react()
                            .model(Model {
                                label: format!("{label}.child"),
                                plan: true,
                                delegate: false,
                                requests: requests.clone(),
                            })
                            .toolkit(toolkit(log.clone(), case.count))
                            .checkpoint(own.clone())
                            .with_max_concurrency(1)
                            .with_max_iters(40)
                            .build()
                            .unwrap(),
                    )
                    .unwrap(),
                );
            }
            builder
                .add_agent_node(
                    label,
                    agent::react()
                        .model(model)
                        .toolkit(toolkit(log.clone(), case.count))
                        .subagents(children)
                        .checkpoint(own.clone())
                        .with_id_fn(case.ids)
                        .with_max_concurrency(1)
                        .with_max_iters(40)
                        .build()
                        .unwrap(),
                )
                .unwrap();
        }
        builder.add_edge(START, label).unwrap();
        if case.branches == 1 {
            builder.add_edge(label, END).unwrap();
        }
    }
    if case.branches == 2 {
        let count = log.clone();
        builder
            .add_node("merge", move |_: Messages, _: Cancellation| {
                let count = count.clone();
                async move {
                    count.merged.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, Error>(Messages::new([Message::function(json!({"merged":true}))]))
                }
            })
            .unwrap();
        builder.add_join(["a", "b"], "merge").unwrap();
        builder.add_edge("merge", END).unwrap();
    }
    let mut graph = builder.compile().unwrap();
    for index in 0..case.depth {
        let mut wrapper = WorkflowBuilder::new(format!("wrapper-{index}"));
        wrapper.add_graph_node("inside", &graph).unwrap();
        wrapper.add_edge(START, "inside").unwrap();
        wrapper.add_edge("inside", END).unwrap();
        graph = wrapper.compile().unwrap();
    }
    graph
}
fn agent_record(scene: &Value, id: i64) -> Option<&Value> {
    fn owner(value: &Value, id: i64) -> Option<&Value> {
        value["steps"]
            .as_array()
            .into_iter()
            .flatten()
            .find_map(|step| step["waits"].get(id.to_string()))
            .or_else(|| nodes(value.get("nodes"), id))
    }
    fn nodes(value: Option<&Value>, id: i64) -> Option<&Value> {
        value
            .and_then(Value::as_object)
            .into_iter()
            .flat_map(|nodes| nodes.values())
            .find_map(|node| {
                let attempt = node["execution"]["attempts"].as_array()?.last()?;
                owner(attempt, id).or_else(|| {
                    attempt
                        .get("agent")
                        .and_then(|scene| agent_record(scene, id))
                })
            })
    }
    owner(scene, id).or_else(|| {
        scene["execution"]["attempts"]
            .as_array()
            .and_then(|attempts| attempts.last())
            .and_then(|attempt| nodes(attempt.get("nodes"), id))
    })
}
fn record(state: &Value, id: i64) -> Option<&Value> {
    state["nodes"]
        .as_object()?
        .values()
        .flat_map(|node| {
            node["executions"]
                .as_object()
                .into_iter()
                .flat_map(|executions| executions.values())
        })
        .find_map(|execution| {
            let attempt = execution["attempts"].as_array()?.last()?;
            attempt
                .get("agent")
                .and_then(|scene| agent_record(scene, id))
                .or_else(|| attempt.get("graph").and_then(|graph| record(graph, id)))
        })
}
fn question_ids(state: &Value) -> Vec<i64> {
    state["waits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|wait| {
            let step = workflow_step(state, wait);
            let records = step["waits"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(_, record)| record["status"] == "waiting")
                .collect::<Vec<_>>();
            assert_eq!(records.len(), 1);
            records[0].0.parse().unwrap()
        })
        .collect()
}
fn workflow_step<'a>(state: &'a Value, wait: &Value) -> &'a Value {
    let coordinates = wait["nodes"].as_array().unwrap();
    let mut graph = state;
    for (index, coordinate) in coordinates.iter().enumerate() {
        let id = coordinate["node_id"].as_str().unwrap();
        let exec_id = coordinate["exec_id"].as_i64().unwrap();
        let execution = &graph["nodes"][id]["executions"][exec_id.to_string()];
        assert_eq!(execution["exec_id"], exec_id);
        let attempt = execution["attempts"].as_array().unwrap().last().unwrap();
        assert_eq!(attempt["idx"], coordinate["idx"]);
        if let Some(scene) = attempt.get("agent").filter(|scene| scene.is_object()) {
            return halo_step(
                scene,
                &json!({"nodes":coordinates[index+1..], "step":wait["step"]}),
            );
        }
        graph = &attempt["graph"];
        assert!(graph.is_object());
    }
    panic!("Workflow locator must reach an actual Agent Step")
}
fn halo_step<'a>(scene: &'a Value, wait: &Value) -> &'a Value {
    let mut owner = scene;
    for coordinate in wait["nodes"].as_array().unwrap() {
        let id = coordinate["node_id"].as_str().unwrap();
        let node = owner
            .get("nodes")
            .and_then(|nodes| nodes.get(id))
            .or_else(|| {
                owner["execution"]["attempts"]
                    .as_array()
                    .and_then(|attempts| attempts.last())
                    .and_then(|attempt| attempt["nodes"].get(id))
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
        .find(|step| step["id"] == wait["step"])
        .unwrap()
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
    .expect("waiting must release actual Workflow capacity")
}
async fn settled<S: futures_core::Stream<Item = Result<Event, Error>>>(
    events: &mut Pin<Box<S>>,
    store: &core::checkpoint::InMemory,
    id: i64,
    count: usize,
) -> Value {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            if let std::task::Poll::Ready(Some(Err(error))) = poll!(events.next()) {
                panic!("{error:?}");
            }
            let state = store.load(1001, &Cancellation::new()).await.unwrap();
            if record(&state, id)
                .and_then(|record| record["feedback"].as_array())
                .is_some_and(|feedback| feedback.len() == count)
            {
                return state;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("native partial progress must reach the complete outer scene")
}
async fn finish<S: futures_core::Stream<Item = Result<Event, Error>>>(
    events: &mut Pin<Box<S>>,
) -> Vec<Result<Event, Error>> {
    tokio::time::timeout(std::time::Duration::from_secs(3), events.collect())
        .await
        .expect("Workflow continuation")
}

#[tokio::test]
async fn live_workflow_collects_on_actual_agent_steps_and_keeps_join_blocked() {
    for kind in [Kind::Chat, Kind::React, Kind::Plan, Kind::ChildPlan] {
        for depth in [0, 2] {
            for count in [1, 3] {
                for capacity in [1, 2] {
                    let case = Case {
                        kind,
                        depth,
                        branches: 2,
                        count,
                        ids: next_id,
                    };
                    let log = Arc::new(Log::default());
                    let requests = Arc::new(Mutex::new(vec![]));
                    let own = Arc::new(core::checkpoint::InMemory::new());
                    let store = Arc::new(core::checkpoint::InMemory::new());
                    let workflow = Runtime::new(
                        1001,
                        compiled(case, log.clone(), requests.clone(), own.clone()),
                    )
                    .unwrap()
                    .with_checkpoint(store.clone())
                    .with_max_concurrency(capacity)
                    .unwrap();
                    let mut events = Box::pin(
                        workflow
                            .run(initial(), Arc::new(next_id), Cancellation::new())
                            .await
                            .unwrap()
                            .fuse(),
                    );
                    let expected = if matches!(kind, Kind::Plan | Kind::ChildPlan) {
                        4
                    } else {
                        2
                    };
                    let mut ids = vec![];
                    for _ in 0..expected {
                        ids.push(question(&mut events).await);
                    }
                    let original = store.load(1001, &Cancellation::new()).await.unwrap();
                    assert_eq!(question_ids(&original), ids);
                    assert_eq!(log.merged.load(Ordering::SeqCst), 0);
                    for (index, id) in ids.iter().enumerate() {
                        if index + 1 < ids.len() {
                            assert_eq!(
                                workflow
                                    .reply(answer(ids[index + 1], "early"), Cancellation::new())
                                    .await
                                    .unwrap_err()
                                    .code,
                                "CONFLICT"
                            );
                        }
                        for part in 0..count {
                            workflow
                                .reply(
                                    answer(*id, &format!("feedback-{index}-{part}")),
                                    Cancellation::new(),
                                )
                                .await
                                .unwrap();
                            workflow
                                .reply(
                                    answer(*id, &format!("feedback-{index}-{part}")),
                                    Cancellation::new(),
                                )
                                .await
                                .unwrap();
                            let saved = settled(&mut events, &store, *id, part + 1).await;
                            let record = record(&saved, *id).unwrap();
                            assert_eq!(record["count"], count);
                            assert_eq!(
                                record["status"],
                                if part + 1 == count {
                                    "resolved"
                                } else {
                                    "waiting"
                                }
                            );
                            if index + 1 < ids.len() {
                                assert_eq!(log.merged.load(Ordering::SeqCst), 0);
                            }
                        }
                        if index + 1 < ids.len() {
                            workflow
                                .reply(
                                    answer(*id, &format!("feedback-{index}-0")),
                                    Cancellation::new(),
                                )
                                .await
                                .unwrap();
                        }
                    }
                    let output = finish(&mut events).await;
                    assert!(output.iter().all(Result::is_ok), "{output:?}");
                    let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
                    assert_eq!(final_state["status"], "succeeded");
                    assert_eq!(log.merged.load(Ordering::SeqCst), 1);
                    assert_eq!(log.received.lock().unwrap().len(), expected);
                    assert_eq!(
                        final_state["usage"]["total"].as_u64(),
                        Some(13 * requests.lock().unwrap().len() as u64)
                    );
                    assert_eq!(final_state["initial"], original["initial"]);
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
async fn one_embedded_agent_restores_partial_collection_through_nested_graphs() {
    for kind in [Kind::Chat, Kind::React, Kind::Plan, Kind::ChildPlan] {
        for depth in [0, 2] {
            let mut case = Case {
                kind,
                depth,
                branches: 1,
                count: 3,
                ids: next_id,
            };
            let store = Arc::new(core::checkpoint::InMemory::new());
            let log = Arc::new(Log::default());
            let first = Runtime::new(
                1001,
                compiled(case, log.clone(), Default::default(), Default::default()),
            )
            .unwrap()
            .with_checkpoint(store.clone());
            let mut events = Box::pin(
                first
                    .run(initial(), Arc::new(next_id), Cancellation::new())
                    .await
                    .unwrap()
                    .fuse(),
            );
            let mut ids = vec![];
            for _ in 0..if matches!(kind, Kind::Plan | Kind::ChildPlan) {
                2
            } else {
                1
            } {
                ids.push(question(&mut events).await);
            }
            first
                .reply(answer(ids[0], "part-one"), Cancellation::new())
                .await
                .unwrap();
            let saved = settled(&mut events, &store, ids[0], 1).await;
            drop(events);
            drop(first);
            case.ids = no_new_id;
            let requests = Arc::new(Mutex::new(vec![]));
            let own = Arc::new(core::checkpoint::InMemory::new());
            let restored = Runtime::from_snapshot(
                1001,
                compiled(case, log.clone(), requests.clone(), own.clone()),
                saved.clone(),
            )
            .unwrap()
            .with_checkpoint(store.clone());
            restored
                .reply(answer(ids[0], "part-two"), Cancellation::new())
                .await
                .unwrap();
            let mut events = Box::pin(
                restored
                    .run(initial(), Arc::new(no_new_id), Cancellation::new())
                    .await
                    .unwrap()
                    .fuse(),
            );
            let partial = settled(&mut events, &store, ids[0], 2).await;
            assert!(requests.lock().unwrap().is_empty());
            assert_eq!(partial["usage"], saved["usage"]);
            assert_eq!(question_ids(&partial), ids);
            for (index, id) in ids.iter().enumerate() {
                for part in if index == 0 { 2 } else { 0 }..3 {
                    restored
                        .reply(
                            answer(*id, &format!("restored-{index}-{part}")),
                            Cancellation::new(),
                        )
                        .await
                        .unwrap();
                    settled(&mut events, &store, *id, part + 1).await;
                }
            }
            let output = finish(&mut events).await;
            assert!(
                output.iter().all(Result::is_ok),
                "{kind:?}/{depth}: {output:?}"
            );
            let state = store.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(state["status"], "succeeded");
            let attempt = embedded_attempt(&state);
            let expected = match kind {
                Kind::Chat | Kind::React => 3,
                // The built-in plan call has its own real Complete too.
                Kind::Plan => 10,
                Kind::ChildPlan => 13,
            };
            assert_eq!(
                attempt["steps"].as_array().unwrap().len(),
                expected,
                "{kind:?}/{depth}: observations must survive partial saves without duplicate prefixes"
            );
            assert_eq!(attempt["messages"].as_array().unwrap().len(), expected);
            let step_ids = attempt["steps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|step| step["id"].as_str().unwrap())
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(step_ids.len(), expected);
            assert_eq!(
                state["usage"]["total"].as_u64(),
                Some(
                    saved["usage"]["total"].as_u64().unwrap()
                        + 13 * requests.lock().unwrap().len() as u64
                )
            );
            assert_eq!(log.received.lock().unwrap().len(), ids.len());
            assert_eq!(
                own.load(1001, &Cancellation::new()).await.unwrap_err().code,
                "NOT_FOUND"
            );
        }
    }
}

fn embedded_attempt(state: &Value) -> &Value {
    if let Some(node) = state["nodes"].get("inside") {
        let attempt = node["executions"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap()["attempts"]
            .as_array()
            .unwrap()
            .last()
            .unwrap();
        return embedded_attempt(&attempt["graph"]);
    }
    state["nodes"]["a"]["executions"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()["attempts"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
}
fn embedded_scene_mut(state: &mut Value) -> &mut Value {
    let nested = state["nodes"].get("inside").is_some();
    let key = if nested { "inside" } else { "a" };
    let attempt = state["nodes"][key]["executions"]
        .as_object_mut()
        .unwrap()
        .values_mut()
        .next()
        .unwrap()["attempts"]
        .as_array_mut()
        .unwrap()
        .last_mut()
        .unwrap();
    if nested {
        embedded_scene_mut(&mut attempt["graph"])
    } else {
        &mut attempt["agent"]
    }
}
#[derive(Clone)]
struct FailPartial(Arc<core::checkpoint::InMemory>);
impl Checkpoint for FailPartial {
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        if state["status"] == "paused"
            && question_ids(&state).iter().any(|id| {
                record(&state, *id).unwrap()["feedback"]
                    .as_array()
                    .is_some_and(|feedback| !feedback.is_empty())
            })
        {
            return Err(Error::new(
                "SAVE_FAILED",
                "Workflow partial feedback save failed",
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
async fn workflow_cancel_and_partial_save_failure_close_receivers_without_completing_tools() {
    for cancel in [false, true] {
        for depth in [0, 2] {
            let case = Case {
                kind: Kind::ChildPlan,
                depth,
                branches: 1,
                count: 3,
                ids: next_id,
            };
            let store = Arc::new(core::checkpoint::InMemory::new());
            let log = Arc::new(Log::default());
            let workflow = Runtime::new(
                1001,
                compiled(case, log.clone(), Default::default(), Default::default()),
            )
            .unwrap()
            .with_checkpoint(FailPartial(store.clone()));
            let cancellation = Cancellation::new();
            let mut events = Box::pin(
                workflow
                    .run(initial(), Arc::new(next_id), cancellation.clone())
                    .await
                    .unwrap()
                    .fuse(),
            );
            let first = question(&mut events).await;
            let second = question(&mut events).await;
            if cancel {
                cancellation.cancel();
            } else {
                workflow
                    .reply(answer(first, "partial"), Cancellation::new())
                    .await
                    .unwrap();
            }
            let output = finish(&mut events).await;
            let expected = if cancel { "CANCELLED" } else { "SAVE_FAILED" };
            assert!(
                output
                    .iter()
                    .any(|event| event.as_ref().is_err_and(|error| error.code == expected)),
                "{output:?}"
            );
            assert!(log.received.lock().unwrap().is_empty());
            let saved = store.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(saved["status"], if cancel { "cancelled" } else { "failed" });
            assert_eq!(record(&saved, first).unwrap()["status"], "waiting");
            assert_eq!(
                record(&saved, first).unwrap()["feedback"]
                    .as_array()
                    .unwrap()
                    .len(),
                usize::from(!cancel)
            );
            for id in [first, second] {
                assert_eq!(
                    workflow
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
async fn invalid_saved_workflow_feedback_is_rejected_before_execution_or_writes() {
    let mut case = Case {
        kind: Kind::Plan,
        depth: 2,
        branches: 1,
        count: 3,
        ids: next_id,
    };
    let store = Arc::new(core::checkpoint::InMemory::new());
    let first = Runtime::new(
        1001,
        compiled(
            case,
            Default::default(),
            Default::default(),
            Default::default(),
        ),
    )
    .unwrap()
    .with_checkpoint(store.clone());
    let mut events = Box::pin(
        first
            .run(initial(), Arc::new(next_id), Cancellation::new())
            .await
            .unwrap()
            .fuse(),
    );
    let a = question(&mut events).await;
    let b = question(&mut events).await;
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    drop(events);
    drop(first);
    case.ids = no_new_id;
    for invalid in ["identity", "unknown", "order", "exec", "count"] {
        let mut state = saved.clone();
        let mut response = answer(a, "feedback");
        match invalid {
            "identity" => {
                response.0[0]
                    .metadata
                    .insert("session_id".into(), json!(999));
            }
            "unknown" => response = answer(-1, "unknown"),
            "order" => response = answer(b, "non-head"),
            "exec" => embedded_scene_mut(&mut state)["waits"][0]["nodes"][0]["exec_id"] = json!(-1),
            "count" => {
                let scene = embedded_scene_mut(&mut state);
                let task = scene["waits"][0]["nodes"][0]["node_id"]
                    .as_str()
                    .unwrap()
                    .to_owned();
                let step = scene["waits"][0]["step"].as_str().unwrap().to_owned();
                let steps = scene["nodes"][task]["execution"]["attempts"][0]["steps"]
                    .as_array_mut()
                    .unwrap();
                steps.iter_mut().find(|value| value["id"] == step).unwrap()["waits"]
                    [a.to_string()]["count"] = json!(0);
            }
            _ => unreachable!(),
        }
        let requests = Arc::new(Mutex::new(vec![]));
        let log = Arc::new(Log::default());
        let restored = Runtime::from_snapshot(
            1001,
            compiled(case, log.clone(), requests.clone(), Default::default()),
            state,
        );
        if matches!(invalid, "exec" | "count") {
            assert_eq!(
                restored.err().expect("invalid saved facts").code,
                "INVALID_STATE"
            );
            assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
            assert!(requests.lock().unwrap().is_empty());
            continue;
        }
        let workflow = restored.unwrap().with_checkpoint(store.clone());
        let before = workflow.snapshot().unwrap();
        let error = workflow
            .reply(response, Cancellation::new())
            .await
            .unwrap_err();
        assert_eq!(
            error.code,
            match invalid {
                "identity" | "order" => "CONFLICT",
                "unknown" => "NOT_FOUND",
                _ => "INVALID_STATE",
            }
        );
        assert_eq!(workflow.snapshot().unwrap(), before);
        assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
        assert!(requests.lock().unwrap().is_empty());
        assert!(log.received.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn cross_agent_restoration_preserves_saved_fifo_with_concurrent_consumers() {
    for kind in [Kind::Chat, Kind::React, Kind::Plan, Kind::ChildPlan] {
        for depth in [0, 2] {
            for capacity in [1, 2] {
                let mut case = Case {
                    kind,
                    depth,
                    branches: 2,
                    count: 3,
                    ids: next_id,
                };
                let store = Arc::new(core::checkpoint::InMemory::new());
                let first = Runtime::new(
                    1001,
                    compiled(
                        case,
                        Default::default(),
                        Default::default(),
                        Default::default(),
                    ),
                )
                .unwrap()
                .with_checkpoint(store.clone())
                .with_max_concurrency(2)
                .unwrap();
                let mut events = Box::pin(
                    first
                        .run(initial(), Arc::new(next_id), Cancellation::new())
                        .await
                        .unwrap()
                        .fuse(),
                );
                let mut ids = Vec::new();
                for _ in 0..if matches!(kind, Kind::Plan | Kind::ChildPlan) {
                    4
                } else {
                    2
                } {
                    ids.push(question(&mut events).await);
                }
                let before = store.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(question_ids(&before), ids);
                // a sorts first, but b is deliberately presented first.
                let coordinates = before["waits"][0]["nodes"].as_array().unwrap();
                assert_eq!(coordinates[depth]["node_id"], "b", "{kind:?}/{depth}");
                first
                    .reply(answer(ids[0], "part-one"), Cancellation::new())
                    .await
                    .unwrap();
                let saved = settled(&mut events, &store, ids[0], 1).await;
                drop(events);
                drop(first);
                case.ids = no_new_id;
                let requests = Arc::new(Mutex::new(Vec::new()));
                let log = Arc::new(Log::default());
                let own = Arc::new(core::checkpoint::InMemory::new());
                let restored = Runtime::from_snapshot(
                    1001,
                    compiled(case, log.clone(), requests.clone(), own.clone()),
                    saved.clone(),
                )
                .unwrap()
                .with_checkpoint(store.clone())
                .with_max_concurrency(capacity)
                .unwrap();
                assert_eq!(
                    restored
                        .reply(answer(ids[1], "too early"), Cancellation::new())
                        .await
                        .unwrap_err()
                        .code,
                    "CONFLICT"
                );
                assert_eq!(restored.snapshot().unwrap(), saved);
                // A retry of an already persisted contribution must not
                // occupy the staged reply slot needed for the next part.
                restored
                    .reply(answer(ids[0], "part-one"), Cancellation::new())
                    .await
                    .unwrap();
                assert_eq!(restored.snapshot().unwrap(), saved);
                restored
                    .reply(answer(ids[0], "part-two"), Cancellation::new())
                    .await
                    .unwrap();
                let mut events = Box::pin(
                    restored
                        .run(initial(), Arc::new(next_id), Cancellation::new())
                        .await
                        .unwrap()
                        .fuse(),
                );
                let partial = settled(&mut events, &store, ids[0], 2).await;
                assert_eq!(question_ids(&partial), ids);
                assert_eq!(partial["usage"], saved["usage"]);
                assert!(
                    requests.lock().unwrap().is_empty(),
                    "selection models must not replay"
                );
                for (index, id) in ids.iter().enumerate() {
                    for part in if index == 0 { 2 } else { 0 }..3 {
                        restored
                            .reply(
                                answer(*id, &format!("remaining-{index}-{part}")),
                                Cancellation::new(),
                            )
                            .await
                            .unwrap();
                        let committed = settled(&mut events, &store, *id, part + 1).await;
                        assert_eq!(
                            question_ids(&committed),
                            if part == 2 {
                                &ids[index + 1..]
                            } else {
                                &ids[index..]
                            }
                        );
                        if part < 2 && index + 1 < ids.len() {
                            assert_eq!(
                                restored
                                    .reply(answer(ids[index + 1], "too early"), Cancellation::new())
                                    .await
                                    .unwrap_err()
                                    .code,
                                "CONFLICT"
                            );
                        }
                    }
                }
                let output = finish(&mut events).await;
                assert!(
                    output.iter().all(Result::is_ok),
                    "{kind:?}/{depth}/{capacity}: {output:?}"
                );
                let state = store.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(state["status"], "succeeded");
                assert_eq!(state["waits"], json!([]));
                assert_eq!(log.received.lock().unwrap().len(), ids.len());
                assert_eq!(log.merged.load(Ordering::SeqCst), 1);
                assert_eq!(
                    state["usage"]["total"].as_u64(),
                    Some(
                        saved["usage"]["total"].as_u64().unwrap()
                            + 13 * requests.lock().unwrap().len() as u64
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

#[tokio::test]
async fn invalid_workflow_locators_fail_before_execution_or_checkpoint_writes() {
    let case = Case {
        kind: Kind::React,
        depth: 2,
        branches: 2,
        count: 3,
        ids: next_id,
    };
    let store = Arc::new(core::checkpoint::InMemory::new());
    let first = Runtime::new(
        1001,
        compiled(
            case,
            Default::default(),
            Default::default(),
            Default::default(),
        ),
    )
    .unwrap()
    .with_checkpoint(store.clone())
    .with_max_concurrency(2)
    .unwrap();
    let mut events = Box::pin(
        first
            .run(initial(), Arc::new(next_id), Cancellation::new())
            .await
            .unwrap()
            .fuse(),
    );
    question(&mut events).await;
    question(&mut events).await;
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    drop(events);
    drop(first);
    for invalid in [
        "missing",
        "duplicate",
        "execution",
        "attempt",
        "step",
        "owner",
        "order",
    ] {
        let mut state = saved.clone();
        match invalid {
            "missing" => state["waits"] = json!([]),
            "duplicate" => {
                let value = state["waits"][0].clone();
                state["waits"].as_array_mut().unwrap().push(value);
            }
            "execution" => state["waits"][0]["nodes"][0]["exec_id"] = json!(-1),
            "attempt" => state["waits"][0]["nodes"][0]["idx"] = json!(1),
            "step" => state["waits"][0]["step"] = json!("not_a_step"),
            "owner" => state["waits"][0]["nodes"] = json!([]),
            // Parent and child must agree on the relative order in a child
            // graph; otherwise the two snapshots cannot describe one queue.
            "order" => state["waits"].as_array_mut().unwrap().reverse(),
            _ => unreachable!(),
        }
        let requests = Arc::new(Mutex::new(Vec::new()));
        let result = Runtime::from_snapshot(
            1001,
            compiled(
                Case {
                    ids: no_new_id,
                    ..case
                },
                Default::default(),
                requests.clone(),
                Default::default(),
            ),
            state,
        );
        assert_eq!(result.err().expect(invalid).code, "INVALID_STATE");
        assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
        assert!(requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
#[ignore = "explicitly invoked by workflow_feedback_recovers_in_fresh_processes"]
async fn workflow_feedback_process_worker() {
    let path = std::env::var("HEART_WORKFLOW_FEEDBACK_FILE").unwrap();
    let save = std::env::var("HEART_WORKFLOW_FEEDBACK_PHASE").unwrap() == "save";
    let kind = match std::env::var("HEART_WORKFLOW_FEEDBACK_KIND")
        .unwrap()
        .as_str()
    {
        "chat" => Kind::Chat,
        "plan" => Kind::Plan,
        "child" => Kind::ChildPlan,
        _ => panic!("invalid kind"),
    };
    let depth = std::env::var("HEART_WORKFLOW_FEEDBACK_DEPTH")
        .unwrap()
        .parse()
        .unwrap();
    let branches = std::env::var("HEART_WORKFLOW_FEEDBACK_BRANCHES")
        .unwrap()
        .parse()
        .unwrap();
    let case = Case {
        kind,
        depth,
        branches,
        count: 3,
        ids: if save { next_id } else { no_new_id },
    };
    let store = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let requests = Arc::new(Mutex::new(vec![]));
    let own = Arc::new(core::checkpoint::InMemory::new());
    let graph = compiled(case, log.clone(), requests.clone(), own.clone());
    if save {
        let workflow = Runtime::new(1001, graph)
            .unwrap()
            .with_checkpoint(store.clone())
            .with_max_concurrency(2)
            .unwrap();
        let mut events = Box::pin(
            workflow
                .run(initial(), Arc::new(next_id), Cancellation::new())
                .await
                .unwrap()
                .fuse(),
        );
        let expected = branches * if matches!(kind, Kind::Chat) { 1 } else { 2 };
        let mut ids = vec![];
        for _ in 0..expected {
            ids.push(question(&mut events).await);
        }
        workflow
            .reply(answer(ids[0], "part-one"), Cancellation::new())
            .await
            .unwrap();
        let state = settled(&mut events, &store, ids[0], 1).await;
        assert_eq!(question_ids(&state), ids);
        std::fs::write(path, serde_json::to_vec(&state).unwrap()).unwrap();
    } else {
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        store
            .save(1001, saved.clone(), &Cancellation::new())
            .await
            .unwrap();
        let ids = question_ids(&saved);
        let workflow = Runtime::from_snapshot(1001, graph, saved.clone())
            .unwrap()
            .with_checkpoint(store.clone())
            .with_max_concurrency(1)
            .unwrap();
        workflow
            .reply(answer(ids[0], "part-two"), Cancellation::new())
            .await
            .unwrap();
        let mut events = Box::pin(
            workflow
                .run(
                    initial(),
                    Arc::new(|| 1_000_000_000 + next_id()),
                    Cancellation::new(),
                )
                .await
                .unwrap()
                .fuse(),
        );
        let partial = settled(&mut events, &store, ids[0], 2).await;
        assert_eq!(question_ids(&partial), ids);
        assert!(requests.lock().unwrap().is_empty());
        assert_eq!(partial["usage"], saved["usage"]);
        for (index, id) in ids.iter().enumerate() {
            for n in if index == 0 { 2 } else { 0 }..3 {
                workflow
                    .reply(
                        answer(*id, &format!("process-{index}-{n}")),
                        Cancellation::new(),
                    )
                    .await
                    .unwrap();
                settled(&mut events, &store, *id, n + 1).await;
            }
        }
        let output = finish(&mut events).await;
        assert!(output.iter().all(Result::is_ok), "{output:?}");
        let state = store.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(state["status"], "succeeded");
        assert_eq!(state["initial"], saved["initial"]);
        assert_eq!(
            state["usage"]["total"].as_u64(),
            Some(
                saved["usage"]["total"].as_u64().unwrap()
                    + 13 * requests.lock().unwrap().len() as u64
            )
        );
        assert_eq!(log.received.lock().unwrap().len(), ids.len());
        for id in ids {
            assert_eq!(record(&state, id).unwrap()["status"], "resolved");
        }
        assert_eq!(
            own.load(1001, &Cancellation::new()).await.unwrap_err().code,
            "NOT_FOUND"
        );
        std::fs::write(path, serde_json::to_vec(&state).unwrap()).unwrap();
    }
}

#[test]
fn workflow_feedback_recovers_in_fresh_processes() {
    for kind in ["chat", "plan", "child"] {
        for depth in [0, 2] {
            for branches in [1, 2] {
                let path = std::env::temp_dir().join(format!(
                    "heart-workflow-feedback-{}-{}.json",
                    std::process::id(),
                    next_id()
                ));
                for phase in ["save", "restore"] {
                    let result = std::process::Command::new(std::env::current_exe().unwrap())
                        .args([
                            "--exact",
                            "workflow_feedback_process_worker",
                            "--ignored",
                            "--nocapture",
                        ])
                        .env("HEART_WORKFLOW_FEEDBACK_FILE", &path)
                        .env("HEART_WORKFLOW_FEEDBACK_PHASE", phase)
                        .env("HEART_WORKFLOW_FEEDBACK_KIND", kind)
                        .env("HEART_WORKFLOW_FEEDBACK_DEPTH", depth.to_string())
                        .env("HEART_WORKFLOW_FEEDBACK_BRANCHES", branches.to_string())
                        .output()
                        .unwrap();
                    assert!(
                        result.status.success(),
                        "{kind}/{depth}/{branches}/{phase}: {}\n{}",
                        String::from_utf8_lossy(&result.stdout),
                        String::from_utf8_lossy(&result.stderr)
                    );
                }
                let state: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                assert_eq!(state["status"], "succeeded");
                std::fs::remove_file(path).unwrap();
            }
        }
    }
}

struct Ask;
impl Permission for Ask {
    async fn check(&self, _: &PermissionRequest, _: &Cancellation) -> Result<Decision, Error> {
        Ok(Decision {
            behavior: Behavior::Ask,
            reason: Some("confirm the actual collect operation".into()),
        })
    }
}

fn mixed_graph(log: Arc<Log>, requests: Arc<Mutex<Vec<Request>>>) -> CompiledWorkflow {
    let mut parent = WorkflowBuilder::new("mixed-feedback-graphs");
    for (label, graph_id) in [("a", "a_branch"), ("b", "z_branch")] {
        // Same inner node name and Call ID in independent graph branches.
        let mut child = WorkflowBuilder::new(format!("mixed-child-{label}"));
        let mut agent = agent::react()
            .model(Model {
                label: label.into(),
                plan: false,
                delegate: false,
                requests: requests.clone(),
            })
            .toolkit(toolkit(log.clone(), if label == "b" { 3 } else { 1 }))
            .with_id_fn(next_id);
        if label == "a" {
            agent = agent.permission(Ask);
        }
        child
            .add_agent_node("work", agent.build().unwrap())
            .unwrap();
        child.add_edge(START, "work").unwrap();
        child.add_edge("work", END).unwrap();
        parent
            .add_graph_node(graph_id, &child.compile().unwrap())
            .unwrap();
        parent.add_edge(START, graph_id).unwrap();
    }
    let merged = log.clone();
    parent
        .add_node("merge", move |_: Messages, _: Cancellation| {
            let merged = merged.clone();
            async move {
                merged.merged.fetch_add(1, Ordering::SeqCst);
                Ok::<Messages, Error>(Messages::new([Message::function(json!({"done":true}))]))
            }
        })
        .unwrap();
    parent.add_join(["a_branch", "z_branch"], "merge").unwrap();
    parent.add_edge("merge", END).unwrap();
    parent.compile().unwrap()
}

fn approval(id: i64, approved: bool) -> Messages {
    let mut response = answer(id, "");
    response.0[0].content[0] = Part {
        r#type: "json".into(),
        data: json!({"value":{"approved":approved}})
            .as_object()
            .unwrap()
            .clone(),
    };
    response
}

#[tokio::test]
async fn mixed_approval_and_multiple_feedback_restore_across_independent_graphs() {
    for approved in [true, false] {
        for capacity in [1, 2] {
            let store = Arc::new(core::checkpoint::InMemory::new());
            let first = Runtime::new(1001, mixed_graph(Default::default(), Default::default()))
                .unwrap()
                .with_checkpoint(store.clone())
                .with_max_concurrency(2)
                .unwrap();
            let mut events = Box::pin(
                first
                    .run(initial(), Arc::new(next_id), Cancellation::new())
                    .await
                    .unwrap()
                    .fuse(),
            );
            let b = question(&mut events).await;
            let a = question(&mut events).await;
            let initial_scene = store.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(question_ids(&initial_scene), [b, a]);
            assert_eq!(record(&initial_scene, b).unwrap()["kind"], "input");
            assert_eq!(record(&initial_scene, a).unwrap()["kind"], "approval");
            first
                .reply(answer(b, "one"), Cancellation::new())
                .await
                .unwrap();
            let saved = settled(&mut events, &store, b, 1).await;
            drop(events);
            drop(first);
            let log = Arc::new(Log::default());
            let requests = Arc::new(Mutex::new(vec![]));
            let restored =
                Runtime::from_snapshot(1001, mixed_graph(log.clone(), requests.clone()), saved)
                    .unwrap()
                    .with_checkpoint(store.clone())
                    .with_max_concurrency(capacity)
                    .unwrap();
            assert_eq!(
                restored
                    .reply(approval(a, approved), Cancellation::new())
                    .await
                    .unwrap_err()
                    .code,
                "CONFLICT"
            );
            restored
                .reply(answer(b, "two"), Cancellation::new())
                .await
                .unwrap();
            let mut events = Box::pin(
                restored
                    .run(initial(), Arc::new(next_id), Cancellation::new())
                    .await
                    .unwrap()
                    .fuse(),
            );
            settled(&mut events, &store, b, 2).await;
            assert!(requests.lock().unwrap().is_empty());
            restored
                .reply(answer(b, "three"), Cancellation::new())
                .await
                .unwrap();
            let after_b = settled(&mut events, &store, b, 3).await;
            assert_eq!(question_ids(&after_b), [a]);
            let duplicate_requests = Arc::new(Mutex::new(vec![]));
            let duplicate = Runtime::from_snapshot(
                1001,
                mixed_graph(Default::default(), duplicate_requests.clone()),
                after_b.clone(),
            )
            .unwrap();
            let duplicate_before = duplicate.snapshot().unwrap();
            duplicate
                .reply(answer(b, "three"), Cancellation::new())
                .await
                .unwrap();
            assert_eq!(
                duplicate
                    .reply(answer(b, "changed"), Cancellation::new())
                    .await
                    .unwrap_err()
                    .code,
                "CONFLICT"
            );
            assert_eq!(duplicate.snapshot().unwrap(), duplicate_before);
            assert!(duplicate_requests.lock().unwrap().is_empty());
            restored
                .reply(approval(a, approved), Cancellation::new())
                .await
                .unwrap();
            if approved {
                let input = question(&mut events).await;
                // Restoring a pending approval can present it again before
                // its answer is consumed; that is still the original ID.
                let input = if input == a {
                    question(&mut events).await
                } else {
                    input
                };
                assert_ne!(input, a);
                assert_ne!(input, b);
                restored
                    .reply(
                        answer(input, "approved-operation-feedback"),
                        Cancellation::new(),
                    )
                    .await
                    .unwrap();
                settled(&mut events, &store, input, 1).await;
            }
            let output = finish(&mut events).await;
            assert!(
                output.iter().all(Result::is_ok),
                "{approved}/{capacity}: {output:?}"
            );
            let state = store.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(state["status"], "succeeded");
            assert_eq!(state["waits"], json!([]));
            assert_eq!(
                record(&state, a).unwrap()["feedback"][0]["content"][0]["data"]["value"]["approved"],
                approved
            );
            assert_eq!(
                log.received.lock().unwrap().len(),
                if approved { 2 } else { 1 }
            );
            assert_eq!(log.merged.load(Ordering::SeqCst), 1);
        }
    }
}
