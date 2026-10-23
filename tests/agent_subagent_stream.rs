//! I03: the parent consumes the same native events while children are running.
use futures_core::Stream;
use futures_util::{StreamExt, stream};
use halo::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, SubAgent},
    checkpoint::Checkpoint as _,
    core,
    event::Event,
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
};
use serde_json::json;
use std::{
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicI64, AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::Notify;

fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(800_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}

fn text(role: Role, kind: &str, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: kind.into(),
        data: json!({"value":value}).as_object().unwrap().clone(),
    });
    message
}

fn complete(message: Message, tokens: u64) -> Event {
    Event::Complete {
        message,
        usage: Some(Usage {
            input: Some(tokens),
            output: Some(0),
            total: Some(tokens),
            ..Default::default()
        }),
        finish_reason: Some("stop".into()),
    }
}

fn input() -> Input {
    serde_json::from_value(json!({"metadata":{"chat_id":501,"session_id":601},"payload":[{"type":"text","data":{"value":"parent query"}}]})).unwrap()
}

fn delegation(names: &[&str]) -> Message {
    let mut message = Message::new(Role::Assistant);
    for (idx, name) in names.iter().enumerate() {
        message.content.push(Part { r#type:"tool_call".into(), data:json!({
            "call_id":format!("delegate_{idx}"), "name":"agent_spawn",
            "arguments":{"name":name,"input":{"payload":[{"type":"text","data":{"value":"child query"}}]}}
        }).as_object().unwrap().clone() });
    }
    message
}

struct ParentModel {
    first: Message,
    requests: Arc<Mutex<Vec<Request>>>,
}
impl Chat for ParentModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut requests = self.requests.lock().unwrap();
        let message = if requests.is_empty() {
            self.first.clone()
        } else {
            text(Role::Assistant, "text", "parent final")
        };
        requests.push(request);
        Ok(stream::iter(vec![Ok(complete(message, 3))]))
    }
}

struct Probe(Arc<AtomicUsize>);
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct GatedModel {
    name: String,
    gate: Arc<Notify>,
    dropped: Arc<AtomicUsize>,
}
impl Chat for GatedModel {
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        let state = (
            0,
            self.name.clone(),
            self.gate.clone(),
            Probe(self.dropped.clone()),
        );
        Ok(Box::pin(stream::unfold(
            state,
            |(phase, name, gate, probe)| async move {
                let event = match phase {
                    0 => Event::Delta {
                        message: text(Role::Assistant, "think", &format!("{name} thinking")),
                    },
                    1 => Event::Delta {
                        message: text(Role::Assistant, "text", &format!("{name} partial")),
                    },
                    2 => {
                        gate.notified().await;
                        complete(text(Role::Assistant, "text", &format!("{name} final")), 5)
                    }
                    _ => return None,
                };
                Some((Ok(event), (phase + 1, name, gate, probe)))
            },
        )))
    }
}

type TestStream = Pin<Box<dyn Stream<Item = Result<Event, Error>>>>;
async fn start(
    react: bool,
    model: ParentModel,
    children: Vec<SubAgent>,
    checkpoint: Arc<core::checkpoint::InMemory>,
) -> TestStream {
    if react {
        Box::pin(
            agent::react()
                .model(model)
                .subagents(children)
                .with_id_fn(next_id)
                .checkpoint(checkpoint)
                .build()
                .unwrap()
                .run(input(), true, Cancellation::new())
                .await
                .unwrap(),
        )
    } else {
        Box::pin(
            agent::chat()
                .model(model)
                .subagents(children)
                .with_id_fn(next_id)
                .checkpoint(checkpoint)
                .build()
                .unwrap()
                .run(input(), true, Cancellation::new())
                .await
                .unwrap(),
        )
    }
}

async fn next(events: &mut TestStream) -> Result<Event, Error> {
    tokio::time::timeout(Duration::from_secs(2), events.next())
        .await
        .expect("child event must not wait for child completion")
        .expect("event stream remains open")
}

fn event_message(event: &Event) -> &Message {
    match event {
        Event::Delta { message } | Event::Complete { message, .. } => message,
    }
}

fn assert_source(event: &Event) -> (String, i64) {
    let metadata = &event_message(event).metadata;
    assert_eq!(metadata["chat_id"], 501);
    (
        metadata["node_id"].as_str().unwrap().to_owned(),
        metadata["exec_id"].as_i64().unwrap(),
    )
}

#[tokio::test]
async fn real_chat_and_react_children_stream_before_finishing_without_releasing_parent() {
    for parent_react in [false, true] {
        for child_react in [false, true] {
            let gate = Arc::new(Notify::new());
            let dropped = Arc::new(AtomicUsize::new(0));
            let model = GatedModel {
                name: "reviewer".into(),
                gate: gate.clone(),
                dropped: dropped.clone(),
            };
            let child = if child_react {
                SubAgent::new("reviewer", agent::react().model(model).build().unwrap()).unwrap()
            } else {
                SubAgent::new("reviewer", agent::chat().model(model).build().unwrap()).unwrap()
            };
            let requests = Arc::new(Mutex::new(Vec::new()));
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let mut events = start(
                parent_react,
                ParentModel {
                    first: delegation(&["reviewer"]),
                    requests: requests.clone(),
                },
                vec![child],
                checkpoint.clone(),
            )
            .await;
            assert!(matches!(
                next(&mut events).await.unwrap(),
                Event::Complete { .. }
            ));
            let think = next(&mut events).await.unwrap();
            assert!(matches!(&think,Event::Delta {message} if message.content[0].r#type=="think"));
            let source = assert_source(&think);
            let partial = next(&mut events).await.unwrap();
            assert_eq!(assert_source(&partial), source);
            assert_eq!(
                event_message(&partial).content[0].data["value"],
                "reviewer partial"
            );
            assert_eq!(
                requests.lock().unwrap().len(),
                1,
                "partial child events cannot start parent inference"
            );
            assert!(
                tokio::time::timeout(Duration::from_millis(20), events.next())
                    .await
                    .is_err()
            );
            gate.notify_one();
            let child_final = next(&mut events).await.unwrap();
            assert_eq!(assert_source(&child_final), source);
            let rest = events.collect::<Vec<_>>().await;
            assert!(rest.iter().all(Result::is_ok), "{rest:?}");
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
            let requests = requests.lock().unwrap().clone();
            assert_eq!(requests.len(), 2);
            let result = requests[1]
                .messages
                .0
                .iter()
                .find(|m| m.role == Role::Function)
                .unwrap();
            assert_eq!(result.content[0].data["value"], "reviewer final");
            assert!(requests[1].messages.0.iter().all(
                |m| !m.metadata.contains_key("node_id") && !m.metadata.contains_key("exec_id")
            ));
            assert!(
                !serde_json::to_string(&requests[1].messages)
                    .unwrap()
                    .contains("reviewer thinking")
            );
            drop(requests);
            let scene = checkpoint.load(501, &Cancellation::new()).await.unwrap();
            assert_eq!(
                scene["usage"]["total"], 11,
                "3 parent + 5 child + 3 parent, without forwarding double-count"
            );
            let child = &scene["execution"]["attempts"][0]["nodes"][&source.0];
            assert_eq!(child["status"], "succeeded");
            assert_eq!(child["execution"]["exec_id"], source.1);
            assert_eq!(child["usage"]["total"], 5);
        }
    }
}

#[tokio::test]
async fn concurrent_children_emit_live_and_parent_context_keeps_selection_order() {
    for react in [false, true] {
        let mut gates = Vec::new();
        let children = ["first", "second"]
            .into_iter()
            .map(|name| {
                let gate = Arc::new(Notify::new());
                gates.push(gate.clone());
                SubAgent::new(
                    name,
                    agent::chat()
                        .model(GatedModel {
                            name: name.into(),
                            gate,
                            dropped: Default::default(),
                        })
                        .build()
                        .unwrap(),
                )
                .unwrap()
            })
            .collect();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let mut events = start(
            react,
            ParentModel {
                first: delegation(&["first", "second"]),
                requests: requests.clone(),
            },
            children,
            checkpoint.clone(),
        )
        .await;
        next(&mut events).await.unwrap();
        let mut sources = std::collections::BTreeMap::new();
        for _ in 0..4 {
            let event = next(&mut events).await.unwrap();
            assert!(matches!(&event, Event::Delta { .. }));
            let value = event_message(&event).content[0].data["value"]
                .as_str()
                .unwrap();
            sources.insert(
                value.split_whitespace().next().unwrap().to_owned(),
                assert_source(&event),
            );
        }
        assert_eq!(sources.len(), 2);
        assert_ne!(sources["first"], sources["second"]);
        gates[1].notify_one();
        let second = next(&mut events).await.unwrap();
        assert_eq!(
            event_message(&second).content[0].data["value"],
            "second final"
        );
        assert_eq!(requests.lock().unwrap().len(), 1);
        gates[0].notify_one();
        let rest = events.collect::<Vec<_>>().await;
        assert!(rest.iter().all(Result::is_ok), "{rest:?}");
        let requests = requests.lock().unwrap().clone();
        let results = requests[1]
            .messages
            .0
            .iter()
            .filter(|m| m.role == Role::Function)
            .collect::<Vec<_>>();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content[0].data["value"], "first final");
        assert_eq!(results[1].content[0].data["value"], "second final");
        drop(requests);
        assert_eq!(
            checkpoint.load(501, &Cancellation::new()).await.unwrap()["usage"]["total"],
            16
        );
    }
}

struct WaitingChild;
impl Agent for WaitingChild {
    async fn reply(
        &self,
        _: halo::agent::Input,
        _: halo::Cancellation,
    ) -> Result<(), halo::Error> {
        Err(halo::Error::new(
            "NOT_FOUND",
            "fixture has no live feedback consumer",
        ))
    }
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn run(&self, _: Input, _stream: bool, _: Cancellation) -> Result<Self::Stream, Error> {
        Ok(stream::iter(vec![
            Ok(Event::Delta {
                message: text(Role::Assistant, "text", "before waiting"),
            }),
            Err(Error::new("WAITING", "needs approval")),
        ]))
    }
}

#[tokio::test]
async fn child_delta_is_not_lost_when_the_child_then_waits() {
    for react in [false, true] {
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let mut events = start(
            react,
            ParentModel {
                first: delegation(&["reviewer"]),
                requests: requests.clone(),
            },
            vec![SubAgent::new("reviewer", WaitingChild).unwrap()],
            checkpoint.clone(),
        )
        .await;
        next(&mut events).await.unwrap();
        let delta = next(&mut events).await.unwrap();
        assert_eq!(
            event_message(&delta).content[0].data["value"],
            "before waiting"
        );
        let source = assert_source(&delta);
        assert_eq!(next(&mut events).await.unwrap_err().code, "WAITING");
        assert_eq!(requests.lock().unwrap().len(), 1);
        let scene = checkpoint.load(501, &Cancellation::new()).await.unwrap();
        assert_eq!(
            scene["execution"]["attempts"][0]["nodes"][&source.0]["status"],
            "waiting"
        );
        assert!(
            !serde_json::to_string(&scene["messages"])
                .unwrap()
                .contains("before waiting")
        );
    }
}

#[tokio::test]
async fn dropping_the_parent_stream_releases_the_child_without_background_producers() {
    for react in [false, true] {
        let dropped = Arc::new(AtomicUsize::new(0));
        let child = SubAgent::new(
            "reviewer",
            agent::chat()
                .model(GatedModel {
                    name: "reviewer".into(),
                    gate: Default::default(),
                    dropped: dropped.clone(),
                })
                .build()
                .unwrap(),
        )
        .unwrap();
        let mut events = start(
            react,
            ParentModel {
                first: delegation(&["reviewer"]),
                requests: Default::default(),
            },
            vec![child],
            Arc::new(core::checkpoint::InMemory::new()),
        )
        .await;
        next(&mut events).await.unwrap();
        next(&mut events).await.unwrap();
        drop(events);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}

struct SourcedChild;
impl Agent for SourcedChild {
    async fn reply(
        &self,
        _: halo::agent::Input,
        _: halo::Cancellation,
    ) -> Result<(), halo::Error> {
        Err(halo::Error::new(
            "NOT_FOUND",
            "fixture has no live feedback consumer",
        ))
    }
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn run(&self, _: Input, _stream: bool, _: Cancellation) -> Result<Self::Stream, Error> {
        let mut message = text(Role::Assistant, "text", "descendant");
        message
            .metadata
            .insert("node_id".into(), json!("nested_real_node"));
        message.metadata.insert("exec_id".into(), json!(778));
        Ok(stream::iter(vec![
            Ok(Event::Delta {
                message: message.clone(),
            }),
            Ok(complete(message, 5)),
        ]))
    }
}

#[tokio::test]
async fn forwarding_preserves_real_descendant_source_instead_of_overwriting_with_container() {
    for react in [false, true] {
        let mut events = start(
            react,
            ParentModel {
                first: delegation(&["reviewer"]),
                requests: Default::default(),
            },
            vec![SubAgent::new("reviewer", SourcedChild).unwrap()],
            Arc::new(core::checkpoint::InMemory::new()),
        )
        .await;
        next(&mut events).await.unwrap();
        assert_eq!(
            assert_source(&next(&mut events).await.unwrap()),
            ("nested_real_node".into(), 778)
        );
        assert_eq!(
            assert_source(&next(&mut events).await.unwrap()),
            ("nested_real_node".into(), 778)
        );
        let rest = events.collect::<Vec<_>>().await;
        assert!(rest.iter().all(Result::is_ok));
    }
}

struct PlannedModel(Arc<Mutex<Vec<Request>>>);
impl Chat for PlannedModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let planned = request
            .messages
            .0
            .iter()
            .any(|m| m.values.contains_key("plan_node"));
        let mut message = if planned {
            if request
                .messages
                .0
                .iter()
                .any(|m| m.metadata.get("name") == Some(&json!("agent_spawn")))
            {
                text(Role::Assistant, "text", "task final")
            } else {
                delegation(&["reviewer"])
            }
        } else if request
            .messages
            .0
            .last()
            .unwrap()
            .values
            .contains_key("plan_complete")
        {
            text(Role::Assistant, "text", "parent final")
        } else if request.tools.is_empty() {
            let mut message = Message::new(Role::Assistant);
            message.values = json!({"version":1,"nodes":{"a":{"name":"research","objective":"delegate review"}},"edges":[]}).as_object().unwrap().clone();
            message
        } else {
            let mut message = Message::new(Role::Assistant);
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"plan_1","name":"plan","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
            message
        };
        if planned {
            message.values.insert("task_answer".into(), json!(true));
        }
        self.0.lock().unwrap().push(request);
        Ok(stream::iter(vec![Ok(complete(message, 3))]))
    }
}

#[tokio::test]
async fn planned_task_forwards_live_child_events_but_commits_only_its_own_outputs() {
    let gate = Arc::new(Notify::new());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let child = SubAgent::new(
        "reviewer",
        agent::chat()
            .model(GatedModel {
                name: "reviewer".into(),
                gate: gate.clone(),
                dropped: Default::default(),
            })
            .build()
            .unwrap(),
    )
    .unwrap();
    let parent = agent::react()
        .model(PlannedModel(requests.clone()))
        .subagents([child])
        .with_id_fn(next_id)
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let mut events: TestStream = Box::pin(
        parent
            .run(input(), true, Cancellation::new())
            .await
            .unwrap(),
    );
    let source = loop {
        let event = next(&mut events).await.unwrap();
        if matches!(&event,Event::Delta {message} if message.content[0].r#type=="think") {
            break assert_source(&event);
        }
    };
    assert_ne!(source.0, "a");
    assert_eq!(
        requests.lock().unwrap().len(),
        3,
        "no task or parent inference from partial child events"
    );
    let partial = next(&mut events).await.unwrap();
    assert_eq!(assert_source(&partial), source);
    gate.notify_one();
    let child_complete = next(&mut events).await.unwrap();
    assert_eq!(assert_source(&child_complete), source);
    let rest = events.collect::<Vec<_>>().await;
    assert!(rest.iter().all(Result::is_ok), "{rest:?}");
    let scene = checkpoint.load(501, &Cancellation::new()).await.unwrap();
    let node = &scene["nodes"]["a"];
    assert_eq!(node["status"], "succeeded");
    let messages: halo::Messages = serde_json::from_value(node["messages"].clone()).unwrap();
    let child_results = messages
        .0
        .iter()
        .filter(|m| {
            m.content
                .iter()
                .any(|p| p.data.get("value") == Some(&json!("reviewer final")))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        child_results.len(),
        1,
        "forwarded child Complete is not another task output"
    );
    assert_eq!(child_results[0].role, Role::Function);
    assert!(
        !serde_json::to_string(&messages)
            .unwrap()
            .contains("reviewer thinking")
    );
    let child = &node["execution"]["attempts"][0]["nodes"][&source.0];
    assert_eq!(child["execution"]["exec_id"], source.1);
    assert_eq!(child["status"], "succeeded");
    assert_eq!(scene["usage"]["total"], 20);
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 5);
    assert!(requests.iter().all(|r| {
        r.messages
            .0
            .iter()
            .all(|m| !m.metadata.contains_key("node_id") && !m.metadata.contains_key("exec_id"))
    }));
}

fn embedded<A: Agent + 'static>(agent: A) -> halo::runtime::workflow::CompiledWorkflow {
    use halo::runtime::workflow::{
        WorkflowBuilder,
        graph::{END, START},
    };
    let mut builder = WorkflowBuilder::new("live-embedded-agent");
    builder.add_agent_node("agent", agent).unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let mut compiled = builder.compile().unwrap();
    for level in 0..2 {
        let mut builder = WorkflowBuilder::new(format!("wrapper-{level}"));
        builder.add_graph_node("child", &compiled).unwrap();
        builder.add_edge(START, "child").unwrap();
        builder.add_edge("child", END).unwrap();
        compiled = builder.compile().unwrap();
    }
    compiled
}

#[tokio::test]
async fn nested_workflow_forwards_the_same_live_child_source_without_extra_subscription() {
    for react in [false, true] {
        let gate = Arc::new(Notify::new());
        let child = SubAgent::new(
            "reviewer",
            agent::chat()
                .model(GatedModel {
                    name: "reviewer".into(),
                    gate: gate.clone(),
                    dropped: Default::default(),
                })
                .build()
                .unwrap(),
        )
        .unwrap();
        let model = ParentModel {
            first: delegation(&["reviewer"]),
            requests: Default::default(),
        };
        let graph = if react {
            embedded(
                agent::react()
                    .model(model)
                    .subagents([child])
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
            )
        } else {
            embedded(
                agent::chat()
                    .model(model)
                    .subagents([child])
                    .with_id_fn(next_id)
                    .build()
                    .unwrap(),
            )
        };
        let mut runtime = halo::runtime::workflow::Runtime::new(501, graph).unwrap();
        let mut initial = Message::new(Role::User);
        initial.values = json!({"metadata":input().metadata,"payload":input().payload})
            .as_object()
            .unwrap()
            .clone();
        let mut events = Box::pin(
            runtime
                .run(
                    halo::Messages::new([initial]),
                    Arc::new(next_id),
                    Cancellation::new(),
                )
                .await
                .unwrap(),
        );
        let source = loop {
            let event = tokio::time::timeout(Duration::from_secs(2), events.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            if matches!(&event,Event::Delta {message} if message.content[0].r#type=="think") {
                break assert_source(&event);
            }
        };
        assert_ne!(source.0, "child");
        assert_ne!(source.0, "agent");
        let partial = events.next().await.unwrap().unwrap();
        assert_eq!(assert_source(&partial), source);
        gate.notify_one();
        let child_complete = events.next().await.unwrap().unwrap();
        assert_eq!(assert_source(&child_complete), source);
        let rest = events.collect::<Vec<_>>().await;
        assert!(rest.iter().all(Result::is_ok), "{rest:?}");
        assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(11));
    }
}

struct MultiStageChild(Arc<Notify>);
impl Agent for MultiStageChild {
    async fn reply(
        &self,
        _: halo::agent::Input,
        _: halo::Cancellation,
    ) -> Result<(), halo::Error> {
        Err(halo::Error::new(
            "NOT_FOUND",
            "fixture has no live feedback consumer",
        ))
    }
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;
    async fn run(&self, _: Input, _stream: bool, _: Cancellation) -> Result<Self::Stream, Error> {
        Ok(Box::pin(stream::unfold(
            (0, self.0.clone()),
            |(phase, gate)| async move {
                let event = match phase {
                    0 => complete(text(Role::Assistant, "text", "intermediate child model"), 2),
                    1 => Event::Complete {
                        message: Message::function(json!({"tool":"complete"})),
                        usage: None,
                        finish_reason: Some("tool".into()),
                    },
                    2 => Event::Delta {
                        message: text(Role::Assistant, "text", "next child model partial"),
                    },
                    3 => {
                        gate.notified().await;
                        complete(text(Role::Assistant, "text", "child task final"), 3)
                    }
                    _ => return None,
                };
                Some((Ok(event), (phase + 1, gate)))
            },
        )))
    }
}

#[tokio::test]
async fn intermediate_model_and_tool_completes_are_live_observations_not_child_terminal() {
    for react in [false, true] {
        let gate = Arc::new(Notify::new());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let mut events = start(
            react,
            ParentModel {
                first: delegation(&["reviewer"]),
                requests: requests.clone(),
            },
            vec![SubAgent::new("reviewer", MultiStageChild(gate.clone())).unwrap()],
            checkpoint.clone(),
        )
        .await;
        next(&mut events).await.unwrap();
        let model = next(&mut events).await.unwrap();
        assert!(matches!(&model, Event::Complete { .. }));
        let source = assert_source(&model);
        let tool = next(&mut events).await.unwrap();
        assert!(matches!(&tool,Event::Complete {finish_reason:Some(reason),..} if reason=="tool"));
        assert_eq!(assert_source(&tool), source);
        let partial = next(&mut events).await.unwrap();
        assert!(matches!(&partial, Event::Delta { .. }));
        assert_eq!(
            requests.lock().unwrap().len(),
            1,
            "internal Complete does not release the parent"
        );
        gate.notify_one();
        let rest = events.collect::<Vec<_>>().await;
        assert!(rest.iter().all(Result::is_ok));
        let requests = requests.lock().unwrap().clone();
        let results = requests[1]
            .messages
            .0
            .iter()
            .filter(|m| m.role == Role::Function)
            .collect::<Vec<_>>();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content[0].data["value"], "child task final");
        assert!(
            !serde_json::to_string(&requests[1].messages)
                .unwrap()
                .contains("intermediate child model")
        );
        drop(requests);
        assert_eq!(
            checkpoint.load(501, &Cancellation::new()).await.unwrap()["usage"]["total"],
            11
        );
    }
}
