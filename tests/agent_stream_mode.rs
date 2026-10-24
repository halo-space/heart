//! A run chooses outward delivery once; every actual call Complete survives.
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicI64, Ordering},
    },
};

use futures_core::Stream;
use futures_util::{StreamExt, poll, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::checkpoint::Checkpoint,
    agent::event::Event,
    agent::message::Role,
    agent::model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
    agent::tool::{Call, Tool, ToolDefinition},
    agent::{self, Agent, Input, SubAgent},
    core,
};
use serde_json::json;

fn input() -> Input {
    serde_json::from_value(json!({
        "metadata":{"chat_id":1001},
        "payload":[{"type":"text","data":{"value":"question"}}]
    }))
    .unwrap()
}

fn text(value: &str) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":value}).as_object().unwrap().clone(),
    });
    message
}

#[derive(Clone, Default)]
struct Model(Arc<Mutex<Vec<Request>>>);

impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let first = {
            let mut requests = self.0.lock().unwrap();
            requests.push(request);
            requests.len() == 1
        };
        let mut message = text("complete");
        if first {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"c1","name":"lookup","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        }
        // Deliberately emit Delta even when the Provider was asked for a full
        // result: run's own outward boundary must still enforce its mode.
        Ok(stream::iter(vec![
            Ok(Event::Delta {
                message: text("partial"),
            }),
            Ok(Event::Complete {
                message,
                usage: Some(Usage {
                    input: Some(10),
                    output: Some(3),
                    total: Some(13),
                    ..Default::default()
                }),
                finish_reason: Some(if first { "tool_calls" } else { "stop" }.into()),
            }),
        ]))
    }
}

struct ScriptedModel {
    replies: Mutex<VecDeque<Message>>,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl ScriptedModel {
    fn new(replies: Vec<Message>) -> Self {
        Self {
            replies: Mutex::new(replies.into()),
            requests: Default::default(),
        }
    }
}

impl Chat for ScriptedModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.requests.lock().unwrap().push(request);
        let message = self.replies.lock().unwrap().pop_front().unwrap();
        Ok(stream::iter(vec![
            Ok(Event::Delta {
                message: text("partial"),
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

fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(1_000_000);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn selected(name: &str, arguments: serde_json::Value) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "tool_call".into(),
        data: json!({"call_id":"c1","name":name,"arguments":arguments})
            .as_object()
            .unwrap()
            .clone(),
    });
    message
}

struct Lookup;

impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: LazyLock<ToolDefinition> = LazyLock::new(|| ToolDefinition {
            name: "lookup".into(),
            description: "lookup".into(),
            parameters: json!({"type":"object","properties":{}})
                .as_object()
                .unwrap()
                .clone(),
        });
        &DEFINITION
    }

    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        Ok(Message::function(json!({"found": true})))
    }
}

async fn collect<A: Agent>(agent: A, mode: bool) -> Vec<Event> {
    agent
        .run(input(), mode, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<_, _>>()
        .unwrap()
}

#[tokio::test]
async fn both_modes_retain_each_model_and_tool_complete_and_same_usage() {
    for react in [false, true] {
        for mode in [false, true] {
            let model = Model::default();
            let requests = model.0.clone();
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let capabilities = vec![Lookup];
            let events = if react {
                collect(
                    agent::react()
                        .model(model)
                        .tools(capabilities)
                        .checkpoint(checkpoint.clone())
                        .build()
                        .unwrap(),
                    mode,
                )
                .await
            } else {
                collect(
                    agent::chat()
                        .model(model)
                        .tools(capabilities)
                        .checkpoint(checkpoint.clone())
                        .build()
                        .unwrap(),
                    mode,
                )
                .await
            };
            {
                let requests = requests.lock().unwrap();
                assert_eq!(requests.len(), 2);
                assert!(requests.iter().all(|request| request.stream == mode));
            }
            let deltas = events
                .iter()
                .filter(|e| matches!(e, Event::Delta { .. }))
                .count();
            assert_eq!(deltas, if mode { 2 } else { 0 });
            let completions = events
                .iter()
                .filter(|e| matches!(e, Event::Complete { .. }))
                .count();
            assert_eq!(completions, 3, "do not collapse intermediate results");
            let model_usage: u64 = events
                .iter()
                .filter_map(|event| match event {
                    Event::Complete { usage, .. } => usage.as_ref().and_then(|usage| usage.total),
                    Event::Delta { .. } => None,
                })
                .sum();
            assert_eq!(model_usage, 26);
            let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(state["status"], "succeeded");
            assert_eq!(state["usage"]["total"], 26);
        }
    }
}

#[tokio::test]
async fn planned_task_model_requests_inherit_the_run_mode() {
    for mode in [false, true] {
        let mut plan = Message::new(Role::Assistant);
        plan.values = json!({
            "version":1,
            "nodes":{"a":{"name":"research","objective":"research question"}},
            "edges":[]
        })
        .as_object()
        .unwrap()
        .clone();
        let model = ScriptedModel::new(vec![
            selected("plan", json!({})),
            plan,
            text("task result"),
            text("final"),
        ]);
        let requests = model.requests.clone();
        let events = collect(
            agent::react()
                .model(model)
                .with_id_fn(next_id)
                .build()
                .unwrap(),
            mode,
        )
        .await;
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert!(requests.iter().all(|request| request.stream == mode));
        assert!(requests[2].messages.as_slice().iter().any(|message| {
            message.values.get("objective") == Some(&json!("research question"))
        }));
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Event::Delta { .. }))
                .count(),
            if mode { 4 } else { 0 }
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Event::Complete { .. }))
                .count(),
            5
        );
        assert_eq!(
            events
                .iter()
                .filter_map(|event| match event {
                    Event::Complete { usage, .. } => usage.as_ref().and_then(|usage| usage.total),
                    Event::Delta { .. } => None,
                })
                .sum::<u64>(),
            52
        );
    }
}

#[tokio::test]
async fn non_streaming_parent_retains_child_completes_without_child_deltas() {
    for parent_react in [false, true] {
        for child_react in [false, true] {
            let child_model = ScriptedModel::new(vec![text("child final")]);
            let child = if child_react {
                SubAgent::new(
                    "reviewer",
                    agent::react().model(child_model).build().unwrap(),
                )
            } else {
                SubAgent::new(
                    "reviewer",
                    agent::chat().model(child_model).build().unwrap(),
                )
            }
            .unwrap();
            let model = ScriptedModel::new(vec![
                selected(
                    "agent_spawn",
                    json!({
                        "name":"reviewer",
                        "input":{"payload":[{"type":"text","data":{"value":"review"}}]}
                    }),
                ),
                text("parent final"),
            ]);
            let events = if parent_react {
                collect(
                    agent::react()
                        .model(model)
                        .subagents([child])
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                    false,
                )
                .await
            } else {
                collect(
                    agent::chat()
                        .model(model)
                        .subagents([child])
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                    false,
                )
                .await
            };
            assert!(
                events
                    .iter()
                    .all(|event| matches!(event, Event::Complete { .. }))
            );
            assert_eq!(events.len(), 4);
            assert!(events.iter().any(|event| {
                matches!(event, Event::Complete { message, .. }
                    if message.content.first().is_some_and(|part| part.data.get("value") == Some(&json!("child final"))))
            }));
            assert_eq!(
                events
                    .iter()
                    .filter_map(|event| match event {
                        Event::Complete { usage, .. } =>
                            usage.as_ref().and_then(|usage| usage.total),
                        Event::Delta { .. } => None,
                    })
                    .sum::<u64>(),
                39
            );
        }
    }
}

struct PendingModel;

impl Chat for PendingModel {
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;

    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        Ok(Box::pin(
            stream::iter(vec![Ok(Event::Delta {
                message: text("partial"),
            })])
            .chain(stream::pending()),
        ))
    }
}

async fn cancel<A: Agent>(agent: A) {
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(input(), false, cancellation.clone())
            .await
            .unwrap(),
    );
    let mut next = Box::pin(events.next());
    assert!(poll!(next.as_mut()).is_pending());
    cancellation.cancel();
    let Event::Complete {
        message,
        usage,
        finish_reason,
    } = next.await.unwrap().unwrap()
    else {
        panic!("non-streaming cancellation still closes with Complete");
    };
    assert_eq!(finish_reason.as_deref(), Some("cancelled"));
    assert_eq!(message.content[0].data["value"], "partial");
    assert!(usage.is_some());
    assert!(events.next().await.is_none());
}

#[tokio::test]
async fn non_streaming_cancellation_keeps_partial_content_and_usage() {
    cancel(agent::chat().model(PendingModel).build().unwrap()).await;
    cancel(agent::react().model(PendingModel).build().unwrap()).await;
}

struct FailingModel;

impl Chat for FailingModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        Ok(stream::iter(vec![
            Ok(Event::Delta {
                message: text("partial"),
            }),
            Err(Error::new("TRANSPORT", "connection lost").with_details("original diagnostic")),
        ]))
    }
}

async fn failure<A: Agent>(agent: A) {
    let events = agent
        .run(input(), false, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(events.len(), 1, "suppress Delta, never suppress the error");
    let error = events.into_iter().next().unwrap().unwrap_err();
    assert_eq!(error.code, "TRANSPORT");
    assert_eq!(error.message, "connection lost");
    assert_eq!(error.details.as_deref(), Some("original diagnostic"));
}

#[tokio::test]
async fn non_streaming_preserves_direct_errors_without_fabricated_complete() {
    failure(agent::chat().model(FailingModel).build().unwrap()).await;
    failure(agent::react().model(FailingModel).build().unwrap()).await;
}

async fn reject_wait<A: Agent>(agent: A, checkpoint: Arc<core::checkpoint::InMemory>) {
    let events = agent
        .run(input(), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    let mut snapshot = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    snapshot["status"] = json!("waiting");
    // Corrupt only the waiting record, not the independently validated native
    // model lifecycle facts, so this still tests the intended error boundary.
    snapshot["steps"]
        .as_array_mut()
        .unwrap()
        .last_mut()
        .unwrap()["waits"] = serde_json::to_value(std::collections::BTreeMap::from([(
        5001,
        halo::runtime::agent::node::Wait {
            count: 1,
            kind: "ask".into(),
            status: "received".into(),
            question: text("question"),
            feedback: Default::default(),
        },
    )]))
    .unwrap();
    checkpoint
        .save(1001, snapshot.clone(), &Cancellation::new())
        .await
        .unwrap();
    let error = match agent.run(input(), true, Cancellation::new()).await {
        Ok(_) => panic!("invalid wait must not dispatch another model call"),
        Err(error) => error,
    };
    assert_eq!(error.code, "INVALID_STATE");
    assert_eq!(error.message, "invalid Agent Step waiting record");
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        snapshot
    );
}

#[tokio::test]
async fn recovery_rejects_unapproved_wait_status_without_overwriting_scene() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    reject_wait(
        agent::chat()
            .model(CompleteModel)
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap(),
        checkpoint,
    )
    .await;
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    reject_wait(
        agent::react()
            .model(CompleteModel)
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap(),
        checkpoint,
    )
    .await;
}

struct CompleteModel;

impl Chat for CompleteModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: text("done"),
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}
