//! Deterministic admission, dependency and cancellation checks for ReAct plans.
//! Gates control progress without sleeps, threads or paid model calls.
use std::{
    collections::BTreeMap,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicI64, Ordering},
    },
    task::{Context, Poll},
};

use futures_core::Stream;
use futures_executor::block_on;
use futures_util::{StreamExt, poll};
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::checkpoint::Checkpoint as _,
    agent::event::Event,
    agent::message::Role,
    agent::model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
    agent::{self, Agent, Input, Payload},
    runtime::workflow::{
        Runtime, WorkflowBuilder,
        graph::{END, START},
    },
};
use serde_json::{Value, json};
use tokio::sync::oneshot;

#[derive(Default)]
struct Probe {
    requests: Vec<Request>,
    started: Vec<String>,
    active: usize,
    peak: usize,
    gates: BTreeMap<String, oneshot::Receiver<Result<(), Error>>>,
}
type Shared = Arc<Mutex<Probe>>;
type Gates = BTreeMap<String, oneshot::Sender<Result<(), Error>>>;

struct Active(Shared);
impl Drop for Active {
    fn drop(&mut self) {
        self.0.lock().unwrap().active -= 1;
    }
}

struct Events {
    gate: Option<oneshot::Receiver<Result<(), Error>>>,
    event: Option<Result<Event, Error>>,
    _active: Option<Active>,
}
impl Stream for Events {
    type Item = Result<Event, Error>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(gate) = self.gate.as_mut() {
            match Pin::new(gate).poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(Ok(()))) => self.gate = None,
                Poll::Ready(Ok(Err(error))) => {
                    self.gate = None;
                    self.event = None;
                    return Poll::Ready(Some(Err(error)));
                }
                Poll::Ready(Err(_)) => panic!("test gate dropped before resolution"),
            }
        }
        Poll::Ready(self.event.take())
    }
}

struct Model(Shared);
impl Chat for Model {
    type Stream = Events;
    async fn stream(&self, request: Request) -> Result<Events, Error> {
        request.validate()?;
        let data = &request.messages.as_slice().last().unwrap().values;
        let node = data
            .get("plan_node")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let mut message = Message::new(Role::Assistant);
        let mut probe = self.0.lock().unwrap();
        if let Some(node) = &node {
            probe.started.push(node.clone());
            probe.active += 1;
            probe.peak = probe.peak.max(probe.active);
            message.values.insert("result".into(), json!(node));
        } else if data.contains_key("plan_complete") {
            message.values.insert("answer".into(), json!("done"));
        } else if probe.requests.is_empty() {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"plan_1","name":"plan","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        } else {
            message.values = plan().as_object().unwrap().clone();
        }
        let gate = node.as_ref().and_then(|id| probe.gates.remove(id));
        probe.requests.push(request);
        Ok(Events {
            gate,
            event: Some(Ok(Event::Complete {
                message,
                usage: Some(Usage {
                    input: Some(2),
                    output: Some(1),
                    total: Some(3),
                    ..Usage::default()
                }),
                finish_reason: Some("stop".into()),
            })),
            _active: node.map(|_| Active(self.0.clone())),
        })
    }
}

fn plan() -> Value {
    json!({"version":1,"nodes":{
        "a":{"name":"北京","objective":"查询北京"},
        "b":{"name":"上海","objective":"查询上海"},
        "c":{"name":"汇总","objective":"汇总两地"},
        "d":{"name":"独立任务","objective":"查询深圳"}
    },"edges":[{"from":"a","to":"c"},{"from":"b","to":"c"}]})
}
fn input() -> Input {
    Input {
        metadata: Default::default(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"查询天气并汇总"}),
            mime_type: None,
        }],
    }
}
fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(50_000);
    NEXT.fetch_add(1, Ordering::Relaxed)
}
fn setup() -> (Model, Shared, Gates) {
    let probe: Shared = Default::default();
    let gates = ["a", "b", "c", "d"]
        .into_iter()
        .map(|id| {
            let (sender, receiver) = oneshot::channel();
            probe.lock().unwrap().gates.insert(id.into(), receiver);
            (id.into(), sender)
        })
        .collect();
    (Model(probe.clone()), probe, gates)
}
fn message(event: Result<Event, Error>) -> Message {
    match event.unwrap() {
        Event::Complete { message, .. } => message,
        _ => panic!("expected Complete"),
    }
}

fn assert_original_messages_have_no_execution_source(value: &Value) {
    match value {
        Value::Object(fields) => {
            if fields.contains_key("role") && fields.contains_key("content") {
                let metadata = fields.get("metadata").unwrap().as_object().unwrap();
                for key in ["chat_id", "node_id", "exec_id"] {
                    assert!(
                        !metadata.contains_key(key),
                        "source {key} leaked into {value}"
                    );
                }
            }
            for child in fields.values() {
                assert_original_messages_have_no_execution_source(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                assert_original_messages_have_no_execution_source(item);
            }
        }
        _ => {}
    }
}

#[test]
fn execution_source_is_only_in_events_not_plan_history_checkpoint_or_next_model_input() {
    block_on(async {
        let (model, probe, gates) = setup();
        for gate in gates.into_values() {
            gate.send(Ok(())).unwrap();
        }
        let checkpoint = Arc::new(halo::core::checkpoint::InMemory::new());
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_id)
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap();
        let mut input = input();
        input.metadata.insert("chat_id".into(), json!(8101));
        let events = agent
            .run(input, true, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        let mut sources = BTreeMap::new();
        for event in events {
            let outgoing = message(event);
            assert_eq!(outgoing.metadata["chat_id"], 8101);
            if let Some(id) = outgoing.metadata.get("node_id").and_then(Value::as_str) {
                sources.insert(
                    id.to_owned(),
                    outgoing.metadata["exec_id"].as_i64().unwrap(),
                );
            }
        }
        assert_eq!(sources.len(), 4);
        let scene = checkpoint.load(8101, &Cancellation::new()).await.unwrap();
        assert_original_messages_have_no_execution_source(&scene);
        for (node_id, exec_id) in sources {
            assert_eq!(scene["nodes"][&node_id]["execution"]["exec_id"], exec_id);
        }
        for request in &probe.lock().unwrap().requests {
            assert_original_messages_have_no_execution_source(
                &serde_json::to_value(&request.messages).unwrap(),
            );
        }
    });
}

#[test]
fn embedded_react_keeps_plan_node_sources_instead_of_overwriting_them_with_the_workflow_container()
{
    block_on(async {
        let (model, probe, gates) = setup();
        for gate in gates.into_values() {
            gate.send(Ok(())).unwrap();
        }
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_id)
            .build()
            .unwrap();
        let mut builder = WorkflowBuilder::new("embedded-plan-observation");
        builder.add_agent_node("worker", agent).unwrap();
        builder.add_edge(START, "worker").unwrap();
        builder.add_edge("worker", END).unwrap();
        let runtime = Runtime::new(8102, builder.compile().unwrap()).unwrap();
        let events = {
            let events = runtime
                .run(
                    Messages::default(),
                    Arc::new(|| 900_001),
                    Cancellation::new(),
                )
                .await
                .unwrap();
            events.collect::<Vec<_>>().await
        };
        let mut sources = BTreeMap::new();
        for event in events {
            let outgoing = message(event);
            assert_eq!(outgoing.metadata["chat_id"], 8102);
            let id = outgoing.metadata["node_id"].as_str().unwrap();
            let exec_id = outgoing.metadata["exec_id"].as_i64().unwrap();
            if id == "worker" {
                assert_eq!(exec_id, 900_001);
            } else {
                sources.insert(id.to_owned(), exec_id);
            }
        }
        assert_eq!(sources.len(), 4);
        let scene = runtime.snapshot().unwrap();
        assert_original_messages_have_no_execution_source(&scene);
        let embedded = &scene["nodes"]["worker"]["executions"]["900001"]["attempts"][0]["agent"];
        for (node_id, exec_id) in sources {
            assert_eq!(embedded["nodes"][&node_id]["execution"]["exec_id"], exec_id);
        }
        for request in &probe.lock().unwrap().requests {
            assert_original_messages_have_no_execution_source(
                &serde_json::to_value(&request.messages).unwrap(),
            );
        }
    });
}

#[test]
fn slow_source_is_not_lost_and_join_receives_both_committed_results() {
    block_on(async {
        let (model, probe, mut gates) = setup();
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_id)
            .with_max_concurrency(2)
            .build()
            .unwrap();
        let mut events = agent.run(input(), true, Cancellation::new()).await.unwrap();
        for _ in 0..3 {
            message(events.next().await.unwrap());
        }
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        assert_eq!(probe.lock().unwrap().started, ["a", "b"]);
        gates.remove("b").unwrap().send(Ok(())).unwrap();
        assert_eq!(message(next.await.unwrap()).metadata["node_id"], "b");
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        assert_eq!(probe.lock().unwrap().started, ["a", "b", "d"]);
        gates.remove("d").unwrap().send(Ok(())).unwrap();
        assert_eq!(message(next.await.unwrap()).metadata["node_id"], "d");
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        assert!(!probe.lock().unwrap().started.contains(&"c".into()));
        gates.remove("a").unwrap().send(Ok(())).unwrap();
        assert_eq!(message(next.await.unwrap()).metadata["node_id"], "a");
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        {
            let probe = probe.lock().unwrap();
            assert_eq!(probe.peak, 2);
            let data = &probe
                .requests
                .last()
                .unwrap()
                .messages
                .as_slice()
                .last()
                .unwrap()
                .values;
            assert_eq!(data["plan_node"], "c");
            assert_eq!(data["sources"]["a"][0]["values"]["result"], "a");
            assert_eq!(data["sources"]["b"][0]["values"]["result"], "b");
            assert!(data["sources"].get("d").is_none());
        }
        gates.remove("c").unwrap().send(Ok(())).unwrap();
        assert_eq!(message(next.await.unwrap()).metadata["node_id"], "c");
        assert_eq!(
            message(events.next().await.unwrap()).values["answer"],
            "done"
        );
        assert!(events.next().await.is_none());
        assert_eq!(probe.lock().unwrap().active, 0);
    });
}

#[test]
fn failed_task_waits_for_independent_work_without_activating_its_downstream() {
    block_on(async {
        let (model, probe, mut gates) = setup();
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_id)
            .with_max_concurrency(2)
            .build()
            .unwrap();
        let mut events = agent.run(input(), true, Cancellation::new()).await.unwrap();
        for _ in 0..3 {
            message(events.next().await.unwrap());
        }
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        gates
            .remove("a")
            .unwrap()
            .send(Err(Error::new("TRANSPORT", "source failed")))
            .unwrap();
        assert!(poll!(next.as_mut()).is_pending());
        assert_eq!(probe.lock().unwrap().started, ["a", "b", "d"]);
        gates.remove("d").unwrap().send(Ok(())).unwrap();
        assert_eq!(message(next.await.unwrap()).metadata["node_id"], "d");
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        gates.remove("b").unwrap().send(Ok(())).unwrap();
        assert_eq!(message(next.await.unwrap()).metadata["node_id"], "b");
        assert_eq!(events.next().await.unwrap().unwrap_err().code, "TRANSPORT");
        assert!(events.next().await.is_none());
        let probe = probe.lock().unwrap();
        assert!(!probe.started.contains(&"c".into()));
        assert_eq!(probe.requests.len(), 5); // No final synthesis of a failed graph.
        assert_eq!(probe.active, 0);
    });
}

#[test]
fn cancelling_parallel_tasks_settles_each_admitted_task_and_never_starts_queued_work() {
    block_on(async {
        let (model, probe, _gates) = setup();
        let checkpoint = Arc::new(halo::core::checkpoint::InMemory::new());
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_id)
            .with_max_concurrency(2)
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap();
        let cancellation = Cancellation::new();
        let mut query = input();
        query.metadata.insert("chat_id".into(), json!(8103));
        let mut events = agent.run(query, true, cancellation.clone()).await.unwrap();
        for _ in 0..3 {
            message(events.next().await.unwrap());
        }
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        cancellation.cancel();
        let mut cancelled = vec![next.await.unwrap().unwrap()];
        while let Some(event) = events.next().await {
            cancelled.push(event.unwrap());
        }
        assert_eq!(cancelled.len(), 2);
        for event in cancelled {
            assert!(
                matches!(event, Event::Complete { message, finish_reason: Some(reason), .. }
                if reason == "cancelled" && message.metadata.contains_key("exec_id"))
            );
        }
        {
            let probe = probe.lock().unwrap();
            assert_eq!(probe.started, ["a", "b"]);
            assert_eq!(probe.active, 0);
        }
        let scene = checkpoint.load(8103, &Cancellation::new()).await.unwrap();
        assert_original_messages_have_no_execution_source(&scene);
    });
}

#[test]
fn dropping_parallel_plan_stream_releases_all_owned_task_futures() {
    block_on(async {
        let (model, probe, _gates) = setup();
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_id)
            .with_max_concurrency(2)
            .build()
            .unwrap();
        let mut events = agent.run(input(), true, Cancellation::new()).await.unwrap();
        for _ in 0..3 {
            message(events.next().await.unwrap());
        }
        let mut next = Box::pin(events.next());
        assert!(poll!(next.as_mut()).is_pending());
        assert_eq!(probe.lock().unwrap().active, 2);
        drop(next);
        drop(events);
        let probe = probe.lock().unwrap();
        assert_eq!(probe.active, 0);
        assert_eq!(probe.started, ["a", "b"]);
    });
}

#[test]
fn one_slot_does_not_eagerly_start_queued_roots_or_dependents() {
    block_on(async {
        let (model, probe, mut gates) = setup();
        let agent = agent::react()
            .model(model)
            .with_id_fn(next_id)
            .with_max_concurrency(1)
            .build()
            .unwrap();
        let mut events = agent.run(input(), true, Cancellation::new()).await.unwrap();
        for _ in 0..3 {
            message(events.next().await.unwrap());
        }
        let mut completed = Vec::new();
        for id in ["a", "b", "c", "d"] {
            let mut next = Box::pin(events.next());
            assert!(poll!(next.as_mut()).is_pending());
            let mut expected = completed.clone();
            expected.push(id.to_owned());
            assert_eq!(probe.lock().unwrap().started, expected);
            gates.remove(id).unwrap().send(Ok(())).unwrap();
            assert_eq!(message(next.await.unwrap()).metadata["node_id"], id);
            completed.push(id.to_owned());
        }
        assert_eq!(
            message(events.next().await.unwrap()).values["answer"],
            "done"
        );
        assert!(events.next().await.is_none());
        assert_eq!(probe.lock().unwrap().peak, 1);
        assert_eq!(probe.lock().unwrap().active, 0);
    });
}
