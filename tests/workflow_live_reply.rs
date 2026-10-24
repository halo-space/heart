//! Actual embedded Agents use the Workflow's original run/reply and snapshot.
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::{self, SubAgent, },
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
    runtime::workflow::{
        CompiledWorkflow, Runtime, WorkflowBuilder,
        graph::{self, END, START},
    },
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicI64, AtomicUsize, Ordering},
};

fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(5_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
#[derive(Default)]
struct Store {
    inner: core::checkpoint::InMemory,
    loads: AtomicUsize,
    saves: AtomicUsize,
}
impl Checkpoint for Store {
    async fn load(&self, id: i64, c: &Cancellation) -> Result<Value, Error> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.inner.load(id, c).await
    }
    async fn save(&self, id: i64, state: Value, c: &Cancellation) -> Result<(), Error> {
        self.saves.fetch_add(1, Ordering::SeqCst);
        self.inner.save(id, state, c).await
    }
    async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, c).await
    }
}
#[derive(Clone, Default)]
struct Model(Arc<Mutex<Vec<Request>>>);
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut requests = self.0.lock().unwrap();
        let mut message = Message::new(Role::Assistant);
        if requests.is_empty() {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"same_call","name":"lookup","arguments":{"city":"Beijing"}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        } else {
            message.values.insert("answer".into(), json!(42));
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
struct Lookup {
    definition: ToolDefinition,
    calls: Arc<Mutex<Vec<Call>>>,
}
impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.calls.lock().unwrap().push(call.clone());
        Ok(Message::function(json!({"weather":"clear"})))
    }
}
struct Ask;
impl Permission for Ask {
    async fn check(&self, _: &PermissionRequest, _: &Cancellation) -> Result<Decision, Error> {
        Ok(Decision {
            behavior: Behavior::Ask,
            reason: Some("Confirm weather lookup".into()),
        })
    }
}
struct Probe {
    model: Model,
    calls: Arc<Mutex<Vec<Call>>>,
    checkpoint: Arc<Store>,
}
fn compiled(react: bool, depth: usize, count: usize) -> (CompiledWorkflow, Vec<Probe>) {
    let mut builder = WorkflowBuilder::new("live-embedded-agents");
    let mut probes = Vec::new();
    for name in ["a", "b"].into_iter().take(count) {
        let model = Model::default();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let checkpoint = Arc::new(Store::default());
        let capabilities=[Lookup{definition:ToolDefinition{name:"lookup".into(),description:"lookup weather".into(),parameters:json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}).as_object().unwrap().clone()},calls:calls.clone()}];
        if react {
            builder
                .add_agent_node(
                    name,
                    agent::react()
                        .model(model.clone())
                        .tools(capabilities)
                        .permission(Ask)
                        .checkpoint(checkpoint.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                )
                .unwrap();
        } else {
            builder
                .add_agent_node(
                    name,
                    agent::chat()
                        .model(model.clone())
                        .tools(capabilities)
                        .permission(Ask)
                        .checkpoint(checkpoint.clone())
                        .with_id_fn(next_id)
                        .build()
                        .unwrap(),
                )
                .unwrap();
        }
        builder.add_edge(START, name).unwrap();
        if count == 1 {
            builder.add_edge(name, END).unwrap();
        }
        probes.push(Probe {
            model,
            calls,
            checkpoint,
        });
    }
    if count == 2 {
        builder
            .add_node("merge", |_: Messages, _: Cancellation| async {
                Ok::<_, Error>(Messages::new([Message::function(json!({"merged":true}))]))
            })
            .unwrap();
        builder.add_join(["a", "b"], "merge").unwrap();
        builder.add_edge("merge", END).unwrap();
    }
    let mut compiled = builder.compile().unwrap();
    for level in 0..depth {
        let mut wrapper = WorkflowBuilder::new(format!("wrapper-{level}"));
        wrapper.add_graph_node("child", &compiled).unwrap();
        wrapper.add_edge(START, "child").unwrap();
        wrapper.add_edge("child", END).unwrap();
        compiled = wrapper.compile().unwrap();
    }
    (compiled, probes)
}
fn initial() -> Messages {
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"weather?"}).as_object().unwrap().clone(),
    });
    Messages::new([message])
}
fn answer(id: i64, approved: bool) -> Messages {
    let mut message = Message::new(Role::User);
    message.metadata.insert("chat_id".into(), json!(1001));
    message.metadata.insert("reply_id".into(), json!(id));
    message.content.push(Part {
        r#type: "json".into(),
        data: json!({"value":{"approved":approved,"arguments":{"city":"untrusted"}}})
            .as_object()
            .unwrap()
            .clone(),
    });
    Messages::new([message])
}
fn leaf(state: &Value) -> &Value {
    if let Some(child) = state["nodes"].get("child") {
        let execution = child["executions"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap();
        leaf(&execution["attempts"][0]["graph"])
    } else {
        state
    }
}
async fn question<S: futures_core::Stream<Item = Result<Event, Error>> + Unpin>(
    events: &mut S,
) -> (i64, Message) {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            let event = events.next().await.expect("live workflow").unwrap();
            if let Event::Complete { message, .. } = event
                && let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64)
            {
                return (id, message);
            }
        }
    })
    .await
    .unwrap()
}
async fn finish<S: futures_core::Stream<Item = Result<Event, Error>> + Unpin>(
    events: &mut S,
) -> Vec<Result<Event, Error>> {
    tokio::time::timeout(std::time::Duration::from_secs(3), events.collect())
        .await
        .unwrap()
}

async fn reply_in_fifo<S: futures_core::Stream<Item = Result<Event, Error>> + Unpin>(
    workflow: &Runtime,
    events: &mut S,
    input: Messages,
) -> Vec<Result<Event, Error>> {
    let mut observed = Vec::new();
    loop {
        tokio::select! {
            biased;
            event = events.next() => match event {
                Some(event) => observed.push(event),
                None => panic!("Workflow ended before the reply became the FIFO head"),
            },
            result = workflow.reply(input.clone(), Cancellation::new()) => match result {
                Ok(()) => return observed,
                Err(error) if error.code == "CONFLICT" => {},
                Err(error) => panic!("unexpected Workflow FIFO reply error: {error:?}"),
            }
        }
    }
}

#[tokio::test]
async fn workflow_reply_continues_real_agent_on_the_same_stream() {
    for react in [false, true] {
        for depth in [0, 2] {
            for approved in [false, true] {
                let (compiled, probes) = compiled(react, depth, 1);
                let store = Arc::new(Store::default());
                let mut workflow = Runtime::new(1001, compiled)
                    .unwrap()
                    .with_checkpoint(store.clone());
                let mut events = Box::pin(
                    workflow
                        .run(initial(), Arc::new(next_id), Cancellation::new())
                        .await
                        .unwrap(),
                );
                let (id, message) = question(&mut events).await;
                assert_eq!(message.metadata["node_id"], "a");
                assert!(message.values.is_empty());
                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(saved["status"], "paused");
                assert_eq!(saved["usage"]["total"], 13);
                let node = &leaf(&saved)["nodes"]["a"];
                let (exec_id, execution) = node["executions"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .next()
                    .unwrap();
                let scene = &execution["attempts"][0]["agent"];
                assert_eq!(execution["status"], "waiting");
                assert_eq!(
                    scene["steps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|step| step["kind"] == "tool")
                        .expect("actual Tool Step")["waits"][id.to_string()]["status"],
                    "waiting"
                );
                assert_eq!(scene["waits"][0]["nodes"], json!([]));
                assert_eq!(store.saves.load(Ordering::SeqCst), 1);
                assert!(
                    matches!(workflow.run(initial(),Arc::new(next_id),Cancellation::new()).await,Err(error) if error.code=="CONFLICT")
                );
                assert!(matches!(workflow.snapshot(),Err(error) if error.code=="CONFLICT"));
                workflow
                    .reply(answer(id, approved), Cancellation::new())
                    .await
                    .unwrap();
                workflow
                    .reply(answer(id, approved), Cancellation::new())
                    .await
                    .unwrap();
                assert_eq!(
                    workflow
                        .reply(answer(id, !approved), Cancellation::new())
                        .await
                        .unwrap_err()
                        .code,
                    "CONFLICT"
                );
                assert!(probes[0].calls.lock().unwrap().is_empty());
                let output = finish(&mut events).await;
                assert!(
                    output.iter().all(Result::is_ok),
                    "{react}/{depth}: {output:?}"
                );
                drop(events);
                assert_eq!(workflow.state().status, graph::Status::Succeeded);
                let final_state = store.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(workflow.snapshot().unwrap(), final_state);
                assert_eq!(final_state["usage"]["total"], 26);
                let final_execution = &leaf(&final_state)["nodes"]["a"]["executions"][exec_id];
                assert_eq!(final_execution["attempts"].as_array().unwrap().len(), 1);
                assert_eq!(
                    final_execution["attempts"][0]["agent"]["steps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|step| step["kind"] == "tool")
                        .expect("actual Tool Step")["waits"][id.to_string()]["status"],
                    "resolved"
                );
                assert_eq!(probes[0].calls.lock().unwrap().len(), usize::from(approved));
                assert_eq!(probes[0].checkpoint.loads.load(Ordering::SeqCst), 0);
                assert_eq!(probes[0].checkpoint.saves.load(Ordering::SeqCst), 0);
                for request in probes[0].model.0.lock().unwrap().iter() {
                    let data = serde_json::to_string(request).unwrap();
                    assert!(!data.contains("reply_id"));
                    assert!(!data.contains("untrusted"));
                }
                assert_eq!(
                    workflow
                        .reply(answer(id, approved), Cancellation::new())
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
async fn workflow_concurrent_questions_are_consumed_in_fifo_order_and_keep_the_join_blocked() {
    for react in [false, true] {
        for depth in [0, 2] {
            let (compiled, probes) = compiled(react, depth, 2);
            let store = Arc::new(Store::default());
            let mut workflow = Runtime::new(1001, compiled)
                .unwrap()
                .with_max_concurrency(1)
                .unwrap()
                .with_checkpoint(store.clone());
            let mut events = Box::pin(
                workflow
                    .run(initial(), Arc::new(next_id), Cancellation::new())
                    .await
                    .unwrap(),
            );
            let (first, a) = question(&mut events).await;
            let (second, b) = question(&mut events).await;
            assert_ne!(first, second);
            assert_ne!(a.metadata["node_id"], b.metadata["node_id"]);
            let saved = store.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(saved["usage"]["total"], 26);
            let first_node = a.metadata["node_id"].as_str().unwrap();
            let second_node = b.metadata["node_id"].as_str().unwrap();
            for (name, id) in [(first_node, first), (second_node, second)] {
                let exec = leaf(&saved)["nodes"][name]["executions"]
                    .as_object()
                    .unwrap()
                    .values()
                    .next()
                    .unwrap();
                assert_eq!(
                    exec["attempts"][0]["agent"]["steps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|step| step["kind"] == "tool")
                        .expect("actual Tool Step")["waits"][id.to_string()]["status"],
                    "waiting"
                );
            }
            assert!(
                leaf(&saved)["nodes"]["merge"]["executions"]
                    .as_object()
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(
                workflow
                    .reply(answer(second, true), Cancellation::new())
                    .await
                    .unwrap_err()
                    .code,
                "CONFLICT",
                "reply_id identifies the later question but cannot skip the FIFO head"
            );
            workflow
                .reply(answer(first, false), Cancellation::new())
                .await
                .unwrap();
            assert_eq!(
                probes
                    .iter()
                    .map(|probe| probe.calls.lock().unwrap().len())
                    .sum::<usize>(),
                0,
                "the later operation cannot run before the head question is consumed"
            );
            loop {
                tokio::select! {
                    biased;
                    event = events.next() => {
                        event.expect("the original Workflow stream remains active").unwrap();
                    }
                    result = workflow.reply(answer(second, true), Cancellation::new()) => {
                        match result {
                            Ok(()) => break,
                            Err(error) if error.code == "CONFLICT" => {}
                            Err(error) => panic!("unexpected FIFO reply error: {error:?}"),
                        }
                    }
                }
            }
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                loop {
                    if let Event::Complete { message, .. } = events.next().await.unwrap().unwrap()
                        && message.metadata.get("node_id") == Some(&json!(second_node))
                        && message.content.is_empty()
                        && !message.metadata.contains_key("reply_id")
                        && message.role == Role::Assistant
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
                    .map(|probe| probe.calls.lock().unwrap().len())
                    .sum::<usize>(),
                1
            );
            let output = finish(&mut events).await;
            assert!(output.iter().all(Result::is_ok), "{output:?}");
            drop(events);
            assert_eq!(workflow.state().status, graph::Status::Succeeded);
            assert_eq!(workflow.state().usage.as_ref().unwrap().total, Some(52));
            assert_eq!(
                store.saves.load(Ordering::SeqCst),
                3,
                "only two waiting presentations and terminal closeout are persisted"
            );
            assert!(
                probes
                    .iter()
                    .all(|probe| probe.checkpoint.loads.load(Ordering::SeqCst) == 0
                        && probe.checkpoint.saves.load(Ordering::SeqCst) == 0)
            );
        }
    }
}

#[tokio::test]
async fn workflow_live_cancellation_and_drop_keep_native_wait_history_without_executing_tools() {
    for react in [false, true] {
        for depth in [0, 2] {
            for drop_stream in [false, true] {
                let (compiled, probes) = compiled(react, depth, 2);
                let store = Arc::new(Store::default());
                let mut workflow = Runtime::new(1001, compiled)
                    .unwrap()
                    .with_max_concurrency(1)
                    .unwrap()
                    .with_checkpoint(store.clone());
                let cancellation = Cancellation::new();
                let mut events = Box::pin(
                    workflow
                        .run(initial(), Arc::new(next_id), cancellation.clone())
                        .await
                        .unwrap(),
                );
                let (first, _) = question(&mut events).await;
                let (second, _) = question(&mut events).await;
                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                if drop_stream {
                    drop(events);
                    assert_eq!(workflow.snapshot().unwrap(), saved);
                } else {
                    cancellation.cancel();
                    let output = finish(&mut events).await;
                    assert!(
                        output
                            .iter()
                            .any(|event| matches!(event,Err(error) if error.code=="CANCELLED")),
                        "{output:?}"
                    );
                    drop(events);
                    assert_eq!(workflow.state().status, graph::Status::Cancelled);
                    assert_eq!(workflow.state().usage.as_ref().unwrap().total, Some(26));
                }
                assert!(
                    probes
                        .iter()
                        .all(|probe| probe.calls.lock().unwrap().is_empty())
                );
                for id in [first, second] {
                    assert_eq!(
                        workflow
                            .reply(answer(id, true), Cancellation::new())
                            .await
                            .unwrap_err()
                            .code,
                        "NOT_FOUND"
                    );
                }
                assert_eq!(workflow.chat_id(), 1001);
            }
        }
    }
}

struct WaitingStore {
    inner: Store,
    fail: bool,
    gate: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    started: AtomicUsize,
}
impl Checkpoint for WaitingStore {
    async fn load(&self, id: i64, c: &Cancellation) -> Result<Value, Error> {
        self.inner.load(id, c).await
    }
    async fn save(&self, id: i64, state: Value, c: &Cancellation) -> Result<(), Error> {
        if state["status"] == "paused" {
            self.started.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(Error::new("SAVE_FAILED", "approval scene save failed"));
            }
            let gate = self.gate.lock().unwrap().take();
            if let Some(gate) = gate {
                gate.await.unwrap();
            }
        }
        self.inner.save(id, state, c).await
    }
    async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, c).await
    }
}

#[tokio::test]
async fn workflow_does_not_present_an_unsaved_or_cancelled_question() {
    use futures_util::poll;
    for react in [false, true] {
        for depth in [0, 2] {
            for fail in [false, true] {
                let (compiled, probes) = compiled(react, depth, 2);
                let (release, gate) = tokio::sync::oneshot::channel();
                let store = Arc::new(WaitingStore {
                    inner: Default::default(),
                    fail,
                    gate: Mutex::new(Some(gate)),
                    started: AtomicUsize::new(0),
                });
                let mut workflow = Runtime::new(1001, compiled)
                    .unwrap()
                    .with_max_concurrency(1)
                    .unwrap()
                    .with_checkpoint(store.clone());
                let cancellation = Cancellation::new();
                let mut events = Box::pin(
                    workflow
                        .run(initial(), Arc::new(next_id), cancellation.clone())
                        .await
                        .unwrap(),
                );
                let output = if fail {
                    finish(&mut events).await
                } else {
                    let mut started = false;
                    for _ in 0..40 {
                        match poll!(events.next()) {
                            std::task::Poll::Pending => {
                                if store.started.load(Ordering::SeqCst) > 0 {
                                    started = true;
                                    break;
                                }
                                tokio::task::yield_now().await;
                            }
                            std::task::Poll::Ready(Some(Ok(Event::Complete {
                                message, ..
                            }))) => assert!(!message.metadata.contains_key("reply_id")),
                            std::task::Poll::Ready(Some(Ok(_))) => {}
                            other => panic!("ended before question save: {other:?}"),
                        }
                    }
                    assert!(started);
                    cancellation.cancel();
                    release.send(()).unwrap();
                    finish(&mut events).await
                };
                assert!(output.iter().all(|event|!matches!(event,Ok(Event::Complete{message,..}) if message.metadata.contains_key("reply_id"))));
                assert!(
                    matches!(output.last(),Some(Err(error)) if error.code==if fail {"SAVE_FAILED"} else {"CANCELLED"}),
                    "{output:?}"
                );
                drop(events);
                assert_eq!(
                    workflow.state().status,
                    if fail {
                        graph::Status::Failed
                    } else {
                        graph::Status::Cancelled
                    }
                );
                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(saved, workflow.snapshot().unwrap());
                let calls = probes
                    .iter()
                    .map(|probe| probe.model.0.lock().unwrap().len())
                    .sum::<usize>();
                assert_eq!(
                    saved["usage"]["total"],
                    13 * calls,
                    "already-returned sibling consumption must be retained"
                );
                assert!(
                    probes
                        .iter()
                        .all(|probe| probe.calls.lock().unwrap().is_empty())
                );
            }
        }
    }
}

#[tokio::test]
async fn workflow_rejects_invalid_feedback_without_touching_the_model_or_checkpoint() {
    for react in [false, true] {
        let (compiled, probes) = compiled(react, 0, 1);
        let store = Arc::new(Store::default());
        let workflow = Runtime::new(1001, compiled)
            .unwrap()
            .with_checkpoint(store.clone());
        let mut events = Box::pin(
            workflow
                .run(initial(), Arc::new(next_id), Cancellation::new())
                .await
                .unwrap(),
        );
        let (id, _) = question(&mut events).await;
        let saves = store.saves.load(Ordering::SeqCst);
        assert_eq!(
            workflow
                .reply(Messages::default(), Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        assert_eq!(
            workflow
                .reply(
                    Messages::new([answer(id, true).0[0].clone(), answer(id, true).0[0].clone()]),
                    Cancellation::new()
                )
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        let mut wrong = answer(id, true);
        wrong.0[0].metadata.insert("chat_id".into(), json!(999));
        assert_eq!(
            workflow
                .reply(wrong, Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
        assert_eq!(
            workflow
                .reply(answer(id + 10000, true), Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
        let mut invalid = answer(id, true);
        invalid.0[0].content[0].data = json!({"value":{"approved":"yes"}})
            .as_object()
            .unwrap()
            .clone();
        assert_eq!(
            workflow
                .reply(invalid, Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        let cancelled = Cancellation::new();
        cancelled.cancel();
        assert_eq!(
            workflow
                .reply(answer(id, true), cancelled)
                .await
                .unwrap_err()
                .code,
            "CANCELLED"
        );
        assert_eq!(store.saves.load(Ordering::SeqCst), saves);
        assert_eq!(probes[0].model.0.lock().unwrap().len(), 1);
        assert!(probes[0].calls.lock().unwrap().is_empty());
        workflow
            .reply(answer(id, true), Cancellation::new())
            .await
            .unwrap();
        assert!(finish(&mut events).await.iter().all(Result::is_ok));
    }
}

struct DelegatingModel(AtomicUsize);
impl Chat for DelegatingModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            message.content.push(Part{r#type:"tool_call".into(),data:json!({"call_id":"spawn","name":"agent_spawn","arguments":{"name":"worker","input":{"payload":[{"type":"text","data":{"value":"weather?"}}]}}}).as_object().unwrap().clone()});
        } else {
            message.values.insert("answer".into(), json!(42));
        }
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

#[tokio::test]
async fn workflow_reply_reaches_a_delegated_child_without_exposing_internal_coordinates() {
    for react in [false, true] {
        for depth in [0, 2] {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let child_checkpoint = Arc::new(Store::default());
            let parent_checkpoint = Arc::new(Store::default());
            let child=SubAgent::new("worker",agent::chat().model(Model::default()).tools([Lookup{definition:ToolDefinition{name:"lookup".into(),description:"lookup weather".into(),parameters:json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}).as_object().unwrap().clone()},calls:calls.clone()}])
            .permission(Ask).checkpoint(child_checkpoint.clone()).build().unwrap()).unwrap();
            let mut builder = WorkflowBuilder::new("delegating-embedded");
            if react {
                builder
                    .add_agent_node(
                        "a",
                        agent::react()
                            .model(DelegatingModel(AtomicUsize::new(0)))
                            .subagents(vec![child])
                            .checkpoint(parent_checkpoint.clone())
                            .with_id_fn(next_id)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();
            } else {
                builder
                    .add_agent_node(
                        "a",
                        agent::chat()
                            .model(DelegatingModel(AtomicUsize::new(0)))
                            .subagents(vec![child])
                            .checkpoint(parent_checkpoint.clone())
                            .with_id_fn(next_id)
                            .build()
                            .unwrap(),
                    )
                    .unwrap();
            }
            builder.add_edge(START, "a").unwrap();
            builder.add_edge("a", END).unwrap();
            let mut graph = builder.compile().unwrap();
            for level in 0..depth {
                let mut wrapper = WorkflowBuilder::new(format!("wrapper-{level}"));
                wrapper.add_graph_node("child", &graph).unwrap();
                wrapper.add_edge(START, "child").unwrap();
                wrapper.add_edge("child", END).unwrap();
                graph = wrapper.compile().unwrap();
            }
            let store = Arc::new(Store::default());
            let mut workflow = Runtime::new(1001, graph)
                .unwrap()
                .with_checkpoint(store.clone());
            let mut events = Box::pin(
                workflow
                    .run(initial(), Arc::new(next_id), Cancellation::new())
                    .await
                    .unwrap(),
            );
            let (id, message) = question(&mut events).await;
            let saved = store.load(1001, &Cancellation::new()).await.unwrap();
            let execution = leaf(&saved)["nodes"]["a"]["executions"]
                .as_object()
                .unwrap()
                .values()
                .next()
                .unwrap();
            let scene = &execution["attempts"][0]["agent"];
            assert_eq!(saved["usage"]["total"], 26);
            let path = scene["waits"][0]["nodes"].as_array().unwrap();
            assert_eq!(path.len(), 1);
            assert_ne!(message.metadata["node_id"], "a");
            let child_id = path[0]["node_id"].as_str().unwrap();
            let child_scene = &scene["execution"]["attempts"][0]["nodes"][child_id]["execution"]["attempts"]
                [0]["agent"];
            assert_eq!(
                child_scene["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|step| step["kind"] == "tool")
                    .expect("actual Tool Step")["waits"][id.to_string()]["status"],
                "waiting"
            );
            workflow
                .reply(answer(id, true), Cancellation::new())
                .await
                .unwrap();
            assert!(finish(&mut events).await.iter().all(Result::is_ok));
            drop(events);
            assert_eq!(workflow.state().status, graph::Status::Succeeded);
            assert_eq!(workflow.state().usage.as_ref().unwrap().total, Some(52));
            assert_eq!(calls.lock().unwrap().len(), 1);
            for checkpoint in [parent_checkpoint, child_checkpoint] {
                assert_eq!(checkpoint.loads.load(Ordering::SeqCst), 0);
                assert_eq!(checkpoint.saves.load(Ordering::SeqCst), 0);
            }
        }
    }
}

#[tokio::test]
async fn idle_workflow_remains_send_and_sync_without_a_second_state() {
    fn check<T: Send + Sync>() {}
    check::<Runtime>();
    let (compiled, probes) = compiled(false, 0, 1);
    let workflow = Runtime::new(1001, compiled).unwrap();
    let before = workflow.snapshot().unwrap();
    for _ in 0..2 {
        let events = workflow
            .run(initial(), Arc::new(next_id), Cancellation::new())
            .await
            .unwrap();
        assert_eq!(workflow.snapshot().unwrap_err().code, "CONFLICT");
        drop(events);
        assert_eq!(workflow.snapshot().unwrap(), before);
    }
    assert!(probes[0].model.0.lock().unwrap().is_empty());
}

#[tokio::test]
async fn workflow_retains_consumed_feedback_and_successful_tool_when_its_stream_is_dropped() {
    for react in [false, true] {
        for depth in [0, 2] {
            let (compiled, probes) = compiled(react, depth, 1);
            let store = Arc::new(Store::default());
            let mut workflow = Runtime::new(1001, compiled)
                .unwrap()
                .with_checkpoint(store.clone());
            let mut events = Box::pin(
                workflow
                    .run(initial(), Arc::new(next_id), Cancellation::new())
                    .await
                    .unwrap(),
            );
            let (id, _) = question(&mut events).await;
            workflow
                .reply(answer(id, true), Cancellation::new())
                .await
                .unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                loop {
                    if let Event::Complete { message, .. } = events.next().await.unwrap().unwrap()
                        && message.metadata.get("call_id") == Some(&json!("same_call"))
                        && message.metadata.get("name") == Some(&json!("lookup"))
                    {
                        break;
                    }
                }
            })
            .await
            .unwrap();
            drop(events);
            let snapshot = workflow.snapshot().unwrap();
            let mut scene = &snapshot;
            for _ in 0..depth {
                let execution = scene["nodes"]["child"]["executions"]
                    .as_object()
                    .unwrap()
                    .values()
                    .next()
                    .unwrap();
                scene = &execution["attempts"][0]["graph"];
            }
            let execution = scene["nodes"]["a"]["executions"]
                .as_object()
                .unwrap()
                .values()
                .next()
                .unwrap();
            let scene = &execution["attempts"][0]["agent"];
            assert_eq!(scene["waits"], json!([]));
            assert_eq!(
                scene["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|step| step["kind"] == "tool")
                    .expect("actual Tool Step")["waits"][id.to_string()]["status"],
                "resolved"
            );
            assert_eq!(probes[0].calls.lock().unwrap().len(), 1);
            assert_eq!(
                store.saves.load(Ordering::SeqCst),
                1,
                "updating the live scene is not another Checkpoint write"
            );
            let output = workflow
                .run(
                    initial(),
                    Arc::new(|| panic!("continuation must retain the execution ID")),
                    Cancellation::new(),
                )
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert!(output.iter().all(Result::is_ok), "{react}: {output:?}");
            assert_eq!(workflow.state().status, graph::Status::Succeeded);
            assert_eq!(probes[0].calls.lock().unwrap().len(), 1);
            assert_eq!(workflow.state().usage.as_ref().unwrap().total, Some(26));
        }
    }
}

#[tokio::test]
async fn dropping_a_nested_workflow_after_a_node_complete_retains_its_success() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let mut builder = WorkflowBuilder::new("nested-result");
    builder
        .add_node("seed", move |_: Messages, _: Cancellation| {
            let observed = observed.clone();
            async move {
                observed.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Error>(Messages::new([Message::function(json!({"seed": 42}))]))
            }
        })
        .unwrap();
    builder.add_edge(START, "seed").unwrap();
    builder.add_edge("seed", END).unwrap();
    let mut compiled = builder.compile().unwrap();
    for level in 0..2 {
        let mut wrapper = WorkflowBuilder::new(format!("result-wrapper-{level}"));
        wrapper.add_graph_node("child", &compiled).unwrap();
        wrapper.add_edge(START, "child").unwrap();
        wrapper.add_edge("child", END).unwrap();
        compiled = wrapper.compile().unwrap();
    }
    let mut workflow = Runtime::new(1001, compiled).unwrap();
    let mut events = Box::pin(
        workflow
            .run(initial(), Arc::new(next_id), Cancellation::new())
            .await
            .unwrap(),
    );
    assert!(matches!(
        events.next().await.unwrap().unwrap(),
        Event::Complete { .. }
    ));
    drop(events);
    let snapshot = workflow.snapshot().unwrap();
    let mut scene = &snapshot;
    for _ in 0..2 {
        let execution = scene["nodes"]["child"]["executions"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap();
        scene = &execution["attempts"][0]["graph"];
    }
    let seed = scene["nodes"]["seed"]["executions"]
        .as_object()
        .expect("the already-presented success must be in the native scene")
        .values()
        .next()
        .unwrap();
    assert_eq!(seed["status"], "succeeded");
    let output = workflow
        .run(initial(), Arc::new(next_id), Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().all(Result::is_ok), "{output:?}");
    assert_eq!(workflow.state().status, graph::Status::Succeeded);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn saved_workflow_approval_replies_restore_the_actual_agent_through_nested_graphs() {
    for react in [false, true] {
        for depth in [0, 2] {
            for approved in [false, true] {
                let (compiled, probes) = compiled(react, depth, 1);
                let store = Arc::new(Store::default());
                let first = Runtime::new(1001, compiled.clone())
                    .unwrap()
                    .with_checkpoint(store.clone());
                let mut original = Box::pin(
                    first
                        .run(initial(), Arc::new(next_id), Cancellation::new())
                        .await
                        .unwrap(),
                );
                let (reply_id, _) = question(&mut original).await;
                drop(original);
                drop(first);

                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                let scene = &leaf(&saved)["nodes"]["a"]["executions"]
                    .as_object()
                    .unwrap()
                    .values()
                    .next()
                    .unwrap()["attempts"][0]["agent"];
                assert_eq!(scene["status"], "waiting");
                assert!(!scene["waits"].as_array().unwrap().is_empty());
                assert_eq!(saved["usage"]["total"], 13);

                let mut restored = Runtime::from_snapshot(1001, compiled, saved)
                    .unwrap()
                    .with_checkpoint(store.clone());
                restored
                    .reply(answer(reply_id, approved), Cancellation::new())
                    .await
                    .unwrap();
                let output = restored
                    .run(
                        initial(),
                        Arc::new(|| panic!("approval recovery must reuse saved execution ids")),
                        Cancellation::new(),
                    )
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                assert!(
                    output.iter().all(Result::is_ok),
                    "{react}/{depth}/{approved}: {output:?}"
                );
                assert_eq!(restored.state().status, graph::Status::Succeeded);
                assert_eq!(restored.state().usage.as_ref().unwrap().total, Some(26));
                assert_eq!(probes[0].calls.lock().unwrap().len(), usize::from(approved));
                assert_eq!(probes[0].model.0.lock().unwrap().len(), 2);
                assert_eq!(probes[0].checkpoint.loads.load(Ordering::SeqCst), 0);
                assert_eq!(probes[0].checkpoint.saves.load(Ordering::SeqCst), 0);
            }
        }
    }
}

#[tokio::test]
async fn saved_workflow_approval_rejects_invalid_or_unmatched_reply_before_state_changes() {
    let (compiled, probes) = compiled(true, 1, 1);
    let store = Arc::new(Store::default());
    let first = Runtime::new(1001, compiled.clone())
        .unwrap()
        .with_checkpoint(store.clone());
    let mut original = Box::pin(
        first
            .run(initial(), Arc::new(next_id), Cancellation::new())
            .await
            .unwrap(),
    );
    let (reply_id, _) = question(&mut original).await;
    drop(original);
    drop(first);
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    let restored = Runtime::from_snapshot(1001, compiled, saved).unwrap();
    let before = restored.snapshot().unwrap();

    let mut invalid = answer(reply_id, true);
    invalid.0[0].content[0]
        .data
        .insert("value".into(), json!({"approved":"yes"}));
    assert_eq!(
        restored
            .reply(invalid, Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(
        restored
            .reply(answer(reply_id + 1, true), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert_eq!(restored.snapshot().unwrap(), before);
    assert_eq!(probes[0].model.0.lock().unwrap().len(), 1);
    assert!(probes[0].calls.lock().unwrap().is_empty());
}

#[derive(Default)]
struct PlanningModel(Mutex<Vec<Request>>);
impl Chat for PlanningModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let task = |request: &Request| {
            request.messages.as_slice().iter().find_map(|message| {
                message
                    .values
                    .get("plan_node")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
        };
        let mut requests = self.0.lock().unwrap();
        let mut message = Message::new(Role::Assistant);
        if let Some(id) = task(&request) {
            if requests
                .iter()
                .any(|request| task(request).as_ref() == Some(&id))
            {
                message.values.insert("result".into(), json!(id));
            } else {
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: json!({"call_id":"same_call","name":"lookup","arguments":{"city":id}})
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
            message.values.insert("answer".into(), json!(42));
        } else {
            message.values=json!({"version":1,"nodes":{"a":{"name":"research","objective":"lookup a"},"b":{"name":"research","objective":"lookup b"}},"edges":[]}).as_object().unwrap().clone();
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

#[tokio::test]
async fn workflow_reply_targets_actual_agent_plan_tasks_even_when_node_names_repeat() {
    for depth in [0, 2] {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let child_checkpoint = Arc::new(Store::default());
        let agent=agent::react().model(PlanningModel::default()).tools([Lookup{definition:ToolDefinition{name:"lookup".into(),description:"lookup weather".into(),parameters:json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}).as_object().unwrap().clone()},calls:calls.clone()}])
            .permission(Ask).with_max_concurrency(1).with_id_fn(next_id).checkpoint(child_checkpoint.clone()).build().unwrap();
        let mut builder = WorkflowBuilder::new("planned-agent-live");
        builder.add_agent_node("a", agent).unwrap();
        builder.add_edge(START, "a").unwrap();
        builder.add_edge("a", END).unwrap();
        let mut compiled = builder.compile().unwrap();
        for level in 0..depth {
            let mut wrapper = WorkflowBuilder::new(format!("wrapper-{level}"));
            wrapper.add_graph_node("child", &compiled).unwrap();
            wrapper.add_edge(START, "child").unwrap();
            wrapper.add_edge("child", END).unwrap();
            compiled = wrapper.compile().unwrap();
        }
        let store = Arc::new(Store::default());
        let mut workflow = Runtime::new(1001, compiled)
            .unwrap()
            .with_checkpoint(store.clone());
        let mut events = Box::pin(
            workflow
                .run(initial(), Arc::new(next_id), Cancellation::new())
                .await
                .unwrap(),
        );
        let (first, _) = question(&mut events).await;
        let (second, _) = question(&mut events).await;
        let saved = store.load(1001, &Cancellation::new()).await.unwrap();
        let outer = leaf(&saved)["nodes"]["a"]["executions"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap();
        let scene = &outer["attempts"][0]["agent"];
        assert_eq!(saved["usage"]["total"], 52);
        let waits = scene["waits"].as_array().unwrap();
        assert_eq!(waits.len(), 2);
        for wait in waits {
            let path = wait["nodes"].as_array().unwrap();
            assert_eq!(path.len(), 1);
            assert_ne!(path[0]["exec_id"], outer["exec_id"]);
            let name = path[0]["node_id"].as_str().unwrap();
            assert_eq!(
                scene["nodes"][name]["execution"]["attempts"][0]["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|step| step["kind"] == "tool")
                    .expect("actual Tool Step")["call_id"],
                "same_call"
            );
        }
        let mut output = reply_in_fifo(&workflow, &mut events, answer(first, true)).await;
        output.extend(reply_in_fifo(&workflow, &mut events, answer(second, true)).await);
        output.extend(finish(&mut events).await);
        assert!(output.iter().all(Result::is_ok), "{output:?}");
        drop(events);
        assert_eq!(workflow.state().status, graph::Status::Succeeded);
        assert_eq!(workflow.state().usage.as_ref().unwrap().total, Some(91));
        assert_eq!(calls.lock().unwrap().len(), 2);
        let final_state = workflow.snapshot().unwrap();
        let outer = leaf(&final_state)["nodes"]["a"]["executions"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap();
        assert_eq!(outer["attempts"].as_array().unwrap().len(), 1);
        assert_eq!(child_checkpoint.loads.load(Ordering::SeqCst), 0);
        assert_eq!(child_checkpoint.saves.load(Ordering::SeqCst), 0);
    }
}
