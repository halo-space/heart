//! Ordinary feedback belongs to each planned task's current native Step.
use futures_util::{StreamExt, poll, stream};
use halo_agents::{
    Cancellation, Error, Message, Messages,
    agent::{self, Agent, Input, toolkit::Toolkit},
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
    static NEXT: AtomicI64 = AtomicI64::new(9_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
fn no_new_id() -> i64 {
    panic!("restored tasks and questions already have IDs")
}
fn continuation_id() -> i64 {
    // Simulate the application's globally unique generator after a process
    // restart, rather than resetting to the IDs in the saved executions.
    static NEXT: AtomicI64 = AtomicI64::new(99_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
fn input() -> Input {
    serde_json::from_value(
        json!({"metadata":{"chat_id":1001,"session_id":2001,"agent_id":3001},
        "payload":[{"type":"text","data":{"value":"complete both tasks"}}]}),
    )
    .unwrap()
}
fn answer(id: i64, value: &str) -> Input {
    let mut input = input();
    input.metadata.insert("reply_id".into(), id.into());
    input.payload[0].data = json!({"value":value});
    input
}
fn task(request: &Request) -> Option<&str> {
    request
        .messages
        .as_slice()
        .iter()
        .find_map(|m| m.values.get("plan_node").and_then(Value::as_str))
}
#[derive(Clone, Default)]
struct Model(Arc<Mutex<Vec<Request>>>);
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut requests = self.0.lock().unwrap();
        let mut message = Message::new(Role::Assistant);
        if let Some(node) = task(&request) {
            if node == "c"
                || request
                    .messages
                    .as_slice()
                    .iter()
                    .any(|m| m.values.get("task").and_then(Value::as_str) == Some(node))
            {
                message.values.insert("result".into(), json!(node));
            } else {
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: json!({
                        "call_id":"same_call","name":"collect","arguments":{"task":node}
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                });
            }
        } else if request
            .messages
            .as_slice()
            .iter()
            .any(|m| m.values.contains_key("plan_complete"))
        {
            message.values.insert("answer".into(), json!("done"));
        } else if requests.is_empty() {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({
                    "call_id":"planning","name":"plan","arguments":{}
                })
                .as_object()
                .unwrap()
                .clone(),
            });
        } else {
            message.values = json!({"version":1,"nodes":{
                "a":{"name":"first","objective":"collect a"},
                "b":{"name":"second","objective":"collect b"},
                "c":{"name":"combine","objective":"combine both results"}
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
#[derive(Default)]
struct Log {
    started: Mutex<Vec<String>>,
    received: Mutex<BTreeMap<String, Messages>>,
}
struct Collect {
    definition: ToolDefinition,
    log: Arc<Log>,
    count: usize,
}
impl Collect {
    fn new(log: Arc<Log>, count: usize) -> Self {
        Self { log, count, definition:ToolDefinition {name:"collect".into(),description:"collect required feedback".into(),
            parameters:json!({"type":"object","properties":{"task":{"type":"string"}},"required":["task"],"additionalProperties":false}).as_object().unwrap().clone()} }
    }
}
impl Tool for Collect {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let node = call.arguments["task"].as_str().unwrap().to_owned();
        self.log.started.lock().unwrap().push(node.clone());
        let mut question = Message::new(Role::Assistant);
        // Deliberately identical question text and Call ID across both tasks.
        question.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"请提交所需反馈"})
                .as_object()
                .unwrap()
                .clone(),
        });
        let feedback = wait::ask(question, self.count, cancellation).await?;
        let count = feedback.as_slice().len();
        self.log
            .received
            .lock()
            .unwrap()
            .insert(node.clone(), feedback);
        Ok(Message::function(json!({"task":node,"count":count})))
    }
}
fn build<C: Checkpoint + 'static>(
    checkpoint: C,
    model: Model,
    log: Arc<Log>,
    count: usize,
    capacity: usize,
    ids: fn() -> i64,
) -> impl Agent {
    agent::react()
        .model(model)
        .toolkit(Toolkit::new().tool(Collect::new(log, count)))
        .checkpoint(checkpoint)
        .with_id_fn(ids)
        .with_max_concurrency(capacity)
        .build()
        .unwrap()
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
    .expect("a waiting task must release capacity for another ready task")
}
fn execution<'a>(scene: &'a Value, node: &str) -> &'a Value {
    &scene["nodes"][node]["execution"]
}
fn record<'a>(scene: &'a Value, node: &str, id: i64) -> &'a Value {
    &execution(scene, node)["attempts"][0]["steps"]
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
        let mut result = vec![];
        while let Some(event) = events.next().await {
            result.push(event.unwrap());
        }
        result
    })
    .await
    .expect("submitted tasks must finish")
}

#[tokio::test]
async fn planned_feedback_collects_on_native_steps_in_fifo_without_receipt_events() {
    for stream in [false, true] {
        for capacity in [1, 2] {
            for count in [1, 3] {
                let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                let model = Model::default();
                let log = Arc::new(Log::default());
                let agent = build(
                    checkpoint.clone(),
                    model.clone(),
                    log.clone(),
                    count,
                    capacity,
                    next_id,
                );
                let mut events = Box::pin(
                    agent
                        .run(input(), stream, Cancellation::new())
                        .await
                        .unwrap(),
                );
                let (a, first) = question(&mut events).await;
                let (b, second) = question(&mut events).await;
                let original = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
                assert_ne!(a, b);
                assert_ne!(first, second);
                assert_eq!(original["waits"][0]["nodes"][0]["node_id"], first);
                assert_eq!(original["waits"][1]["nodes"][0]["node_id"], second);
                let identities = [
                    execution(&original, &first).clone(),
                    execution(&original, &second).clone(),
                ];
                assert_eq!(
                    agent
                        .reply(answer(b, "out of order"), Cancellation::new())
                        .await
                        .unwrap_err()
                        .code,
                    "CONFLICT"
                );
                assert_eq!(
                    checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
                    original
                );
                for idx in 0..count {
                    let value = format!("first-{idx}");
                    agent
                        .reply(answer(a, &value), Cancellation::new())
                        .await
                        .unwrap();
                    agent
                        .reply(answer(a, &value), Cancellation::new())
                        .await
                        .unwrap();
                    if idx + 1 < count {
                        assert!(poll!(events.next()).is_pending());
                        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
                        assert_eq!(record(&saved, &first, a)["count"], count);
                        assert_eq!(
                            record(&saved, &first, a)["feedback"]
                                .as_array()
                                .unwrap()
                                .len(),
                            idx + 1
                        );
                        assert_eq!(record(&saved, &second, b)["feedback"], json!([]));
                        assert_eq!(saved["nodes"][&first]["status"], "waiting");
                        assert_eq!(saved["query"], original["query"]);
                        assert_eq!(saved["usage"], original["usage"]);
                        assert!(log.received.lock().unwrap().is_empty());
                    }
                }
                let mut prefix = vec![];
                tokio::time::timeout(std::time::Duration::from_secs(3),async {
            loop {
                let event = events.next().await.unwrap().unwrap();
                let done = matches!(&event,Event::Complete{message,..} if message.values.get("result").and_then(Value::as_str)==Some(first.as_str()));
                prefix.push(event);
                if done {break;}
            }
        }).await.unwrap();
                assert_eq!(log.received.lock().unwrap().len(), 1);
                assert!(!model.0.lock().unwrap().iter().any(|r| task(r) == Some("c")));
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
                assert!(!prefix.iter().any(|e|matches!(e,Event::Complete{message,..} if message.metadata.contains_key("reply_id"))));
                if !stream {
                    assert!(!prefix.iter().any(|e| matches!(e, Event::Delta { .. })));
                }
                let terminal = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(terminal["status"], "succeeded");
                assert_eq!(
                    terminal["usage"]["total"],
                    model.0.lock().unwrap().len() as u64 * 13
                );
                for (node, id, old) in [(&first, a, &identities[0]), (&second, b, &identities[1])] {
                    assert_eq!(execution(&terminal, node)["exec_id"], old["exec_id"]);
                    assert_eq!(
                        execution(&terminal, node)["attempts"]
                            .as_array()
                            .unwrap()
                            .len(),
                        1
                    );
                    assert_eq!(record(&terminal, node, id)["status"], "resolved");
                    assert_eq!(
                        record(&terminal, node, id)["feedback"]
                            .as_array()
                            .unwrap()
                            .len(),
                        count
                    );
                    assert_eq!(log.received.lock().unwrap()[node].as_slice().len(), count);
                }
                assert!(
                    !model
                        .0
                        .lock()
                        .unwrap()
                        .iter()
                        .any(|r| r
                            .messages
                            .as_slice()
                            .iter()
                            .any(|m| m.content.iter().any(|p| p
                                .data
                                .get("value")
                                .and_then(Value::as_str)
                                .is_some_and(
                                    |s| s.starts_with("first-") || s.starts_with("second-")
                                ))))
                );
            }
        }
    }
}

async fn partial(
    checkpoint: Arc<core::checkpoint::InMemory>,
    stream: bool,
) -> (i64, String, i64, String, Value) {
    let agent = build(
        checkpoint.clone(),
        Model::default(),
        Default::default(),
        3,
        1,
        next_id,
    );
    let mut events = Box::pin(
        agent
            .run(input(), stream, Cancellation::new())
            .await
            .unwrap(),
    );
    let (a, first) = question(&mut events).await;
    let (b, second) = question(&mut events).await;
    agent
        .reply(answer(a, "first"), Cancellation::new())
        .await
        .unwrap();
    assert!(poll!(events.next()).is_pending());
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        record(&scene, &first, a)["feedback"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    drop(events);
    (a, first, b, second, scene)
}
async fn restore(
    checkpoint: Arc<core::checkpoint::InMemory>,
    a: i64,
    first: String,
    b: i64,
    second: String,
    previous: Value,
    stream: bool,
) {
    let model = Model::default();
    let log = Arc::new(Log::default());
    let agent = build(
        checkpoint.clone(),
        model.clone(),
        log.clone(),
        3,
        1,
        continuation_id, // Only pending downstream task c needs a fresh exec ID.
    );
    let mut events = Box::pin(
        agent
            .run(answer(a, "second"), !stream, Cancellation::new())
            .await
            .unwrap(),
    );
    // The other task's saved question may be re-presented; it is not a new ID.
    let mut prefix = vec![];
    for _ in 0..4 {
        match poll!(events.next()) {
            std::task::Poll::Ready(Some(Ok(event))) => prefix.push(event),
            std::task::Poll::Pending => break,
            other => panic!("unexpected restore {other:?}"),
        }
    }
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        record(&scene, &first, a)["feedback"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(record(&scene, &second, b)["feedback"], json!([]));
    assert_eq!(scene["stream"], stream);
    assert_eq!(scene["query"], previous["query"]);
    assert_eq!(scene["usage"], previous["usage"]);
    assert!(
        model.0.lock().unwrap().is_empty(),
        "do not replay planning or selecting models"
    );
    agent
        .reply(answer(a, "first"), Cancellation::new())
        .await
        .unwrap();
    assert!(poll!(events.next()).is_pending());
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        scene
    );
    agent
        .reply(answer(a, "third"), Cancellation::new())
        .await
        .unwrap();
    // Consume the first task result before submitting to the next FIFO head.
    tokio::time::timeout(std::time::Duration::from_secs(3),async {
        loop { let event = events.next().await.unwrap().unwrap();
            let done=matches!(&event,Event::Complete{message,..} if message.values.get("result").and_then(Value::as_str)==Some(first.as_str()));
            prefix.push(event); if done {break;}
        }
    }).await.unwrap();
    for text in ["one", "two", "three"] {
        agent
            .reply(answer(b, text), Cancellation::new())
            .await
            .unwrap();
    }
    prefix.extend(finish(&mut events).await);
    if !stream {
        assert!(!prefix.iter().any(|e| matches!(e, Event::Delta { .. })));
    }
    let terminal = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(terminal["status"], "succeeded");
    assert_eq!(
        terminal["usage"]["total"],
        previous["usage"]["total"].as_u64().unwrap() + model.0.lock().unwrap().len() as u64 * 13
    );
    for (node, id) in [(&first, a), (&second, b)] {
        assert_eq!(
            execution(&terminal, node)["exec_id"],
            execution(&previous, node)["exec_id"]
        );
        assert_eq!(
            record(&terminal, node, id)["question"],
            record(&previous, node, id)["question"]
        );
        assert_eq!(
            record(&terminal, node, id)["feedback"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }
}
#[tokio::test]
async fn planned_partial_feedback_restores_without_replanning_or_new_execution_ids() {
    for stream in [false, true] {
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let (a, first, b, second, scene) = partial(checkpoint.clone(), stream).await;
        restore(checkpoint, a, first, b, second, scene, stream).await;
    }
}

#[tokio::test]
async fn planned_feedback_rejects_stale_or_wrong_targets_without_changing_scene() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let (a, _, b, _, scene) = partial(checkpoint.clone(), false).await;
    for (mut candidate, mut response, code) in [
        (scene.clone(), answer(b, "wrong head"), "CONFLICT"),
        (scene.clone(), answer(a, "wrong owner"), "CONFLICT"),
        (scene.clone(), answer(a, "stale"), "INVALID_STATE"),
    ] {
        if code == "CONFLICT" && response.payload[0].data["value"] == "wrong owner" {
            response.metadata.insert("session_id".into(), json!(9999));
        }
        if code == "INVALID_STATE" {
            candidate["waits"][0]["nodes"][0]["idx"] = json!(8);
        }
        checkpoint
            .save(1001, candidate.clone(), &Cancellation::new())
            .await
            .unwrap();
        let model = Model::default();
        let log = Arc::new(Log::default());
        let agent = build(
            checkpoint.clone(),
            model.clone(),
            log.clone(),
            3,
            1,
            no_new_id,
        );
        let error = match agent.run(response, false, Cancellation::new()).await {
            Ok(_) => panic!("invalid input accepted"),
            Err(error) => error,
        };
        assert_eq!(error.code, code);
        assert!(model.0.lock().unwrap().is_empty());
        assert!(log.started.lock().unwrap().is_empty());
        assert_eq!(
            checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
            candidate
        );
    }
}

#[tokio::test]
async fn cancelling_a_plan_with_partial_feedback_keeps_facts_but_closes_all_consumers() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let model = Model::default();
    let agent = build(
        checkpoint.clone(),
        model.clone(),
        log.clone(),
        3,
        1,
        next_id,
    );
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(input(), true, cancellation.clone())
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
    let previous = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    cancellation.cancel();
    let tail = finish(&mut events).await;
    assert!(tail.iter().any(
        |e| matches!(e,Event::Complete{finish_reason:Some(reason),..} if reason=="cancelled")
    ));
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["status"], "stopped");
    assert_eq!(scene["waits"], json!([]));
    assert_eq!(record(&scene, &first, a), record(&previous, &first, a));
    assert_eq!(scene["usage"], previous["usage"]);
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

#[derive(Clone)]
struct FailPartial(Arc<core::checkpoint::InMemory>);
impl Checkpoint for FailPartial {
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        let partial = state["nodes"].as_object().is_some_and(|nodes| {
            nodes.values().any(|node| {
                node["execution"]["attempts"][0]["steps"]
                    .as_array()
                    .is_some_and(|steps| {
                        steps.iter().any(|step| {
                            step["waits"].as_object().is_some_and(|waits| {
                                waits.values().any(|record| {
                                    record["kind"] == "input"
                                        && record["feedback"]
                                            .as_array()
                                            .is_some_and(|feedback| feedback.len() == 1)
                                })
                            })
                        })
                    })
            })
        });
        if state["status"] == "waiting" && partial {
            return Err(Error::new("SAVE_FAILED", "partial feedback save failed"));
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
async fn failure_to_save_partial_task_feedback_stops_without_fabricating_completion() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let agent = build(
        FailPartial(checkpoint.clone()),
        Model::default(),
        log.clone(),
        3,
        1,
        next_id,
    );
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
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["status"], "failed");
    assert_eq!(scene["waits"], json!([]));
    assert_eq!(
        record(&scene, &first, a)["feedback"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(record(&scene, &first, a)["status"], "waiting");
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
#[ignore = "explicitly invoked by planned_feedback_restores_across_fresh_processes"]
async fn planned_feedback_process_worker() {
    let path = std::env::var("HEART_PLAN_FEEDBACK_STATE").unwrap();
    let phase = std::env::var("HEART_PLAN_FEEDBACK_PHASE").unwrap();
    let mode = std::env::var("HEART_PLAN_FEEDBACK_STREAM").unwrap() == "true";
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    if phase == "save" {
        let (_, _, _, _, scene) = partial(checkpoint, mode).await;
        std::fs::write(path, serde_json::to_vec(&scene).unwrap()).unwrap();
    } else {
        let scene: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let pairs = scene["waits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|position| {
                let node = position["nodes"][0]["node_id"].as_str().unwrap().to_owned();
                let id = execution(&scene, &node)["attempts"][0]["steps"]
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
                (id, node)
            })
            .collect::<Vec<_>>();
        checkpoint
            .save(1001, scene.clone(), &Cancellation::new())
            .await
            .unwrap();
        restore(
            checkpoint,
            pairs[0].0,
            pairs[0].1.clone(),
            pairs[1].0,
            pairs[1].1.clone(),
            scene,
            mode,
        )
        .await;
    }
}
#[test]
fn planned_feedback_restores_across_fresh_processes() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "heart-plan-feedback-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    for mode in ["false", "true"] {
        let path = directory.join(format!("{mode}.json"));
        for phase in ["save", "restore"] {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "planned_feedback_process_worker",
                    "--nocapture",
                ])
                .env("HEART_PLAN_FEEDBACK_STATE", &path)
                .env("HEART_PLAN_FEEDBACK_PHASE", phase)
                .env("HEART_PLAN_FEEDBACK_STREAM", mode)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{mode}/{phase}: {}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        }
        std::fs::remove_file(path).unwrap();
    }
    std::fs::remove_dir(directory).unwrap();
}
