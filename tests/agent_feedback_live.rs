//! Independent root ordinary feedback through the real Agent run/reply path.
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
    permission::{Behavior, Decision, Permission, Request as PermissionRequest},
    runtime::agent::wait,
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicI64, AtomicUsize, Ordering},
};

fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(7_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
fn no_new_id() -> i64 {
    panic!("restoration must retain the saved question ID")
}
fn input() -> Input {
    serde_json::from_value(
        json!({"metadata":{"chat_id":1001,"session_id":2001,"agent_id":3001},
        "payload":[{"type":"text","data":{"value":"collect feedback"}}]}),
    )
    .unwrap()
}
fn answer(id: i64, text: &str) -> Input {
    let mut result = input();
    result.metadata.insert("reply_id".into(), id.into());
    result.payload[0].data = json!({"value":text});
    result
}
fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":value}).as_object().unwrap().clone(),
    });
    message
}

#[derive(Clone)]
struct Model {
    requests: Arc<Mutex<Vec<Request>>>,
    select: bool,
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let first = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request);
            requests.len() == 1 && self.select
        };
        let mut message = Message::new(Role::Assistant);
        if first {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({
                    "call_id":"c1","name":"collect","arguments":{}
                })
                .as_object()
                .unwrap()
                .clone(),
            });
        } else {
            message.values.insert("answer".into(), json!("done"));
        }
        Ok(stream::iter(vec![
            Ok(Event::Delta {
                message: text(Role::Assistant, "delta"),
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
    started: AtomicUsize,
    completed: AtomicUsize,
    received: Mutex<Messages>,
}
struct Collect {
    definition: ToolDefinition,
    count: usize,
    log: Arc<Log>,
}
impl Collect {
    fn new(count: usize, log: Arc<Log>) -> Self {
        Self {
            count,
            log,
            definition: ToolDefinition {
                name: "collect".into(),
                description: "Collect user feedback".into(),
                parameters: json!({"type":"object","properties":{},"additionalProperties":false})
                    .as_object()
                    .unwrap()
                    .clone(),
            },
        }
    }
}
impl Tool for Collect {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, _: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        self.log.started.fetch_add(1, Ordering::SeqCst);
        let received = wait::ask(
            text(Role::Assistant, "请提供所需反馈"),
            self.count,
            cancellation,
        )
        .await?;
        *self.log.received.lock().unwrap() = received.clone();
        self.log.completed.fetch_add(1, Ordering::SeqCst);
        Ok(Message::function(json!({"received":received})))
    }
}

async fn question<A: Agent>(agent: &A, stream: bool) -> (std::pin::Pin<Box<A::Stream>>, i64) {
    let mut events = Box::pin(
        agent
            .run(input(), stream, Cancellation::new())
            .await
            .unwrap(),
    );
    let id = loop {
        let event = events.next().await.expect("question event").unwrap();
        if !stream {
            assert!(!matches!(event, Event::Delta { .. }));
        }
        if let Event::Complete { message, .. } = event
            && let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64)
        {
            break id;
        }
    };
    (events, id)
}

fn record(state: &Value) -> &Value {
    state["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["kind"] == "tool")
        .expect("actual Tool Step")["waits"]
        .as_object()
        .unwrap()
        .values()
        .find(|record| record["kind"] == "input")
        .unwrap()
}

struct Ask;
impl Permission for Ask {
    async fn check(&self, _: &PermissionRequest, _: &Cancellation) -> Result<Decision, Error> {
        Ok(Decision {
            behavior: Behavior::Ask,
            reason: Some("approve collect".into()),
        })
    }
}

async fn approved_saved<A: Agent>(
    agent: &A,
    checkpoint: &core::checkpoint::InMemory,
) -> (i64, Value) {
    let (mut events, approval) = question(agent, false).await;
    let mut response = answer(approval, "");
    response.payload[0].r#type = "json".into();
    response.payload[0].data = json!({"value":{"approved":true}});
    agent.reply(response, Cancellation::new()).await.unwrap();
    let id = loop {
        let event = events.next().await.unwrap().unwrap();
        if let Event::Complete { message, .. } = event
            && let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64)
        {
            assert_ne!(id, approval);
            break id;
        }
    };
    agent
        .reply(answer(id, "first"), Cancellation::new())
        .await
        .unwrap();
    assert!(poll!(events.next()).is_pending());
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][approval.to_string()]["status"],
        "resolved"
    );
    assert_eq!(record(&scene)["feedback"].as_array().unwrap().len(), 1);
    drop(events);
    (id, scene)
}

#[tokio::test]
async fn restoring_tool_input_after_approval_preserves_the_exact_approved_operation() {
    macro_rules! verify {
        ($constructor:expr) => {{
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let old = $constructor
                .model(Model {
                    requests: Default::default(),
                    select: true,
                })
                .toolkit(Toolkit::new().tool(Collect::new(3, Default::default())))
                .permission(Ask)
                .checkpoint(checkpoint.clone())
                .with_id_fn(next_id)
                .build()
                .unwrap();
            let (id, previous) = approved_saved(&old, &checkpoint).await;
            let log = Arc::new(Log::default());
            let requests = Arc::new(Mutex::new(vec![]));
            let agent = $constructor
                .model(Model {
                    requests: requests.clone(),
                    select: false,
                })
                .toolkit(Toolkit::new().tool(Collect::new(3, log.clone())))
                .permission(Ask)
                .checkpoint(checkpoint.clone())
                .with_id_fn(no_new_id)
                .build()
                .unwrap();
            tokio::time::timeout(
                std::time::Duration::from_secs(3),
                restored(agent, checkpoint, log, requests, id, previous, false),
            )
            .await
            .expect("restore must not ask for the same approval behind ordinary feedback");
        }};
    }
    verify!(agent::chat());
    verify!(agent::react());
}

struct Two(Model);
impl Chat for Two {
    type Stream = <Model as Chat>::Stream;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut events = self.0.stream(request).await?.collect::<Vec<_>>().await;
        for event in &mut events {
            if let Ok(Event::Complete { message, .. }) = event
                && let Some(first) = message
                    .content
                    .first()
                    .filter(|part| part.r#type == "tool_call")
            {
                let mut second = first.clone();
                second.data.insert("call_id".into(), json!("c2"));
                message.content.push(second);
            }
        }
        Ok(stream::iter(events))
    }
}

async fn fifo<A: Agent>(agent: A, checkpoint: Arc<core::checkpoint::InMemory>, log: Arc<Log>) {
    let (mut events, first) = question(&agent, true).await;
    let second = loop {
        if let Event::Complete { message, .. } =
            tokio::time::timeout(std::time::Duration::from_secs(3), events.next())
                .await
                .expect("second question must not be blocked by capacity one")
                .unwrap()
                .unwrap()
            && let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64)
        {
            break id;
        }
    };
    assert_ne!(first, second);
    let before = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        agent
            .reply(answer(second, "early"), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        before
    );
    for idx in 0..3 {
        agent
            .reply(answer(first, &format!("first-{idx}")), Cancellation::new())
            .await
            .unwrap();
    }
    assert_eq!(
        agent
            .reply(answer(first, "too many"), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    loop {
        match poll!(events.next()) {
            std::task::Poll::Pending => break,
            std::task::Poll::Ready(Some(event)) => {
                event.unwrap();
            }
            std::task::Poll::Ready(None) => panic!("second question must remain pending"),
        }
    }
    assert_eq!(log.completed.load(Ordering::SeqCst), 1);
    assert_eq!(log.started.load(Ordering::SeqCst), 2);
    agent
        .reply(answer(first, "first-0"), Cancellation::new())
        .await
        .unwrap();
    for idx in 0..3 {
        agent
            .reply(
                answer(second, &format!("second-{idx}")),
                Cancellation::new(),
            )
            .await
            .unwrap();
        if idx < 2 {
            assert!(poll!(events.next()).is_pending());
            let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(state["waits"].as_array().unwrap().len(), 1);
            assert_eq!(log.completed.load(Ordering::SeqCst), 1);
        }
    }
    let tail = events.collect::<Vec<_>>().await;
    assert!(tail.iter().all(Result::is_ok), "{tail:?}");
    assert_eq!(log.completed.load(Ordering::SeqCst), 2);
    let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(state["status"], "succeeded");
    assert_eq!(state["waits"], json!([]));
    for step in state["steps"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|step| step["kind"] == "tool")
    {
        for wait in step["waits"].as_object().unwrap().values() {
            assert_eq!(wait["status"], "resolved");
            assert_eq!(wait["feedback"].as_array().unwrap().len(), 3);
        }
    }
}

#[tokio::test]
async fn concurrent_questions_accept_only_the_fifo_head_and_queue_each_part_once() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let agent = agent::chat()
        .model(Two(Model {
            requests: Default::default(),
            select: true,
        }))
        .toolkit(Toolkit::new().tool(Collect::new(3, log.clone())))
        .with_max_concurrency(1)
        .checkpoint(checkpoint.clone())
        .with_id_fn(next_id)
        .build()
        .unwrap();
    tokio::time::timeout(
        std::time::Duration::from_secs(8),
        fifo(agent, checkpoint, log),
    )
    .await
    .expect("Chat FIFO at capacity one must finish");
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let agent = agent::react()
        .model(Two(Model {
            requests: Default::default(),
            select: true,
        }))
        .toolkit(Toolkit::new().tool(Collect::new(3, log.clone())))
        .with_max_concurrency(1)
        .checkpoint(checkpoint.clone())
        .with_id_fn(next_id)
        .build()
        .unwrap();
    tokio::time::timeout(
        std::time::Duration::from_secs(8),
        fifo(agent, checkpoint, log),
    )
    .await
    .expect("ReAct FIFO at capacity one must finish");
}

async fn cancelled<A: Agent>(agent: A, checkpoint: Arc<core::checkpoint::InMemory>, log: Arc<Log>) {
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(input(), true, cancellation.clone())
            .await
            .unwrap(),
    );
    let id = loop {
        if let Event::Complete { message, .. } = events.next().await.unwrap().unwrap()
            && let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64)
        {
            break id;
        }
    };
    agent
        .reply(answer(id, "first"), Cancellation::new())
        .await
        .unwrap();
    assert!(poll!(events.next()).is_pending());
    let before = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    let mut wrong = answer(id, "second");
    wrong.metadata.insert("session_id".into(), json!(0));
    assert_eq!(
        agent
            .reply(wrong, Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    let cancelled_reply = Cancellation::new();
    cancelled_reply.cancel();
    assert_eq!(
        agent
            .reply(answer(id, "second"), cancelled_reply)
            .await
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        before
    );
    cancellation.cancel();
    let tail = events.collect::<Vec<_>>().await;
    assert!(tail.iter().all(Result::is_ok));
    assert!(tail.iter().any(|event| matches!(event, Ok(Event::Complete {finish_reason:Some(reason), ..}) if reason=="cancelled")));
    assert_eq!(log.completed.load(Ordering::SeqCst), 0);
    let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(state["status"], "stopped");
    assert_eq!(record(&state)["status"], "waiting");
    assert_eq!(record(&state)["feedback"].as_array().unwrap().len(), 1);
    assert_eq!(
        agent
            .reply(answer(id, "second"), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
}

#[tokio::test]
async fn cancellation_keeps_partial_feedback_without_completing_the_tool_or_accepting_late_replies()
{
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let agent = agent::chat()
        .model(Model {
            requests: Default::default(),
            select: true,
        })
        .toolkit(Toolkit::new().tool(Collect::new(3, log.clone())))
        .checkpoint(checkpoint.clone())
        .with_id_fn(next_id)
        .build()
        .unwrap();
    cancelled(agent, checkpoint, log).await;
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let agent = agent::react()
        .model(Model {
            requests: Default::default(),
            select: true,
        })
        .toolkit(Toolkit::new().tool(Collect::new(3, log.clone())))
        .checkpoint(checkpoint.clone())
        .with_id_fn(next_id)
        .build()
        .unwrap();
    cancelled(agent, checkpoint, log).await;
}

#[tokio::test]
async fn feedback_without_an_actual_agent_step_fails_instead_of_creating_a_detached_wait() {
    assert_eq!(
        wait::ask(text(Role::Assistant, "question"), 3, &Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "INVALID_STATE"
    );
}

struct FailPartial(Arc<core::checkpoint::InMemory>);
impl Checkpoint for FailPartial {
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        if state["status"] == "waiting"
            && state["steps"].as_array().is_some_and(|steps| {
                steps.iter().any(|step| {
                    step["waits"].as_object().is_some_and(|waits| {
                        waits.values().any(|wait| {
                            wait["feedback"]
                                .as_array()
                                .is_some_and(|parts| !parts.is_empty())
                        })
                    })
                })
            })
        {
            return Err(Error::new(
                "SAVE_FAILED",
                "partial waiting snapshot save failed",
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

async fn failed_save<A: Agent>(
    agent: A,
    checkpoint: Arc<core::checkpoint::InMemory>,
    log: Arc<Log>,
) {
    let (mut events, id) = question(&agent, true).await;
    agent
        .reply(answer(id, "first"), Cancellation::new())
        .await
        .unwrap();
    assert_eq!(
        events.next().await.unwrap().unwrap_err().code,
        "SAVE_FAILED"
    );
    assert!(events.next().await.is_none());
    assert_eq!(log.completed.load(Ordering::SeqCst), 0);
    assert_eq!(
        agent
            .reply(answer(id, "second"), Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(state["status"], "failed");
    assert_eq!(record(&state)["feedback"].as_array().unwrap().len(), 1);
    assert_eq!(record(&state)["status"], "waiting");
}

#[tokio::test]
async fn partial_save_failure_stops_the_original_run_before_tool_completion() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let agent = agent::chat()
        .model(Model {
            requests: Default::default(),
            select: true,
        })
        .toolkit(Toolkit::new().tool(Collect::new(3, log.clone())))
        .checkpoint(FailPartial(checkpoint.clone()))
        .with_id_fn(next_id)
        .build()
        .unwrap();
    failed_save(agent, checkpoint, log).await;
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let log = Arc::new(Log::default());
    let agent = agent::react()
        .model(Model {
            requests: Default::default(),
            select: true,
        })
        .toolkit(Toolkit::new().tool(Collect::new(3, log.clone())))
        .checkpoint(FailPartial(checkpoint.clone()))
        .with_id_fn(next_id)
        .build()
        .unwrap();
    failed_save(agent, checkpoint, log).await;
}

#[tokio::test]
#[ignore = "explicitly invoked by ordinary_feedback_recovers_across_fresh_os_processes"]
async fn ordinary_feedback_recovery_worker() {
    let path = std::env::var("HEART_FEEDBACK_STATE").unwrap();
    let phase = std::env::var("HEART_FEEDBACK_PHASE").unwrap();
    let strategy = std::env::var("HEART_FEEDBACK_AGENT").unwrap();
    let stream = std::env::var("HEART_FEEDBACK_STREAM").unwrap() == "true";
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    if phase == "save" {
        let (_, state) = if strategy == "chat" {
            let agent = agent::chat()
                .model(Model {
                    requests: Default::default(),
                    select: true,
                })
                .toolkit(Toolkit::new().tool(Collect::new(3, Default::default())))
                .checkpoint(checkpoint.clone())
                .with_id_fn(next_id)
                .build()
                .unwrap();
            saved(&agent, &checkpoint, stream).await
        } else {
            let agent = agent::react()
                .model(Model {
                    requests: Default::default(),
                    select: true,
                })
                .toolkit(Toolkit::new().tool(Collect::new(3, Default::default())))
                .checkpoint(checkpoint.clone())
                .with_id_fn(next_id)
                .build()
                .unwrap();
            saved(&agent, &checkpoint, stream).await
        };
        std::fs::write(path, serde_json::to_vec(&state).unwrap()).unwrap();
    } else {
        let state: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let id = record(&state)["question"]["metadata"]["reply_id"]
            .as_i64()
            .unwrap();
        checkpoint
            .save(1001, state.clone(), &Cancellation::new())
            .await
            .unwrap();
        let log = Arc::new(Log::default());
        let requests = Arc::new(Mutex::new(vec![]));
        if strategy == "chat" {
            let agent = agent::chat()
                .model(Model {
                    requests: requests.clone(),
                    select: false,
                })
                .toolkit(Toolkit::new().tool(Collect::new(3, log.clone())))
                .checkpoint(checkpoint.clone())
                .with_id_fn(no_new_id)
                .build()
                .unwrap();
            restored(agent, checkpoint, log, requests, id, state, stream).await;
        } else {
            let agent = agent::react()
                .model(Model {
                    requests: requests.clone(),
                    select: false,
                })
                .toolkit(Toolkit::new().tool(Collect::new(3, log.clone())))
                .checkpoint(checkpoint.clone())
                .with_id_fn(no_new_id)
                .build()
                .unwrap();
            restored(agent, checkpoint, log, requests, id, state, stream).await;
        }
    }
}

#[test]
fn ordinary_feedback_recovers_across_fresh_os_processes() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "heart-feedback-process-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    for strategy in ["chat", "react"] {
        for stream in ["false", "true"] {
            let path = directory.join(format!("{strategy}-{stream}.json"));
            for phase in ["save", "restore"] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--ignored",
                        "--exact",
                        "ordinary_feedback_recovery_worker",
                        "--nocapture",
                    ])
                    .env("HEART_FEEDBACK_STATE", &path)
                    .env("HEART_FEEDBACK_PHASE", phase)
                    .env("HEART_FEEDBACK_AGENT", strategy)
                    .env("HEART_FEEDBACK_STREAM", stream)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{strategy}/{stream}/{phase}: {}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            std::fs::remove_file(path).unwrap();
        }
    }
    std::fs::remove_dir(directory).unwrap();
}

async fn live<A: Agent>(
    agent: A,
    checkpoint: Arc<core::checkpoint::InMemory>,
    log: Arc<Log>,
    requests: Arc<Mutex<Vec<Request>>>,
    count: usize,
    stream: bool,
) {
    let (mut events, id) = question(&agent, stream).await;
    let before = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(record(&before)["count"], count);
    assert_eq!(requests.lock().unwrap().len(), 1);
    for idx in 0..count {
        let reply = answer(id, &format!("part-{idx}"));
        agent
            .reply(reply.clone(), Cancellation::new())
            .await
            .unwrap();
        if idx + 1 < count {
            assert!(poll!(events.next()).is_pending());
            let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(
                record(&state)["feedback"].as_array().unwrap().len(),
                idx + 1
            );
            assert_eq!(record(&state)["status"], "waiting");
            assert_eq!(state["status"], "waiting");
            assert_eq!(state["query"], before["query"]);
            assert_eq!(requests.lock().unwrap().len(), 1);
            assert_eq!(log.completed.load(Ordering::SeqCst), 0);
            let mut retry = reply;
            retry.metadata.insert("count".into(), json!(1));
            agent.reply(retry, Cancellation::new()).await.unwrap();
            assert!(poll!(events.next()).is_pending());
            assert_eq!(
                checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
                state
            );
        }
    }
    let tail = events.collect::<Vec<_>>().await;
    assert!(tail.iter().all(Result::is_ok), "{tail:?}");
    assert_eq!(
        tail.iter()
            .filter(|event| matches!(event, Ok(Event::Complete { .. })))
            .count(),
        2,
        "only actual Tool result and final model, no feedback receipts"
    );
    if !stream {
        assert!(
            tail.iter()
                .all(|event| !matches!(event, Ok(Event::Delta { .. })))
        );
    }
    assert_eq!(log.started.load(Ordering::SeqCst), 1);
    assert_eq!(log.completed.load(Ordering::SeqCst), 1);
    assert_eq!(log.received.lock().unwrap().as_slice().len(), count);
    let final_state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(final_state["status"], "succeeded");
    assert_eq!(record(&final_state)["status"], "resolved");
    assert_eq!(final_state["waits"], json!([]));
    assert_eq!(final_state["usage"]["total"], 26);
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn chat_and_react_collect_one_or_three_parts_on_the_original_stream_in_both_modes() {
    for count in [1, 3] {
        for stream in [false, true] {
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let log = Arc::new(Log::default());
            let requests = Arc::new(Mutex::new(vec![]));
            let agent = agent::chat()
                .model(Model {
                    requests: requests.clone(),
                    select: true,
                })
                .toolkit(Toolkit::new().tool(Collect::new(count, log.clone())))
                .checkpoint(checkpoint.clone())
                .with_id_fn(next_id)
                .build()
                .unwrap();
            live(agent, checkpoint, log, requests, count, stream).await;
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let log = Arc::new(Log::default());
            let requests = Arc::new(Mutex::new(vec![]));
            let agent = agent::react()
                .model(Model {
                    requests: requests.clone(),
                    select: true,
                })
                .toolkit(Toolkit::new().tool(Collect::new(count, log.clone())))
                .checkpoint(checkpoint.clone())
                .with_id_fn(next_id)
                .build()
                .unwrap();
            live(agent, checkpoint, log, requests, count, stream).await;
        }
    }
}

async fn saved<A: Agent>(
    agent: &A,
    checkpoint: &core::checkpoint::InMemory,
    stream: bool,
) -> (i64, Value) {
    let (mut events, id) = question(agent, stream).await;
    agent
        .reply(answer(id, "first"), Cancellation::new())
        .await
        .unwrap();
    assert!(poll!(events.next()).is_pending());
    let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(record(&state)["feedback"].as_array().unwrap().len(), 1);
    drop(events);
    (id, state)
}

async fn restored<A: Agent>(
    agent: A,
    checkpoint: Arc<core::checkpoint::InMemory>,
    log: Arc<Log>,
    requests: Arc<Mutex<Vec<Request>>>,
    id: i64,
    previous: Value,
    stream: bool,
) {
    let mut events = Box::pin(
        agent
            .run(answer(id, "second"), !stream, Cancellation::new())
            .await
            .unwrap(),
    );
    assert!(poll!(events.next()).is_pending());
    let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(state["stream"], stream);
    assert_eq!(state["query"], previous["query"]);
    assert_eq!(state["usage"], previous["usage"]);
    assert_eq!(record(&state)["count"], 3);
    assert_eq!(record(&state)["question"], record(&previous)["question"]);
    assert_eq!(record(&state)["feedback"].as_array().unwrap().len(), 2);
    assert!(
        requests.lock().unwrap().is_empty(),
        "do not replay the selecting model"
    );
    assert_eq!(log.completed.load(Ordering::SeqCst), 0);
    agent
        .reply(answer(id, "first"), Cancellation::new())
        .await
        .unwrap();
    assert!(poll!(events.next()).is_pending());
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        state
    );
    agent
        .reply(answer(id, "third"), Cancellation::new())
        .await
        .unwrap();
    let tail = events.collect::<Vec<_>>().await;
    assert!(tail.iter().all(Result::is_ok), "{tail:?}");
    assert_eq!(
        tail.iter()
            .filter(|event| matches!(event, Ok(Event::Complete { .. })))
            .count(),
        2
    );
    if !stream {
        assert!(
            tail.iter()
                .all(|event| !matches!(event, Ok(Event::Delta { .. })))
        );
    }
    assert_eq!(log.completed.load(Ordering::SeqCst), 1);
    assert_eq!(log.received.lock().unwrap().as_slice().len(), 3);
    assert_eq!(requests.lock().unwrap().len(), 1);
    let terminal = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(terminal["usage"]["total"], 26);
    let cached = agent
        .run(answer(id, "first"), !stream, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(cached.iter().all(Result::is_ok));
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert_eq!(log.completed.load(Ordering::SeqCst), 1);
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        terminal
    );
}

#[tokio::test]
async fn fresh_chat_and_react_restore_partial_feedback_without_model_replay_or_new_question_ids() {
    for stream in [false, true] {
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let old = agent::chat()
            .model(Model {
                requests: Default::default(),
                select: true,
            })
            .toolkit(Toolkit::new().tool(Collect::new(3, Default::default())))
            .checkpoint(checkpoint.clone())
            .with_id_fn(next_id)
            .build()
            .unwrap();
        let (id, previous) = saved(&old, &checkpoint, stream).await;
        let log = Arc::new(Log::default());
        let requests = Arc::new(Mutex::new(vec![]));
        let agent = agent::chat()
            .model(Model {
                requests: requests.clone(),
                select: false,
            })
            .toolkit(Toolkit::new().tool(Collect::new(3, log.clone())))
            .checkpoint(checkpoint.clone())
            .with_id_fn(no_new_id)
            .build()
            .unwrap();
        restored(agent, checkpoint, log, requests, id, previous, stream).await;

        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let old = agent::react()
            .model(Model {
                requests: Default::default(),
                select: true,
            })
            .toolkit(Toolkit::new().tool(Collect::new(3, Default::default())))
            .checkpoint(checkpoint.clone())
            .with_id_fn(next_id)
            .build()
            .unwrap();
        let (id, previous) = saved(&old, &checkpoint, stream).await;
        let log = Arc::new(Log::default());
        let requests = Arc::new(Mutex::new(vec![]));
        let agent = agent::react()
            .model(Model {
                requests: requests.clone(),
                select: false,
            })
            .toolkit(Toolkit::new().tool(Collect::new(3, log.clone())))
            .checkpoint(checkpoint.clone())
            .with_id_fn(no_new_id)
            .build()
            .unwrap();
        restored(agent, checkpoint, log, requests, id, previous, stream).await;
    }
}
