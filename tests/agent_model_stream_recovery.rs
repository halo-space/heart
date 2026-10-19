//! Unfinished output belongs to the actual Model Attempt, not a successful
//! Complete or a Tool dispatch. Closeout uses the existing checkpoint boundary.
use std::{
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use futures_core::Stream;
use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, SubAgent, toolkit::Toolkit},
    checkpoint::Checkpoint,
    core,
    event::Event,
    message::Role,
    middleware::Middleware,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Value, json};

#[derive(Default)]
struct Log {
    requests: Mutex<Vec<Request>>,
    before: AtomicUsize,
    after: AtomicUsize,
    tools: AtomicUsize,
}
#[derive(Default)]
struct Store {
    backend: core::checkpoint::InMemory,
    saves: AtomicUsize,
}
impl Checkpoint for Store {
    async fn save(&self, id: i64, state: Value, c: &Cancellation) -> Result<(), Error> {
        self.saves.fetch_add(1, Ordering::SeqCst);
        self.backend.save(id, state, c).await
    }
    async fn load(&self, id: i64, c: &Cancellation) -> Result<Value, Error> {
        self.backend.load(id, c).await
    }
    async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
        self.backend.delete(id, c).await
    }
}
struct Hooks(Arc<Log>);
impl Middleware for Hooks {
    async fn before_model(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.before.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn after_model(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.after.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
struct Lookup(Arc<Log>, ToolDefinition);
impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        &self.1
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.0.tools.fetch_add(1, Ordering::SeqCst);
        panic!("unfinished tool_call must not execute")
    }
}
fn part(kind: &str, data: Value) -> Part {
    Part {
        r#type: kind.into(),
        data: data.as_object().unwrap().clone(),
    }
}
fn partial(usage: Option<Usage>, padding: bool) -> Message {
    let mut m = Message::new(Role::Assistant);
    m.content = vec![
        part("think", json!({"value":"thinking"})),
        part("text", json!({"value":"unfinished answer"})),
        part(
            "tool_call",
            json!({"name":"lookup","call_id":"c1","arguments":"{\"query\":"}),
        ),
    ];
    m.usage = usage;
    if padding {
        m.metadata
            .insert("irrelevant".into(), json!("m".repeat(20_000)));
        m.values
            .insert("irrelevant".into(), json!("v".repeat(20_000)));
    }
    m
}
struct Model {
    log: Arc<Log>,
    error: Option<&'static str>,
    usage: Option<Usage>,
    cancel: Option<Cancellation>,
    padding: bool,
}
impl Chat for Model {
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let first = {
            let mut requests = self.log.requests.lock().unwrap();
            requests.push(request);
            requests.len() == 1
        };
        if !first {
            let mut message = Message::new(Role::Assistant);
            message
                .content
                .push(part("text", json!({"value":"authoritative answer"})));
            return Ok(Box::pin(stream::iter(vec![Ok(Event::Complete {
                message,
                usage: Some(Usage {
                    input: Some(12),
                    output: Some(8),
                    total: Some(20),
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
            })])));
        }
        let delta = stream::iter(vec![Ok(Event::Delta {
            message: partial(self.usage.clone(), self.padding),
        })]);
        let cancel = self.cancel.clone();
        let error = self.error;
        let end = stream::once(async move {
            if let Some(c) = cancel {
                c.cancel();
                std::future::pending::<()>().await;
            }
            error.map(|code| {
                Err(Error::new(code, "original failure").with_details("original detail"))
            })
        })
        .filter_map(std::future::ready);
        Ok(Box::pin(delta.chain(end)))
    }
}
fn input(text: &str) -> Input {
    serde_json::from_value(json!({
        "metadata":{"chat_id":1001,"session_id":2001,"agent_id":3001},
        "payload":[{"type":"text","data":{"value":text}}]
    }))
    .unwrap()
}
#[allow(clippy::too_many_arguments)]
async fn run(
    react: bool,
    log: Arc<Log>,
    store: Arc<Store>,
    mode: bool,
    error: Option<&'static str>,
    usage: Option<Usage>,
    cancel: bool,
    padding: bool,
    text: &str,
) -> Vec<Result<Event, Error>> {
    let c = Cancellation::new();
    let model = Model {
        log: log.clone(),
        error,
        usage,
        cancel: cancel.then(|| c.clone()),
        padding,
    };
    let tools = Toolkit::new().tool(Lookup(
        log.clone(),
        ToolDefinition {
            name: "lookup".into(),
            description: "lookup".into(),
            parameters: json!({"type":"object","properties":{}})
                .as_object()
                .unwrap()
                .clone(),
        },
    ));
    if react {
        agent::react()
            .model(model)
            .toolkit(tools)
            .middleware(Hooks(log))
            .with_max_iters(2)
            .checkpoint(store)
            .build()
            .unwrap()
            .run(input(text), mode, c)
            .await
            .unwrap()
            .collect()
            .await
    } else {
        agent::chat()
            .model(model)
            .toolkit(tools)
            .middleware(Hooks(log))
            .checkpoint(store)
            .build()
            .unwrap()
            .run(input(text), mode, c)
            .await
            .unwrap()
            .collect()
            .await
    }
}
fn model_step(scene: &Value) -> &Value {
    scene["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["kind"] == "model")
        .unwrap()
}
fn reported() -> Usage {
    Usage {
        input: Some(30),
        output: Some(10),
        total: Some(40),
        details: json!({"provider":"retained"}).as_object().unwrap().clone(),
        ..Default::default()
    }
}

#[tokio::test]
async fn waiting_keeps_partial_on_attempt_without_success_and_resumes_with_authoritative_complete()
{
    for react in [false, true] {
        for mode in [false, true] {
            for usage in [
                None,
                Some(reported()),
                Some(Usage {
                    input: Some(30),
                    ..Default::default()
                }),
            ] {
                let log = Arc::new(Log::default());
                let store = Arc::new(Store::default());
                let events = run(
                    react,
                    log.clone(),
                    store.clone(),
                    mode,
                    Some("WAITING"),
                    usage.clone(),
                    false,
                    false,
                    "original",
                )
                .await;
                assert!(
                    matches!(events.last(), Some(Err(e)) if e.code == "WAITING"),
                    "{events:?}"
                );
                assert!(
                    !events
                        .iter()
                        .any(|e| matches!(e, Ok(Event::Complete { .. })))
                );
                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                let step = model_step(&saved);
                assert_eq!(step["stage"], "call");
                assert!(step["message"].is_null());
                let attempt = &step["attempts"][0];
                assert_eq!(attempt["status"], "waiting");
                let first_usage: Usage = serde_json::from_value(attempt["usage"].clone()).unwrap();
                assert!(
                    first_usage.input.is_some()
                        && first_usage.output.is_some()
                        && first_usage.total.is_some()
                );
                if let Some(usage) = &usage {
                    assert_eq!(first_usage.input, usage.input);
                    assert_eq!(first_usage.details, usage.details);
                    if usage.total.is_some() {
                        assert_eq!(first_usage, *usage);
                    }
                }
                assert_eq!(
                    attempt["message"]["content"],
                    serde_json::to_value(partial(None, false).content).unwrap()
                );
                assert_eq!(attempt["message"]["usage"], attempt["usage"]);
                assert_eq!(log.after.load(Ordering::SeqCst), 0);
                assert_eq!(store.saves.load(Ordering::SeqCst), 1);
                let next = run(
                    react,
                    log.clone(),
                    store.clone(),
                    mode,
                    Some("WAITING"),
                    None,
                    false,
                    false,
                    "external input",
                )
                .await;
                assert!(next.iter().all(Result::is_ok), "{next:?}");
                let complete = next
                    .iter()
                    .filter_map(|event| match event {
                        Ok(Event::Complete { message, .. }) => Some(message),
                        _ => None,
                    })
                    .next_back()
                    .unwrap();
                assert_eq!(
                    complete.content,
                    vec![part("text", json!({"value":"authoritative answer"}))]
                );
                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                let attempts = model_step(&saved)["attempts"].as_array().unwrap();
                assert_eq!(attempts.len(), 2);
                assert_eq!(attempts[0], *attempt);
                assert_eq!(attempts[1]["status"], "succeeded");
                assert!(attempts[1].get("message").is_none());
                assert_eq!(saved["usage"]["total"], first_usage.total.unwrap() + 20);
                assert_eq!(
                    saved["query"],
                    serde_json::to_value(input("original")).unwrap()
                );
                let requests = log.requests.lock().unwrap();
                assert_eq!(requests.len(), 2);
                let messages = serde_json::to_string(&requests[1].messages).unwrap();
                assert!(messages.contains("external input"));
                assert!(!messages.contains("unfinished answer"));
                assert_eq!(log.before.load(Ordering::SeqCst), 2);
                assert_eq!(log.after.load(Ordering::SeqCst), 1);
                assert_eq!(log.tools.load(Ordering::SeqCst), 0);
                assert_eq!(store.saves.load(Ordering::SeqCst), 2);
            }
        }
    }
}

#[tokio::test]
async fn failure_and_eof_keep_received_output_cost_without_fabricating_success() {
    for react in [false, true] {
        for mode in [false, true] {
            for error in [Some("NETWORK"), None] {
                let log = Arc::new(Log::default());
                let store = Arc::new(Store::default());
                let events = run(
                    react,
                    log.clone(),
                    store.clone(),
                    mode,
                    error,
                    Some(reported()),
                    false,
                    false,
                    "original",
                )
                .await;
                let e = events.last().unwrap().as_ref().unwrap_err();
                assert_eq!(e.code, error.unwrap_or("INVALID_RESPONSE"));
                if error.is_some() {
                    assert_eq!(e.message, "original failure");
                    assert_eq!(e.details, Some("original detail".into()));
                }
                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                let step = model_step(&saved);
                assert_eq!(step["status"], "failed");
                assert_eq!(step["attempts"][0]["status"], "failed");
                assert_eq!(step["attempts"][0]["usage"]["total"], 40);
                assert_eq!(
                    step["attempts"][0]["message"]["content"],
                    serde_json::to_value(partial(None, false).content).unwrap()
                );
                assert_eq!(saved["usage"]["total"], 40);
                assert!(step["message"].is_null());
                assert_eq!(log.after.load(Ordering::SeqCst), 0);
                assert_eq!(log.tools.load(Ordering::SeqCst), 0);
                assert_eq!(store.saves.load(Ordering::SeqCst), 1);
            }
        }
    }
}

#[tokio::test]
async fn cancellation_retains_partial_usage_and_ignores_runtime_metadata_in_estimates() {
    for react in [false, true] {
        for mode in [false, true] {
            for usage in [None, Some(reported())] {
                let mut estimates = Vec::new();
                for padding in [false, true] {
                    let log = Arc::new(Log::default());
                    let store = Arc::new(Store::default());
                    let events = run(
                        react,
                        log.clone(),
                        store.clone(),
                        mode,
                        None,
                        usage.clone(),
                        true,
                        padding,
                        "original",
                    )
                    .await;
                    let (message, event_usage) = match events.last().unwrap().as_ref().unwrap() {
                        Event::Complete {
                            message,
                            usage,
                            finish_reason,
                        } => {
                            assert_eq!(finish_reason.as_deref(), Some("cancelled"));
                            (message, usage.as_ref().unwrap())
                        }
                        other => panic!("{other:?}"),
                    };
                    assert_eq!(message.content, partial(None, false).content);
                    if let Some(usage) = &usage {
                        assert_eq!(event_usage, usage);
                    }
                    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                    let step = model_step(&saved);
                    assert_eq!(step["attempts"][0]["status"], "cancelled");
                    assert_eq!(
                        step["attempts"][0]["usage"],
                        serde_json::to_value(event_usage).unwrap()
                    );
                    assert_eq!(
                        step["attempts"][0]["message"]["usage"],
                        step["attempts"][0]["usage"]
                    );
                    assert_eq!(saved["usage"]["total"], event_usage.total.unwrap());
                    assert!(step["message"].is_null());
                    assert_eq!(log.after.load(Ordering::SeqCst), 0);
                    assert_eq!(log.tools.load(Ordering::SeqCst), 0);
                    assert_eq!(store.saves.load(Ordering::SeqCst), 1);
                    estimates.push(event_usage.clone());
                }
                assert_eq!(estimates[0], estimates[1]);
            }
        }
    }
}

#[tokio::test]
async fn malformed_partial_facts_fail_before_new_work_or_checkpoint_writeback() {
    for react in [false, true] {
        for succeeded in [false, true] {
            let log = Arc::new(Log::default());
            let store = Arc::new(Store::default());
            run(
                react,
                log.clone(),
                store.clone(),
                true,
                Some("WAITING"),
                Some(reported()),
                false,
                false,
                "original",
            )
            .await;
            let mut saved = store.load(1001, &Cancellation::new()).await.unwrap();
            let attempt = &mut saved["steps"][0]["attempts"][0];
            if succeeded {
                attempt["status"] = json!("succeeded");
            } else {
                attempt["message"]["usage"]["total"] = json!(999);
            }
            store
                .save(1001, saved.clone(), &Cancellation::new())
                .await
                .unwrap();
            let model = Model {
                log: log.clone(),
                error: None,
                usage: None,
                cancel: None,
                padding: false,
            };
            let error = if react {
                agent::react()
                    .model(model)
                    .middleware(Hooks(log.clone()))
                    .checkpoint(store.clone())
                    .build()
                    .unwrap()
                    .run(input("next"), true, Cancellation::new())
                    .await
                    .err()
                    .unwrap()
            } else {
                agent::chat()
                    .model(model)
                    .middleware(Hooks(log.clone()))
                    .checkpoint(store.clone())
                    .build()
                    .unwrap()
                    .run(input("next"), true, Cancellation::new())
                    .await
                    .err()
                    .unwrap()
            };
            assert_eq!(error.code, "INVALID_STATE");
            assert_eq!(log.requests.lock().unwrap().len(), 1);
            assert_eq!(log.before.load(Ordering::SeqCst), 1);
            assert_eq!(store.saves.load(Ordering::SeqCst), 2);
            assert_eq!(store.load(1001, &Cancellation::new()).await.unwrap(), saved);
        }
    }
}

fn nested(
    react: bool,
    log: Arc<Log>,
    child_store: Arc<Store>,
) -> halo_agents::runtime::workflow::CompiledWorkflow {
    use halo_agents::runtime::workflow::{
        WorkflowBuilder,
        graph::{END, START},
    };
    let model = Model {
        log: log.clone(),
        error: Some("WAITING"),
        usage: Some(reported()),
        cancel: None,
        padding: false,
    };
    let mut child = WorkflowBuilder::new("partial-child");
    if react {
        child
            .add_agent_node(
                "model",
                agent::react()
                    .model(model)
                    .middleware(Hooks(log))
                    .checkpoint(child_store)
                    .build()
                    .unwrap(),
            )
            .unwrap();
    } else {
        child
            .add_agent_node(
                "model",
                agent::chat()
                    .model(model)
                    .middleware(Hooks(log))
                    .checkpoint(child_store)
                    .build()
                    .unwrap(),
            )
            .unwrap();
    }
    child.add_node("sibling", halo_agents::node::Echo).unwrap();
    child.add_node("merge", halo_agents::node::Echo).unwrap();
    child.add_edge(START, "model").unwrap();
    child.add_edge(START, "sibling").unwrap();
    child.add_join(["model", "sibling"], "merge").unwrap();
    child.add_edge("merge", END).unwrap();
    let child = child.compile().unwrap();
    let mut parent = WorkflowBuilder::new("partial-parent");
    parent.add_graph_node("child", &child).unwrap();
    parent.add_edge(START, "child").unwrap();
    parent.add_edge("child", END).unwrap();
    parent.compile().unwrap()
}
fn attempt<'a>(scene: &'a Value, node: &str) -> &'a Value {
    &scene["nodes"][node]["executions"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()["attempts"][0]
}
fn ids() -> i64 {
    static NEXT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(89_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
#[tokio::test]
async fn nested_workflow_retains_same_attempt_partial_cost_and_successful_sibling() {
    use halo_agents::{Messages, runtime::workflow::Runtime};
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let child_store = Arc::new(Store::default());
        let runtime = Runtime::new(1001, nested(react, log.clone(), child_store.clone())).unwrap();
        let mut message = Message::new(Role::User);
        message.metadata = input("original").metadata;
        let messages = Messages::new([message]);
        let events = runtime
            .run(messages.clone(), Arc::new(ids), Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(
            matches!(events.last(), Some(Err(e)) if e.code == "WAITING"),
            "{events:?}"
        );
        let saved = runtime.snapshot().unwrap();
        let graph = &attempt(&saved, "child")["graph"];
        let native = &attempt(graph, "model")["agent"];
        assert_eq!(model_step(native)["attempts"][0]["usage"]["total"], 40);
        assert_eq!(native["usage"]["total"], 40);
        let sibling = graph["nodes"]["sibling"].clone();
        let mut restored = Runtime::from_snapshot(
            1001,
            nested(react, log.clone(), child_store.clone()),
            serde_json::from_str(&saved.to_string()).unwrap(),
        )
        .unwrap();
        restored.resume("child", Messages::default()).unwrap();
        let events = restored
            .run(messages, Arc::new(ids), Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        let done = restored.snapshot().unwrap();
        let child = &attempt(&done, "child")["graph"];
        assert_eq!(child["nodes"]["sibling"], sibling);
        let before = attempt(graph, "model");
        let after = attempt(child, "model");
        assert_eq!(after["idx"], before["idx"]);
        assert_eq!(
            after["agent"]["steps"][0]["attempts"][0],
            native["steps"][0]["attempts"][0]
        );
        assert_eq!(done["usage"]["total"], 60);
        assert_eq!(log.requests.lock().unwrap().len(), 2);
        assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
    }
}

fn complete(value: &str, tokens: u64) -> Result<Event, Error> {
    let mut message = Message::new(Role::Assistant);
    message.content.push(part("text", json!({"value":value})));
    Ok(Event::Complete {
        message,
        usage: Some(Usage {
            total: Some(tokens),
            ..Default::default()
        }),
        finish_reason: Some("stop".into()),
    })
}
fn task(request: &Request) -> Option<&str> {
    request
        .messages
        .as_slice()
        .iter()
        .rev()
        .find_map(|m| m.values.get("plan_node").and_then(Value::as_str))
}
struct Planned {
    log: Arc<Log>,
    error: Option<&'static str>,
}
impl Chat for Planned {
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut log = self.log.requests.lock().unwrap();
        let events = if task(&request) == Some("a") && !log.iter().any(|r| task(r) == Some("a")) {
            let mut events = vec![Ok(Event::Delta {
                message: partial(Some(reported()), false),
            })];
            if let Some(code) = self.error {
                events.push(Err(
                    Error::new(code, "task original failure").with_details("task detail")
                ));
            }
            events
        } else if let Some(node) = task(&request) {
            vec![complete(
                &format!("{node} authoritative"),
                if node == "a" { 7 } else { 5 },
            )]
        } else if request.response_format.is_some() {
            let mut message = Message::new(Role::Assistant);
            message.values = json!({"version":1,"nodes":{
                "a":{"name":"research","objective":"first task"},
                "b":{"name":"research","objective":"second task"}
            },"edges":[]})
            .as_object()
            .unwrap()
            .clone();
            vec![Ok(Event::Complete {
                message,
                usage: Some(Usage {
                    total: Some(3),
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
            })]
        } else if log.is_empty() {
            let mut message = Message::new(Role::Assistant);
            message.content.push(part(
                "tool_call",
                json!({"name":"plan","call_id":"plan","arguments":{}}),
            ));
            vec![Ok(Event::Complete {
                message,
                usage: Some(Usage {
                    total: Some(2),
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
            })]
        } else {
            vec![complete("final authoritative", 11)]
        };
        log.push(request);
        Ok(Box::pin(stream::iter(events)))
    }
}
async fn run_plan(
    log: Arc<Log>,
    store: Arc<Store>,
    mode: bool,
    error: Option<&'static str>,
) -> Vec<Result<Event, Error>> {
    agent::react()
        .model(Planned {
            log: log.clone(),
            error,
        })
        .middleware(Hooks(log))
        .checkpoint(store)
        .with_id_fn(ids)
        .with_max_iters(6)
        .with_max_concurrency(2)
        .build()
        .unwrap()
        .run(input("original"), mode, Cancellation::new())
        .await
        .unwrap()
        .collect()
        .await
}
fn current(node: &Value) -> &Value {
    node["execution"]["attempts"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
}

#[tokio::test]
async fn planned_partial_wait_preserves_native_attempt_and_successful_sibling_without_replanning() {
    for mode in [false, true] {
        let log = Arc::new(Log::default());
        let store = Arc::new(Store::default());
        let events = run_plan(log.clone(), store.clone(), mode, Some("WAITING")).await;
        assert!(
            matches!(events.last(), Some(Err(e)) if e.code == "WAITING"),
            "{events:?}"
        );
        let saved = store.load(1001, &Cancellation::new()).await.unwrap();
        let a = &saved["nodes"]["a"];
        let old_attempt = current(a);
        let call = &model_step(old_attempt)["attempts"][0];
        assert_eq!(call["status"], "waiting");
        assert_eq!(call["usage"]["total"], 40);
        assert_eq!(
            call["message"]["content"],
            serde_json::to_value(partial(None, false).content).unwrap()
        );
        assert!(model_step(old_attempt)["message"].is_null());
        assert_eq!(saved["nodes"]["b"]["status"], "succeeded");
        assert_eq!(saved["usage"]["total"], 50);
        let events = run_plan(log.clone(), store.clone(), !mode, Some("WAITING")).await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        let done = store.load(1001, &Cancellation::new()).await.unwrap();
        let new_attempt = current(&done["nodes"]["a"]);
        assert_eq!(done["nodes"]["b"], saved["nodes"]["b"]);
        assert_eq!(
            done["nodes"]["a"]["execution"]["exec_id"],
            a["execution"]["exec_id"]
        );
        assert_eq!(new_attempt["idx"], old_attempt["idx"]);
        assert_eq!(new_attempt["created_time"], old_attempt["created_time"]);
        assert_eq!(
            done["nodes"]["a"]["execution"]["attempts"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(model_step(new_attempt)["attempts"][0], *call);
        assert_eq!(
            model_step(new_attempt)["attempts"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(done["usage"]["total"], 68);
        assert_eq!(done["query"], saved["query"]);
        assert_eq!(done["stream"], mode);
        let requests = log.requests.lock().unwrap();
        assert_eq!(requests.len(), 6);
        assert_eq!(
            requests
                .iter()
                .filter(|r| r.response_format.is_some())
                .count(),
            1
        );
        assert_eq!(requests.iter().filter(|r| task(r) == Some("b")).count(), 1);
        assert!(requests.iter().all(|r| {
            !serde_json::to_string(&r.messages)
                .unwrap()
                .contains("unfinished answer")
        }));
    }
}

#[tokio::test]
async fn planned_failure_and_eof_keep_partial_on_its_own_task_without_affecting_sibling() {
    for mode in [false, true] {
        for error in [Some("NETWORK"), None] {
            let log = Arc::new(Log::default());
            let store = Arc::new(Store::default());
            let events = run_plan(log.clone(), store.clone(), mode, error).await;
            let err = events.last().unwrap().as_ref().unwrap_err();
            assert_eq!(err.code, error.unwrap_or("INVALID_RESPONSE"));
            let saved = store.load(1001, &Cancellation::new()).await.unwrap();
            let a = current(&saved["nodes"]["a"]);
            assert_eq!(model_step(a)["attempts"][0]["status"], "failed");
            assert_eq!(model_step(a)["attempts"][0]["usage"]["total"], 40);
            assert!(model_step(a)["attempts"][0]["message"].is_object());
            assert_eq!(saved["nodes"]["b"]["status"], "succeeded");
            assert_eq!(saved["usage"]["total"], 50);
            assert_eq!(log.requests.lock().unwrap().len(), 4);
        }
    }
}

struct Parent(Arc<Log>);
impl Chat for Parent {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut log = self.0.requests.lock().unwrap();
        let event = if log.is_empty() {
            let mut message = Message::new(Role::Assistant);
            for name in ["a", "b"] {
                message.content.push(part("tool_call", json!({
                    "name":"agent_spawn","call_id":format!("delegate_{name}"),
                    "arguments":{"name":name,"input":{"payload":[{"type":"text","data":{"value":"child query"}}]}}
                })));
            }
            Ok(Event::Complete {
                message,
                usage: Some(Usage {
                    total: Some(2),
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
            })
        } else {
            complete("parent authoritative", 2)
        };
        log.push(request);
        Ok(stream::iter(vec![event]))
    }
}
struct Sibling(Arc<AtomicUsize>);
impl Chat for Sibling {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(stream::iter(vec![complete("sibling authoritative", 5)]))
    }
}
#[allow(clippy::too_many_arguments)]
async fn run_children(
    parent_react: bool,
    child_react: bool,
    parent: Arc<Log>,
    child: Arc<Log>,
    sibling: Arc<AtomicUsize>,
    store: Arc<Store>,
    child_store: Arc<Store>,
    mode: bool,
    error: Option<&'static str>,
) -> Vec<Result<Event, Error>> {
    let model = Model {
        log: child.clone(),
        error,
        usage: Some(reported()),
        cancel: None,
        padding: false,
    };
    let a = if child_react {
        SubAgent::new(
            "a",
            agent::react()
                .model(model)
                .middleware(Hooks(child))
                .checkpoint(child_store)
                .build()
                .unwrap(),
        )
        .unwrap()
    } else {
        SubAgent::new(
            "a",
            agent::chat()
                .model(model)
                .middleware(Hooks(child))
                .checkpoint(child_store)
                .build()
                .unwrap(),
        )
        .unwrap()
    };
    let b = SubAgent::new("b", agent::chat().model(Sibling(sibling)).build().unwrap()).unwrap();
    if parent_react {
        agent::react()
            .model(Parent(parent))
            .subagents(vec![a, b])
            .checkpoint(store)
            .with_id_fn(ids)
            .with_max_concurrency(2)
            .build()
            .unwrap()
            .run(input("original"), mode, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await
    } else {
        agent::chat()
            .model(Parent(parent))
            .subagents(vec![a, b])
            .checkpoint(store)
            .with_id_fn(ids)
            .with_max_concurrency(2)
            .build()
            .unwrap()
            .run(input("original"), mode, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await
    }
}
fn children(scene: &Value) -> &serde_json::Map<String, Value> {
    scene["execution"]["attempts"][0]["nodes"]
        .as_object()
        .unwrap()
}
#[tokio::test]
async fn delegated_partial_wait_retains_child_scene_identity_cost_and_successful_sibling() {
    for parent_react in [false, true] {
        for child_react in [false, true] {
            for mode in [false, true] {
                let parent = Arc::new(Log::default());
                let child = Arc::new(Log::default());
                let sibling = Arc::new(AtomicUsize::new(0));
                let store = Arc::new(Store::default());
                let child_store = Arc::new(Store::default());
                let events = run_children(
                    parent_react,
                    child_react,
                    parent.clone(),
                    child.clone(),
                    sibling.clone(),
                    store.clone(),
                    child_store.clone(),
                    mode,
                    Some("WAITING"),
                )
                .await;
                assert!(
                    matches!(events.last(), Some(Err(e)) if e.code == "WAITING"),
                    "{events:?}"
                );
                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                let (id, a) = children(&saved)
                    .iter()
                    .find(|(_, n)| n["status"] == "waiting")
                    .unwrap();
                let child_scene = &current(a)["agent"];
                let call = &model_step(child_scene)["attempts"][0];
                assert_eq!(call["usage"]["total"], 40);
                assert_eq!(
                    call["message"]["content"],
                    serde_json::to_value(partial(None, false).content).unwrap()
                );
                assert!(model_step(child_scene)["message"].is_null());
                let siblings = children(&saved)
                    .iter()
                    .filter(|(key, _)| *key != id)
                    .collect::<Vec<_>>();
                assert_eq!(siblings.len(), 1);
                assert_eq!(siblings[0].1["status"], "succeeded");
                assert_eq!(saved["usage"]["total"], 47);
                let events = run_children(
                    parent_react,
                    child_react,
                    parent.clone(),
                    child.clone(),
                    sibling.clone(),
                    store.clone(),
                    child_store.clone(),
                    !mode,
                    Some("WAITING"),
                )
                .await;
                assert!(events.iter().all(Result::is_ok), "{events:?}");
                let done = store.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(children(&done)[siblings[0].0], *siblings[0].1);
                let new = &children(&done)[id];
                assert_eq!(new["execution"]["exec_id"], a["execution"]["exec_id"]);
                assert_eq!(current(new)["idx"], current(a)["idx"]);
                assert_eq!(new["execution"]["attempts"].as_array().unwrap().len(), 1);
                let native = &current(new)["agent"];
                assert_eq!(model_step(native)["attempts"][0], *call);
                assert_eq!(model_step(native)["attempts"].as_array().unwrap().len(), 2);
                assert_eq!(native["usage"]["total"], 60);
                assert_eq!(done["usage"]["total"], 69);
                assert_eq!(done["query"], saved["query"]);
                assert_eq!(done["stream"], mode);
                assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
                assert_eq!(sibling.load(Ordering::SeqCst), 1);
                assert_eq!(parent.requests.lock().unwrap().len(), 2);
                assert_eq!(child.requests.lock().unwrap().len(), 2);
                assert_eq!(child.after.load(Ordering::SeqCst), 1);
                assert!(parent.requests.lock().unwrap().iter().all(|r| {
                    !serde_json::to_string(&r.messages)
                        .unwrap()
                        .contains("unfinished answer")
                }));
            }
        }
    }
}

#[tokio::test]
async fn delegated_technical_failure_keeps_partial_in_child_not_parent_context() {
    for parent_react in [false, true] {
        for child_react in [false, true] {
            for mode in [false, true] {
                let parent = Arc::new(Log::default());
                let child = Arc::new(Log::default());
                let sibling = Arc::new(AtomicUsize::new(0));
                let store = Arc::new(Store::default());
                let child_store = Arc::new(Store::default());
                let events = run_children(
                    parent_react,
                    child_react,
                    parent.clone(),
                    child.clone(),
                    sibling.clone(),
                    store.clone(),
                    child_store.clone(),
                    mode,
                    Some("NETWORK"),
                )
                .await;
                // A child failure is ordinary delegation feedback; the parent
                // can still reason about it instead of reporting child success.
                assert!(events.iter().all(Result::is_ok), "{events:?}");
                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                let a = children(&saved)
                    .values()
                    .find(|node| node["status"] == "failed")
                    .unwrap();
                let native = &current(a)["agent"];
                let attempt = &model_step(native)["attempts"][0];
                assert_eq!(attempt["status"], "failed");
                assert_eq!(attempt["usage"]["total"], 40);
                assert_eq!(attempt["error"]["code"], "NETWORK");
                assert_eq!(attempt["error"]["details"], "original detail");
                assert_eq!(
                    attempt["message"]["content"],
                    serde_json::to_value(partial(None, false).content).unwrap()
                );
                assert_eq!(saved["status"], "succeeded");
                assert_eq!(saved["usage"]["total"], 49);
                assert_eq!(child.after.load(Ordering::SeqCst), 0);
                assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
                assert_eq!(sibling.load(Ordering::SeqCst), 1);
                let requests = parent.requests.lock().unwrap();
                assert_eq!(requests.len(), 2);
                let context = serde_json::to_string(&requests[1].messages).unwrap();
                assert!(context.contains("NETWORK"));
                assert!(!context.contains("unfinished answer"));
            }
        }
    }
}

#[tokio::test]
async fn cancellation_after_planned_delta_keeps_actual_task_attempt_and_consumption() {
    let log = Arc::new(Log::default());
    let store = Arc::new(Store::default());
    let cancellation = Cancellation::new();
    let agent = agent::react()
        .model(Planned {
            log: log.clone(),
            error: None,
        })
        .middleware(Hooks(log.clone()))
        .checkpoint(store.clone())
        .with_id_fn(ids)
        .with_max_iters(6)
        .with_max_concurrency(2)
        .build()
        .unwrap();
    let mut events = agent
        .run(input("original"), true, cancellation.clone())
        .await
        .unwrap();
    loop {
        let event = events
            .next()
            .await
            .expect("planned task must produce its partial output")
            .unwrap();
        if matches!(event, Event::Delta { .. }) {
            break;
        }
    }
    cancellation.cancel();
    let tail = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        events.collect::<Vec<_>>(),
    )
    .await
    .unwrap();
    assert!(
        matches!(tail.last(), Some(Ok(Event::Complete { finish_reason: Some(reason), .. })) if reason == "cancelled"),
        "{tail:?}"
    );
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    let a = &saved["nodes"]["a"];
    let attempt = &model_step(current(a))["attempts"][0];
    assert_eq!(attempt["status"], "cancelled");
    assert_eq!(attempt["usage"]["total"], 40);
    assert_eq!(
        attempt["message"]["content"],
        serde_json::to_value(partial(None, false).content).unwrap()
    );
    assert!(model_step(current(a))["message"].is_null());
    assert_eq!(a["execution"]["attempts"].as_array().unwrap().len(), 1);
    assert_eq!(saved["status"], "stopped");
    let task_usage = saved["nodes"]
        .as_object()
        .unwrap()
        .values()
        .map(|n| n["usage"]["total"].as_u64().unwrap_or(0))
        .sum::<u64>();
    assert_eq!(saved["usage"]["total"], 5 + task_usage);
    assert_eq!(
        log.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| task(r) == Some("a"))
            .count(),
        1
    );
}
