//! Model input middleware resumes from the actual native Model Step.
use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, Payload, toolkit::Toolkit},
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
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Default)]
struct Log {
    requests: Mutex<Vec<Request>>,
    first: AtomicUsize,
    gate: AtomicUsize,
    tools: AtomicUsize,
    allow: AtomicBool,
    fail: AtomicBool,
    cancel: AtomicBool,
    second: bool,
}
fn has_tool(request: &Request) -> bool {
    request
        .messages
        .as_slice()
        .iter()
        .any(|m| m.role == Role::Tool)
}
struct Model(Arc<Log>);
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        assert_eq!(
            request.options["marker"], 1,
            "successful before must not run twice"
        );
        let tool = self.0.second && !has_tool(&request);
        self.0.requests.lock().unwrap().push(request);
        let mut message = Message::new(Role::Assistant);
        message.content.push(Part {
            r#type: if tool { "tool_call" } else { "text" }.into(),
            data: if tool {
                json!({"name":"lookup","call_id":"c1","arguments":{}})
            } else {
                json!({"value":"answer"})
            }
            .as_object()
            .unwrap()
            .clone(),
        });
        Ok(stream::iter(
            [Ok(Event::Complete {
                message,
                usage: Some(Usage {
                    input: Some(7),
                    output: Some(3),
                    total: Some(10),
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
            })]
            .into_iter()
            .collect::<Vec<_>>(),
        ))
    }
}
struct First(Arc<Log>);
impl Middleware for First {
    async fn before_model(&self, request: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.first.fetch_add(1, Ordering::SeqCst);
        let marker = request["options"]["marker"].as_u64().unwrap_or(0);
        request["options"]["marker"] = json!(marker + 1);
        Ok(())
    }
}
struct Gate(Arc<Log>);
impl Middleware for Gate {
    async fn before_model(&self, request: &mut Value, c: &Cancellation) -> Result<(), Error> {
        let actual: Request = serde_json::from_value(request.clone()).unwrap();
        if self.0.second && !has_tool(&actual) {
            return Ok(());
        }
        self.0.gate.fetch_add(1, Ordering::SeqCst);
        assert_eq!(request["options"]["marker"], 1);
        if !self.0.allow.load(Ordering::SeqCst) {
            request["options"]["marker"] = json!(999);
            if self.0.cancel.load(Ordering::SeqCst) {
                c.cancel();
                std::future::pending().await
            } else if self.0.fail.load(Ordering::SeqCst) {
                Err(Error::new("INPUT_HOOK", "original input error")
                    .with_details("original detail"))
            } else {
                Err(Error::new("WAITING", "input waits"))
            }
        } else {
            Ok(())
        }
    }
}
struct Lookup(Arc<Log>, ToolDefinition);
impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        &self.1
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.0.tools.fetch_add(1, Ordering::SeqCst);
        let mut message = Message::new(Role::Tool);
        message.values.insert("found".into(), json!(true));
        Ok(message)
    }
}
fn toolkit(log: Arc<Log>) -> Toolkit {
    Toolkit::new().tools(Lookup(
        log,
        ToolDefinition {
            name: "lookup".into(),
            description: "lookup".into(),
            parameters: json!({"type":"object","properties":{}})
                .as_object()
                .unwrap()
                .clone(),
        },
    ))
}
fn input(text: &str) -> Input {
    Input {
        metadata: json!({"chat_id":1001,"session_id":2001,"agent_id":3001})
            .as_object()
            .unwrap()
            .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":text}),
            mime_type: None,
        }],
    }
}
async fn run(
    react: bool,
    log: Arc<Log>,
    checkpoint: Arc<core::checkpoint::InMemory>,
    mode: bool,
    text: &str,
) -> Result<Vec<Result<Event, Error>>, Error> {
    if react {
        Ok(agent::react()
            .model(Model(log.clone()))
            .toolkit(toolkit(log.clone()))
            .middleware(First(log.clone()))
            .middleware(Gate(log))
            .with_max_iters(2)
            .checkpoint(checkpoint)
            .build()?
            .run(input(text), mode, Cancellation::new())
            .await?
            .collect()
            .await)
    } else {
        Ok(agent::chat()
            .model(Model(log.clone()))
            .toolkit(toolkit(log.clone()))
            .middleware(First(log.clone()))
            .middleware(Gate(log))
            .checkpoint(checkpoint)
            .build()?
            .run(input(text), mode, Cancellation::new())
            .await?
            .collect()
            .await)
    }
}

#[tokio::test]
async fn before_wait_preserves_successful_request_without_replaying_prior_work() {
    for react in [false, true] {
        for mode in [false, true] {
            for second in [false, true] {
                let log = Arc::new(Log {
                    second,
                    ..Default::default()
                });
                let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                let first = run(react, log.clone(), checkpoint.clone(), mode, "original")
                    .await
                    .unwrap();
                assert!(
                    matches!(first.last(),Some(Err(error)) if error.code=="WAITING"),
                    "{first:?}"
                );
                let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
                let models = scene["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|s| s["kind"] == "model")
                    .collect::<Vec<_>>();
                let current = models.last().unwrap();
                assert_eq!(current["stage"], "before");
                assert_eq!(current["idx"], 1);
                assert_eq!(current["request"]["options"]["marker"], 1);
                assert!(current["attempts"].as_array().unwrap().is_empty());
                assert!(current["message"].is_null());
                assert_eq!(
                    scene["query"],
                    serde_json::to_value(input("original")).unwrap()
                );
                assert_eq!(log.requests.lock().unwrap().len(), usize::from(second));
                let repeated = run(
                    react,
                    log.clone(),
                    checkpoint.clone(),
                    !mode,
                    "not model context",
                )
                .await
                .unwrap();
                assert!(matches!(repeated.last(),Some(Err(error)) if error.code=="WAITING"));
                assert_eq!(log.first.load(Ordering::SeqCst), 1 + usize::from(second));
                log.allow.store(true, Ordering::SeqCst);
                let done = run(
                    react,
                    log.clone(),
                    checkpoint.clone(),
                    !mode,
                    "not model context",
                )
                .await
                .unwrap();
                assert!(done.iter().all(Result::is_ok), "{done:?}");
                assert_eq!(log.first.load(Ordering::SeqCst), 1 + usize::from(second));
                assert_eq!(log.tools.load(Ordering::SeqCst), usize::from(second));
                {
                    let requests = log.requests.lock().unwrap();
                    assert_eq!(requests.len(), 1 + usize::from(second));
                    assert!(requests.iter().all(|request| request.stream == mode));
                    assert!(requests.iter().all(|r| {
                        !serde_json::to_string(&r.messages)
                            .unwrap()
                            .contains("not model context")
                    }));
                }
                let final_scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(final_scene["stream"], mode);
                assert_eq!(
                    final_scene["usage"]["total"],
                    10 * (1 + usize::from(second))
                );
                let completed = final_scene["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|s| s["id"] == current["id"])
                    .unwrap();
                assert_eq!(completed["stage"], "committed");
                assert_eq!(completed["attempts"].as_array().unwrap().len(), 1);
            }
        }
    }
}

#[tokio::test]
async fn before_cancel_has_no_model_usage_or_fake_successful_attempt() {
    for react in [false, true] {
        for second in [false, true] {
            let log = Arc::new(Log {
                second,
                ..Default::default()
            });
            log.cancel.store(true, Ordering::SeqCst);
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let events = run(react, log.clone(), checkpoint.clone(), true, "original")
                .await
                .unwrap();
            assert!(
                matches!(events.last(),Some(Ok(Event::Complete{usage:None,finish_reason:Some(reason),..})) if reason=="cancelled"),
                "{events:?}"
            );
            let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(log.requests.lock().unwrap().len(), usize::from(second));
            if second {
                assert_eq!(saved["usage"]["total"], 10);
            } else {
                assert!(saved["usage"].is_null());
            }
            let current = saved["steps"]
                .as_array()
                .unwrap()
                .iter()
                .rev()
                .find(|s| s["kind"] == "model")
                .unwrap();
            assert_eq!(current["status"], "cancelled");
            assert!(current["attempts"].as_array().unwrap().is_empty());
            assert_eq!(current["request"]["options"]["marker"], 1);
        }
    }
}

#[tokio::test]
async fn before_error_preserves_the_original_error_and_successful_input() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        log.fail.store(true, Ordering::SeqCst);
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let result = run(react, log.clone(), checkpoint.clone(), true, "original").await;
        let error = match result {
            Err(error) => error,
            Ok(events) => events.into_iter().last().unwrap().unwrap_err(),
        };
        assert_eq!(
            error,
            Error::new("INPUT_HOOK", "original input error").with_details("original detail")
        );
        assert!(log.requests.lock().unwrap().is_empty());
        let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(
            scene["steps"][0]["error"],
            serde_json::to_value(error).unwrap()
        );
        assert_eq!(scene["steps"][0]["request"]["options"]["marker"], 1);
    }
}

#[tokio::test]
async fn malformed_before_scene_is_rejected_without_work_or_writeback() {
    for react in [false, true] {
        for corrupt in ["idx", "request", "attempts"] {
            let log = Arc::new(Log::default());
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            run(react, log.clone(), checkpoint.clone(), true, "original")
                .await
                .unwrap();
            let mut scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            match corrupt {
                "idx" => scene["steps"][0]["idx"] = json!(3),
                "request" => {
                    scene["steps"][0]["request"]["tool_choice"] =
                        json!({"Named":{"name":"missing"}})
                }
                _ => scene["steps"][0]["attempts"] = json!([{"status":"succeeded"}]),
            }
            checkpoint
                .save(1001, scene.clone(), &Cancellation::new())
                .await
                .unwrap();
            log.allow.store(true, Ordering::SeqCst);
            let result = run(react, log.clone(), checkpoint.clone(), true, "ignored").await;
            assert!(matches!(result,Err(error) if error.code=="INVALID_STATE"));
            assert_eq!(log.first.load(Ordering::SeqCst), 1);
            assert!(log.requests.lock().unwrap().is_empty());
            assert_eq!(
                checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
                scene
            );
        }
    }
}

struct Planned(Arc<Log>);
impl Chat for Planned {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        assert_eq!(request.options["marker"], 1);
        let task = request
            .messages
            .as_slice()
            .iter()
            .rev()
            .find_map(|m| m.values.get("plan_node").and_then(Value::as_str));
        let mut message = Message::new(Role::Assistant);
        if let Some(task) = task {
            message.values.insert("task".into(), json!(task));
        } else if request.response_format.is_some() {
            message.values = json!({"version":1,"nodes":{
                "a":{"name":"research","objective":"first task"},
                "b":{"name":"research","objective":"second task"}
            },"edges":[]})
            .as_object()
            .unwrap()
            .clone();
        } else if self.0.requests.lock().unwrap().is_empty() {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({
                    "name":"plan","call_id":"plan","arguments":{}
                })
                .as_object()
                .unwrap()
                .clone(),
            });
        } else {
            message.values.insert("final".into(), json!(true));
        }
        self.0.requests.lock().unwrap().push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                total: Some(10),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}
struct PlanGate(Arc<Log>, bool);
impl Middleware for PlanGate {
    async fn before_model(&self, request: &mut Value, c: &Cancellation) -> Result<(), Error> {
        let actual: Request = serde_json::from_value(request.clone()).unwrap();
        let target = if self.1 {
            actual.response_format.is_some()
        } else {
            actual
                .messages
                .as_slice()
                .iter()
                .any(|m| m.values.get("plan_node") == Some(&json!("a")))
        };
        if target {
            Gate(self.0.clone()).before_model(request, c).await
        } else {
            Ok(())
        }
    }
}
fn ids() -> i64 {
    static IDS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(72_000_000);
    IDS.fetch_add(1, Ordering::SeqCst)
}

#[tokio::test]
async fn plan_and_task_before_wait_keep_purpose_identity_and_successful_sibling() {
    for planner in [false, true] {
        let log = Arc::new(Log::default());
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let build = || {
            agent::react()
                .model(Planned(log.clone()))
                .middleware(First(log.clone()))
                .middleware(PlanGate(log.clone(), planner))
                .with_max_iters(5)
                .with_id_fn(ids)
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
        };
        let events = build()
            .run(input("original"), true, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(
            matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
            "{events:?}"
        );
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        let first_count = if planner { 2 } else { 4 };
        assert_eq!(log.first.load(Ordering::SeqCst), first_count);
        if planner {
            assert!(!saved["planning"].is_null());
            assert_eq!(log.requests.lock().unwrap().len(), 1);
        } else {
            assert_eq!(saved["nodes"]["a"]["status"], "waiting");
            assert_eq!(saved["nodes"]["b"]["status"], "succeeded");
            assert_eq!(
                saved["nodes"]["a"]["execution"]["attempts"][0]["steps"][0]["stage"],
                "before"
            );
            assert_eq!(log.requests.lock().unwrap().len(), 3);
        }
        log.allow.store(true, Ordering::SeqCst);
        let events = build()
            .run(input("not model context"), false, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        let done = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(done["status"], "succeeded");
        assert_eq!(done["usage"]["total"], 50);
        assert_eq!(log.requests.lock().unwrap().len(), 5);
        assert_eq!(log.first.load(Ordering::SeqCst), 5);
        if !planner {
            assert_eq!(done["nodes"]["b"], saved["nodes"]["b"]);
            assert_eq!(
                done["nodes"]["a"]["execution"]["exec_id"],
                saved["nodes"]["a"]["execution"]["exec_id"]
            );
            assert_eq!(
                done["nodes"]["a"]["execution"]["attempts"]
                    .as_array()
                    .unwrap()
                    .len(),
                1
            );
        }
        assert!(log.requests.lock().unwrap().iter().all(|r| {
            !serde_json::to_string(&r.messages)
                .unwrap()
                .contains("not model context")
        }));
    }
}

struct OwnerGate(Arc<Log>);
impl Middleware for OwnerGate {
    async fn before_model(&self, request: &mut Value, c: &Cancellation) -> Result<(), Error> {
        let actual: Request = serde_json::from_value(request.clone()).unwrap();
        if actual.messages.as_slice().iter().any(|m| {
            m.role == Role::System
                && m.content
                    .iter()
                    .any(|p| p.data.get("value") == Some(&json!("A")))
        }) {
            Gate(self.0.clone()).before_model(request, c).await?;
        }
        Ok(())
    }
}
fn nested(react: bool, log: Arc<Log>) -> halo_agents::runtime::workflow::CompiledWorkflow {
    use halo_agents::runtime::workflow::{
        WorkflowBuilder,
        graph::{END, START},
    };
    let mut child = WorkflowBuilder::new("model-before-child");
    for (node, prompt) in [("a", "A"), ("b", "B")] {
        if react {
            child
                .add_agent_node(
                    node,
                    agent::react()
                        .model(Model(log.clone()))
                        .system_prompt(prompt)
                        .middleware(First(log.clone()))
                        .middleware(OwnerGate(log.clone()))
                        .build()
                        .unwrap(),
                )
                .unwrap();
        } else {
            child
                .add_agent_node(
                    node,
                    agent::chat()
                        .model(Model(log.clone()))
                        .system_prompt(prompt)
                        .middleware(First(log.clone()))
                        .middleware(OwnerGate(log.clone()))
                        .build()
                        .unwrap(),
                )
                .unwrap();
        }
        child.add_edge(START, node).unwrap();
    }
    child.add_node("merge", halo_agents::node::Echo).unwrap();
    child.add_join(["a", "b"], "merge").unwrap();
    child.add_edge("merge", END).unwrap();
    let child = child.compile().unwrap();
    let mut parent = WorkflowBuilder::new("model-before-parent");
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
#[tokio::test]
async fn nested_workflow_before_restore_reuses_own_input_and_preserves_sibling() {
    use halo_agents::{Messages, runtime::workflow::Runtime};
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let runtime = Runtime::new(1001, nested(react, log.clone())).unwrap();
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
            matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
            "{events:?}"
        );
        let saved = runtime.snapshot().unwrap();
        let child = &attempt(&saved, "child")["graph"];
        let a = &attempt(child, "a")["agent"];
        assert_eq!(a["steps"][0]["stage"], "before");
        assert_eq!(a["steps"][0]["request"]["options"]["marker"], 1);
        let b = child["nodes"]["b"].clone();
        assert_eq!(log.requests.lock().unwrap().len(), 1);
        assert_eq!(log.first.load(Ordering::SeqCst), 2);
        log.allow.store(true, Ordering::SeqCst);
        let mut restored = Runtime::from_snapshot(
            1001,
            nested(react, log.clone()),
            serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap(),
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
        assert_eq!(attempt(&done, "child")["graph"]["nodes"]["b"], b);
        assert_eq!(log.requests.lock().unwrap().len(), 2);
        assert_eq!(log.first.load(Ordering::SeqCst), 2);
        assert_eq!(done["usage"]["total"], 20);
    }
}

struct FinalGate(Arc<Log>);
impl Middleware for FinalGate {
    async fn before_model(&self, request: &mut Value, c: &Cancellation) -> Result<(), Error> {
        let actual: Request = serde_json::from_value(request.clone()).unwrap();
        if actual
            .messages
            .as_slice()
            .iter()
            .any(|m| m.values.get("plan_complete") == Some(&json!(true)))
        {
            Gate(self.0.clone()).before_model(request, c).await?;
        }
        Ok(())
    }
}

#[tokio::test]
async fn final_model_before_resumes_when_all_plan_nodes_have_succeeded() {
    let log = Arc::new(Log::default());
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let build = || {
        agent::react()
            .model(Planned(log.clone()))
            .middleware(First(log.clone()))
            .middleware(FinalGate(log.clone()))
            .with_max_iters(5)
            .with_id_fn(ids)
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap()
    };
    let events = build()
        .run(input("original"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
        "{events:?}"
    );
    let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert!(
        saved["nodes"]
            .as_object()
            .unwrap()
            .values()
            .all(|node| node["status"] == "succeeded")
    );
    assert_eq!(log.requests.lock().unwrap().len(), 4);
    assert_eq!(log.first.load(Ordering::SeqCst), 5);
    log.allow.store(true, Ordering::SeqCst);
    let events = build()
        .run(input("not model context"), false, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    let done = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(done["nodes"], saved["nodes"]);
    assert_eq!(done["usage"]["total"], 50);
    assert_eq!(log.requests.lock().unwrap().len(), 5);
    assert_eq!(log.first.load(Ordering::SeqCst), 5);
}

struct ExternalModel(Arc<Log>);
impl Chat for ExternalModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        assert_eq!(request.options["marker"], 1);
        let first = {
            let mut requests = self.0.requests.lock().unwrap();
            let first = requests.is_empty();
            requests.push(request);
            first
        };
        let event = if first {
            Err(Error::new("WAITING", "Provider requests external input"))
        } else {
            Ok(Event::Complete {
                message: Message::new(Role::Assistant),
                usage: Some(Usage {
                    total: Some(10),
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
            })
        };
        Ok(stream::iter(vec![event]))
    }
}

#[tokio::test]
async fn provider_origin_wait_keeps_external_input_semantics_and_rechecks_new_input() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        log.allow.store(true, Ordering::SeqCst);
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        for (text, waiting) in [("original", true), ("new external input", false)] {
            let events = if react {
                agent::react()
                    .model(ExternalModel(log.clone()))
                    .middleware(First(log.clone()))
                    .middleware(Gate(log.clone()))
                    .checkpoint(checkpoint.clone())
                    .with_max_iters(2)
                    .build()
                    .unwrap()
                    .run(input(text), true, Cancellation::new())
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await
            } else {
                agent::chat()
                    .model(ExternalModel(log.clone()))
                    .middleware(First(log.clone()))
                    .middleware(Gate(log.clone()))
                    .checkpoint(checkpoint.clone())
                    .build()
                    .unwrap()
                    .run(input(text), true, Cancellation::new())
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await
            };
            if waiting {
                assert!(matches!(events.last(),Some(Err(error)) if error.code=="WAITING"));
            } else {
                assert!(events.iter().all(Result::is_ok), "{events:?}");
            }
        }
        assert_eq!(log.first.load(Ordering::SeqCst), 2);
        assert_eq!(log.gate.load(Ordering::SeqCst), 2);
        assert!(
            serde_json::to_string(&log.requests.lock().unwrap()[1].messages)
                .unwrap()
                .contains("new external input")
        );
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(
            saved["query"],
            serde_json::to_value(input("original")).unwrap()
        );
        assert_eq!(saved["steps"][0]["attempts"].as_array().unwrap().len(), 2);
        assert_eq!(saved["steps"][0]["attempts"][0]["status"], "waiting");
        assert_eq!(saved["steps"][0]["attempts"][1]["status"], "succeeded");
        assert_eq!(saved["usage"]["total"], 10);
    }
}
