//! Permission progress is native to the actual Tool Step, never a grant cache.
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::{
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
    permission::{Behavior, Decision, Permission, Request as PermissionRequest},
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Clone, Copy, Debug)]
enum Phase {
    Before,
    Check,
    After,
}
struct Log {
    phase: Phase,
    changed: bool,
    only_city: Option<&'static str>,
    before: AtomicUsize,
    after: AtomicUsize,
    tools_before: AtomicUsize,
    tools: AtomicUsize,
    allow: AtomicBool,
    fail: AtomicBool,
    cancel: AtomicBool,
    deny: AtomicBool,
    requests: Mutex<Vec<Request>>,
    checks: Mutex<Vec<PermissionRequest>>,
}
impl Log {
    fn new(phase: Phase, changed: bool) -> Self {
        Self {
            phase,
            changed,
            only_city: None,
            before: AtomicUsize::new(0),
            after: AtomicUsize::new(0),
            tools_before: AtomicUsize::new(0),
            tools: AtomicUsize::new(0),
            allow: AtomicBool::new(false),
            fail: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            deny: AtomicBool::new(false),
            requests: Mutex::new(Vec::new()),
            checks: Mutex::new(Vec::new()),
        }
    }
}
fn targets(log: &Log, request: &PermissionRequest) -> bool {
    if let Some(city) = log.only_city {
        return request.metadata["call"]["arguments"]["city"] == city;
    }
    !log.changed || request.metadata["call"]["arguments"]["city"] == "Paris"
}
struct Model(Arc<Log>);
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let complete = request
            .messages
            .as_slice()
            .iter()
            .any(|m| m.role == Role::Tool);
        let city = if request.messages.as_slice().iter().any(|m| {
            m.role == Role::System
                && m.content
                    .iter()
                    .any(|p| p.data.get("value") == Some(&json!("B")))
        }) {
            "Moscow"
        } else {
            "Beijing"
        };
        self.0.requests.lock().unwrap().push(request);
        let mut message = Message::new(Role::Assistant);
        message.content.push(Part {
            r#type: if complete { "text" } else { "tool_call" }.into(),
            data: if complete {
                json!({"value":"answer"})
            } else {
                json!({"call_id":"same_call","name":"lookup","arguments":{"city":city}})
            }
            .as_object()
            .unwrap()
            .clone(),
        });
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                input: Some(7),
                output: Some(3),
                total: Some(10),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}
struct Policy(Arc<Log>);
impl Permission for Policy {
    async fn check(
        &self,
        request: &PermissionRequest,
        _: &Cancellation,
    ) -> Result<Decision, Error> {
        self.0.checks.lock().unwrap().push(request.clone());
        assert_eq!(request.metadata["before"], 1);
        assert!(
            request.metadata.get("after").is_none(),
            "after changes must not become policy input"
        );
        if matches!(self.0.phase, Phase::Check)
            && targets(&self.0, request)
            && !self.0.allow.load(Ordering::SeqCst)
        {
            return Err(Error::new("WAITING", "policy itself waits"));
        }
        Ok(Decision {
            behavior: if self.0.deny.load(Ordering::SeqCst) {
                Behavior::Deny
            } else {
                Behavior::Allow
            },
            reason: Some("current policy".into()),
        })
    }
}
struct Before(Arc<Log>);
impl Middleware for Before {
    async fn before_permission(&self, input: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.before.fetch_add(1, Ordering::SeqCst);
        assert!(input["metadata"].get("before").is_none());
        input["metadata"]["before"] = json!(1);
        Ok(())
    }
}
struct After(Arc<Log>);
impl Middleware for After {
    async fn after_permission(&self, output: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.after.fetch_add(1, Ordering::SeqCst);
        assert!(output["metadata"].get("after").is_none());
        output["metadata"]["after"] = json!(1);
        Ok(())
    }
}
struct Gate(Arc<Log>);
impl Gate {
    async fn act(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        let request: PermissionRequest = serde_json::from_value(value.clone()).unwrap();
        if !targets(&self.0, &request) || self.0.allow.load(Ordering::SeqCst) {
            return Ok(());
        }
        value["metadata"]["uncommitted"] = json!(true);
        if self.0.cancel.load(Ordering::SeqCst) {
            c.cancel();
            std::future::pending().await
        } else if self.0.fail.load(Ordering::SeqCst) {
            Err(Error::new("HOOK_FAILED", "original permission error")
                .with_details("original detail"))
        } else {
            Err(Error::new("WAITING", "permission Hook waits"))
        }
    }
}
impl Middleware for Gate {
    async fn before_permission(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        if matches!(self.0.phase, Phase::Before) {
            self.act(value, c).await
        } else {
            Ok(())
        }
    }
    async fn after_permission(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        if matches!(self.0.phase, Phase::After) {
            self.act(value, c).await
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
    async fn before(&self, call: &mut Call, _: &Cancellation) -> Result<(), Error> {
        self.0.tools_before.fetch_add(1, Ordering::SeqCst);
        if self.0.changed {
            call.arguments.insert("city".into(), json!("Paris"));
        }
        Ok(())
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.0.tools.fetch_add(1, Ordering::SeqCst);
        let mut result = Message::new(Role::Tool);
        result.values.insert("found".into(), json!(true));
        Ok(result)
    }
}
fn capabilities(log: Arc<Log>) -> Toolkit {
    Toolkit::new().tools(Lookup(
        log,
        ToolDefinition {
            name: "lookup".into(),
            description: "lookup".into(),
            parameters:
                json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})
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
fn ids() -> i64 {
    static IDS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(83_000_000);
    IDS.fetch_add(1, Ordering::SeqCst)
}
async fn run(
    react: bool,
    mode: bool,
    log: Arc<Log>,
    checkpoint: Arc<core::checkpoint::InMemory>,
    text: &str,
) -> Result<Vec<Result<Event, Error>>, Error> {
    let work = async {
        if react {
            Ok(agent::react()
                .model(Model(log.clone()))
                .toolkit(capabilities(log.clone()))
                .permission(Policy(log.clone()))
                .middleware(Before(log.clone()))
                .middleware(Gate(log.clone()))
                .middleware(After(log))
                .checkpoint(checkpoint)
                .with_id_fn(ids)
                .with_max_iters(2)
                .build()?
                .run(input(text), mode, Cancellation::new())
                .await?
                .collect()
                .await)
        } else {
            Ok(agent::chat()
                .model(Model(log.clone()))
                .toolkit(capabilities(log.clone()))
                .permission(Policy(log.clone()))
                .middleware(Before(log.clone()))
                .middleware(Gate(log.clone()))
                .middleware(After(log))
                .checkpoint(checkpoint)
                .with_id_fn(ids)
                .build()?
                .run(input(text), mode, Cancellation::new())
                .await?
                .collect()
                .await)
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(10), work)
        .await
        .expect("permission must yield waiting/error, not invent a live question")
}
fn tool(scene: &Value) -> &Value {
    scene["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["kind"] == "tool")
        .unwrap()
}

#[tokio::test]
async fn permission_wait_resumes_native_methods_and_rechecks_current_policy() {
    for react in [false, true] {
        for mode in [false, true] {
            for phase in [Phase::Before, Phase::Check, Phase::After] {
                for changed in [false, true] {
                    let log = Arc::new(Log::new(phase, changed));
                    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                    let events = run(react, mode, log.clone(), checkpoint.clone(), "original")
                        .await
                        .unwrap();
                    assert!(
                        matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
                        "{react}/{phase:?}/{events:?}"
                    );
                    let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
                    let state = &tool(&saved)["permission"];
                    assert_eq!(
                        state["stage"],
                        match phase {
                            Phase::Before => "before",
                            Phase::Check => "check",
                            Phase::After => "after",
                        }
                    );
                    assert_eq!(
                        state["idx"],
                        if matches!(phase, Phase::Check) { 0 } else { 1 }
                    );
                    assert_eq!(state["input"]["metadata"]["before"], 1);
                    assert!(state["input"]["metadata"]["uncommitted"].is_null());
                    assert!(state["output"]["metadata"]["uncommitted"].is_null());
                    assert_eq!(log.tools.load(Ordering::SeqCst), 0);
                    assert!(saved["waits"].as_array().unwrap().is_empty());
                    let before = log.before.load(Ordering::SeqCst);
                    let after = log.after.load(Ordering::SeqCst);
                    let events = run(
                        react,
                        !mode,
                        log.clone(),
                        checkpoint.clone(),
                        "not model context",
                    )
                    .await
                    .unwrap();
                    assert!(matches!(events.last(),Some(Err(error)) if error.code=="WAITING"));
                    assert_eq!(log.before.load(Ordering::SeqCst), before);
                    assert_eq!(log.after.load(Ordering::SeqCst), after);
                    log.allow.store(true, Ordering::SeqCst);
                    let events = run(
                        react,
                        !mode,
                        log.clone(),
                        checkpoint.clone(),
                        "not model context",
                    )
                    .await
                    .unwrap();
                    assert!(
                        events.iter().all(Result::is_ok),
                        "{react}/{phase:?}/{events:?}"
                    );
                    assert_eq!(log.before.load(Ordering::SeqCst), before);
                    assert_eq!(log.after.load(Ordering::SeqCst), 1 + usize::from(changed));
                    assert_eq!(log.tools_before.load(Ordering::SeqCst), 1);
                    assert_eq!(log.tools.load(Ordering::SeqCst), 1);
                    let done = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
                    assert_eq!(done["usage"]["total"], 20);
                    assert_eq!(
                        done["query"],
                        serde_json::to_value(input("original")).unwrap()
                    );
                    assert_eq!(done["stream"], mode);
                    assert_eq!(tool(&done)["permission"]["stage"], "complete");
                    assert_eq!(tool(&done)["id"], tool(&saved)["id"]);
                    assert_eq!(log.requests.lock().unwrap().len(), 2);
                    assert!(log.requests.lock().unwrap().iter().all(|r| {
                        !serde_json::to_string(&r.messages)
                            .unwrap()
                            .contains("not model context")
                    }));
                }
            }
        }
    }
}

#[tokio::test]
async fn saved_allow_cannot_override_current_deny_or_replay_after_methods() {
    for react in [false, true] {
        let log = Arc::new(Log::new(Phase::After, false));
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        run(react, true, log.clone(), checkpoint.clone(), "original")
            .await
            .unwrap();
        log.deny.store(true, Ordering::SeqCst);
        log.allow.store(true, Ordering::SeqCst);
        let events = run(
            react,
            true,
            log.clone(),
            checkpoint.clone(),
            "not model context",
        )
        .await
        .unwrap();
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert_eq!(log.tools_before.load(Ordering::SeqCst), 0);
        assert_eq!(log.tools.load(Ordering::SeqCst), 0);
        assert_eq!(log.checks.lock().unwrap().len(), 2);
        assert_eq!(log.before.load(Ordering::SeqCst), 1);
        assert_eq!(log.after.load(Ordering::SeqCst), 1);
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(tool(&saved)["permission"]["decision"]["behavior"], "deny");
        assert!(
            log.requests.lock().unwrap()[1]
                .messages
                .as_slice()
                .iter()
                .any(|m| m
                    .values
                    .get("error")
                    .is_some_and(|error| error["code"] == "PERMISSION_DENIED"))
        );
    }
}

#[tokio::test]
async fn permission_hook_errors_remain_original_control_errors() {
    for react in [false, true] {
        for phase in [Phase::Before, Phase::After] {
            let log = Arc::new(Log::new(phase, false));
            log.fail.store(true, Ordering::SeqCst);
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let result = run(react, true, log.clone(), checkpoint.clone(), "original").await;
            let error = match result {
                Err(error) => error,
                Ok(events) => events.into_iter().last().unwrap().unwrap_err(),
            };
            assert_eq!(
                error,
                Error::new("HOOK_FAILED", "original permission error")
                    .with_details("original detail")
            );
            assert_eq!(log.requests.lock().unwrap().len(), 1);
            assert_eq!(log.tools.load(Ordering::SeqCst), 0);
            let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(tool(&saved)["error"], serde_json::to_value(error).unwrap());
            assert!(tool(&saved)["permission"]["input"]["metadata"]["uncommitted"].is_null());
        }
    }
}

#[tokio::test]
async fn permission_hook_cancel_preserves_model_usage_without_tool_success() {
    for react in [false, true] {
        for phase in [Phase::Before, Phase::After] {
            let log = Arc::new(Log::new(phase, false));
            log.cancel.store(true, Ordering::SeqCst);
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let events = run(react, true, log.clone(), checkpoint.clone(), "original")
                .await
                .unwrap();
            assert!(
                matches!(events.last(),Some(Ok(Event::Complete{finish_reason:Some(reason),..})) if reason=="cancelled"),
                "{events:?}"
            );
            assert_eq!(log.tools.load(Ordering::SeqCst), 0);
            let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(saved["usage"]["total"], 10);
            assert_eq!(tool(&saved)["status"], "cancelled");
        }
    }
}

#[tokio::test]
async fn invalid_permission_scene_is_rejected_before_work_and_writeback() {
    for react in [false, true] {
        for field in ["idx", "call", "decision", "output"] {
            let log = Arc::new(Log::new(Phase::Before, false));
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            run(react, true, log.clone(), checkpoint.clone(), "original")
                .await
                .unwrap();
            let mut saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            let step = saved["steps"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|s| s["kind"] == "tool")
                .unwrap();
            match field {
                "idx" => step["permission"]["idx"] = json!(4),
                "call" => {
                    step["permission"]["input"]["metadata"]["call"]["arguments"]["city"] =
                        json!("wrong")
                }
                "decision" => {
                    step["permission"]["decision"] = json!({"behavior":"allow","reason":null})
                }
                _ => step["permission"]["output"] = step["permission"]["input"].clone(),
            }
            checkpoint
                .save(1001, saved.clone(), &Cancellation::new())
                .await
                .unwrap();
            log.allow.store(true, Ordering::SeqCst);
            let result = run(react, true, log.clone(), checkpoint.clone(), "ignored").await;
            assert!(matches!(result,Err(error) if error.code=="INVALID_STATE"));
            assert_eq!(
                checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
                saved
            );
            assert_eq!(log.before.load(Ordering::SeqCst), 1);
            assert_eq!(log.requests.lock().unwrap().len(), 1);
            assert_eq!(log.tools.load(Ordering::SeqCst), 0);
        }
    }
}

#[tokio::test]
async fn pending_permission_cannot_silently_restore_without_its_policy() {
    for react in [false, true] {
        let log = Arc::new(Log::new(Phase::After, false));
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        run(react, true, log.clone(), checkpoint.clone(), "original")
            .await
            .unwrap();
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        let result = if react {
            agent::react()
                .model(Model(log.clone()))
                .toolkit(capabilities(log.clone()))
                .middleware(Before(log.clone()))
                .middleware(Gate(log.clone()))
                .middleware(After(log.clone()))
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("ignored"), true, Cancellation::new())
                .await
                .map(|_| ())
        } else {
            agent::chat()
                .model(Model(log.clone()))
                .toolkit(capabilities(log.clone()))
                .middleware(Before(log.clone()))
                .middleware(Gate(log.clone()))
                .middleware(After(log.clone()))
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("ignored"), true, Cancellation::new())
                .await
                .map(|_| ())
        };
        let error = result.expect_err("missing policy");
        assert_eq!(error.code, "INVALID_STATE");
        assert_eq!(
            checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
            saved
        );
        assert_eq!(log.requests.lock().unwrap().len(), 1);
        assert_eq!(log.tools.load(Ordering::SeqCst), 0);
    }
}

struct Planned(Arc<Log>);
impl Chat for Planned {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let node = request
            .messages
            .as_slice()
            .iter()
            .rev()
            .find_map(|m| m.values.get("plan_node").and_then(Value::as_str));
        let mut message = Message::new(Role::Assistant);
        if let Some(node) = node {
            if request
                .messages
                .as_slice()
                .iter()
                .any(|m| m.role == Role::Tool && m.metadata.get("name") == Some(&json!("lookup")))
            {
                message.values.insert("task".into(), json!(node));
            } else {
                message.content.push(Part{r#type:"tool_call".into(),data:json!({"call_id":"same_call","name":"lookup","arguments":{"city":if node=="a" {"Beijing"} else {"Moscow"}}}).as_object().unwrap().clone()});
            }
        } else if request.response_format.is_some() {
            message.values=json!({"version":1,"nodes":{"a":{"name":"research","objective":"first task"},"b":{"name":"research","objective":"second task"}},"edges":[]}).as_object().unwrap().clone();
        } else if self.0.requests.lock().unwrap().is_empty() {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"planning","name":"plan","arguments":{}})
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

#[tokio::test]
async fn planned_permission_wait_keeps_native_attempt_and_same_call_sibling_isolation() {
    let mut setup = Log::new(Phase::After, false);
    setup.only_city = Some("Beijing");
    let log = Arc::new(setup);
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let build = || {
        agent::react()
            .model(Planned(log.clone()))
            .toolkit(capabilities(log.clone()))
            .permission(Policy(log.clone()))
            .middleware(Before(log.clone()))
            .middleware(Gate(log.clone()))
            .middleware(After(log.clone()))
            .checkpoint(checkpoint.clone())
            .with_id_fn(ids)
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
    assert_eq!(saved["nodes"]["a"]["status"], "waiting");
    assert_eq!(saved["nodes"]["b"]["status"], "succeeded");
    let a = &saved["nodes"]["a"]["execution"];
    assert_eq!(
        tool(&a["attempts"][0])["permission"]["input"]["metadata"]["node_id"],
        "a"
    );
    assert_eq!(log.tools.load(Ordering::SeqCst), 1);
    assert_eq!(log.before.load(Ordering::SeqCst), 2);
    assert_eq!(log.after.load(Ordering::SeqCst), 2);
    log.allow.store(true, Ordering::SeqCst);
    let events = build()
        .run(input("not model context"), false, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    let done = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(done["nodes"]["b"], saved["nodes"]["b"]);
    assert_eq!(done["nodes"]["a"]["execution"]["exec_id"], a["exec_id"]);
    assert_eq!(
        done["nodes"]["a"]["execution"]["attempts"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(log.tools.load(Ordering::SeqCst), 2);
    assert_eq!(log.before.load(Ordering::SeqCst), 2);
    assert_eq!(log.after.load(Ordering::SeqCst), 2);
    assert_eq!(log.checks.lock().unwrap().len(), 3);
    assert_eq!(log.requests.lock().unwrap().len(), 7);
    assert_eq!(done["usage"]["total"], 70);
}

fn nested(react: bool, log: Arc<Log>) -> halo::runtime::workflow::CompiledWorkflow {
    use halo::runtime::workflow::{
        WorkflowBuilder,
        graph::{END, START},
    };
    let mut child = WorkflowBuilder::new("permission-child");
    for (name, prompt) in [("a", "A"), ("b", "B")] {
        if react {
            child
                .add_agent_node(
                    name,
                    agent::react()
                        .model(Model(log.clone()))
                        .system_prompt(prompt)
                        .toolkit(capabilities(log.clone()))
                        .permission(Policy(log.clone()))
                        .middleware(Before(log.clone()))
                        .middleware(Gate(log.clone()))
                        .middleware(After(log.clone()))
                        .build()
                        .unwrap(),
                )
                .unwrap();
        } else {
            child
                .add_agent_node(
                    name,
                    agent::chat()
                        .model(Model(log.clone()))
                        .system_prompt(prompt)
                        .toolkit(capabilities(log.clone()))
                        .permission(Policy(log.clone()))
                        .middleware(Before(log.clone()))
                        .middleware(Gate(log.clone()))
                        .middleware(After(log.clone()))
                        .build()
                        .unwrap(),
                )
                .unwrap();
        }
        child.add_edge(START, name).unwrap();
    }
    child.add_node("merge", halo::node::Echo).unwrap();
    child.add_join(["a", "b"], "merge").unwrap();
    child.add_edge("merge", END).unwrap();
    let child = child.compile().unwrap();
    let mut parent = WorkflowBuilder::new("permission-parent");
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
async fn nested_workflow_permission_progress_belongs_to_actual_agent_owner() {
    use halo::agent::Messages;
    use halo::runtime::workflow::Runtime;
    for react in [false, true] {
        let mut setup = Log::new(Phase::After, false);
        setup.only_city = Some("Beijing");
        let log = Arc::new(setup);
        let runtime = Runtime::new(1001, nested(react, log.clone())).unwrap();
        let mut user = Message::new(Role::User);
        user.metadata = input("original").metadata;
        let messages = Messages::new([user]);
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
        let b = child["nodes"]["b"].clone();
        assert_eq!(tool(a)["permission"]["stage"], "after");
        assert_eq!(tool(a)["permission"]["idx"], 1);
        assert_eq!(log.tools.load(Ordering::SeqCst), 1);
        assert_eq!(log.before.load(Ordering::SeqCst), 2);
        assert_eq!(log.after.load(Ordering::SeqCst), 2);
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
        assert_eq!(log.tools.load(Ordering::SeqCst), 2);
        assert_eq!(log.before.load(Ordering::SeqCst), 2);
        assert_eq!(log.after.load(Ordering::SeqCst), 2);
        assert_eq!(log.checks.lock().unwrap().len(), 3);
        assert_eq!(log.requests.lock().unwrap().len(), 4);
        assert_eq!(done["usage"]["total"], 40);
    }
}
