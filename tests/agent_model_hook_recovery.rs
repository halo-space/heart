//! A successful model is not replayed to finish its Middleware or Agent after.
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::{self, Agent, Input, Payload, },
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
    model: AtomicUsize,
    execute: AtomicUsize,
    first: AtomicUsize,
    allow: AtomicBool,
    cancel: AtomicBool,
    fail: AtomicBool,
    omit_usage: AtomicBool,
    nonstream: AtomicBool,
    only_owner: AtomicBool,
    only_final: AtomicBool,
}
#[derive(Clone)]
struct Model {
    log: Arc<Log>,
    tools: bool,
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let tool = self.tools
            && !request
                .messages
                .as_slice()
                .iter()
                .any(|message| message.role == Role::Tool);
        let owner = request
            .messages
            .as_slice()
            .iter()
            .find(|message| message.role == Role::System)
            .and_then(|message| message.content.first())
            .and_then(|part| part.data.get("value"))
            .cloned();
        self.log.requests.lock().unwrap().push(request);
        let mut message = Message::new(Role::Assistant);
        if let Some(owner) = owner {
            message.values.insert("owner".into(), owner);
        }
        message.values.insert("data".into(), json!("raw"));
        message.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"raw model answer"})
                .as_object()
                .unwrap()
                .clone(),
        });
        if tool {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"name":"weather","call_id":"same_call","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: (!self.log.omit_usage.load(Ordering::SeqCst)).then_some(Usage {
                input: Some(10),
                output: Some(5),
                total: Some(15),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}
#[derive(Clone, Copy)]
enum Phase {
    Model,
    Execute,
}
struct Block {
    log: Arc<Log>,
    phase: Phase,
}
impl Block {
    async fn act(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        if self.log.only_final.load(Ordering::SeqCst) && value["values"]["final"] != true {
            return Ok(());
        }
        if self.log.only_owner.load(Ordering::SeqCst) && value["values"]["owner"] != "A" {
            return Ok(());
        }
        if value["values"]["owner"] == "B" {
            return Ok(());
        }
        if !self.log.allow.load(Ordering::SeqCst) {
            // Let intermediate selections proceed; this gate exercises final
            // processing after more than one real settled model call too.
            if value["content"]
                .as_array()
                .is_some_and(|parts| parts.iter().any(|part| part["type"] == "tool_call"))
            {
                return Ok(());
            }
            value["values"]["data"] = json!("uncommitted mutation");
            if self.log.cancel.load(Ordering::SeqCst) {
                c.cancel();
                std::future::pending().await
            } else if self.log.fail.load(Ordering::SeqCst) {
                Err(Error::new("HOOK_FAILED", "original after failure")
                    .with_details("user hook detail"))
            } else {
                Err(Error::new("WAITING", "after waits"))
            }
        } else {
            Ok(())
        }
    }
}
impl Middleware for Block {
    async fn after_model(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        self.log.model.fetch_add(1, Ordering::SeqCst);
        if matches!(self.phase, Phase::Model) {
            self.act(value, c).await?;
        }
        Ok(())
    }
    async fn after_execute(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        self.log.execute.fetch_add(1, Ordering::SeqCst);
        if matches!(self.phase, Phase::Execute) {
            self.act(value, c).await?;
        }
        Ok(())
    }
}
struct Transform(Arc<Log>);
impl Middleware for Transform {
    async fn after_model(&self, value: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.first.fetch_add(1, Ordering::SeqCst);
        // A valid Plan is a strict business schema, not an arbitrary Message
        // values bag. Preserve it while exercising the same lifecycle.
        if value["values"].get("version").is_none() {
            value["values"]["data"] = json!("model transformed");
        }
        // Reported consumption cannot be rewritten by middleware.
        if value["usage"].is_object() {
            value["usage"]["total"] = json!(9999);
        }
        Ok(())
    }
    async fn after_execute(&self, value: &mut Value, _: &Cancellation) -> Result<(), Error> {
        value["values"]["data"] = json!("agent transformed");
        Ok(())
    }
}
struct Weather(ToolDefinition);
impl Tool for Weather {
    fn definition(&self) -> &ToolDefinition {
        &self.0
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        let mut message = Message::new(Role::Tool);
        message.values.insert("data".into(), json!("sunny"));
        Ok(message)
    }
}
fn capabilities() -> impl halo::tool::Tool {
    Weather(ToolDefinition {
        name: "weather".into(),
        description: "weather".into(),
        parameters: json!({"type":"object","properties":{}})
            .as_object()
            .unwrap()
            .clone(),
    })
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
    phase: Phase,
    log: Arc<Log>,
    checkpoint: Arc<core::checkpoint::InMemory>,
    tools: bool,
    text: &str,
) -> Vec<Result<Event, Error>> {
    let model = Model {
        log: log.clone(),
        tools,
    };
    let stream = !log.nonstream.load(Ordering::SeqCst);
    if react {
        agent::react()
            .model(model)
            .tools(capabilities())
            .middleware(Block {
                log: log.clone(),
                phase,
            })
            .middleware(Transform(log))
            .checkpoint(checkpoint)
            .build()
            .unwrap()
            .run(input(text), stream, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await
    } else {
        agent::chat()
            .model(model)
            .tools(capabilities())
            .middleware(Block {
                log: log.clone(),
                phase,
            })
            .middleware(Transform(log))
            .checkpoint(checkpoint)
            .build()
            .unwrap()
            .run(input(text), stream, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await
    }
}
#[tokio::test]
async fn unfinished_after_reuses_the_raw_model_and_successful_middleware() {
    for react in [false, true] {
        for phase in [Phase::Model, Phase::Execute] {
            let log = Arc::new(Log::default());
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let events = run(
                react,
                phase,
                log.clone(),
                checkpoint.clone(),
                false,
                "original",
            )
            .await;
            assert!(
                matches!(events.last(), Some(Err(error)) if error.code == "WAITING"),
                "{events:?}"
            );
            let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(saved["status"], "waiting");
            let models = saved["steps"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|step| step["kind"] == "model")
                .collect::<Vec<_>>();
            assert_eq!(models.len(), 1);
            assert_eq!(models[0]["attempts"][0]["usage"]["total"], 15);
            assert_eq!(models[0]["message"]["values"]["data"], "model transformed");
            let repeated = run(
                react,
                phase,
                log.clone(),
                checkpoint.clone(),
                false,
                "still waiting",
            )
            .await;
            assert!(matches!(repeated.last(),Some(Err(error)) if error.code=="WAITING"));
            assert_eq!(log.requests.lock().unwrap().len(), 1);
            assert_eq!(log.first.load(Ordering::SeqCst), 1);
            log.allow.store(true, Ordering::SeqCst);
            let events = run(
                react,
                phase,
                log.clone(),
                checkpoint.clone(),
                false,
                "not model context",
            )
            .await;
            assert!(events.iter().all(Result::is_ok), "{events:?}");
            assert!(
                matches!(events.last(), Some(Ok(Event::Complete { message, .. })) if message.values["data"] == "agent transformed")
            );
            assert_eq!(
                log.requests.lock().unwrap().len(),
                1,
                "must not replay model"
            );
            assert_eq!(log.first.load(Ordering::SeqCst), 1);
            assert_eq!(
                checkpoint.load(1001, &Cancellation::new()).await.unwrap()["usage"]["total"],
                15
            );
        }
    }
}
#[tokio::test]
async fn after_execute_does_not_process_intermediate_tool_selection() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        log.allow.store(true, Ordering::SeqCst);
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let events = run(
            react,
            Phase::Execute,
            log.clone(),
            checkpoint.clone(),
            true,
            "weather",
        )
        .await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert_eq!(log.requests.lock().unwrap().len(), 2);
        assert_eq!(log.model.load(Ordering::SeqCst), 2);
        assert_eq!(log.execute.load(Ordering::SeqCst), 1);
        assert_eq!(
            checkpoint.load(1001, &Cancellation::new()).await.unwrap()["usage"]["total"],
            30
        );
    }
}

#[tokio::test]
async fn cancellation_after_restore_preserves_all_settled_calls() {
    for react in [false, true] {
        for phase in [Phase::Model, Phase::Execute] {
            for nonstream in [false, true] {
                for omit_usage in [false, true] {
                    let log = Arc::new(Log::default());
                    log.nonstream.store(nonstream, Ordering::SeqCst);
                    log.omit_usage.store(omit_usage, Ordering::SeqCst);
                    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                    let events = run(
                        react,
                        phase,
                        log.clone(),
                        checkpoint.clone(),
                        true,
                        "weather",
                    )
                    .await;
                    assert!(
                        matches!(events.last(), Some(Err(error)) if error.code == "WAITING"),
                        "{events:?}"
                    );
                    assert_eq!(log.requests.lock().unwrap().len(), 2);
                    log.cancel.store(true, Ordering::SeqCst);
                    let events = run(
                        react,
                        phase,
                        log.clone(),
                        checkpoint.clone(),
                        true,
                        "resume",
                    )
                    .await;
                    assert!(events.iter().all(Result::is_ok), "{events:?}");
                    let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
                    assert_eq!(saved["status"], "stopped");
                    assert_eq!(
                        log.requests.lock().unwrap().len(),
                        2,
                        "cancellation cannot replay model"
                    );
                    let expected = match phase {
                        Phase::Model => "model transformed",
                        Phase::Execute => "agent transformed",
                    };
                    assert!(
                        matches!(events.last(), Some(Ok(Event::Complete { message, finish_reason, .. })) if finish_reason.as_deref()==Some("cancelled") && message.values["data"]==expected),
                        "{react} {nonstream} {omit_usage}: {events:?}"
                    );
                    if !omit_usage {
                        assert_eq!(
                            saved["usage"]["total"], 30,
                            "must retain both settled calls"
                        );
                    } else {
                        assert!(
                            saved["usage"]["input"].as_u64().unwrap() > 0,
                            "use actual saved request to estimate missing input"
                        );
                        assert!(saved["usage"]["output"].as_u64().unwrap() > 0);
                        let model = saved["steps"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .rev()
                            .find(|step| step["kind"] == "model")
                            .unwrap();
                        assert_eq!(model["message"]["usage"], model["attempts"][0]["usage"]);
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn failed_after_preserves_original_error_and_successful_facts() {
    for react in [false, true] {
        for phase in [Phase::Model, Phase::Execute] {
            let log = Arc::new(Log::default());
            log.fail.store(true, Ordering::SeqCst);
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let events = run(
                react,
                phase,
                log.clone(),
                checkpoint.clone(),
                false,
                "original",
            )
            .await;
            assert!(
                matches!(events.last(), Some(Err(error)) if error.code=="HOOK_FAILED" && error.details.as_deref()==Some("user hook detail")),
                "{events:?}"
            );
            assert_eq!(log.requests.lock().unwrap().len(), 1);
            let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(saved["status"], "failed");
            let model = &saved["steps"][0];
            assert_eq!(model["message"]["values"]["data"], "model transformed");
            assert_eq!(model["attempts"][0]["status"], "succeeded");
            assert_eq!(saved["usage"]["total"], 15);
        }
    }
}

#[tokio::test]
async fn live_cancellation_delivers_the_last_acknowledged_transformation() {
    for react in [false, true] {
        for phase in [Phase::Model, Phase::Execute] {
            let log = Arc::new(Log::default());
            log.cancel.store(true, Ordering::SeqCst);
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let events = run(
                react,
                phase,
                log.clone(),
                checkpoint.clone(),
                false,
                "original",
            )
            .await;
            assert!(events.iter().all(Result::is_ok), "{events:?}");
            let expected = match phase {
                Phase::Model => "model transformed",
                Phase::Execute => "agent transformed",
            };
            assert!(
                matches!(events.last(),Some(Ok(Event::Complete{message,finish_reason,..})) if finish_reason.as_deref()==Some("cancelled") && message.values["data"]==expected),
                "{events:?}"
            );
            let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(saved["usage"]["total"], 15);
            assert_eq!(log.requests.lock().unwrap().len(), 1);
        }
    }
}

struct WaitSelection(Arc<Log>);
impl Middleware for WaitSelection {
    async fn after_model(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        if self.0.allow.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err(Error::new("WAITING", "selection after waits"))
        }
    }
}
struct CancelNext {
    log: Arc<Log>,
    cancellation: Cancellation,
}
impl Chat for CancelNext {
    type Stream = std::pin::Pin<Box<dyn futures_core::Stream<Item = Result<Event, Error>> + Send>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        if !request
            .messages
            .as_slice()
            .iter()
            .any(|message| message.role == Role::Tool)
        {
            return Ok(Box::pin(
                Model {
                    log: self.log.clone(),
                    tools: true,
                }
                .stream(request)
                .await?,
            ));
        }
        self.log.requests.lock().unwrap().push(request);
        let mut message = Message::new(Role::Assistant);
        message
            .values
            .insert("data".into(), json!("new call partial"));
        message.usage = Some(Usage {
            input: Some(7),
            output: Some(2),
            total: Some(9),
            ..Default::default()
        });
        let cancellation = self.cancellation.clone();
        Ok(Box::pin(
            stream::iter(vec![Ok(Event::Delta { message })]).chain(stream::once(async move {
                cancellation.cancel();
                std::future::pending::<Result<Event, Error>>().await
            })),
        ))
    }
}
#[tokio::test]
async fn new_call_after_restored_selection_settles_its_reported_partial_usage() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        for restoring in [false, true] {
            let cancellation = Cancellation::new();
            log.allow.store(restoring, Ordering::SeqCst);
            let model = CancelNext {
                log: log.clone(),
                cancellation: cancellation.clone(),
            };
            let events = if react {
                agent::react()
                    .model(model)
                    .middleware(WaitSelection(log.clone()))
                    .tools(capabilities())
                    .checkpoint(checkpoint.clone())
                    .build()
                    .unwrap()
                    .run(input("original"), true, cancellation)
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await
            } else {
                agent::chat()
                    .model(model)
                    .middleware(WaitSelection(log.clone()))
                    .tools(capabilities())
                    .checkpoint(checkpoint.clone())
                    .build()
                    .unwrap()
                    .run(input("original"), true, cancellation)
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await
            };
            if !restoring {
                assert!(matches!(events.last(),Some(Err(error)) if error.code=="WAITING"));
            } else {
                assert!(events.iter().all(Result::is_ok), "{events:?}");
                assert!(
                    matches!(events.last(),Some(Ok(Event::Complete{usage:Some(usage),finish_reason,..})) if usage.total==Some(9) && finish_reason.as_deref()==Some("cancelled"))
                );
                let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(
                    saved["usage"]["total"], 24,
                    "the new unsettled call's known usage is not an old refinement"
                );
                assert_eq!(saved["steps"][0]["attempts"][0]["usage"]["total"], 15);
                assert_eq!(log.requests.lock().unwrap().len(), 2);
            }
        }
    }
}

fn graph(react: bool, phase: Phase, log: Arc<Log>) -> halo::runtime::workflow::CompiledWorkflow {
    use halo::runtime::workflow::{
        WorkflowBuilder,
        graph::{END, START},
    };
    let mut builder = WorkflowBuilder::new("model-after-owners");
    for (node, prompt) in [("a", "A"), ("b", "B")] {
        let model = Model {
            log: log.clone(),
            tools: false,
        };
        if react {
            builder
                .add_agent_node(
                    node,
                    agent::react()
                        .model(model)
                        .system_prompt(prompt)
                        .middleware(Block {
                            log: log.clone(),
                            phase,
                        })
                        .middleware(Transform(log.clone()))
                        .build()
                        .unwrap(),
                )
                .unwrap();
        } else {
            builder
                .add_agent_node(
                    node,
                    agent::chat()
                        .model(model)
                        .system_prompt(prompt)
                        .middleware(Block {
                            log: log.clone(),
                            phase,
                        })
                        .middleware(Transform(log.clone()))
                        .build()
                        .unwrap(),
                )
                .unwrap();
        }
        builder.add_edge(START, node).unwrap();
    }
    builder.add_node("merge", halo::node::Echo).unwrap();
    builder.add_join(["a", "b"], "merge").unwrap();
    builder.add_edge("merge", END).unwrap();
    let child = builder.compile().unwrap();
    let mut parent = WorkflowBuilder::new("model-after-parent");
    parent.add_graph_node("child", &child).unwrap();
    parent.add_edge(START, "child").unwrap();
    parent.add_edge("child", END).unwrap();
    parent.compile().unwrap()
}
fn ids() -> i64 {
    static IDS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(60_500_000);
    IDS.fetch_add(1, Ordering::SeqCst)
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
async fn nested_workflow_retains_actual_owner_and_successful_sibling() {
    use halo::agent::Messages;
    use halo::runtime::workflow::Runtime;
    for react in [false, true] {
        for phase in [Phase::Model, Phase::Execute] {
            let log = Arc::new(Log::default());
            let runtime = Runtime::new(1001, graph(react, phase, log.clone())).unwrap();
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
            let model = &a["steps"][0];
            assert_eq!(model["kind"], "model");
            assert_eq!(model["message"]["values"]["owner"], "A");
            assert_eq!(model["attempts"][0]["status"], "succeeded");
            assert_eq!(
                model["stage"],
                match phase {
                    Phase::Model => "after",
                    Phase::Execute => "committed",
                }
            );
            assert_eq!(attempt(child, "b")["agent"]["status"], "succeeded");
            let b = child["nodes"]["b"].clone();
            assert_eq!(log.requests.lock().unwrap().len(), 2);
            log.allow.store(true, Ordering::SeqCst);
            let mut restored = Runtime::from_snapshot(
                1001,
                graph(react, phase, log.clone()),
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
            assert_eq!(done["status"], "succeeded");
            assert_eq!(attempt(&done, "child")["graph"]["nodes"]["b"], b);
            assert_eq!(
                log.requests.lock().unwrap().len(),
                2,
                "do not replay either model"
            );
            assert_eq!(
                log.first.load(Ordering::SeqCst),
                2,
                "do not replay successful after_model"
            );
            assert_eq!(done["usage"]["total"], 30);
        }
    }
}

#[tokio::test]
async fn malformed_model_after_is_rejected_without_work_or_checkpoint_writeback() {
    for react in [false, true] {
        let log = Arc::new(Log::default());
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        run(
            react,
            Phase::Model,
            log.clone(),
            checkpoint.clone(),
            false,
            "original",
        )
        .await;
        let original = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        for field in ["idx", "stage", "request", "message", "attempts"] {
            let mut invalid = original.clone();
            invalid["steps"][0][field] = match field {
                "idx" => json!(3),
                "stage" => json!("unknown"),
                "attempts" => json!([]),
                _ => Value::Null,
            };
            let backend = Arc::new(core::checkpoint::InMemory::new());
            backend
                .save(1001, invalid.clone(), &Cancellation::new())
                .await
                .unwrap();
            let model = Model {
                log: log.clone(),
                tools: false,
            };
            let error = if react {
                agent::react()
                    .model(model)
                    .middleware(Block {
                        log: log.clone(),
                        phase: Phase::Model,
                    })
                    .middleware(Transform(log.clone()))
                    .checkpoint(backend.clone())
                    .build()
                    .unwrap()
                    .run(input("resume"), true, Cancellation::new())
                    .await
                    .err()
                    .expect("invalid scene must fail")
            } else {
                agent::chat()
                    .model(model)
                    .middleware(Block {
                        log: log.clone(),
                        phase: Phase::Model,
                    })
                    .middleware(Transform(log.clone()))
                    .checkpoint(backend.clone())
                    .build()
                    .unwrap()
                    .run(input("resume"), true, Cancellation::new())
                    .await
                    .expect_err("invalid scene must fail")
            };
            assert_eq!(error.code, "INVALID_STATE", "{field}: {error:?}");
            assert_eq!(
                backend.load(1001, &Cancellation::new()).await.unwrap(),
                invalid
            );
            assert_eq!(log.requests.lock().unwrap().len(), 1);
            assert_eq!(log.first.load(Ordering::SeqCst), 1);
        }
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
            .find_map(|message| message.values.get("plan_node").and_then(Value::as_str));
        let mut message = Message::new(Role::Assistant);
        if let Some(node) = node {
            message
                .values
                .insert("owner".into(), json!(if node == "a" { "A" } else { "B" }));
        } else if request
            .messages
            .as_slice()
            .last()
            .is_some_and(|message| message.values.contains_key("plan_complete"))
        {
            message.values.insert("final".into(), json!(true));
        } else if self.0.requests.lock().unwrap().is_empty() {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"name":"plan","call_id":"plan","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        } else {
            message.values=json!({"version":1,"nodes":{"a":{"name":"research","objective":"first task"},"b":{"name":"research","objective":"second task"}},"edges":[]}).as_object().unwrap().clone();
        }
        self.0.requests.lock().unwrap().push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                input: Some(2),
                output: Some(1),
                total: Some(3),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}
#[tokio::test]
async fn planned_attempt_after_retains_models_and_does_not_replan() {
    for phase in [Phase::Model, Phase::Execute] {
        let log = Arc::new(Log::default());
        log.only_owner.store(true, Ordering::SeqCst);
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let build = || {
            agent::react()
                .model(Planned(log.clone()))
                .middleware(Block {
                    log: log.clone(),
                    phase,
                })
                .middleware(Transform(log.clone()))
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
        assert_eq!(saved["nodes"]["a"]["status"], "waiting");
        assert_eq!(saved["nodes"]["b"]["status"], "succeeded");
        let b = saved["nodes"]["b"].clone();
        let a = &saved["nodes"]["a"]["execution"];
        assert_eq!(a["attempts"][0]["steps"][0]["kind"], "model");
        assert_eq!(log.requests.lock().unwrap().len(), 4);
        log.allow.store(true, Ordering::SeqCst);
        let events = build()
            .run(input("resume"), false, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        let done = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(done["status"], "succeeded");
        assert_eq!(done["nodes"]["b"], b);
        assert_eq!(done["nodes"]["a"]["execution"]["exec_id"], a["exec_id"]);
        assert_eq!(
            done["nodes"]["a"]["execution"]["attempts"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            log.requests.lock().unwrap().len(),
            5,
            "only final inference is new"
        );
        assert_eq!(done["usage"]["total"], 15);
        assert_eq!(
            done["nodes"]["a"]["execution"]["attempts"][0]["usage"]["total"],
            3
        );
    }
}

#[tokio::test]
async fn planner_model_after_restores_its_original_planning_purpose() {
    let log = Arc::new(Log::default());
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let build = || {
        agent::react()
            .model(Planned(log.clone()))
            .middleware(Block {
                log: log.clone(),
                phase: Phase::Model,
            })
            .middleware(Transform(log.clone()))
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
    assert!(!saved["planning"].is_null());
    assert_eq!(log.requests.lock().unwrap().len(), 2);
    log.allow.store(true, Ordering::SeqCst);
    let events = build()
        .run(input("resume"), false, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    let done = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(done["status"], "succeeded");
    assert_eq!(done["nodes"]["a"]["status"], "succeeded");
    assert_eq!(done["nodes"]["b"]["status"], "succeeded");
    assert_eq!(
        log.requests.lock().unwrap().len(),
        5,
        "planning request must not replay"
    );
    assert_eq!(done["usage"]["total"], 15);
}

#[tokio::test]
async fn final_after_wait_restores_even_when_all_planned_nodes_already_succeeded() {
    for phase in [Phase::Model, Phase::Execute] {
        let log = Arc::new(Log::default());
        log.only_final.store(true, Ordering::SeqCst);
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let build = || {
            agent::react()
                .model(Planned(log.clone()))
                .middleware(Block {
                    log: log.clone(),
                    phase,
                })
                .middleware(Transform(log.clone()))
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
        assert_eq!(log.requests.lock().unwrap().len(), 5);
        log.allow.store(true, Ordering::SeqCst);
        let events = build()
            .run(input("resume"), false, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        let done = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(done["status"], "succeeded");
        assert_eq!(done["nodes"], saved["nodes"]);
        assert_eq!(done["usage"]["total"], 15);
        assert_eq!(
            log.requests.lock().unwrap().len(),
            5,
            "final output after-processing cannot rerun the plan or final model"
        );
        assert_eq!(log.first.load(Ordering::SeqCst), 5);
    }
}
