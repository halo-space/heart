//! Preparation belongs to the actual Agent invocation, not its first Model.
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
};

use chat::Store as _;
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::checkpoint::Checkpoint,
    agent::event::Event,
    agent::memory::{
        self,
        short::{chat, session, trace},
    },
    agent::message::Role,
    agent::middleware::Middleware,
    agent::model::chat::{Chat, Request},
    agent::{self, Agent, Input, Payload},
    core,
    runtime::workflow::{
        CompiledWorkflow, Runtime, WorkflowBuilder,
        graph::{END, START},
    },
};
use serde_json::{Value, json};
use session::Store as _;
use trace::Store as _;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Stage {
    Execute,
    BeforePrompt,
    AfterPrompt,
    History,
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum Stop {
    Waiting,
    Failure,
    Cancel,
}
#[derive(Clone)]
struct Gate {
    stage: Stage,
    stop: Stop,
    hits: Arc<AtomicUsize>,
    execute: Arc<AtomicUsize>,
    prompt: Arc<AtomicUsize>,
}
impl Gate {
    fn new(stage: Stage, stop: Stop) -> Self {
        Self {
            stage,
            stop,
            hits: Default::default(),
            execute: Default::default(),
            prompt: Default::default(),
        }
    }
    async fn stop(&self, stage: Stage, c: &Cancellation) -> Result<(), Error> {
        if self.stage != stage || self.hits.fetch_add(1, Ordering::SeqCst) > 0 {
            return Ok(());
        }
        match self.stop {
            Stop::Waiting => Err(Error::new("WAITING", "preparation not ready")),
            Stop::Failure => {
                Err(Error::new("BACKEND", "preparation failed").with_details("original details"))
            }
            Stop::Cancel => {
                c.cancel();
                std::future::pending().await
            }
        }
    }
}
impl Middleware for Gate {
    async fn before_execute(&self, input: &mut Value, c: &Cancellation) -> Result<(), Error> {
        let original = input.clone();
        input["payload"][0]["data"]["value"] = json!("uncommitted mutation");
        self.stop(Stage::Execute, c).await?;
        *input = original;
        Ok(())
    }
    async fn before_system_prompt(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        let original = value.clone();
        *value = json!("uncommitted prompt");
        self.stop(Stage::BeforePrompt, c).await?;
        *value = original;
        Ok(())
    }
    async fn after_system_prompt(&self, _: &mut Value, c: &Cancellation) -> Result<(), Error> {
        self.stop(Stage::AfterPrompt, c).await
    }
}
struct Transform(Gate);
impl Middleware for Transform {
    async fn before_execute(&self, input: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.execute.fetch_add(1, Ordering::SeqCst);
        input["payload"][0]["data"]["value"] = json!("effective input");
        Ok(())
    }
    async fn before_system_prompt(&self, value: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.prompt.fetch_add(1, Ordering::SeqCst);
        *value = json!("effective prompt");
        Ok(())
    }
}
#[derive(Clone, Default)]
struct Model(Arc<Mutex<Vec<Request>>>);
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.0.lock().unwrap().push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: Message::new(Role::Assistant),
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}
struct Sessions {
    inner: core::memory::short::session::InMemory,
    gate: Gate,
}
impl session::Store for Sessions {
    async fn create(
        &self,
        value: session::Session,
        c: &Cancellation,
    ) -> Result<session::Session, Error> {
        self.inner.create(value, c).await
    }
    async fn read(&self, id: i64, c: &Cancellation) -> Result<session::Session, Error> {
        self.gate.stop(Stage::History, c).await?;
        self.inner.read(id, c).await
    }
    async fn update(
        &self,
        value: session::Session,
        c: &Cancellation,
    ) -> Result<session::Session, Error> {
        self.inner.update(value, c).await
    }
    async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, c).await
    }
}
type Memory = memory::Config<
    Sessions,
    core::memory::short::chat::InMemory,
    core::memory::short::trace::InMemory,
    core::memory::InMemory,
>;
async fn memory(gate: Gate) -> Memory {
    let memory = Memory::new(
        Sessions {
            inner: Default::default(),
            gate,
        },
        Default::default(),
        Default::default(),
        Default::default(),
    );
    memory
        .session
        .create(
            session::Session {
                id: 4,
                tenant_id: 1,
                user_id: 2,
                r#type: memory::short::Type::Agent,
                title: None,
                summary: None,
                usage: None,
                status: session::Status::Active,
                metadata: Default::default(),
                created_time: 0,
                updated_time: 0,
            },
            &Cancellation::new(),
        )
        .await
        .unwrap();
    memory
}
fn input() -> Input {
    Input {
        metadata: json!({"tenant_id":1,"user_id":2,"agent_id":3,"session_id":4,"chat_id":1001})
            .as_object()
            .unwrap()
            .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"original input"}),
            mime_type: None,
        }],
    }
}
async fn run(
    react: bool,
    gate: Gate,
    model: Model,
    memory: Memory,
    checkpoint: Arc<core::checkpoint::InMemory>,
    input: Input,
) -> Result<Vec<Result<Event, Error>>, Error> {
    // Rebuild the Agent each time: continuation must come from its snapshot.
    if react {
        let agent = agent::react()
            .model(model)
            .middleware(Transform(gate.clone()))
            .middleware(gate)
            .memory(memory)
            .checkpoint(checkpoint)
            .build()?;
        Ok(agent
            .run(input, true, Cancellation::new())
            .await?
            .collect()
            .await)
    } else {
        let agent = agent::chat()
            .model(model)
            .middleware(Transform(gate.clone()))
            .middleware(gate)
            .memory(memory)
            .checkpoint(checkpoint)
            .build()?;
        Ok(agent
            .run(input, true, Cancellation::new())
            .await?
            .collect()
            .await)
    }
}

#[tokio::test]
async fn waiting_preparation_resumes_only_unfinished_methods_without_consuming_feedback() {
    for react in [false, true] {
        for stage in [
            Stage::Execute,
            Stage::BeforePrompt,
            Stage::AfterPrompt,
            Stage::History,
        ] {
            let gate = Gate::new(stage, Stop::Waiting);
            let memory = memory(gate.clone()).await;
            let checkpoint = Arc::new(core::checkpoint::InMemory::new());
            let model = Model::default();
            let first = run(
                react,
                gate.clone(),
                model.clone(),
                memory.clone(),
                checkpoint.clone(),
                input(),
            )
            .await
            .unwrap();
            assert_eq!(
                first.last().unwrap().as_ref().unwrap_err().code,
                "WAITING",
                "{react}/{stage:?}"
            );
            let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
            assert_eq!(saved["status"], "waiting");
            assert_eq!(saved["query"], serde_json::to_value(input()).unwrap());
            assert_eq!(
                saved["preparation"]["input"]["payload"][0]["data"]["value"],
                "effective input"
            );
            assert!(saved["steps"].as_array().is_none_or(Vec::is_empty));
            assert!(saved["execution"].is_null());
            assert!(saved["waits"].as_array().unwrap().is_empty());
            assert_eq!(
                memory
                    .chat
                    .read(1001, &Cancellation::new())
                    .await
                    .unwrap()
                    .status,
                chat::Status::Running
            );
            assert_eq!(
                memory
                    .trace
                    .read(1001, &Cancellation::new())
                    .await
                    .unwrap_err()
                    .code,
                "NOT_FOUND"
            );
            let mut continuation = input();
            continuation.payload[0].data = json!({"value":"not a model message"});
            let resumed = run(
                react,
                gate.clone(),
                model.clone(),
                memory.clone(),
                checkpoint.clone(),
                continuation,
            )
            .await
            .unwrap();
            assert!(
                resumed.iter().all(Result::is_ok),
                "{react}/{stage:?}: {resumed:?}"
            );
            assert_eq!(gate.execute.load(Ordering::SeqCst), 1);
            assert_eq!(gate.prompt.load(Ordering::SeqCst), 1);
            let json = {
                let requests = model.0.lock().unwrap();
                assert_eq!(requests.len(), 1);
                serde_json::to_string(&requests[0]).unwrap()
            };
            assert!(
                json.contains("effective input") && json.contains("effective prompt"),
                "{json}"
            );
            assert!(
                !json.contains("uncommitted") && !json.contains("not a model message"),
                "{json}"
            );
            assert_eq!(
                checkpoint.load(1001, &Cancellation::new()).await.unwrap()["status"],
                "succeeded"
            );
        }
    }
}

#[tokio::test]
async fn preparation_failures_and_cancellation_close_the_original_chat_and_checkpoint() {
    for react in [false, true] {
        for stage in [
            Stage::Execute,
            Stage::BeforePrompt,
            Stage::AfterPrompt,
            Stage::History,
        ] {
            for stop in [Stop::Failure, Stop::Cancel] {
                let gate = Gate::new(stage, stop);
                let memory = memory(gate.clone()).await;
                let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                let model = Model::default();
                let result = run(
                    react,
                    gate.clone(),
                    model.clone(),
                    memory.clone(),
                    checkpoint.clone(),
                    input(),
                )
                .await;
                if stop == Stop::Failure {
                    let error = match result {
                        Err(error) => error,
                        Ok(events) => events.into_iter().find_map(Result::err).unwrap(),
                    };
                    assert_eq!(error.code, "BACKEND");
                    assert_eq!(error.details.as_deref(), Some("original details"));
                } else {
                    assert!(
                        matches!(result.unwrap().last(), Some(Ok(Event::Complete { finish_reason: Some(reason), .. })) if reason == "cancelled")
                    );
                }
                let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(
                    saved["status"],
                    if stop == Stop::Cancel {
                        "stopped"
                    } else {
                        "failed"
                    }
                );
                let chat = memory.chat.read(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(
                    chat.status,
                    if stop == Stop::Cancel {
                        chat::Status::Stopped
                    } else {
                        chat::Status::Failed
                    }
                );
                assert_eq!(
                    chat.input.unwrap().payload[0].data,
                    json!({"value":"original input"})
                );
                let trace = memory.trace.read(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(trace.state["preparation"], saved["preparation"]);
                assert!(model.0.lock().unwrap().is_empty());
                let hits = gate.hits.load(Ordering::SeqCst);
                let _ = run(
                    react,
                    gate.clone(),
                    model.clone(),
                    memory.clone(),
                    checkpoint.clone(),
                    input(),
                )
                .await;
                assert_eq!(
                    gate.hits.load(Ordering::SeqCst),
                    hits,
                    "terminal replay must not rerun preparation"
                );
                assert_eq!(
                    checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
                    saved
                );
            }
        }
    }
}

#[tokio::test]
async fn invalid_preparation_is_rejected_before_hooks_or_memory_writeback() {
    for react in [false, true] {
        let gate = Gate::new(Stage::BeforePrompt, Stop::Waiting);
        let memory = memory(gate.clone()).await;
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let model = Model::default();
        run(
            react,
            gate.clone(),
            model.clone(),
            memory.clone(),
            checkpoint.clone(),
            input(),
        )
        .await
        .unwrap();
        let original = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        let mut foreign_input = original["preparation"]["input"].clone();
        foreign_input["metadata"]["agent_id"] = json!(999);
        for invalid in [
            json!({"idx":999}),
            json!({"idx":"bad cursor"}),
            json!({"input":foreign_input}),
            json!({"stage":"unknown"}),
            json!({"stage":"history","idx":1}),
        ] {
            let mut saved = original.clone();
            saved["preparation"]
                .as_object_mut()
                .unwrap()
                .extend(invalid.as_object().unwrap().clone());
            checkpoint
                .save(1001, saved.clone(), &Cancellation::new())
                .await
                .unwrap();
            let error = run(
                react,
                gate.clone(),
                model.clone(),
                memory.clone(),
                checkpoint.clone(),
                input(),
            )
            .await
            .unwrap_err();
            assert_eq!(error.code, "INVALID_STATE");
            assert_eq!(
                checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
                saved
            );
            assert_eq!(gate.hits.load(Ordering::SeqCst), 1);
            assert!(model.0.lock().unwrap().is_empty());
            assert_eq!(
                memory
                    .chat
                    .read(1001, &Cancellation::new())
                    .await
                    .unwrap()
                    .status,
                chat::Status::Running
            );
        }
    }
}

#[tokio::test]
async fn repeated_prompt_waits_keep_the_previous_successful_transformations() {
    for react in [false, true] {
        let first = Gate::new(Stage::BeforePrompt, Stop::Waiting);
        let memory = memory(first.clone()).await;
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let model = Model::default();
        run(
            react,
            first.clone(),
            model.clone(),
            memory.clone(),
            checkpoint.clone(),
            input(),
        )
        .await
        .unwrap();
        let second = Gate {
            stage: Stage::AfterPrompt,
            hits: Default::default(),
            ..first.clone()
        };
        let result = run(
            react,
            second.clone(),
            model.clone(),
            memory.clone(),
            checkpoint.clone(),
            input(),
        )
        .await
        .unwrap();
        assert!(matches!(result.last(), Some(Err(error)) if error.code == "WAITING"));
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(saved["preparation"]["stage"], "after_prompt");
        assert_eq!(saved["preparation"]["prompt"], "effective prompt");
        let result = run(
            react,
            second,
            model.clone(),
            memory.clone(),
            checkpoint,
            input(),
        )
        .await
        .unwrap();
        assert!(result.iter().all(Result::is_ok));
        assert_eq!(first.execute.load(Ordering::SeqCst), 1);
        assert_eq!(first.prompt.load(Ordering::SeqCst), 1);
        assert_eq!(model.0.lock().unwrap().len(), 1);
    }
}

#[test]
fn native_run_futures_do_not_inline_the_execution_closeout_machine() {
    let model = Model::default();
    let chat = agent::chat().model(model.clone()).build().unwrap();
    let react = agent::react().model(model.clone()).build().unwrap();
    let chat_run = chat.run(input(), true, Cancellation::new());
    let react_run = react.run(input(), true, Cancellation::new());
    let chat_bytes = std::mem::size_of_val(&chat_run);
    let react_bytes = std::mem::size_of_val(&react_run);
    assert!(
        chat_bytes < 16 * 1024,
        "Chat run future: {chat_bytes} bytes"
    );
    assert!(
        react_bytes < 16 * 1024,
        "ReAct run future: {react_bytes} bytes"
    );
    assert!(model.0.lock().unwrap().is_empty());
}

#[derive(Clone, Default)]
struct Isolated(Arc<Isolation>);
#[derive(Default)]
struct Isolation {
    execute: AtomicUsize,
    prompts: Mutex<std::collections::BTreeMap<String, usize>>,
    allow: AtomicBool,
}
impl Middleware for Isolated {
    async fn before_execute(&self, input: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.execute.fetch_add(1, Ordering::SeqCst);
        input["payload"][0]["data"]["value"] = json!("isolated effective input");
        Ok(())
    }
    async fn before_system_prompt(
        &self,
        prompt: &mut Value,
        _: &Cancellation,
    ) -> Result<(), Error> {
        let owner = prompt.as_str().unwrap().to_owned();
        *self
            .0
            .prompts
            .lock()
            .unwrap()
            .entry(owner.clone())
            .or_default() += 1;
        if owner == "A" && !self.0.allow.load(Ordering::SeqCst) {
            *prompt = json!("unfinished local mutation");
            return Err(Error::new("WAITING", "A prompt waits"));
        }
        Ok(())
    }
}
fn graph(react: bool, hooks: Isolated, model: Model) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("preparation-owner-isolation");
    for (node, prompt) in [("a", "A"), ("b", "B")] {
        if react {
            builder
                .add_agent_node(
                    node,
                    agent::react()
                        .model(model.clone())
                        .system_prompt(prompt)
                        .middleware(hooks.clone())
                        .build()
                        .unwrap(),
                )
                .unwrap();
        } else {
            builder
                .add_agent_node(
                    node,
                    agent::chat()
                        .model(model.clone())
                        .system_prompt(prompt)
                        .middleware(hooks.clone())
                        .build()
                        .unwrap(),
                )
                .unwrap();
        }
        builder.add_edge(START, node).unwrap();
    }
    builder.add_node("merge", halo::agent::node::Echo).unwrap();
    builder.add_join(["a", "b"], "merge").unwrap();
    builder.add_edge("merge", END).unwrap();
    let child = builder.compile().unwrap();
    let mut parent = WorkflowBuilder::new("preparation-wrapper");
    parent.add_graph_node("child", &child).unwrap();
    parent.add_edge(START, "child").unwrap();
    parent.add_edge("child", END).unwrap();
    parent.compile().unwrap()
}
fn ids() -> i64 {
    static IDS: AtomicI64 = AtomicI64::new(60_000_000);
    IDS.fetch_add(1, Ordering::SeqCst)
}
fn attempt<'a>(snapshot: &'a Value, node: &str) -> &'a Value {
    &snapshot["nodes"][node]["executions"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()["attempts"][0]
}
fn workflow_input() -> Messages {
    let mut message = Message::new(Role::User);
    message.metadata = input().metadata;
    message
        .content
        .push(halo::agent::model::content::part::Part {
            r#type: "text".into(),
            data: json!({"value":"original input"})
                .as_object()
                .unwrap()
                .clone(),
        });
    Messages::new([message])
}
#[tokio::test]
async fn preparation_is_owned_by_the_actual_nested_agent_attempt_not_shared_components() {
    for react in [false, true] {
        let hooks = Isolated::default();
        let model = Model::default();
        let runtime = Runtime::new(1001, graph(react, hooks.clone(), model.clone())).unwrap();
        let events = runtime
            .run(workflow_input(), Arc::new(ids), Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(
            matches!(events.last(), Some(Err(error)) if error.code == "WAITING"),
            "{events:?}"
        );
        let saved = runtime.snapshot().unwrap();
        let inner = &attempt(&saved, "child")["graph"];
        let a = &attempt(inner, "a")["agent"];
        assert_eq!(a["preparation"]["stage"], "before_prompt");
        assert_eq!(a["preparation"]["prompt"], "A");
        assert_eq!(
            a["preparation"]["input"]["payload"][0]["data"]["value"],
            "isolated effective input"
        );
        assert_eq!(attempt(inner, "b")["agent"]["status"], "succeeded");
        let b = inner["nodes"]["b"].clone();
        assert_eq!(model.0.lock().unwrap().len(), 1);
        hooks.0.allow.store(true, Ordering::SeqCst);
        let mut restored = Runtime::from_snapshot(
            1001,
            graph(react, hooks.clone(), model.clone()),
            serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap(),
        )
        .unwrap();
        restored.resume("child", Messages::default()).unwrap();
        let events = restored
            .run(workflow_input(), Arc::new(ids), Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        let done = restored.snapshot().unwrap();
        assert_eq!(done["status"], "succeeded");
        assert_eq!(attempt(&done, "child")["graph"]["nodes"]["b"], b);
        assert_eq!(
            hooks.0.execute.load(Ordering::SeqCst),
            2,
            "A before_execute and successful B cannot replay"
        );
        assert_eq!(hooks.0.prompts.lock().unwrap()["A"], 2);
        assert_eq!(hooks.0.prompts.lock().unwrap()["B"], 1);
        assert_eq!(model.0.lock().unwrap().len(), 2);
    }
}
