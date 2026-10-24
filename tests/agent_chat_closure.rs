//! Per-run Chat closure uses the Chat record's final state, not an Event log.
use chat::Store as _;
use futures_core::Stream;
use futures_executor::block_on;
use futures_util::{StreamExt, poll, stream};
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::checkpoint::Checkpoint as _,
    agent::event::Event,
    agent::memory::{
        self,
        short::{chat, session},
    },
    agent::message::Role,
    agent::middleware::Middleware,
    agent::model::{
        chat::{Chat, Request},
        content::part::Part,
        token::{InputDetails, OutputDetails, Usage},
    },
    agent::tool::{Call, Tool, ToolDefinition},
    agent::{self, Agent, Input, Payload},
    core,
    runtime::workflow::{
        Runtime, WorkflowBuilder,
        graph::{END, START},
    },
};
use serde_json::{Map, Value, json};
use session::Store as _;
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Default)]
struct Chats {
    inner: core::memory::short::chat::InMemory,
    writes: Mutex<Vec<chat::Status>>,
    fail_create: bool,
    fail_update: bool,
    fail_after_update: bool,
    updates: AtomicUsize,
    wait_update: Option<Arc<tokio::sync::Notify>>,
}
impl chat::Store for Chats {
    async fn create(&self, value: chat::Chat, c: &Cancellation) -> Result<chat::Chat, Error> {
        if self.fail_create {
            return Err(storage_error());
        }
        let value = self.inner.create(value, c).await?;
        self.writes.lock().unwrap().push(value.status.clone());
        Ok(value)
    }
    async fn read(&self, id: i64, c: &Cancellation) -> Result<chat::Chat, Error> {
        self.inner.read(id, c).await
    }
    async fn update(&self, value: chat::Chat, c: &Cancellation) -> Result<chat::Chat, Error> {
        self.updates.fetch_add(1, Ordering::SeqCst);
        assert!(
            !c.is_cancelled(),
            "Stopped persistence must not use execution cancellation"
        );
        if self.fail_update {
            return Err(storage_error());
        }
        if let Some(gate) = &self.wait_update {
            gate.notified().await;
        }
        let value = self.inner.update(value, c).await?;
        self.writes.lock().unwrap().push(value.status.clone());
        if self.fail_after_update {
            return Err(storage_error());
        }
        Ok(value)
    }
    async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, c).await
    }
    async fn list(
        &self,
        options: Map<String, Value>,
        c: &Cancellation,
    ) -> Result<Vec<chat::Chat>, Error> {
        self.inner.list(options, c).await
    }
}
fn storage_error() -> Error {
    Error::new("BACKEND", "original storage failure").with_details("original details")
}

type Config = memory::Config<
    memory::short::Pair<core::memory::short::session::InMemory, Chats>,
    core::memory::InMemory,
>;
fn config(chats: Chats) -> Config {
    config_for(chats, memory::short::Type::Agent)
}
fn config_for(chats: Chats, owner: memory::short::Type) -> Config {
    let config = Config::new(
        memory::short::Pair::new(core::memory::short::session::InMemory::new(), chats),
        core::memory::InMemory::new(),
    );
    block_on(config.session().create(
        session::Session {
            id: 10,
            tenant_id: 1,
            user_id: 2,
            r#type: owner,
            title: None,
            summary: None,
            usage: None,
            status: session::Status::Active,
            metadata: Map::new(),
            created_time: 0,
            updated_time: 0,
        },
        &Cancellation::new(),
    ))
    .unwrap();
    config
}
fn input(id: i64) -> Input {
    Input {
        metadata: json!({"tenant_id":1,"user_id":2,"agent_id":3,"session_id":10,"chat_id":id})
            .as_object()
            .unwrap()
            .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"question"}),
            mime_type: None,
        }],
    }
}
fn text(value: &str) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":value}).as_object().unwrap().clone(),
    });
    message
}
fn usage(input: u64, output: u64) -> Usage {
    Usage {
        input: Some(input),
        output: Some(output),
        total: Some(input + output),
        input_details: Some(InputDetails { cached: Some(2) }),
        output_details: Some(OutputDetails { think: Some(1) }),
        details: json!({"provider_private":"kept on original only"})
            .as_object()
            .unwrap()
            .clone(),
    }
}
fn complete(message: Message, usage: Option<Usage>) -> Result<Event, Error> {
    Ok(Event::Complete {
        message,
        usage,
        finish_reason: Some("stop".into()),
    })
}
fn tool_call() -> Message {
    let mut message = text("intermediate");
    message.content.push(Part {
        r#type: "tool_call".into(),
        data: json!({"name":"lookup","call_id":"c1","arguments":{}})
            .as_object()
            .unwrap()
            .clone(),
    });
    message
}
struct Step {
    events: Result<Vec<Result<Event, Error>>, Error>,
    pending: bool,
}
impl Step {
    fn events(events: Vec<Result<Event, Error>>) -> Self {
        Self {
            events: Ok(events),
            pending: false,
        }
    }
    fn error() -> Self {
        Self {
            events: Err(Error::new("TRANSPORT", "original model failure").with_details("unchanged")),
            pending: false,
        }
    }
    fn pending(delta: Message) -> Self {
        Self {
            events: Ok(vec![Ok(Event::Delta { message: delta })]),
            pending: true,
        }
    }
}
#[derive(Clone)]
struct Model {
    steps: Arc<Mutex<VecDeque<Step>>>,
    requests: Arc<Mutex<Vec<Request>>>,
}
impl Model {
    fn new(steps: Vec<Step>) -> Self {
        Self {
            steps: Arc::new(Mutex::new(steps.into())),
            requests: Arc::default(),
        }
    }
}
impl Chat for Model {
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.requests.lock().unwrap().push(request);
        let step = self
            .steps
            .lock()
            .unwrap()
            .pop_front()
            .expect("no hidden extra model calls");
        let events = stream::iter(step.events?);
        if step.pending {
            Ok(Box::pin(events.chain(stream::pending())))
        } else {
            Ok(Box::pin(events))
        }
    }
}
struct Lookup {
    definition: ToolDefinition,
    cancel: bool,
}
impl Lookup {
    fn new(cancel: bool) -> Self {
        Self {
            definition: ToolDefinition {
                name: "lookup".into(),
                description: "lookup".into(),
                parameters: json!({"type":"object"}).as_object().unwrap().clone(),
            },
            cancel,
        }
    }
}
impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, _: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        if self.cancel {
            cancellation.cancel();
            return Err(Error::new("CANCELLED", "cancel tool"));
        }
        let mut message = Message::new(Role::Tool);
        message.values.insert("found".into(), json!(true));
        Ok(message)
    }
}
fn capabilities(cancel: bool) -> Vec<Lookup> {
    vec![Lookup::new(cancel)]
}
fn saved(config: &Config, id: i64) -> chat::Chat {
    block_on(config.chat().read(id, &Cancellation::new())).unwrap()
}
fn assert_no_state(config: &Config, id: i64) {
    match block_on(config.chat().read(id, &Cancellation::new())) {
        Ok(record) => assert!(record.state.is_none()),
        Err(error) => assert_eq!(error.code, "NOT_FOUND"),
    }
    let session = block_on(config.session().read(10, &Cancellation::new())).unwrap();
    assert!(session.summary.is_none());
}

fn assert_state(config: &Config, id: i64) {
    assert!(
        saved(config, id)
            .state
            .is_some_and(|state| state.is_object())
    );
}

#[test]
fn chat_creates_once_and_saves_provider_complete_not_deltas() {
    let config = config(Chats::default());
    let model = Model::new(vec![Step::events(vec![
        Ok(Event::Delta {
            message: text("wrong partial"),
        }),
        complete(text("final"), Some(usage(10, 3))),
    ])]);
    let agent = agent::chat()
        .model(model)
        .memory(config.clone())
        .build()
        .unwrap();
    let mut events = block_on(agent.run(input(100), true, Cancellation::new())).unwrap();
    assert_eq!(saved(&config, 100).status, chat::Status::Running);
    assert!(matches!(
        block_on(events.next()),
        Some(Ok(Event::Delta { .. }))
    ));
    assert_eq!(saved(&config, 100).status, chat::Status::Running);
    let Some(Ok(Event::Complete {
        message,
        usage: event_usage,
        ..
    })) = block_on(events.next())
    else {
        panic!("Complete");
    };
    let saved = saved(&config, 100);
    assert_eq!(saved.status, chat::Status::Completed);
    let mut original = message;
    assert_eq!(original.metadata.remove("chat_id"), Some(json!(100)));
    assert_eq!(saved.message, Some(original));
    assert_eq!(
        saved.message.as_ref().unwrap().content[0].data["value"],
        "final"
    );
    assert_eq!(event_usage, Some(usage(10, 3)));
    assert_eq!(saved.usage.as_ref().unwrap().total, Some(13));
    assert!(saved.usage.unwrap().details.is_empty());
    assert!(saved.graph.is_none());
    assert!(saved.started_time.is_some() && saved.completed_time.is_some());
    assert!(block_on(events.next()).is_none());
    assert_eq!(
        *config.chat().writes.lock().unwrap(),
        [chat::Status::Running, chat::Status::Completed]
    );
    assert_state(&config, 100);
}

#[test]
fn react_aggregates_two_model_calls_but_keeps_per_call_usage() {
    let config = config(Chats::default());
    let model = Model::new(vec![
        Step::events(vec![complete(tool_call(), Some(usage(10, 3)))]),
        Step::events(vec![complete(text("final"), Some(usage(20, 5)))]),
    ]);
    let agent = agent::react()
        .model(model)
        .tools(capabilities(false))
        .memory(config.clone())
        .build()
        .unwrap();
    let events = block_on(
        block_on(agent.run(input(101), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert_eq!(events.len(), 3);
    let totals: Vec<_> = events
        .iter()
        .map(|e| match e.as_ref().unwrap() {
            Event::Complete { usage, .. } => usage.as_ref().and_then(|u| u.total),
            _ => panic!(),
        })
        .collect();
    assert_eq!(totals, [Some(13), None, Some(25)]);
    let record = saved(&config, 101);
    assert_eq!(record.message.unwrap().content[0].data["value"], "final");
    let sum = record.usage.unwrap();
    assert_eq!(
        (sum.input, sum.output, sum.total),
        (Some(30), Some(8), Some(38))
    );
    assert_eq!(sum.input_details.unwrap().cached, Some(4));
    assert_eq!(sum.output_details.unwrap().think, Some(2));
    assert!(sum.details.is_empty());
    assert_eq!(config.chat().writes.lock().unwrap().len(), 2);
    assert_state(&config, 101);
}

#[test]
fn history_usage_is_not_billed_to_the_new_run() {
    let config = config(Chats::default());
    let first = agent::chat()
        .model(Model::new(vec![Step::events(vec![complete(
            text("old"),
            Some(usage(1000, 500)),
        )])]))
        .memory(config.clone())
        .build()
        .unwrap();
    block_on(
        block_on(first.run(input(102), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    let second = agent::react()
        .model(Model::new(vec![Step::events(vec![complete(
            text("new"),
            Some(usage(20, 5)),
        )])]))
        .memory(config.clone())
        .build()
        .unwrap();
    block_on(
        block_on(second.run(input(103), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert_eq!(saved(&config, 103).usage.unwrap().total, Some(25));
}

#[test]
fn chat_cancellation_preserves_received_text_and_known_usage() {
    let config = config(Chats::default());
    let mut delta = text("partial");
    delta.usage = Some(Usage {
        input: Some(77),
        ..Default::default()
    });
    let agent = agent::chat()
        .model(Model::new(vec![Step::pending(delta)]))
        .memory(config.clone())
        .build()
        .unwrap();
    let cancellation = Cancellation::new();
    let mut events = block_on(agent.run(input(104), true, cancellation.clone())).unwrap();
    block_on(events.next()).unwrap().unwrap();
    cancellation.cancel();
    let Some(Ok(Event::Complete {
        message,
        usage,
        finish_reason,
    })) = block_on(events.next())
    else {
        panic!()
    };
    assert_eq!(finish_reason.as_deref(), Some("cancelled"));
    assert_eq!(message.content[0].data["value"], "partial");
    let usage = usage.unwrap();
    assert_eq!(usage.input, Some(77));
    assert!(usage.output.unwrap() > 0);
    let record = saved(&config, 104);
    assert_eq!(record.status, chat::Status::Stopped);
    assert_eq!(record.usage.unwrap().total, usage.total);
    let mut original = message;
    assert_eq!(original.metadata.remove("chat_id"), Some(json!(104)));
    assert_eq!(record.message, Some(original));
    assert!(block_on(events.next()).is_none());
}

#[test]
fn react_tool_cancellation_does_not_double_count_a_completed_model() {
    let config = config(Chats::default());
    let model = Model::new(vec![Step::events(vec![complete(
        tool_call(),
        Some(usage(10, 3)),
    )])]);
    let agent = agent::react()
        .model(model)
        .tools(capabilities(true))
        .memory(config.clone())
        .build()
        .unwrap();
    let events = block_on(
        block_on(agent.run(input(105), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert_eq!(events.len(), 2);
    assert!(
        matches!(&events[1],Ok(Event::Complete {usage:None,finish_reason:Some(reason),..}) if reason=="cancelled")
    );
    let record = saved(&config, 105);
    assert_eq!(record.status, chat::Status::Stopped);
    assert_eq!(record.usage.unwrap().total, Some(13));
}

#[test]
fn react_active_second_call_cancellation_adds_only_that_partial_call() {
    let config = config(Chats::default());
    let mut delta = text("partial second");
    delta.usage = Some(usage(20, 4));
    let model = Model::new(vec![
        Step::events(vec![complete(tool_call(), Some(usage(10, 3)))]),
        Step::pending(delta),
    ]);
    let agent = agent::react()
        .model(model)
        .tools(capabilities(false))
        .memory(config.clone())
        .build()
        .unwrap();
    let c = Cancellation::new();
    let mut events = block_on(agent.run(input(106), true, c.clone())).unwrap();
    for _ in 0..3 {
        block_on(events.next()).unwrap().unwrap();
    }
    c.cancel();
    let Some(Ok(Event::Complete { usage, .. })) = block_on(events.next()) else {
        panic!()
    };
    assert_eq!(usage.unwrap().total, Some(24));
    assert_eq!(saved(&config, 106).usage.unwrap().total, Some(37));
    assert!(block_on(events.next()).is_none());
}

#[test]
fn model_creation_failure_closes_chat_as_failed() {
    let config = config(Chats::default());
    let agent = agent::chat()
        .model(Model::new(vec![Step::error()]))
        .memory(config.clone())
        .build()
        .unwrap();
    let error = block_on(agent.run(input(107), true, Cancellation::new())).unwrap_err();
    assert_eq!(
        (error.code.as_str(), error.details.as_deref()),
        ("TRANSPORT", Some("unchanged"))
    );
    let record = saved(&config, 107);
    assert_eq!(record.status, chat::Status::Failed);
    assert!(record.message.is_none() && record.usage.is_none());
    let state = saved(&config, 107).state.unwrap();
    assert_eq!(state["chat_id"], 107);
    assert_eq!(state["status"], "failed");
    assert_eq!(state["query"], json!(input(107)));
    assert_eq!(state["error"]["code"], "TRANSPORT");
    assert_eq!(state["error"]["details"], "unchanged");
    assert!(state["usage"].is_null());
}

#[test]
fn stream_failure_does_not_persist_delta_as_a_final_answer() {
    let config = config(Chats::default());
    let model = Model::new(vec![Step::events(vec![
        Ok(Event::Delta {
            message: text("unfinished"),
        }),
        Err(Error::new("TRANSPORT", "original")),
    ])]);
    let agent = agent::chat()
        .model(model)
        .memory(config.clone())
        .build()
        .unwrap();
    let events = block_on(
        block_on(agent.run(input(108), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert_eq!(events.len(), 2);
    assert!(events[1].is_err());
    let record = saved(&config, 108);
    assert_eq!(record.status, chat::Status::Failed);
    assert!(record.message.is_none());
    let usage = record.usage.unwrap();
    assert!(usage.input.unwrap() > 0);
    assert!(usage.output.unwrap() > 0);
    assert_eq!(
        usage.total,
        Some(usage.input.unwrap() + usage.output.unwrap())
    );
    let state = saved(&config, 108).state.unwrap();
    assert!(state["final_message"].is_null());
    assert_eq!(state["steps"][0]["status"], "failed");
    assert_eq!(
        state["steps"][0]["attempts"][0]["message"]["content"][0]["data"]["value"],
        "unfinished"
    );
}

#[test]
fn react_failure_keeps_previous_settled_usage_not_intermediate_message() {
    let config = config(Chats::default());
    let agent = agent::react()
        .model(Model::new(vec![
            Step::events(vec![complete(tool_call(), Some(usage(10, 3)))]),
            Step::error(),
        ]))
        .tools(capabilities(false))
        .memory(config.clone())
        .build()
        .unwrap();
    let events = block_on(
        block_on(agent.run(input(109), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        events.last().unwrap().as_ref().unwrap_err().code,
        "TRANSPORT"
    );
    let record = saved(&config, 109);
    assert_eq!(record.status, chat::Status::Failed);
    assert!(record.message.is_none());
    assert_eq!(record.usage.unwrap().total, Some(13));
}

struct Hook {
    fail: bool,
    cancel: bool,
}
impl Middleware for Hook {
    async fn before_execute(&self, value: &mut Value, _: &Cancellation) -> Result<(), Error> {
        value["payload"][0]["data"] = json!({"value":"hook input"});
        Ok(())
    }
    async fn after_model(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        if self.fail {
            return Err(Error::new("HOOK", "original hook"));
        }
        if self.cancel {
            c.cancel();
        }
        value["content"][0]["data"] = json!({"value":"hook answer"});
        Ok(())
    }
}
#[test]
fn original_input_is_immutable_but_final_output_includes_hooks() {
    let config = config(Chats::default());
    let model = Model::new(vec![Step::events(vec![complete(
        text("raw"),
        Some(usage(10, 3)),
    )])]);
    let agent = agent::chat()
        .model(model.clone())
        .middleware(Hook {
            fail: false,
            cancel: false,
        })
        .memory(config.clone())
        .build()
        .unwrap();
    block_on(
        block_on(agent.run(input(110), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        model.requests.lock().unwrap()[0].messages.as_slice()[0].content[0].data["value"],
        "hook input"
    );
    let record = saved(&config, 110);
    assert_eq!(record.input.unwrap().payload[0].data["value"], "question");
    assert_eq!(
        record.message.unwrap().content[0].data["value"],
        "hook answer"
    );
}

#[test]
fn output_hook_failure_does_not_erase_provider_usage() {
    for react in [false, true] {
        let config = config(Chats::default());
        let model = Model::new(vec![Step::events(vec![complete(
            text("raw"),
            Some(usage(10, 3)),
        )])]);
        if react {
            let agent = agent::react()
                .model(model)
                .middleware(Hook {
                    fail: true,
                    cancel: false,
                })
                .memory(config.clone())
                .build()
                .unwrap();
            let events = block_on(
                block_on(agent.run(input(111), true, Cancellation::new()))
                    .unwrap()
                    .collect::<Vec<_>>(),
            );
            assert_eq!(events.last().unwrap().as_ref().unwrap_err().code, "HOOK");
        } else {
            let agent = agent::chat()
                .model(model)
                .middleware(Hook {
                    fail: true,
                    cancel: false,
                })
                .memory(config.clone())
                .build()
                .unwrap();
            let events = block_on(
                block_on(agent.run(input(111), true, Cancellation::new()))
                    .unwrap()
                    .collect::<Vec<_>>(),
            );
            assert_eq!(events.last().unwrap().as_ref().unwrap_err().code, "HOOK");
        }
        let record = saved(&config, 111);
        assert_eq!(record.status, chat::Status::Failed);
        assert_eq!(record.usage.unwrap().total, Some(13));
        assert!(record.message.is_none());
    }
}

#[test]
fn cancellation_in_output_hook_refines_same_call_without_double_counting() {
    for react in [false, true] {
        let config = config(Chats::default());
        let model = Model::new(vec![Step::events(vec![complete(
            text("raw"),
            Some(usage(10, 3)),
        )])]);
        if react {
            let agent = agent::react()
                .model(model)
                .middleware(Hook {
                    fail: false,
                    cancel: true,
                })
                .memory(config.clone())
                .build()
                .unwrap();
            block_on(
                block_on(agent.run(input(112), true, Cancellation::new()))
                    .unwrap()
                    .collect::<Vec<_>>(),
            );
        } else {
            let agent = agent::chat()
                .model(model)
                .middleware(Hook {
                    fail: false,
                    cancel: true,
                })
                .memory(config.clone())
                .build()
                .unwrap();
            block_on(
                block_on(agent.run(input(112), true, Cancellation::new()))
                    .unwrap()
                    .collect::<Vec<_>>(),
            );
        }
        let record = saved(&config, 112);
        assert_eq!(record.status, chat::Status::Stopped);
        assert_eq!(record.usage.unwrap().total, Some(13));
    }
}

#[test]
fn storage_create_and_update_errors_are_direct_and_not_retried() {
    for fail_create in [false, true] {
        let config = config(Chats {
            fail_create,
            fail_update: !fail_create,
            ..Default::default()
        });
        let model = Model::new(vec![Step::events(vec![complete(
            text("raw"),
            Some(usage(10, 3)),
        )])]);
        let agent = agent::chat()
            .model(model.clone())
            .memory(config.clone())
            .build()
            .unwrap();
        let output = block_on(agent.run(input(113), true, Cancellation::new()));
        if fail_create {
            assert_eq!(output.unwrap_err(), storage_error());
            assert!(model.requests.lock().unwrap().is_empty());
        } else {
            let events = block_on(output.unwrap().collect::<Vec<_>>());
            assert_eq!(events.len(), 1);
            assert_eq!(events[0], Err(storage_error()));
            assert_eq!(saved(&config, 113).status, chat::Status::Running);
        }
        assert_no_state(&config, 113);
    }
}

fn failing_closeout(stage: usize) -> Config {
    failing_closeout_for(stage, memory::short::Type::Agent)
}
fn failing_closeout_for(stage: usize, owner: memory::short::Type) -> Config {
    config_for(
        Chats {
            fail_update: stage == 0,
            fail_after_update: stage == 1,
            ..Default::default()
        },
        owner,
    )
}

fn assert_closeout_failure(config: &Config, stage: usize, scene: &Value, status: chat::Status) {
    assert_eq!(config.chat().updates.load(Ordering::SeqCst), 1);
    let record = saved(config, 130);
    if stage == 0 {
        assert_eq!(record.status, chat::Status::Running);
        assert!(record.message.is_none());
        assert!(record.completed_time.is_none());
    } else {
        assert_eq!(record.status, status);
        assert_eq!(record.usage.as_ref().unwrap().total, Some(13));
        assert!(record.completed_time.is_some());
    }
    if stage == 0 {
        assert!(record.state.is_none());
    } else {
        assert_eq!(record.state.unwrap(), *scene);
    }
}

async fn check_agent_closeout_failure<A: Agent>(
    agent: A,
    model: Model,
    config: &Config,
    checkpoint: &core::checkpoint::InMemory,
    stage: usize,
    streaming: bool,
    cancelled: bool,
) {
    let c = Cancellation::new();
    let mut events = Box::pin(agent.run(input(130), streaming, c.clone()).await.unwrap());
    if cancelled {
        if streaming {
            assert!(matches!(events.next().await, Some(Ok(Event::Delta { .. }))));
        } else {
            assert!(poll!(events.next()).is_pending());
        }
        c.cancel();
    }
    let tail = events.collect::<Vec<_>>().await;
    assert_eq!(tail.last(), Some(&Err(storage_error())));
    let scene = checkpoint.load(130, &Cancellation::new()).await.unwrap();
    assert_eq!(
        scene["status"],
        if cancelled { "stopped" } else { "succeeded" }
    );
    assert_closeout_failure(
        config,
        stage,
        &scene,
        if cancelled {
            chat::Status::Stopped
        } else {
            chat::Status::Completed
        },
    );
    if stage != 0 {
        assert_eq!(
            saved(config, 130).message.unwrap().content[0].data["value"],
            if cancelled { "partial" } else { "answer" }
        );
    }
    // The runtime snapshot remains the recovery truth. Terminal restoration
    // does not repeat business work or silently retry a failed audit write.
    let _ = agent
        .run(input(130), streaming, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(
        checkpoint.load(130, &Cancellation::new()).await.unwrap(),
        scene
    );
    assert_eq!(model.requests.lock().unwrap().len(), 1);
    assert_closeout_failure(
        config,
        stage,
        &scene,
        if cancelled {
            chat::Status::Stopped
        } else {
            chat::Status::Completed
        },
    );
}

#[tokio::test]
async fn audit_write_failures_do_not_replace_chat_or_react_checkpoint_facts() {
    for react in [false, true] {
        for streaming in [false, true] {
            for cancelled in [false, true] {
                for stage in 0..2 {
                    let config = failing_closeout(stage);
                    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                    let model = Model::new(vec![if cancelled {
                        let mut partial = text("partial");
                        partial.usage = Some(usage(10, 3));
                        Step::pending(partial)
                    } else {
                        Step::events(vec![complete(text("answer"), Some(usage(10, 3)))])
                    }]);
                    if react {
                        let agent = agent::react()
                            .model(model.clone())
                            .memory(config.clone())
                            .checkpoint(checkpoint.clone())
                            .build()
                            .unwrap();
                        check_agent_closeout_failure(
                            agent,
                            model,
                            &config,
                            &checkpoint,
                            stage,
                            streaming,
                            cancelled,
                        )
                        .await;
                    } else {
                        let agent = agent::chat()
                            .model(model.clone())
                            .memory(config.clone())
                            .checkpoint(checkpoint.clone())
                            .build()
                            .unwrap();
                        check_agent_closeout_failure(
                            agent,
                            model,
                            &config,
                            &checkpoint,
                            stage,
                            streaming,
                            cancelled,
                        )
                        .await;
                    }
                }
            }
        }
    }
}

struct FailingCheckpoint {
    inner: core::checkpoint::InMemory,
    saves: AtomicUsize,
    after_write: bool,
}
impl halo::agent::checkpoint::Checkpoint for FailingCheckpoint {
    async fn save(&self, id: i64, state: Value, c: &Cancellation) -> Result<(), Error> {
        assert!(!c.is_cancelled());
        self.saves.fetch_add(1, Ordering::SeqCst);
        if self.after_write {
            self.inner.save(id, state, c).await?;
        }
        Err(storage_error())
    }
    async fn load(&self, id: i64, c: &Cancellation) -> Result<Value, Error> {
        self.inner.load(id, c).await
    }
    async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, c).await
    }
}

#[tokio::test]
async fn checkpoint_write_failure_does_not_create_a_false_failed_audit() {
    for react in [false, true] {
        for after_write in [false, true] {
            let config = config(Chats::default());
            let checkpoint = Arc::new(FailingCheckpoint {
                inner: core::checkpoint::InMemory::new(),
                saves: AtomicUsize::new(0),
                after_write,
            });
            let model = Model::new(vec![Step::events(vec![complete(
                text("answer"),
                Some(usage(10, 3)),
            )])]);
            let mut events: Pin<Box<dyn Stream<Item = Result<Event, Error>>>> = if react {
                Box::pin(
                    agent::react()
                        .model(model.clone())
                        .memory(config.clone())
                        .checkpoint(checkpoint.clone())
                        .build()
                        .unwrap()
                        .run(input(130), false, Cancellation::new())
                        .await
                        .unwrap(),
                )
            } else {
                Box::pin(
                    agent::chat()
                        .model(model.clone())
                        .memory(config.clone())
                        .checkpoint(checkpoint.clone())
                        .build()
                        .unwrap()
                        .run(input(130), false, Cancellation::new())
                        .await
                        .unwrap(),
                )
            };
            assert_eq!(events.next().await, Some(Err(storage_error())));
            assert!(events.next().await.is_none());
            assert_eq!(checkpoint.saves.load(Ordering::SeqCst), 1);
            assert_eq!(model.requests.lock().unwrap().len(), 1);
            assert_eq!(config.chat().updates.load(Ordering::SeqCst), 0);
            assert_eq!(saved(&config, 130).status, chat::Status::Running);
            assert_no_state(&config, 130);
            let saved = checkpoint.load(130, &Cancellation::new()).await;
            if after_write {
                let scene = saved.unwrap();
                assert_eq!(scene["status"], "succeeded");
                assert!(scene["error"].is_null());
                assert_eq!(scene["usage"]["total"], 13);
            } else {
                assert_eq!(saved.unwrap_err().code, "NOT_FOUND");
            }
        }
    }
}

#[tokio::test]
async fn workflow_audit_failure_preserves_outer_state_without_replaying_nodes() {
    for nested in [false, true] {
        for failed in [false, true] {
            for stage in 0..2 {
                let config = failing_closeout_for(stage, memory::short::Type::Workflow);
                let checkpoint = Arc::new(core::checkpoint::InMemory::new());
                let model = Model::new(vec![if failed {
                    Step::error()
                } else {
                    Step::events(vec![complete(text("answer"), Some(usage(10, 3)))])
                }]);
                let mut builder = WorkflowBuilder::new("audit-closeout");
                builder
                    .add_agent_node("agent", agent::chat().model(model.clone()).build().unwrap())
                    .unwrap();
                builder.add_edge(START, "agent").unwrap();
                builder.add_edge("agent", END).unwrap();
                let mut workflow = builder.compile().unwrap();
                if nested {
                    let mut parent = WorkflowBuilder::new("outer-audit-closeout");
                    parent.add_graph_node("child", &workflow).unwrap();
                    parent.add_edge(START, "child").unwrap();
                    parent.add_edge("child", END).unwrap();
                    workflow = parent.compile().unwrap();
                }
                let runtime = Runtime::new(130, workflow)
                    .unwrap()
                    .memory(config.clone())
                    .unwrap()
                    .with_checkpoint(checkpoint.clone());
                let mut message = Message::new(Role::User);
                message.metadata = input(130).metadata;
                let initial = Messages::new([message]);
                let events = runtime
                    .run(initial.clone(), Arc::new(|| 1_001), Cancellation::new())
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                assert_eq!(events.last(), Some(&Err(storage_error())));
                let scene = checkpoint.load(130, &Cancellation::new()).await.unwrap();
                assert_eq!(runtime.snapshot().unwrap(), scene);
                assert_eq!(scene["status"], if failed { "failed" } else { "succeeded" });
                if failed {
                    // A storage failure must not replace the model's original
                    // execution error in the recoverable runtime scene.
                    assert_eq!(
                        saved(&config, 130).status,
                        if stage == 0 {
                            chat::Status::Running
                        } else {
                            chat::Status::Failed
                        }
                    );
                    assert_eq!(config.chat().updates.load(Ordering::SeqCst), 1);
                    let saved_state = saved(&config, 130).state;
                    if stage == 0 {
                        assert!(saved_state.is_none());
                    } else {
                        assert_eq!(saved_state.unwrap(), scene);
                    }
                } else {
                    assert_closeout_failure(&config, stage, &scene, chat::Status::Completed);
                }
                let restored = runtime
                    .run(
                        initial,
                        Arc::new(|| panic!("no new execution")),
                        Cancellation::new(),
                    )
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                if failed {
                    assert!(
                        matches!(restored.last(), Some(Err(error)) if error.code == "TRANSPORT")
                    );
                } else {
                    assert!(restored.iter().all(Result::is_ok));
                }
                assert_eq!(model.requests.lock().unwrap().len(), 1);
                assert_eq!(config.chat().updates.load(Ordering::SeqCst), 1);
                assert_eq!(runtime.snapshot().unwrap(), scene);
            }
        }
    }
}

#[test]
fn abandoned_stream_does_not_spawn_background_closure() {
    let config = config(Chats::default());
    let agent = agent::chat()
        .model(Model::new(vec![Step::pending(text("partial"))]))
        .memory(config.clone())
        .build()
        .unwrap();
    let mut events = block_on(agent.run(input(114), true, Cancellation::new())).unwrap();
    block_on(events.next()).unwrap().unwrap();
    drop(events);
    assert_eq!(saved(&config, 114).status, chat::Status::Running);
    assert_eq!(
        *config.chat().writes.lock().unwrap(),
        [chat::Status::Running]
    );
    assert_no_state(&config, 114);
}

#[test]
fn duplicate_chat_id_is_conflict_not_resume_or_overwrite() {
    let config = config(Chats::default());
    let model = Model::new(vec![Step::events(vec![complete(text("final"), None)])]);
    let agent = agent::chat()
        .model(model.clone())
        .memory(config.clone())
        .build()
        .unwrap();
    block_on(
        block_on(agent.run(input(115), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        block_on(agent.run(input(115), true, Cancellation::new()))
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    assert_eq!(model.requests.lock().unwrap().len(), 1);
    assert_eq!(saved(&config, 115).status, chat::Status::Completed);
}

#[test]
fn unknown_usage_stays_unknown_and_no_memory_requires_no_ids() {
    let config = config(Chats::default());
    let agent = agent::chat()
        .model(Model::new(vec![Step::events(vec![complete(
            text("final"),
            None,
        )])]))
        .memory(config.clone())
        .build()
        .unwrap();
    block_on(
        block_on(agent.run(input(116), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert!(saved(&config, 116).usage.is_none());
    let agent = agent::react()
        .model(Model::new(vec![Step::events(vec![complete(
            text("final"),
            Some(usage(10, 3)),
        )])]))
        .build()
        .unwrap();
    let events = block_on(
        block_on(agent.run(Input::default(), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert!(events.iter().all(Result::is_ok));
}

#[test]
fn independent_runs_of_one_agent_do_not_share_a_usage_ledger() {
    let config = config(Chats::default());
    let model = Model::new(vec![
        Step::events(vec![complete(text("first"), Some(usage(10, 3)))]),
        Step::events(vec![complete(text("second"), Some(usage(20, 5)))]),
    ]);
    let agent = agent::chat()
        .model(model)
        .memory(config.clone())
        .build()
        .unwrap();
    let first = block_on(agent.run(input(120), true, Cancellation::new())).unwrap();
    let second = block_on(agent.run(input(121), true, Cancellation::new())).unwrap();
    block_on(second.collect::<Vec<_>>());
    block_on(first.collect::<Vec<_>>());
    assert_eq!(saved(&config, 120).usage.unwrap().total, Some(13));
    assert_eq!(saved(&config, 121).usage.unwrap().total, Some(25));
}

#[test]
fn max_iters_and_missing_complete_are_failed_not_successful_closure() {
    let config = config(Chats::default());
    let agent = agent::react()
        .model(Model::new(vec![Step::events(vec![complete(
            tool_call(),
            Some(usage(10, 3)),
        )])]))
        .with_max_iters(1)
        .tools(capabilities(false))
        .memory(config.clone())
        .build()
        .unwrap();
    let events = block_on(
        block_on(agent.run(input(122), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        events.last().unwrap().as_ref().unwrap_err().code,
        "MAX_ITERS"
    );
    let record = saved(&config, 122);
    assert_eq!(record.status, chat::Status::Failed);
    assert!(record.message.is_none());
    assert_eq!(record.usage.unwrap().total, Some(13));
    let agent = agent::chat()
        .model(Model::new(vec![Step::events(vec![Ok(Event::Delta {
            message: text("partial"),
        })])]))
        .memory(config.clone())
        .build()
        .unwrap();
    let events = block_on(
        block_on(agent.run(input(123), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        events.last().unwrap().as_ref().unwrap_err().code,
        "INVALID_RESPONSE"
    );
    assert_eq!(saved(&config, 123).status, chat::Status::Failed);
}

#[test]
fn known_usage_on_a_failed_stream_is_retained_and_missing_consumption_is_estimated() {
    for react in [false, true] {
        let config = config(Chats::default());
        let mut delta = text("unfinished");
        delta.usage = Some(Usage {
            input: Some(77),
            ..Default::default()
        });
        let model = Model::new(vec![Step::events(vec![
            Ok(Event::Delta { message: delta }),
            Err(Error::new("TRANSPORT", "original")),
        ])]);
        if react {
            let agent = agent::react()
                .model(model)
                .memory(config.clone())
                .build()
                .unwrap();
            block_on(
                block_on(agent.run(input(124), true, Cancellation::new()))
                    .unwrap()
                    .collect::<Vec<_>>(),
            );
        } else {
            let agent = agent::chat()
                .model(model)
                .memory(config.clone())
                .build()
                .unwrap();
            block_on(
                block_on(agent.run(input(124), true, Cancellation::new()))
                    .unwrap()
                    .collect::<Vec<_>>(),
            );
        }
        let record = saved(&config, 124);
        assert_eq!(record.status, chat::Status::Failed);
        let usage = record.usage.unwrap();
        assert_eq!(usage.input, Some(77));
        assert!(usage.output.unwrap() > 0);
        assert_eq!(usage.total, Some(77 + usage.output.unwrap()));
        let state = saved(&config, 124).state.unwrap();
        let attempt = &state["steps"][0]["attempts"][0];
        assert_eq!(attempt["usage"]["input"], 77);
        assert_eq!(attempt["usage"]["total"], usage.total.unwrap());
        assert_eq!(
            attempt["message"]["content"][0]["data"]["value"],
            "unfinished"
        );
        assert!(state["final_message"].is_null());
    }
}

struct RedirectIdentity;
impl Middleware for RedirectIdentity {
    async fn before_execute(&self, value: &mut Value, _: &Cancellation) -> Result<(), Error> {
        value["metadata"]["agent_id"] = json!(999);
        Ok(())
    }
}
#[test]
fn hooks_cannot_redirect_the_persisted_chat_to_a_different_owner() {
    let config = config(Chats::default());
    let model = Model::new(vec![]);
    let agent = agent::chat()
        .model(model.clone())
        .middleware(RedirectIdentity)
        .memory(config.clone())
        .build()
        .unwrap();
    assert_eq!(
        block_on(agent.run(input(125), true, Cancellation::new()))
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert!(config.chat().writes.lock().unwrap().is_empty());
    assert!(model.requests.lock().unwrap().is_empty());
}

#[test]
fn cancellation_during_final_write_does_not_spin_or_restart_the_update() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};
    struct Wake(AtomicUsize);
    impl futures_util::task::ArcWake for Wake {
        fn wake_by_ref(value: &Arc<Self>) {
            value.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let gate = Arc::new(tokio::sync::Notify::new());
    let config = config(Chats {
        wait_update: Some(gate.clone()),
        ..Default::default()
    });
    let agent = agent::chat()
        .model(Model::new(vec![Step::events(vec![complete(
            text("final"),
            Some(usage(10, 3)),
        )])]))
        .memory(config.clone())
        .build()
        .unwrap();
    let c = Cancellation::new();
    let mut events = block_on(agent.run(input(126), true, c.clone())).unwrap();
    let wake = Arc::new(Wake(AtomicUsize::new(0)));
    let waker = futures_util::task::waker(wake.clone());
    let mut cx = Context::from_waker(&waker);
    assert!(matches!(
        Pin::new(&mut events).poll_next(&mut cx),
        Poll::Pending
    ));
    c.cancel();
    let count = wake.0.load(Ordering::SeqCst);
    for _ in 0..5 {
        assert!(matches!(
            Pin::new(&mut events).poll_next(&mut cx),
            Poll::Pending
        ));
    }
    assert_eq!(wake.0.load(Ordering::SeqCst), count);
    gate.notify_one();
    assert!(matches!(
        block_on(events.next()),
        Some(Ok(Event::Complete { .. }))
    ));
    assert_eq!(saved(&config, 126).status, chat::Status::Completed);
    assert_eq!(
        *config.chat().writes.lock().unwrap(),
        [chat::Status::Running, chat::Status::Completed]
    );
}

#[test]
fn chat_persistence_rejects_nonassistant_final_without_changing_unconfigured_runs() {
    let config = config(Chats::default());
    let message = Message::function(json!({"answer":"not an assistant"}));
    let model = Model::new(vec![Step::events(vec![complete(
        message.clone(),
        Some(usage(10, 3)),
    )])]);
    let agent = agent::chat()
        .model(model)
        .memory(config.clone())
        .build()
        .unwrap();
    let events = block_on(
        block_on(agent.run(input(127), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert_eq!(events[0].as_ref().unwrap_err().code, "INVALID_RESPONSE");
    let record = saved(&config, 127);
    assert_eq!(record.status, chat::Status::Failed);
    assert!(record.message.is_none());
    assert_eq!(record.usage.unwrap().total, Some(13));
    let model = Model::new(vec![Step::events(vec![complete(message.clone(), None)])]);
    let agent = agent::react().model(model).build().unwrap();
    let events = block_on(
        block_on(agent.run(Input::default(), true, Cancellation::new()))
            .unwrap()
            .collect::<Vec<_>>(),
    );
    assert!(matches!(&events[0],Ok(Event::Complete{message:actual,..}) if actual==&message));
}
