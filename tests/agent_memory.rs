//! History spies exercise initialization independently of final Chat writes.
use std::{
    future::pending,
    sync::{Arc, Mutex},
    time::Duration,
};

use futures_executor::block_on;
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::{self, Agent, Input, Payload, },
    event::Event,
    memory::{
        self, Memory,
        short::{self, chat, session, trace},
    },
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Map, Value, json};

use session::Store as _;

#[derive(Clone, Default)]
struct State {
    calls: Arc<Mutex<Vec<String>>>,
    chats: Arc<Mutex<Vec<chat::Chat>>>,
    items: Arc<Mutex<Vec<memory::Item>>>,
    fail: Option<&'static str>,
    wait: Option<&'static str>,
    cancel: Option<&'static str>,
}

impl State {
    async fn stage(&self, name: &'static str, cancellation: &Cancellation) -> Result<(), Error> {
        self.calls.lock().unwrap().push(name.into());
        if self.fail == Some(name) {
            return Err(
                Error::new("BACKEND", "original backend error").with_details("original details")
            );
        }
        if self.wait == Some(name) {
            pending::<()>().await;
        }
        if self.cancel == Some(name) {
            cancellation.cancel();
        }
        Ok(())
    }
}

struct Sessions {
    value: session::Session,
    state: State,
}
impl session::Store for Sessions {
    async fn create(
        &self,
        _: session::Session,
        _: &Cancellation,
    ) -> Result<session::Session, Error> {
        panic!("initialization must not create Session")
    }
    async fn read(&self, id: i64, cancellation: &Cancellation) -> Result<session::Session, Error> {
        assert_eq!(id, 101);
        self.state.stage("session", cancellation).await?;
        Ok(self.value.clone())
    }
    async fn update(
        &self,
        _: session::Session,
        _: &Cancellation,
    ) -> Result<session::Session, Error> {
        panic!("initialization must not update Session")
    }
    async fn delete(&self, _: i64, _: &Cancellation) -> Result<(), Error> {
        panic!("initialization must not delete Session")
    }
}
struct Chats(State);
impl chat::Store for Chats {
    async fn create(
        &self,
        value: chat::Chat,
        cancellation: &Cancellation,
    ) -> Result<chat::Chat, Error> {
        assert!(!cancellation.is_cancelled());
        assert_eq!(value.status, chat::Status::Running);
        Ok(value)
    }
    async fn read(&self, _: i64, _: &Cancellation) -> Result<chat::Chat, Error> {
        panic!("use standard history list")
    }
    async fn update(
        &self,
        value: chat::Chat,
        cancellation: &Cancellation,
    ) -> Result<chat::Chat, Error> {
        assert!(!cancellation.is_cancelled());
        assert_ne!(value.status, chat::Status::Running);
        Ok(value)
    }
    async fn delete(&self, _: i64, _: &Cancellation) -> Result<(), Error> {
        panic!("history must be retained")
    }
    async fn list(
        &self,
        options: Map<String, Value>,
        cancellation: &Cancellation,
    ) -> Result<Vec<chat::Chat>, Error> {
        assert_eq!(
            Value::Object(options),
            json!({"filter":{"tenant_id":1,"user_id":2,"agent_id":3,"session_id":101,"type":"agent"},"sort":{"field":"created_time","order":"asc"}})
        );
        self.0.stage("chat", cancellation).await?;
        Ok(self.0.chats.lock().unwrap().clone())
    }
}
struct Traces;
impl trace::Store for Traces {
    async fn create(&self, value: trace::Trace, _: &Cancellation) -> Result<trace::Trace, Error> {
        Ok(value)
    }
    async fn read(&self, _: i64, _: &Cancellation) -> Result<trace::Trace, Error> {
        Err(Error::new("NOT_FOUND", "trace not found"))
    }
    async fn update(&self, value: trace::Trace, _: &Cancellation) -> Result<trace::Trace, Error> {
        Ok(value)
    }
    async fn delete(&self, _: i64, _: &Cancellation) -> Result<(), Error> {
        Ok(())
    }
}

struct Long {
    state: State,
    wrong: bool,
}
impl Memory for Long {
    async fn write(&self, _: memory::Request, _: &Cancellation) -> Result<memory::Item, Error> {
        panic!("do not copy chats into long Memory")
    }
    async fn update(
        &self,
        _: memory::Request,
        _: u64,
        _: &Cancellation,
    ) -> Result<memory::Item, Error> {
        panic!("no implicit extraction")
    }
    async fn delete(&self, _: i64, _: &Cancellation) -> Result<(), Error> {
        panic!("no implicit delete")
    }
    async fn get(
        &self,
        user: i64,
        agent: Option<i64>,
        session: Option<i64>,
        kind: memory::Type,
        cancellation: &Cancellation,
    ) -> Result<Option<memory::Item>, Error> {
        self.state.stage("long", cancellation).await?;
        // Native async contracts must allow a non-Send component future.
        let local = std::rc::Rc::new(());
        futures_util::future::ready(()).await;
        assert_eq!(std::rc::Rc::strong_count(&local), 1);
        self.state
            .calls
            .lock()
            .unwrap()
            .push(format!("get:{user}:{agent:?}:{session:?}:{kind:?}"));
        let values = self.state.items.lock().unwrap();
        if self.wrong {
            return Ok(values.first().cloned());
        }
        Ok(values
            .iter()
            .find(|value| {
                value.user_id == user
                    && value.agent_id == agent
                    && value.session_id == session
                    && value.r#type == kind
            })
            .cloned())
    }
    async fn list(
        &self,
        _: i64,
        _: Option<i64>,
        _: Option<i64>,
        _: memory::Type,
        _: &Cancellation,
    ) -> Result<Vec<memory::Item>, Error> {
        panic!("initialization uses exact get")
    }
    async fn keyword(
        &self,
        _: memory::keyword::Request,
        _: &Cancellation,
    ) -> Result<Vec<memory::Item>, Error> {
        panic!("no implicit query or vector fallback")
    }
    async fn vector(
        &self,
        _: memory::vector::Request,
        _: &Cancellation,
    ) -> Result<Vec<memory::Item>, Error> {
        panic!("no implicit Embed")
    }
    async fn hybrid(
        &self,
        _: memory::hybrid::Request,
        _: &Cancellation,
    ) -> Result<Vec<memory::Item>, Error> {
        panic!("no implicit hybrid")
    }
}

type Config = memory::Config<Sessions, Chats, Traces, Long>;
fn session() -> session::Session {
    session::Session {
        id: 101,
        tenant_id: 1,
        user_id: 2,
        r#type: short::Type::Agent,
        title: None,
        summary: None,
        usage: None,
        status: session::Status::Active,
        metadata: Map::new(),
        created_time: 1,
        updated_time: 1,
    }
}
fn config(value: session::Session, state: &State) -> Config {
    memory::Config::new(
        Sessions {
            value,
            state: state.clone(),
        },
        Chats(state.clone()),
        Traces,
        Long {
            state: state.clone(),
            wrong: false,
        },
    )
}
fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":value}).as_object().unwrap().clone(),
    });
    message
}
fn history(id: i64, time: i64, question: &str, answer: &str) -> chat::Chat {
    chat::Chat {
        id,
        tenant_id: 1,
        user_id: 2,
        agent_id: 3,
        session_id: 101,
        r#type: short::Type::Agent,
        status: chat::Status::Completed,
        input: Some(chat::Request {
            metadata: json!({"old":true}).as_object().unwrap().clone(),
            payload: vec![chat::Payload {
                r#type: "text".into(),
                data: json!({"value":question}),
                mime_type: None,
            }],
        }),
        message: Some(text(Role::Assistant, answer)),
        usage: None,
        graph: None,
        metadata: Map::new(),
        started_time: None,
        completed_time: None,
        created_time: time,
        updated_time: time,
    }
}
fn input() -> Input {
    Input {
        metadata: json!({"tenant_id":1,"user_id":2,"agent_id":3,"session_id":101,"chat_id":999})
            .as_object()
            .unwrap()
            .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"camping?"}),
            mime_type: None,
        }],
    }
}
fn item(agent: Option<i64>, kind: memory::Type, content: &str) -> memory::Item {
    memory::Item {
        id: 401,
        r#type: kind,
        user_id: 2,
        agent_id: agent,
        session_id: None,
        from: 1,
        to: 2,
        content: content.into(),
        version: 1,
        created_time: 1,
        updated_time: 1,
        metadata: Some(json!({"private":"not prompt"}).as_object().unwrap().clone()),
    }
}

#[derive(Clone, Default)]
struct Model {
    requests: Arc<Mutex<Vec<Request>>>,
    tool: bool,
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut requests = self.requests.lock().unwrap();
        let first = requests.is_empty();
        requests.push(request);
        let message = if self.tool && first {
            let mut message = Message::new(Role::Assistant);
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"call_1","name":"search","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
            message
        } else {
            text(Role::Assistant, "final")
        };
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: None,
        })]))
    }
}
fn values(model: &Model, index: usize) -> Vec<String> {
    model.requests.lock().unwrap()[index]
        .messages
        .as_slice()
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|part| {
            part.data
                .get("value")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

#[test]
fn chat_initializes_ordered_history_before_current_input() {
    let state = State::default();
    *state.chats.lock().unwrap() = vec![
        history(202, 20, "second?", "second!"),
        history(201, 10, "first?", "first!"),
    ];
    let model = Model::default();
    let agent = agent::chat()
        .memory(config(session(), &state))
        .system_prompt("instruction")
        .model(model.clone())
        .build()
        .unwrap();
    let output = block_on(agent.run(input(), true, Cancellation::new())).unwrap();
    block_on(output.collect::<Vec<_>>());
    assert_eq!(
        values(&model, 0),
        [
            "instruction",
            "first?",
            "first!",
            "second?",
            "second!",
            "camping?"
        ]
    );
    assert_eq!(*state.calls.lock().unwrap(), ["session", "chat"]);
}

#[test]
fn react_uses_the_same_initialization_contract() {
    let state = State::default();
    state
        .chats
        .lock()
        .unwrap()
        .push(history(201, 1, "weather?", "sunny"));
    let model = Model::default();
    let agent = agent::react()
        .memory(config(session(), &state))
        .model(model.clone())
        .build()
        .unwrap();
    let events = block_on(agent.run(input(), true, Cancellation::new())).unwrap();
    assert!(
        block_on(events.collect::<Vec<_>>())
            .iter()
            .all(Result::is_ok)
    );
    assert_eq!(values(&model, 0), ["weather?", "sunny", "camping?"]);
}

#[test]
fn summary_coverage_resolves_history_order_not_numeric_id_ranges() {
    let state = State::default();
    *state.chats.lock().unwrap() = vec![
        history(900, 1, "covered1", "a"),
        history(50, 2, "covered2", "b"),
        history(100, 3, "recent", "c"),
    ];
    let mut s = session();
    s.summary = Some("previous conversation".into());
    s.metadata = json!({"from":900,"to":50}).as_object().unwrap().clone();
    let model = Model::default();
    let agent = agent::chat()
        .model(model.clone())
        .memory(config(s, &state))
        .build()
        .unwrap();
    block_on(agent.run(input(), true, Cancellation::new())).unwrap();
    assert_eq!(
        values(&model, 0),
        [
            "[Session summary]\nprevious conversation",
            "recent",
            "c",
            "camping?"
        ]
    );
    assert_eq!(state.chats.lock().unwrap().len(), 3);
}

#[test]
fn recall_scopes_are_exact_and_only_affect_long_memory() {
    let state = State::default();
    state
        .chats
        .lock()
        .unwrap()
        .push(history(201, 1, "short", "answer"));
    *state.items.lock().unwrap() = vec![
        item(None, memory::Type::Preference, "user preference"),
        item(Some(3), memory::Type::Fact, "agent fact"),
        item(Some(4), memory::Type::Summary, "other agent"),
    ];
    let model = Model::default();
    let agent = agent::chat()
        .model(model.clone())
        .memory(config(session(), &state).with_scopes(["user", "agent"]))
        .build()
        .unwrap();
    block_on(agent.run(input(), true, Cancellation::new())).unwrap();
    assert_eq!(
        values(&model, 0),
        [
            "short",
            "answer",
            "[Long-term memory]\nuser preference",
            "[Long-term memory]\nagent fact",
            "camping?"
        ]
    );
    let gets: Vec<_> = state
        .calls
        .lock()
        .unwrap()
        .iter()
        .filter(|call| call.starts_with("get:"))
        .cloned()
        .collect();
    assert_eq!(
        gets,
        [
            "get:2:None:None:Summary",
            "get:2:None:None:Fact",
            "get:2:None:None:Preference",
            "get:2:Some(3):None:Summary",
            "get:2:Some(3):None:Fact",
            "get:2:Some(3):None:Preference"
        ]
    );
    let requests = model.requests.lock().unwrap();
    let recalled = &requests[0].messages.as_slice()[3];
    assert_eq!(recalled.role, Role::User);
    assert_eq!(recalled.metadata["private"], "not prompt");
    assert!(recalled.values.is_empty());
}

#[test]
fn no_configuration_keeps_agent_independent_of_memory_identity() {
    let model = Model::default();
    let agent = agent::chat().model(model.clone()).build().unwrap();
    let mut question = input();
    question.metadata.clear();
    block_on(agent.run(question, true, Cancellation::new())).unwrap();
    assert_eq!(values(&model, 0), ["camping?"]);
}

#[test]
fn invalid_scopes_fail_build_without_reading_backends() {
    for scopes in [
        vec!["unknown"],
        vec!["session"],
        vec!["user", "user"],
        vec![" USER"],
    ] {
        let state = State::default();
        let configured = config(session(), &state).with_scopes(scopes);
        let error = agent::chat()
            .model(Model::default())
            .memory(configured)
            .build()
            .err()
            .unwrap();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
        assert!(state.calls.lock().unwrap().is_empty());
    }
}

#[test]
fn configured_memory_requires_exact_i64_business_metadata_before_io() {
    for field in ["tenant_id", "user_id", "agent_id", "session_id", "chat_id"] {
        for invalid in [Value::Null, json!("123"), json!(0)] {
            let state = State::default();
            let model = Model::default();
            let agent = agent::chat()
                .model(model.clone())
                .memory(config(session(), &state))
                .build()
                .unwrap();
            let mut question = input();
            question.metadata.insert(field.into(), invalid);
            assert_eq!(
                block_on(agent.run(question, true, Cancellation::new()))
                    .err()
                    .unwrap()
                    .code,
                "INVALID_ARGUMENTS"
            );
            assert!(state.calls.lock().unwrap().is_empty());
            assert!(model.requests.lock().unwrap().is_empty());
        }
    }
}

#[test]
fn mismatched_session_identity_stops_before_history_and_model() {
    for index in 0..4 {
        let state = State::default();
        let mut s = session();
        match index {
            0 => s.id = 9,
            1 => s.tenant_id = 9,
            2 => s.user_id = 9,
            _ => s.r#type = short::Type::Workflow,
        }
        let model = Model::default();
        let agent = agent::chat()
            .model(model.clone())
            .memory(config(s, &state))
            .build()
            .unwrap();
        assert_eq!(
            block_on(agent.run(input(), true, Cancellation::new()))
                .err()
                .unwrap()
                .code,
            "INVALID_ARGUMENTS"
        );
        assert_eq!(*state.calls.lock().unwrap(), ["session"]);
        assert!(model.requests.lock().unwrap().is_empty());
    }
}

#[test]
fn incorrect_backend_chat_filtering_is_not_trusted() {
    for index in 0..6 {
        let state = State::default();
        let mut wrong = history(201, 1, "secret", "answer");
        match index {
            0 => wrong.tenant_id = 9,
            1 => wrong.user_id = 9,
            2 => wrong.agent_id = 9,
            3 => wrong.session_id = 9,
            4 => wrong.r#type = short::Type::Workflow,
            _ => {}
        }
        state.chats.lock().unwrap().push(wrong.clone());
        if index == 5 {
            state.chats.lock().unwrap().push(wrong);
        }
        let model = Model::default();
        let agent = agent::react()
            .model(model.clone())
            .memory(config(session(), &state))
            .build()
            .unwrap();
        assert_eq!(
            block_on(agent.run(input(), true, Cancellation::new()))
                .err()
                .unwrap()
                .code,
            "INVALID_ARGUMENTS"
        );
        assert!(model.requests.lock().unwrap().is_empty());
    }
}

#[test]
fn backend_errors_are_direct_without_retries_or_prompt_injection() {
    for stage in ["session", "chat", "long"] {
        let state = State {
            fail: Some(stage),
            ..Default::default()
        };
        let model = Model::default();
        let agent = agent::chat()
            .model(model.clone())
            .memory(config(session(), &state).with_scopes(["user"]))
            .build()
            .unwrap();
        let error = block_on(agent.run(input(), true, Cancellation::new()))
            .err()
            .unwrap();
        assert_eq!(error.code, "BACKEND");
        assert_eq!(error.message, "original backend error");
        assert_eq!(error.details.as_deref(), Some("original details"));
        assert_eq!(
            state
                .calls
                .lock()
                .unwrap()
                .iter()
                .filter(|call| call.as_str() == stage)
                .count(),
            1
        );
        assert!(model.requests.lock().unwrap().is_empty());
    }
}

#[test]
fn running_failed_stopped_incomplete_and_current_chats_are_not_replayed() {
    let state = State::default();
    let mut rows = vec![history(201, 1, "good", "answer")];
    for (index, status) in [
        chat::Status::Running,
        chat::Status::Failed,
        chat::Status::Stopped,
    ]
    .into_iter()
    .enumerate()
    {
        let mut row = history(210 + index as i64, 2, "bad", "bad");
        row.status = status;
        rows.push(row);
    }
    let mut incomplete = history(220, 3, "incomplete", "answer");
    incomplete.message = None;
    rows.push(incomplete);
    rows.push(history(999, 4, "current saved copy", "duplicate"));
    *state.chats.lock().unwrap() = rows;
    let model = Model::default();
    let agent = agent::chat()
        .model(model.clone())
        .memory(config(session(), &state))
        .build()
        .unwrap();
    block_on(agent.run(input(), true, Cancellation::new())).unwrap();
    assert_eq!(values(&model, 0), ["good", "answer", "camping?"]);
}

#[test]
fn original_media_and_complete_assistant_fields_are_preserved() {
    let state = State::default();
    let mut row = history(201, 1, "caption", "answer");
    row.input.as_mut().unwrap().payload.push(chat::Payload {
        r#type: "image".into(),
        data: json!({"value":"stable-image-ref"}),
        mime_type: Some("image/png".into()),
    });
    let message = row.message.as_mut().unwrap();
    message.values.insert("result".into(), json!({"x":1}));
    message.metadata.insert("original".into(), json!(true));
    message.usage = Some(Usage {
        total: Some(100),
        ..Default::default()
    });
    let original = message.clone();
    state.chats.lock().unwrap().push(row);
    let model = Model::default();
    let agent = agent::chat()
        .model(model.clone())
        .memory(config(session(), &state))
        .build()
        .unwrap();
    block_on(agent.run(input(), true, Cancellation::new())).unwrap();
    let requests = model.requests.lock().unwrap();
    let messages = requests[0].messages.as_slice();
    assert_eq!(
        messages[0].content[1].data,
        json!({"value":"stable-image-ref","mime_type":"image/png"})
            .as_object()
            .unwrap()
            .clone()
    );
    assert_eq!(messages[1], original);
    let mut message_metadata = input().metadata;
    for key in ["chat_id", "node_id", "exec_id"] {
        message_metadata.remove(key);
    }
    assert_eq!(messages[2].metadata, message_metadata);
}

#[test]
fn malformed_or_unresolved_summary_coverage_is_not_silently_replayed() {
    for markers in [
        json!({}),
        json!({"from":201}),
        json!({"from":201,"to":404}),
        json!({"from":202,"to":201}),
        json!({"from":"201","to":202}),
        json!({"from":null,"to":202}),
    ] {
        let state = State::default();
        *state.chats.lock().unwrap() = vec![
            history(201, 1, "first", "a"),
            history(202, 2, "second", "b"),
        ];
        let mut s = session();
        s.summary = Some("summary".into());
        s.metadata = markers.as_object().unwrap().clone();
        let agent = agent::chat()
            .model(Model::default())
            .memory(config(s, &state))
            .build()
            .unwrap();
        assert_eq!(
            block_on(agent.run(input(), true, Cancellation::new()))
                .err()
                .unwrap()
                .code,
            "INVALID_ARGUMENTS"
        );
    }
    let state = State::default();
    let mut s = session();
    s.metadata.insert("from".into(), json!(201));
    let agent = agent::chat()
        .model(Model::default())
        .memory(config(s, &state))
        .build()
        .unwrap();
    assert!(block_on(agent.run(input(), true, Cancellation::new())).is_err());
}

#[test]
fn wrong_long_scope_or_type_never_reaches_the_model() {
    let state = State::default();
    state
        .items
        .lock()
        .unwrap()
        .push(item(Some(99), memory::Type::Summary, "other agent secret"));
    let mut configured = config(session(), &state);
    configured.long = Arc::new(Long {
        state: state.clone(),
        wrong: true,
    });
    let model = Model::default();
    let agent = agent::chat()
        .model(model.clone())
        .memory(configured.with_scopes(["user"]))
        .build()
        .unwrap();
    assert_eq!(
        block_on(agent.run(input(), true, Cancellation::new()))
            .err()
            .unwrap()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert!(model.requests.lock().unwrap().is_empty());
}

#[test]
fn config_clone_shares_stores_and_each_run_reads_a_fresh_snapshot() {
    let state = State::default();
    state
        .chats
        .lock()
        .unwrap()
        .push(history(201, 1, "first", "a"));
    let configured = config(session(), &state);
    let retained = configured.clone();
    assert!(Arc::ptr_eq(&configured.session, &retained.session));
    assert!(Arc::ptr_eq(&configured.chat, &retained.chat));
    assert!(Arc::ptr_eq(&configured.trace, &retained.trace));
    assert!(Arc::ptr_eq(&configured.long, &retained.long));
    let model = Model::default();
    let agent = agent::chat()
        .memory(configured)
        .model(model.clone())
        .build()
        .unwrap();
    let output = block_on(agent.run(input(), true, Cancellation::new())).unwrap();
    block_on(output.collect::<Vec<_>>());
    state
        .chats
        .lock()
        .unwrap()
        .push(history(202, 2, "new", "b"));
    block_on(agent.run(input(), true, Cancellation::new())).unwrap();
    assert_eq!(values(&model, 0), ["first", "a", "camping?"]);
    assert_eq!(values(&model, 1), ["first", "a", "new", "b", "camping?"]);
    assert_eq!(
        *state.calls.lock().unwrap(),
        ["session", "chat", "session", "chat"]
    );
}

struct Search {
    definition: ToolDefinition,
    state: State,
}
impl Tool for Search {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.state.chats.lock().unwrap().push(history(
            202,
            2,
            "new concurrent history",
            "must not appear",
        ));
        Ok(Message::function(json!({"found":true})))
    }
}

#[test]
fn react_model_iterations_reuse_one_initial_memory_snapshot() {
    let state = State::default();
    state
        .chats
        .lock()
        .unwrap()
        .push(history(201, 1, "first", "a"));
    let capabilities = vec![Search {
        definition: ToolDefinition {
            name: "search".into(),
            description: "search".into(),
            parameters: json!({"type":"object"}).as_object().unwrap().clone(),
        },
        state: state.clone(),
    }];
    let model = Model {
        tool: true,
        ..Default::default()
    };
    let agent = agent::react()
        .model(model.clone())
        .memory(config(session(), &state).with_scopes(["user"]))
        .tools(capabilities)
        .build()
        .unwrap();
    let output = block_on(agent.run(input(), true, Cancellation::new())).unwrap();
    let events = block_on(output.collect::<Vec<_>>());
    assert_eq!(events.len(), 3);
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(model.requests.lock().unwrap().len(), 2);
    assert!(
        !values(&model, 1)
            .iter()
            .any(|value| value == "new concurrent history")
    );
    assert_eq!(
        state
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|call| call.as_str() == "session")
            .count(),
        1
    );
    assert_eq!(
        state
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|call| call.as_str() == "chat")
            .count(),
        1
    );
    assert_eq!(
        state
            .calls
            .lock()
            .unwrap()
            .iter()
            .filter(|call| call.as_str() == "long")
            .count(),
        3
    );
}

#[test]
fn later_memory_configuration_replaces_earlier_components() {
    let first = State::default();
    let last = State::default();
    last.chats
        .lock()
        .unwrap()
        .push(history(201, 1, "last configured", "answer"));
    let model = Model::default();
    let agent = agent::chat()
        .memory(config(session(), &first))
        .model(model.clone())
        .memory(config(session(), &last))
        .build()
        .unwrap();
    block_on(agent.run(input(), true, Cancellation::new())).unwrap();
    assert!(first.calls.lock().unwrap().is_empty());
    assert_eq!(values(&model, 0), ["last configured", "answer", "camping?"]);
}

#[tokio::test]
async fn pending_memory_reads_are_cancellable_at_every_stage() {
    for stage in ["session", "chat", "long"] {
        let state = State {
            wait: Some(stage),
            ..Default::default()
        };
        let model = Model::default();
        let agent = agent::chat()
            .model(model.clone())
            .memory(config(session(), &state).with_scopes(["user"]))
            .build()
            .unwrap();
        let cancellation = Cancellation::new();
        let stop = cancellation.clone();
        let cancel = async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            stop.cancel();
        };
        let (result, _) = tokio::time::timeout(
            Duration::from_secs(1),
            futures_util::future::join(agent.run(input(), true, cancellation), cancel),
        )
        .await
        .unwrap();
        let events = result.unwrap().collect::<Vec<_>>().await;
        assert!(
            matches!(events.last(), Some(Ok(Event::Complete { usage: None, finish_reason: Some(reason), .. })) if reason == "cancelled")
        );
        assert!(model.requests.lock().unwrap().is_empty());
    }
}

#[test]
fn synchronous_backend_cancellation_stops_later_reads_and_model() {
    for stage in ["session", "chat", "long"] {
        let state = State {
            cancel: Some(stage),
            ..Default::default()
        };
        let model = Model::default();
        let agent = agent::react()
            .model(model.clone())
            .memory(config(session(), &state).with_scopes(["user"]))
            .build()
            .unwrap();
        let events = block_on(async {
            agent
                .run(input(), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        });
        assert!(
            matches!(events.last(), Some(Ok(Event::Complete { usage: None, finish_reason: Some(reason), .. })) if reason == "cancelled")
        );
        assert_eq!(
            state
                .calls
                .lock()
                .unwrap()
                .iter()
                .filter(|call| call.as_str() == stage)
                .count(),
            1
        );
        assert!(model.requests.lock().unwrap().is_empty());
    }
}

#[test]
fn in_memory_backends_remain_directly_usable_through_the_config() {
    let configured = memory::Config::new(
        halo::core::memory::short::session::InMemory::new(),
        halo::core::memory::short::chat::InMemory::new(),
        halo::core::memory::short::trace::InMemory::new(),
        halo::core::memory::InMemory::new(),
    );
    block_on(configured.session.create(session(), &Cancellation::new())).unwrap();
    let model = Model::default();
    let agent = agent::chat()
        .model(model.clone())
        .memory(configured.clone().with_scopes(["agent"]))
        .build()
        .unwrap();
    block_on(agent.run(input(), true, Cancellation::new())).unwrap();
    assert_eq!(values(&model, 0), ["camping?"]);
    assert!(
        block_on(
            configured
                .long
                .get(2, Some(3), None, memory::Type::Fact, &Cancellation::new())
        )
        .unwrap()
        .is_none()
    );
}
