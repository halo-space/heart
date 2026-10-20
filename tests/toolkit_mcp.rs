use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use futures_executor::block_on;
use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, toolkit::Toolkit},
    event::Event,
    mcp::{Definition, Server},
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
    },
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Map, json};

#[derive(Clone, Copy)]
enum Mode {
    Normal,
    CallError,
    InvalidSchema,
    Dynamic,
}

struct Mcp {
    definition: Definition,
    mode: Mode,
    calls: Arc<AtomicUsize>,
    tools: Vec<ToolDefinition>,
}

fn definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.into(), description: "look up weather".into(),
        parameters: serde_json::from_value(json!({"type":"object", "properties":{"city":{"type":"string"}}, "required":["city"], "additionalProperties":false})).unwrap(),
    }
}

impl Server for Mcp {
    fn definition(&self) -> &Definition {
        &self.definition
    }

    fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    async fn call(&self, call: Call, _cancellation: &Cancellation) -> Result<Message, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(call.arguments["city"], "Beijing");
        if matches!(self.mode, Mode::CallError) {
            return Err(Error::new("TIMEOUT", "remote timeout"));
        }
        let mut message = Message::new(Role::Tool);
        message.values.insert("temperature".into(), json!(22));
        message.metadata.insert("origin".into(), json!("mcp"));
        Ok(message)
    }
}

fn server(mode: Mode) -> (Mcp, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let lists = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let mut tools = vec![definition("weather")];
    if matches!(mode, Mode::InvalidSchema) {
        tools[0].parameters = Map::from_iter([("type".into(), json!("array"))]);
    }
    (
        Mcp {
            definition: Definition {
                name: "weather-server".into(),
                ..Default::default()
            },
            mode,
            calls: Arc::clone(&calls),
            tools,
        },
        lists,
        calls,
    )
}

fn call() -> Call {
    Call {
        call_id: "call_1".into(),
        name: "weather".into(),
        arguments: Map::from_iter([("city".into(), json!("Beijing"))]),
    }
}

#[test]
fn registration_uses_ready_definitions_and_standalone_execution_preserves_direct_message() {
    let (server, lists, calls) = server(Mode::Normal);
    let toolkit = Toolkit::new().mcps(vec![server]);
    assert_eq!(lists.load(Ordering::SeqCst), 0);
    assert_eq!(toolkit.definitions()[0].name, "weather");
    let (message, evaluation) = block_on(toolkit.execute(call(), &Cancellation::new())).unwrap();
    assert_eq!(lists.load(Ordering::SeqCst), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(message.values["temperature"], 22);
    assert_eq!(message.metadata["origin"], "mcp");
    assert!(!message.values.contains_key("tool_result"));
    assert!(evaluation.passed);
    assert_eq!(toolkit.definitions()[0].name, "weather");
}

#[test]
fn calls_are_validated_before_the_server_and_unknown_names_are_rejected() {
    let (server, _, calls) = server(Mode::Normal);
    let toolkit = Toolkit::new().mcps(vec![server]);
    for (call, code) in [
        (
            Call {
                arguments: Map::new(),
                ..call()
            },
            "INVALID_ARGUMENTS",
        ),
        (
            Call {
                call_id: String::new(),
                ..call()
            },
            "INVALID_ARGUMENTS",
        ),
        (
            Call {
                name: "unknown".into(),
                ..call()
            },
            "NOT_FOUND",
        ),
    ] {
        assert_eq!(
            block_on(toolkit.execute(call, &Cancellation::new()))
                .unwrap_err()
                .code,
            code
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

struct LocalTool;
impl Tool for LocalTool {
    fn definition(&self) -> &ToolDefinition {
        static TOOL: std::sync::OnceLock<ToolDefinition> = std::sync::OnceLock::new();
        TOOL.get_or_init(|| definition("weather"))
    }
    async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
        panic!("a colliding capability must never execute");
    }
}

#[test]
fn duplicate_names_across_local_tools_or_mcp_servers_fail_without_dispatch() {
    let (first, _, calls) = server(Mode::Normal);
    let toolkit = Toolkit::new().tools(LocalTool).mcps(vec![first]);
    assert_eq!(
        block_on(toolkit.execute(call(), &Cancellation::new()))
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let (first, _, first_calls) = server(Mode::Normal);
    let (second, _, second_calls) = server(Mode::Normal);
    let toolkit = Toolkit::new().mcps(vec![first, second]);
    assert_eq!(
        block_on(toolkit.execute(call(), &Cancellation::new()))
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(
        first_calls.load(Ordering::SeqCst) + second_calls.load(Ordering::SeqCst),
        0
    );
}

struct Model {
    requests: Arc<Mutex<Vec<Request>>>,
    tools: bool,
    expect_error: bool,
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let index = self.requests.lock().unwrap().len();
        if self.tools && index == 1 {
            let result = request
                .messages
                .as_slice()
                .iter()
                .find(|message| message.role == Role::Tool)
                .unwrap();
            if self.expect_error {
                assert_eq!(result.values["error"]["code"], "TIMEOUT");
            } else {
                assert_eq!(result.values["temperature"], 22);
            }
            assert!(!result.values.contains_key("tool_result"));
        }
        self.requests.lock().unwrap().push(request);
        let mut message = Message::new(Role::Assistant);
        if self.tools && index == 0 {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"call_1", "name":"weather", "arguments":{"city":"Beijing"}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        } else {
            message.values.insert("answer".into(), json!("done"));
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

#[test]
fn chat_uses_prepared_mcp_definitions_without_rediscovery() {
    let (server, lists, calls) = server(Mode::Dynamic);
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = agent::chat()
        .model(Model {
            requests: Arc::clone(&requests),
            tools: false,
            expect_error: false,
        })
        .toolkit(Toolkit::new().mcps(vec![server]))
        .build()
        .unwrap();
    block_on(async {
        for _ in 0..2 {
            let mut events = agent
                .run(Input::default(), true, Cancellation::new())
                .await
                .unwrap();
            assert!(events.next().await.unwrap().is_ok());
        }
    });
    assert_eq!(lists.load(Ordering::SeqCst), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let requests = requests.lock().unwrap();
    assert_eq!(requests[0].tools[0].name, "weather");
    assert_eq!(requests[1].tools[0].name, "weather");
}

#[test]
fn react_uses_ready_mcp_tools_and_preserves_success_or_direct_error_context() {
    for (mode, expect_error) in [(Mode::Normal, false), (Mode::CallError, true)] {
        let (server, lists, calls) = server(mode);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let agent = agent::react()
            .model(Model {
                requests: Arc::clone(&requests),
                tools: true,
                expect_error,
            })
            .toolkit(Toolkit::new().mcps(vec![server]))
            .build()
            .unwrap();
        let events = block_on(async {
            let events = agent
                .run(Input::default(), true, Cancellation::new())
                .await
                .unwrap();
            events.collect::<Vec<_>>().await
        });
        assert!(events.iter().all(Result::is_ok));
        assert_eq!(events.len(), 3);
        assert_eq!(lists.load(Ordering::SeqCst), 0);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(requests.lock().unwrap().len(), 2);
    }
}

#[test]
fn invalid_ready_tool_schema_is_rejected_before_dispatch() {
    let (server, lists, calls) = server(Mode::InvalidSchema);
    let toolkit = Toolkit::new().mcps(vec![server]);
    let error = block_on(toolkit.execute(call(), &Cancellation::new())).unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert_eq!(lists.load(Ordering::SeqCst), 0);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
