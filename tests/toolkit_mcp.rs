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
    DiscoveryError,
    Pending,
    Cancel,
    InvalidSchema,
    Dynamic,
}

struct Mcp {
    definition: Definition,
    mode: Mode,
    lists: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
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

    async fn list_tools(&self, cancellation: &Cancellation) -> Result<Vec<ToolDefinition>, Error> {
        let index = self.lists.fetch_add(1, Ordering::SeqCst) + 1;
        match self.mode {
            Mode::DiscoveryError => return Err(Error::new("UNAVAILABLE", "discovery failed")),
            Mode::Pending => std::future::pending::<()>().await,
            Mode::Cancel => cancellation.cancel(),
            _ => {}
        }
        let mut tool = definition(&if matches!(self.mode, Mode::Dynamic) {
            format!("weather_{index}")
        } else {
            "weather".into()
        });
        if matches!(self.mode, Mode::InvalidSchema) {
            tool.parameters = Map::from_iter([("type".into(), json!("array"))]);
        }
        Ok(vec![tool])
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
    (
        Mcp {
            definition: Definition {
                name: "weather-server".into(),
                ..Default::default()
            },
            mode,
            lists: Arc::clone(&lists),
            calls: Arc::clone(&calls),
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
fn registration_is_deferred_and_standalone_execution_preserves_direct_message() {
    let (server, lists, calls) = server(Mode::Normal);
    let toolkit = Toolkit::new().mcps(vec![server]);
    assert_eq!(lists.load(Ordering::SeqCst), 0);
    assert!(toolkit.definitions().is_empty());
    let (message, evaluation) = block_on(toolkit.execute(call(), &Cancellation::new())).unwrap();
    assert_eq!(lists.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(message.values["temperature"], 22);
    assert_eq!(message.metadata["origin"], "mcp");
    assert!(!message.values.contains_key("tool_result"));
    assert!(evaluation.passed);
    assert!(toolkit.definitions().is_empty()); // Per-use view, no shared mutation.
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
fn chat_advertises_mcp_without_executing_it_and_refreshes_each_run() {
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
    assert_eq!(lists.load(Ordering::SeqCst), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let requests = requests.lock().unwrap();
    assert_eq!(requests[0].tools[0].name, "weather_1");
    assert_eq!(requests[1].tools[0].name, "weather_2");
}

#[test]
fn react_discovers_once_and_preserves_mcp_success_or_direct_error_context() {
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
        assert_eq!(lists.load(Ordering::SeqCst), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(requests.lock().unwrap().len(), 2);
    }
}

#[test]
fn discovery_failure_invalid_schema_or_cancellation_prevents_model_work() {
    for (mode, code) in [
        (Mode::DiscoveryError, "UNAVAILABLE"),
        (Mode::InvalidSchema, "INVALID_ARGUMENTS"),
        (Mode::Cancel, "CANCELLED"),
    ] {
        let (server, lists, calls) = server(mode);
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
        let result = block_on(agent.run(Input::default(), true, Cancellation::new()));
        if code == "CANCELLED" {
            // Agent cancellation closes the constructed run with Complete;
            // the standalone Toolkit API still returns its direct error.
            let events = block_on(result.unwrap().collect::<Vec<_>>());
            assert!(
                matches!(events.as_slice(), [Ok(Event::Complete { message, finish_reason: Some(reason), .. })]
                if reason == "cancelled" && message.content.is_empty())
            );
        } else {
            assert!(matches!(result, Err(ref error) if error.code == code));
        }
        assert_eq!(lists.load(Ordering::SeqCst), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(requests.lock().unwrap().is_empty());
    }
}

#[test]
fn waiting_discovery_is_cancellable_and_precancellation_does_not_discover() {
    let (server, lists, calls) = server(Mode::Pending);
    let toolkit = Toolkit::new().mcps(vec![server]);
    let cancellation = Cancellation::new();
    let (result, ()) = block_on(async {
        futures_util::join!(toolkit.execute(call(), &cancellation), async {
            cancellation.cancel();
        })
    });
    assert_eq!(result.unwrap_err().code, "CANCELLED");
    assert_eq!(lists.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let result = block_on(toolkit.execute(call(), &cancellation));
    assert_eq!(result.unwrap_err().code, "CANCELLED");
    assert_eq!(lists.load(Ordering::SeqCst), 1);
}
