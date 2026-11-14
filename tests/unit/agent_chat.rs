fn test_id() -> i64 {
    static NEXT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(600000);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::task::{Context, Poll};

use crate::runtime::agent::chat::runner::execute_subagent;
use super::*;
use halo::agent::Agent as _;
use halo::agent::event::Event;
use halo::agent::message::Role;
use halo::agent::tool::{Call, Tool, ToolDefinition};
use futures_core::Stream;
use futures_executor::block_on;
use serde_json::json;

#[derive(Debug)]
struct EventFeed {
    events: VecDeque<Result<Event, Error>>,
    pending: bool,
}

impl Stream for EventFeed {
    type Item = Result<Event, Error>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.pending {
            return Poll::Pending;
        }
        Poll::Ready(self.events.pop_front())
    }
}

struct WaitingSubAgent;

struct CountingSubAgent {
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl halo::agent::Agent for WaitingSubAgent {
    async fn reply(
        &self,
        _: halo::agent::Input,
        _: halo::agent::Cancellation,
    ) -> Result<(), halo::agent::Error> {
        Err(halo::agent::Error::new(
            "NOT_FOUND",
            "fixture has no live feedback consumer",
        ))
    }
    type Stream = futures_util::stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn run(
        &self,
        _input: Input,
        _stream: bool,
        _cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        if crate::runtime::agent::scope::restore().is_none() {
            assert!(crate::runtime::agent::scope::capture(json!({
                "owner": "chat-child"
            })));
            return Err(Error::new("WAITING", "child is waiting for approval"));
        }
        let mut message = Message::new(Role::Assistant);
        message.content.push(Part {
            r#type: "text".into(),
            data: halo::agent::model::content::part::object(json!({"value": "child approved"})),
        });
        Ok(futures_util::stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

impl halo::agent::Agent for CountingSubAgent {
    async fn reply(
        &self,
        _: halo::agent::Input,
        _: halo::agent::Cancellation,
    ) -> Result<(), halo::agent::Error> {
        Err(halo::agent::Error::new(
            "NOT_FOUND",
            "fixture has no live feedback consumer",
        ))
    }
    type Stream = futures_util::stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn run(
        &self,
        _input: Input,
        _stream: bool,
        _cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut message = Message::new(Role::Assistant);
        message.content.push(Part {
            r#type: "text".into(),
            data: halo::agent::model::content::part::object(json!({"value": "done child"})),
        });
        Ok(futures_util::stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

fn next_now<S: Stream<Item = Result<Event, Error>> + Unpin>(
    stream: &mut S,
) -> Option<Result<Event, Error>> {
    let waker: &std::task::Waker = std::task::Waker::noop();
    let mut context = Context::from_waker(waker);
    match Pin::new(stream).poll_next(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => None,
    }
}

#[test]
fn builder_uses_the_frozen_defaults_and_maps_payload_parts() {
    let config = builder();
    assert_eq!(config.max_concurrency, 100);

    let message = crate::components::agent_input::user_message(Input {
        metadata: Map::from_iter([
            (String::from("chat_id"), json!(101)),
            (String::from("language"), json!("zh")),
        ]),
        payload: vec![halo::agent::Payload {
            r#type: "text".into(),
            data: json!("hello"),
            mime_type: Some("text/plain".into()),
        }],
    });
    assert!(!message.metadata.contains_key("chat_id"));
    assert_eq!(message.metadata["language"], "zh");
    assert_eq!(message.content[0].r#type, "text");
    assert_eq!(message.content[0].data["value"], "hello");
    assert_eq!(message.content[0].data["mime_type"], "text/plain");
}

#[test]
fn public_chat_builder_keeps_system_prompt_before_user_payload() {
    let messages = initial_messages(Input::default(), Some("你是助手"), Vec::new());
    assert_eq!(messages.as_slice()[0].role, Role::System);
    assert_eq!(messages.as_slice()[0].content[0].data["value"], "你是助手");
    assert_eq!(messages.as_slice()[1].role, Role::User);
}

#[test]
fn stream_item_errors_and_normal_completion_are_terminal_once() {
    let mut errored = Events::new(
        EventFeed {
            events: VecDeque::from([Err(Error::new("PROVIDER", "stream failed"))]),
            pending: false,
        },
        Cancellation::new(),
        0,
        Hooks::default(),
    );
    assert_eq!(
        next_now(&mut errored)
            .expect("stream error")
            .unwrap_err()
            .code,
        "PROVIDER"
    );
    assert!(next_now(&mut errored).is_none());

    let cancellation = Cancellation::new();
    let mut completed = Events::new(
        EventFeed {
            events: VecDeque::from([Ok(Event::Complete {
                message: Message::new(Role::Assistant),
                usage: None,
                finish_reason: Some("stop".into()),
            })]),
            pending: false,
        },
        cancellation.clone(),
        0,
        Hooks::default(),
    );
    assert!(matches!(
        next_now(&mut completed),
        Some(Ok(Event::Complete { .. }))
    ));
    cancellation.cancel();
    assert!(next_now(&mut completed).is_none());
}

#[test]
fn cancelled_run_does_not_invoke_provider() {
    let provider = halo::agent::core::provider::vllm::Vllm::builder()
        .api_key("compile-only")
        .build()
        .expect("construct provider without sending a request");
    let chat = provider
        .chat(halo::agent::model::chat::Config::new("compile-only"))
        .expect("construct real Provider Chat");
    let agent = builder().model(chat).build().expect("agent build");
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let error = block_on(agent.run(Input::default(), true, cancellation)).unwrap_err();
    assert_eq!(error.code, "CANCELLED");
}

#[test]
fn chat_subagent_waiting_scene_is_resumed_for_the_same_call() {
    block_on(async {
        let child = SubAgent::new("reviewer", WaitingSubAgent).unwrap();
        let call = Call {
            call_id: "spawn_1".into(),
            name: "agent_spawn".into(),
            arguments: serde_json::from_value(json!({
                "name": "reviewer",
                "input": {"payload": []}
            }))
            .unwrap(),
        };
        let failure = execute_subagent(
            vec![child.clone()],
            Map::new(),
            call.clone(),
            None,
            Cancellation::new(),
            None,
            None,
        )
        .await
        .expect_err("child should wait before approval");
        assert_eq!(failure.error.code, "WAITING");
        assert_eq!(failure.scene, Some(json!({"owner": "chat-child"})));

        let progress = execute_subagent(
            vec![child],
            Map::new(),
            call,
            failure.scene,
            Cancellation::new(),
            None,
            None,
        )
        .await
        .expect("saved child scene should resume");
        assert_eq!(progress.message.role, Role::Function);
        assert_eq!(progress.message.content[0].data["value"], "child approved");
    });
}

#[test]
fn chat_parallel_subagents_keep_completed_siblings_when_one_waits() {
    block_on(async {
        let root = Root::default();
        let steps: Arc<Mutex<Vec<Step>>> = Default::default();
        let completed_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let subagents = vec![
            SubAgent::new(
                "done",
                CountingSubAgent {
                    calls: completed_calls.clone(),
                },
            )
            .unwrap(),
            SubAgent::new("waiting", WaitingSubAgent).unwrap(),
        ];
        let calls = vec![
            Call {
                call_id: "spawn_done".into(),
                name: "agent_spawn".into(),
                arguments: serde_json::from_value(json!({
                    "name": "done",
                    "input": {"payload": []}
                }))
                .unwrap(),
            },
            Call {
                call_id: "spawn_waiting".into(),
                name: "agent_spawn".into(),
                arguments: serde_json::from_value(json!({
                    "name": "waiting",
                    "input": {"payload": []}
                }))
                .unwrap(),
            },
        ];
        let first = execute_tool_batch(
            ToolBatchConfig {
                interaction: None,
                relay: None,
                dispatch: Arc::new(Dispatch::new()),
                steps: steps.clone(),
                subagents: subagents.clone(),
                access: Access::default(),
                hooks: Hooks::default(),
                metadata: Map::new(),
                cancellation: Cancellation::new(),
                max_concurrency: 2,
                root: root.clone(),
                id_fn: Some(test_id),
                trace: TraceRoot::start("agent.chat.test"),
            },
            calls.clone(),
        )
        .await
        .expect_err("one child should wait");
        assert_eq!(first.error.code, "WAITING");
        assert!(root.message(&steps.lock().unwrap(), &calls[0]).is_some());
        let resumed = execute_tool_batch(
            ToolBatchConfig {
                interaction: None,
                relay: None,
                dispatch: Arc::new(Dispatch::new()),
                steps,
                subagents,
                access: Access::default(),
                hooks: Hooks::default(),
                metadata: Map::new(),
                cancellation: Cancellation::new(),
                max_concurrency: 2,
                root,
                id_fn: None,
                trace: TraceRoot::start("agent.chat.test"),
            },
            calls,
        )
        .await
        .expect("waiting child should resume with completed sibling");
        assert_eq!(resumed.0.len(), 2);
        assert_eq!(completed_calls.load(Ordering::SeqCst), 1);
    });
}

struct BatchProbeTool {
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    started: Arc<std::sync::Mutex<Vec<usize>>>,
    fail: Option<usize>,
}

impl Tool for BatchProbeTool {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: std::sync::OnceLock<ToolDefinition> = std::sync::OnceLock::new();
        DEFINITION.get_or_init(|| ToolDefinition {
            name: "batch_probe".into(),
            description: "Record an indexed batch call".into(),
            parameters: serde_json::from_value(json!({
                "type":"object",
                "properties":{"index":{"type":"integer"}},
                "required":["index"],
                "additionalProperties":false
            }))
            .expect("static tool schema"),
        })
    }

    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        let index = call.arguments["index"].as_u64().unwrap() as usize;
        self.started.lock().unwrap().push(index);
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(4)).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        if self.fail == Some(index) {
            return Err(Error::new("BACKEND", format!("failed index {index}")));
        }
        Ok(Message::function(json!({"index":index})))
    }
}

fn batch_calls(indices: &[usize]) -> Vec<Call> {
    indices
        .iter()
        .map(|index| Call {
            call_id: format!("batch_{index}"),
            name: "batch_probe".into(),
            arguments: serde_json::from_value(json!({"index":index})).expect("valid indexed call"),
        })
        .collect()
}

fn batch_config(tool: BatchProbeTool, max_concurrency: usize) -> ToolBatchConfig {
    ToolBatchConfig {
        interaction: None,
        relay: None,
        steps: Arc::new(std::sync::Mutex::new(Vec::new())),
        dispatch: Arc::new(Dispatch::new().tools(tool)),
        subagents: Vec::new(),
        access: Access::default(),
        hooks: Hooks::default(),
        metadata: Map::new(),
        cancellation: Cancellation::new(),
        max_concurrency,
        root: Root::default(),
        id_fn: None,
        trace: TraceRoot::start("agent.chat.test"),
    }
}

#[test]
fn input_estimate_includes_structured_message_fields_and_tool_schema() {
    let base = Request {
        messages: Messages::new([Message::new(Role::User)]),
        ..Request::default()
    };
    let mut structured = Message::new(Role::Tool);
    structured.values.insert(
        "result".into(),
        json!({"documents":[{"title":"a","content":"long result"}]}),
    );
    structured
        .metadata
        .insert("call_id".into(), json!("call_001"));
    let with_structured = Request {
        messages: Messages::new([structured]),
        ..Request::default()
    };
    let with_tool = Request {
        messages: base.messages.clone(),
        tools: vec![ToolDefinition {
            name: "search".into(),
            description: "Search documents by query".into(),
            parameters: serde_json::from_value(json!({
                "type":"object",
                "properties":{"query":{"type":"string"}},
                "required":["query"]
            }))
            .expect("valid tool schema"),
        }],
        ..Request::default()
    };

    assert!(estimate_input_tokens(&with_structured) > estimate_input_tokens(&base));
    assert!(estimate_input_tokens(&with_tool) > estimate_input_tokens(&base));
}

#[test]
fn context_limit_keeps_summary_and_newest_message() {
    fn text(role: Role, value: &str) -> Message {
        let mut message = Message::new(role);
        message.content.push(Part {
            r#type: "text".into(),
            data: halo::agent::model::content::part::object(json!({"value": value})),
        });
        message
    }

    let summary = text(Role::User, "[Session summary]\nprevious context");
    let newest = text(Role::User, "current question");
    let kept = Request {
        messages: Messages::new([summary.clone(), newest.clone()]),
        ..Request::default()
    };
    let limit = estimate_input_tokens(&kept);
    let request = Request {
        messages: Messages::new([
            text(Role::User, "old history that should be removed"),
            summary,
            text(Role::Assistant, "old answer"),
            newest,
        ]),
        ..Request::default()
    };
    let fitted = fit_request(request, Some(limit)).expect("history should be clipped");
    assert_eq!(fitted.messages.as_slice().len(), 2);
    assert!(
        fitted.messages.as_slice()[0].content[0].data["value"]
            .as_str()
            .unwrap()
            .starts_with("[Session summary]")
    );
    assert_eq!(
        fitted.messages.as_slice()[1].content[0].data["value"],
        "current question"
    );
}

#[test]
fn context_limit_reports_when_pinned_messages_do_not_fit() {
    let mut system = Message::new(Role::System);
    system.content.push(Part {
        r#type: "text".into(),
        data: halo::agent::model::content::part::object(json!({"value": "a long system instruction"})),
    });
    let request = Request {
        messages: Messages::new([system]),
        ..Request::default()
    };
    let error = fit_request(request, Some(1)).expect_err("limit must be enforced");
    assert_eq!(error.code, "CONTEXT_LENGTH_EXCEEDED");
}

#[tokio::test]
async fn direct_tool_batch_caps_concurrency_and_returns_source_order() {
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (messages, usage) = execute_tool_batch(
        batch_config(
            BatchProbeTool {
                active,
                peak: Arc::clone(&peak),
                started,
                fail: None,
            },
            2,
        ),
        batch_calls(&[1, 2, 3, 4]),
    )
    .await
    .expect("all direct batch calls should succeed");

    assert_eq!(usage, None);
    assert_eq!(
        messages
            .iter()
            .map(|message| message.values["index"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
    assert_eq!(peak.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn direct_tool_batch_runs_every_call_and_returns_first_source_error() {
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (messages, _) = execute_tool_batch(
        batch_config(
            BatchProbeTool {
                active,
                peak,
                started: Arc::clone(&started),
                fail: Some(2),
            },
            3,
        ),
        batch_calls(&[1, 2, 3]),
    )
    .await
    .expect("tool errors are returned as per-call messages");

    let failed = messages
        .iter()
        .find(|message| message.values.get("call_id") == Some(&json!("batch_2")))
        .expect("failed call message is retained");
    assert_eq!(failed.values["error"]["code"], "BACKEND");
    assert_eq!(failed.values["error"]["message"], "failed index 2");
    let mut started = started.lock().unwrap().clone();
    started.sort_unstable();
    assert_eq!(started, [1, 2, 3]);
}
