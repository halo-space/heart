use std::{
    collections::VecDeque,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};

use futures_core::Stream;
use futures_executor::block_on;
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::compression::Compressor as _,
    agent::event::Event,
    agent::message::Role,
    agent::model::{
        chat::{self, Chat},
        content::part::Part,
        token::Usage,
    },
    agent::reason::Reasoner as _,
    core,
};
use serde_json::json;

fn message(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":value}).as_object().unwrap().clone(),
    });
    message
}

fn complete(value: &str) -> Result<Event, Error> {
    Ok(Event::Complete {
        message: message(Role::Assistant, value),
        usage: None,
        finish_reason: Some("stop".into()),
    })
}

#[derive(Clone, Copy)]
enum Tail {
    End,
    Pending,
    Panic,
}

struct Model {
    requests: Mutex<Vec<chat::Request>>,
    events: Vec<Result<Event, Error>>,
    tail: Tail,
    create_pending: bool,
    create_error: Option<Error>,
    cancel_during_create: Option<Cancellation>,
    create_dropped: Arc<AtomicBool>,
    stream_dropped: Arc<AtomicBool>,
}

#[derive(Clone)]
struct Fake(Arc<Model>);

impl Fake {
    fn new(events: Vec<Result<Event, Error>>, tail: Tail) -> Self {
        Self(Arc::new(Model {
            requests: Mutex::new(Vec::new()),
            events,
            tail,
            create_pending: false,
            create_error: None,
            cancel_during_create: None,
            create_dropped: Arc::new(AtomicBool::new(false)),
            stream_dropped: Arc::new(AtomicBool::new(false)),
        }))
    }
}

struct Guard(Arc<AtomicBool>);
impl Drop for Guard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct Events {
    events: VecDeque<Result<Event, Error>>,
    tail: Tail,
    _guard: Guard,
}

impl Stream for Events {
    type Item = Result<Event, Error>;
    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(event) = self.events.pop_front() {
            return Poll::Ready(Some(event));
        }
        match self.tail {
            Tail::End => Poll::Ready(None),
            Tail::Pending => Poll::Pending,
            Tail::Panic => panic!("must not poll beyond Complete"),
        }
    }
}

impl Chat for Fake {
    type Stream = Events;
    async fn stream(&self, request: chat::Request) -> Result<Events, Error> {
        self.0.requests.lock().unwrap().push(request);
        let _guard = Guard(self.0.create_dropped.clone());
        if let Some(cancellation) = &self.0.cancel_during_create {
            cancellation.cancel();
        }
        if self.0.create_pending {
            std::future::pending::<()>().await;
        }
        if let Some(error) = &self.0.create_error {
            return Err(error.clone());
        }
        Ok(Events {
            events: self.0.events.clone().into(),
            tail: self.0.tail,
            _guard: Guard(self.0.stream_dropped.clone()),
        })
    }
}

fn request() -> chat::Request {
    chat::Request {
        messages: Messages::new([message(Role::System, "按调用方要求处理以下历史。")]),
        stream: true,
        ..Default::default()
    }
}

#[test]
fn compression_size_has_no_implicit_default_or_construction_io() {
    let fake = Fake::new(vec![complete("summary")], Tail::End);
    let compressor = core::compression::Compressor::new(fake.clone(), request()).unwrap();
    assert_eq!(compressor.size(), None);
    assert!(fake.0.requests.lock().unwrap().is_empty());
    let output = block_on(compressor.compress(Messages::default(), &Cancellation::new())).unwrap();
    assert_eq!(output.as_slice()[0].content[0].data["value"], "summary");
    assert_eq!(compressor.size(), None);
}

#[test]
fn compression_size_is_component_owned_and_replaceable_without_io() {
    let fake = Fake::new(vec![complete("summary")], Tail::End);
    let compressor = core::compression::Compressor::new(fake.clone(), request())
        .unwrap()
        .with_size(10_000);
    assert_eq!(compressor.size(), Some(10_000));
    let compressor = compressor.with_size(20_000);
    assert_eq!(compressor.size(), Some(20_000));
    assert!(fake.0.requests.lock().unwrap().is_empty());
}

#[test]
fn compression_size_does_not_change_model_max_tokens_or_output_usage() {
    let known = Usage {
        input: Some(30),
        output: Some(8),
        total: Some(38),
        ..Default::default()
    };
    let fake = Fake::new(
        vec![Ok(Event::Complete {
            message: message(Role::Assistant, "summary"),
            usage: Some(known.clone()),
            finish_reason: None,
        })],
        Tail::End,
    );
    let mut base = request();
    base.max_tokens = Some(128);
    let compressor = core::compression::Compressor::new(fake.clone(), base.clone())
        .unwrap()
        .with_size(10_000);
    let output = block_on(compressor.compress(Messages::default(), &Cancellation::new())).unwrap();
    assert_eq!(fake.0.requests.lock().unwrap()[0], base);
    assert_eq!(output.as_slice()[0].usage, Some(known));
    assert_eq!(compressor.size(), Some(10_000));
}

#[test]
fn compression_zero_size_is_not_silently_normalized_or_a_direct_call_gate() {
    let fake = Fake::new(vec![complete("summary")], Tail::End);
    let compressor = core::compression::Compressor::new(fake, request())
        .unwrap()
        .with_size(0);
    assert_eq!(compressor.size(), Some(0));
    assert!(block_on(compressor.compress(Messages::default(), &Cancellation::new())).is_ok());
}

#[test]
fn custom_compressors_share_size_contract_and_legacy_standalone_remains_compatible() {
    struct Custom {
        size: Option<u64>,
    }
    impl Custom {
        fn with_size(mut self, size: u64) -> Self {
            self.size = Some(size);
            self
        }
    }
    impl halo::agent::compression::Compressor for Custom {
        fn size(&self) -> Option<u64> {
            self.size
        }
        async fn compress(&self, input: Messages, _: &Cancellation) -> Result<Messages, Error> {
            Ok(input)
        }
    }
    struct Standalone;
    impl halo::agent::compression::Compressor for Standalone {
        async fn compress(&self, input: Messages, _: &Cancellation) -> Result<Messages, Error> {
            Ok(input)
        }
    }
    let custom = Custom { size: None }.with_size(10_000);
    assert_eq!(custom.size(), Some(10_000));
    let input = Messages::new([message(Role::User, "input")]);
    assert_eq!(
        block_on(custom.compress(input.clone(), &Cancellation::new())).unwrap(),
        input
    );
    assert_eq!(Standalone.size(), None);
    assert_eq!(
        block_on(Standalone.compress(input.clone(), &Cancellation::new())).unwrap(),
        input
    );
}

#[test]
fn compressor_preserves_input_order_and_all_explicit_request_parameters() {
    let fake = Fake::new(vec![complete("摘要")], Tail::End);
    let mut base = request();
    base.temperature = Some(0.1);
    base.max_tokens = Some(200);
    base.top_p = Some(0.8);
    base.stop = Some(vec!["END".into()]);
    base.response_format = Some(chat::ResponseFormat::JsonObject);
    base.options.insert("reasoning_effort".into(), json!("low"));
    let original = base.clone();
    let component = core::compression::Compressor::new(fake.clone(), base).unwrap();
    let input = Messages::new([
        message(Role::User, "北京天气"),
        message(Role::Assistant, "晴天"),
    ]);
    let result = block_on(component.compress(input.clone(), &Cancellation::new())).unwrap();
    assert_eq!(result, Messages::new([message(Role::Assistant, "摘要")]));
    let calls = fake.0.requests.lock().unwrap();
    assert_eq!(calls.len(), 1);
    let mut expected = original;
    expected.messages.0.extend(input.0);
    assert_eq!(calls[0], expected);
}

#[test]
fn reasoner_is_callable_without_agent_or_runtime_and_preserves_structured_result() {
    let mut final_message = message(Role::Assistant, "结果");
    final_message.values.insert("answer".into(), json!(42));
    final_message
        .metadata
        .insert("model_name".into(), json!("test-model"));
    final_message.content.insert(
        0,
        Part {
            r#type: "think".into(),
            data: json!({"value":"分析"}).as_object().unwrap().clone(),
        },
    );
    let fake = Fake::new(
        vec![Ok(Event::Complete {
            message: final_message.clone(),
            usage: None,
            finish_reason: None,
        })],
        Tail::End,
    );
    let component = core::reason::Reasoner::new(fake.clone(), request()).unwrap();
    let result = block_on(component.reason(
        Messages::new([message(Role::User, "问题")]),
        &Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(result, Messages::new([final_message]));
    assert_eq!(fake.0.requests.lock().unwrap().len(), 1);
}

#[test]
fn complete_is_not_reconstructed_from_deltas_or_polled_past_its_terminal_boundary() {
    let fake = Fake::new(
        vec![
            Ok(Event::Delta {
                message: message(Role::Assistant, "不应拼接的片段"),
            }),
            complete("最终完整摘要"),
        ],
        Tail::Panic,
    );
    let component = core::compression::Compressor::new(fake.clone(), request()).unwrap();
    let result = block_on(component.compress(Messages::default(), &Cancellation::new())).unwrap();
    assert_eq!(
        result,
        Messages::new([message(Role::Assistant, "最终完整摘要")])
    );
    assert!(fake.0.stream_dropped.load(Ordering::SeqCst));
}

#[test]
fn complete_usage_is_copied_to_message_without_delta_accumulation() {
    let delta_usage = Usage {
        total: Some(999),
        ..Default::default()
    };
    let usage = Usage {
        input: Some(10),
        output: Some(4),
        total: Some(14),
        ..Default::default()
    };
    let mut delta = message(Role::Assistant, "片段");
    delta.usage = Some(delta_usage);
    let fake = Fake::new(
        vec![
            Ok(Event::Delta { message: delta }),
            Ok(Event::Complete {
                message: message(Role::Assistant, "摘要"),
                usage: Some(usage.clone()),
                finish_reason: None,
            }),
        ],
        Tail::End,
    );
    let component = core::compression::Compressor::new(fake, request()).unwrap();
    let result = block_on(component.compress(Messages::default(), &Cancellation::new())).unwrap();
    assert_eq!(result.0[0].usage, Some(usage));
}

#[test]
fn absent_event_usage_keeps_actual_message_usage_instead_of_estimating_it() {
    let mut final_message = message(Role::Assistant, "结果");
    final_message.usage = Some(Usage {
        total: Some(12),
        ..Default::default()
    });
    let fake = Fake::new(
        vec![Ok(Event::Complete {
            message: final_message.clone(),
            usage: None,
            finish_reason: None,
        })],
        Tail::End,
    );
    let component = core::reason::Reasoner::new(fake, request()).unwrap();
    assert_eq!(
        block_on(component.reason(Messages::default(), &Cancellation::new()))
            .unwrap()
            .0[0],
        final_message
    );
}

#[test]
fn missing_complete_is_an_error_not_partial_success() {
    for events in [
        vec![],
        vec![Ok(Event::Delta {
            message: message(Role::Assistant, "部分摘要"),
        })],
    ] {
        let fake = Fake::new(events, Tail::End);
        let component = core::compression::Compressor::new(fake.clone(), request()).unwrap();
        let error =
            block_on(component.compress(Messages::default(), &Cancellation::new())).unwrap_err();
        assert_eq!(error.code, "INVALID_RESPONSE");
        assert_eq!(fake.0.requests.lock().unwrap().len(), 1);
        assert!(fake.0.stream_dropped.load(Ordering::SeqCst));
    }
}

#[test]
fn model_error_after_delta_is_preserved_without_retry_or_error_message_wrapping() {
    let error = Error::new("RATE_LIMITED", "rate limited").with_details("retry later");
    let fake = Fake::new(
        vec![
            Ok(Event::Delta {
                message: message(Role::Assistant, "部分"),
            }),
            Err(error.clone()),
        ],
        Tail::Panic,
    );
    let component = core::reason::Reasoner::new(fake.clone(), request()).unwrap();
    assert_eq!(
        block_on(component.reason(Messages::default(), &Cancellation::new())).unwrap_err(),
        error
    );
    assert_eq!(fake.0.requests.lock().unwrap().len(), 1);
    assert!(fake.0.stream_dropped.load(Ordering::SeqCst));
}

#[test]
fn creation_error_is_also_direct_and_not_retried() {
    let error = Error::new("NETWORK_ERROR", "connection failed").with_details("endpoint");
    let mut fake = Fake::new(vec![], Tail::End);
    Arc::get_mut(&mut fake.0).unwrap().create_error = Some(error.clone());
    let component = core::compression::Compressor::new(fake.clone(), request()).unwrap();
    assert_eq!(
        block_on(component.compress(Messages::default(), &Cancellation::new())).unwrap_err(),
        error
    );
    assert_eq!(fake.0.requests.lock().unwrap().len(), 1);
}

#[test]
fn constructor_rejects_invalid_requests_before_any_model_call() {
    let fake = Fake::new(vec![], Tail::Panic);
    let mut invalid = request();
    invalid.tool_choice = Some(chat::ToolChoice::Required);
    assert_eq!(
        core::compression::Compressor::new(fake.clone(), invalid.clone())
            .err()
            .unwrap()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(
        core::reason::Reasoner::new(fake.clone(), invalid)
            .err()
            .unwrap()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert!(fake.0.requests.lock().unwrap().is_empty());
}

#[test]
fn cancellation_before_call_prevents_model_invocation_for_both_components() {
    let fake = Fake::new(vec![], Tail::Panic);
    let compressor = core::compression::Compressor::new(fake.clone(), request()).unwrap();
    let reasoner = core::reason::Reasoner::new(fake.clone(), request()).unwrap();
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        block_on(compressor.compress(Messages::default(), &cancellation))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(
        block_on(reasoner.reason(Messages::default(), &cancellation))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert!(fake.0.requests.lock().unwrap().is_empty());
}

struct Wakes(AtomicUsize);
impl Wake for Wakes {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn cancellation_wakes_pending_stream_creation_and_drops_its_future() {
    let mut fake = Fake::new(vec![], Tail::Panic);
    Arc::get_mut(&mut fake.0).unwrap().create_pending = true;
    let component = core::compression::Compressor::new(fake.clone(), request()).unwrap();
    let cancellation = Cancellation::new();
    let mut future = Box::pin(component.compress(Messages::default(), &cancellation));
    let wakes = Arc::new(Wakes(AtomicUsize::new(0)));
    let waker = Waker::from(wakes.clone());
    let mut cx = Context::from_waker(&waker);
    assert!(future.as_mut().poll(&mut cx).is_pending());
    assert!(!fake.0.create_dropped.load(Ordering::SeqCst));
    cancellation.cancel();
    assert!(wakes.0.load(Ordering::SeqCst) > 0);
    let Poll::Ready(result) = future.as_mut().poll(&mut cx) else {
        panic!("must wake after cancellation")
    };
    assert_eq!(result.unwrap_err().code, "CANCELLED");
    assert!(fake.0.create_dropped.load(Ordering::SeqCst));
}

#[test]
fn cancellation_wakes_silent_stream_after_delta_and_drops_it_without_partial_result() {
    let fake = Fake::new(
        vec![Ok(Event::Delta {
            message: message(Role::Assistant, "部分"),
        })],
        Tail::Pending,
    );
    let component = core::reason::Reasoner::new(fake.clone(), request()).unwrap();
    let cancellation = Cancellation::new();
    let mut future = Box::pin(component.reason(Messages::default(), &cancellation));
    let wakes = Arc::new(Wakes(AtomicUsize::new(0)));
    let waker = Waker::from(wakes.clone());
    let mut cx = Context::from_waker(&waker);
    assert!(future.as_mut().poll(&mut cx).is_pending());
    assert!(!fake.0.stream_dropped.load(Ordering::SeqCst));
    cancellation.cancel();
    assert!(wakes.0.load(Ordering::SeqCst) > 0);
    let Poll::Ready(result) = future.as_mut().poll(&mut cx) else {
        panic!("must wake after cancellation")
    };
    assert_eq!(result.unwrap_err().code, "CANCELLED");
    assert!(fake.0.stream_dropped.load(Ordering::SeqCst));
}

#[test]
fn synchronous_cancellation_during_provider_poll_prevents_complete_delivery() {
    let cancellation = Cancellation::new();
    let mut fake = Fake::new(vec![complete("不能交付")], Tail::End);
    Arc::get_mut(&mut fake.0).unwrap().cancel_during_create = Some(cancellation.clone());
    let component = core::compression::Compressor::new(fake.clone(), request()).unwrap();
    assert_eq!(
        block_on(component.compress(Messages::default(), &cancellation))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(fake.0.requests.lock().unwrap().len(), 1);
    assert!(fake.0.stream_dropped.load(Ordering::SeqCst));
}

#[test]
fn independent_calls_do_not_accumulate_hidden_history_or_modify_configuration() {
    let fake = Fake::new(vec![complete("摘要")], Tail::End);
    let component = core::compression::Compressor::new(fake.clone(), request()).unwrap();
    for text in ["第一轮", "第二轮"] {
        block_on(component.compress(
            Messages::new([message(Role::User, text)]),
            &Cancellation::new(),
        ))
        .unwrap();
    }
    let calls = fake.0.requests.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].messages.0.len(), 2);
    assert_eq!(calls[1].messages.0.len(), 2);
    assert_eq!(calls[0].messages.0[1].content[0].data["value"], "第一轮");
    assert_eq!(calls[1].messages.0[1].content[0].data["value"], "第二轮");
}

#[test]
fn components_can_be_shared_between_independent_application_threads() {
    let fake = Fake::new(vec![complete("摘要")], Tail::End);
    let component = Arc::new(core::compression::Compressor::new(fake.clone(), request()).unwrap());
    let handles: Vec<_> = ["a", "b"]
        .into_iter()
        .map(|text| {
            let component = component.clone();
            std::thread::spawn(move || {
                block_on(component.compress(
                    Messages::new([message(Role::User, text)]),
                    &Cancellation::new(),
                ))
                .unwrap()
            })
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap().0[0].content[0].data["value"], "摘要");
    }
    let calls = fake.0.requests.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|call| call.messages.0.len() == 2));
}

#[test]
fn explicit_tool_call_results_are_returned_but_not_executed_by_the_transform() {
    let mut result = Message::new(Role::Assistant);
    result.content.push(Part {
        r#type: "tool_call".into(),
        data: json!({"call_id":"call1","name":"search","arguments":{"query":"天气"}})
            .as_object()
            .unwrap()
            .clone(),
    });
    let fake = Fake::new(
        vec![Ok(Event::Complete {
            message: result.clone(),
            usage: None,
            finish_reason: Some("tool_calls".into()),
        })],
        Tail::Panic,
    );
    let component = core::reason::Reasoner::new(fake.clone(), request()).unwrap();
    assert_eq!(
        block_on(component.reason(Messages::default(), &Cancellation::new())).unwrap(),
        Messages::new([result])
    );
    assert_eq!(fake.0.requests.lock().unwrap().len(), 1);
}
