use std::{
    collections::VecDeque,
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};

use futures_core::Stream;
use futures_executor::block_on;
use halo_agents::{
    Cancellation, Error, Message, Messages,
    agent::{Input, Payload},
    core,
    evaluation::{Evaluation, Evaluator as _},
    event::Event,
    message::Role,
    model::{
        chat::{self, Chat},
        content::part::Part,
    },
    plan::{Plan, Planner as _},
};
use serde_json::{Value, json};

fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": value}).as_object().unwrap().clone(),
    });
    message
}

fn input() -> Input {
    Input {
        metadata: serde_json::from_value(json!({"chat_id": 123, "session_id": 7})).unwrap(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"查询北京天气并总结"}),
            mime_type: None,
        }],
    }
}

fn plan_json() -> Value {
    json!({
        "version": 2,
        "nodes": {
            "a": {"name": "search", "objective": "查询天气"},
            "b": {"name": "summarize", "objective": "总结天气"}
        },
        "edges": [{"from": "a", "to": "b"}]
    })
}

fn complete(message: Message) -> Result<Event, Error> {
    Ok(Event::Complete {
        message,
        usage: None,
        finish_reason: Some("stop".into()),
    })
}

#[derive(Clone)]
struct Model {
    requests: Arc<Mutex<Vec<chat::Request>>>,
    replies: Arc<Vec<Result<Event, Error>>>,
    create_pending: bool,
    stream_pending: bool,
    create_error: Option<Error>,
    cancellation: Option<Cancellation>,
    create_dropped: Arc<AtomicBool>,
    dropped: Arc<AtomicBool>,
}

impl Model {
    fn new(events: Vec<Result<Event, Error>>) -> Self {
        Self {
            requests: Arc::default(),
            replies: Arc::new(events),
            create_pending: false,
            stream_pending: false,
            create_error: None,
            cancellation: None,
            create_dropped: Arc::new(AtomicBool::new(false)),
            dropped: Arc::new(AtomicBool::new(false)),
        }
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
    pending: bool,
    _guard: Guard,
}
impl Stream for Events {
    type Item = Result<Event, Error>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(event) = self.events.pop_front() {
            return Poll::Ready(Some(event));
        }
        if self.pending {
            Poll::Pending
        } else {
            panic!("must not read past the complete result")
        }
    }
}

impl Chat for Model {
    type Stream = Events;
    async fn stream(&self, request: chat::Request) -> Result<Events, Error> {
        // Deliberately native non-Send creation future.
        let marker = Rc::new(1);
        let _guard = Guard(self.create_dropped.clone());
        self.requests.lock().unwrap().push(request);
        if self.create_pending {
            std::future::pending::<()>().await;
        }
        assert_eq!(*marker, 1);
        if let Some(token) = &self.cancellation {
            token.cancel();
        }
        if let Some(error) = &self.create_error {
            return Err(error.clone());
        }
        Ok(Events {
            events: self.replies.as_ref().clone().into(),
            pending: self.stream_pending,
            _guard: Guard(self.dropped.clone()),
        })
    }
}

fn planner(model: Model) -> core::plan::Planner<Model> {
    core::plan::Planner::new(model, chat::Request::default()).unwrap()
}

fn evaluation(passed: bool) -> Evaluation {
    Evaluation {
        passed,
        feedback: (!passed).then(|| Messages::new([text(Role::User, "需要补充来源")])),
    }
}

#[test]
fn component_contracts_use_unified_paths_and_keep_frozen_json() {
    let plan: Plan = serde_json::from_value(plan_json()).unwrap();
    assert_eq!(serde_json::to_value(plan).unwrap(), plan_json());
    let result: Evaluation = evaluation(false);
    assert!(!result.passed);
    assert_eq!(
        serde_json::from_value::<Evaluation>(serde_json::to_value(&result).unwrap()).unwrap(),
        result
    );
}

#[test]
fn construction_is_pure_and_preserves_explicit_request_parameters() {
    let model = Model::new(vec![complete(text(
        Role::Assistant,
        &plan_json().to_string(),
    ))]);
    let schema = chat::ResponseFormat::JsonSchema {
        name: "plan".into(),
        schema: json!({"type":"object"}),
        strict: true,
    };
    let base = chat::Request {
        messages: Messages::new([text(Role::System, "用户定义规划要求")]),
        temperature: Some(0.2),
        max_tokens: Some(500),
        top_p: Some(0.8),
        stop: Some(vec!["<stop>".into()]),
        stream: false,
        response_format: Some(schema.clone()),
        options: serde_json::from_value(json!({"thinking":true})).unwrap(),
        ..Default::default()
    };
    let planner = core::plan::Planner::new(model.clone(), base.clone()).unwrap();
    assert!(model.requests.lock().unwrap().is_empty());
    block_on(planner.plan(&input(), None, None, &Cancellation::new())).unwrap();
    let requests = model.requests.lock().unwrap();
    let request = &requests[0];
    assert_eq!(request.messages.0[0], base.messages.0[0]);
    assert_eq!(request.response_format, Some(schema));
    assert_eq!(request.options, base.options);
    assert_eq!(request.temperature, base.temperature);
    assert_eq!(request.max_tokens, base.max_tokens);
    assert_eq!(request.top_p, base.top_p);
    assert_eq!(request.stop, base.stop);
    assert!(!request.stream);
}

#[test]
fn default_structured_request_asks_for_json() {
    let model = Model::new(vec![complete(text(
        Role::Assistant,
        &plan_json().to_string(),
    ))]);
    block_on(planner(model.clone()).plan(&input(), None, None, &Cancellation::new())).unwrap();
    let request = &model.requests.lock().unwrap()[0];
    assert_eq!(
        request.response_format,
        Some(chat::ResponseFormat::JsonObject)
    );
    let context: Value = serde_json::from_str(
        request.messages.0.last().unwrap().content[0].data["value"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert!(context["current_plan"].is_null());
    assert!(context["feedback"].is_null());
    assert!(context.get("capabilities").is_none());
}

#[test]
fn constructor_rejects_invalid_request_without_io() {
    let model = Model::new(vec![]);
    let invalid = chat::Request {
        tool_choice: Some(chat::ToolChoice::Required),
        ..Default::default()
    };
    assert!(core::plan::Planner::new(model.clone(), invalid.clone()).is_err());
    assert!(core::evaluation::Evaluator::new(model.clone(), invalid).is_err());
    assert!(model.requests.lock().unwrap().is_empty());
}

#[test]
fn replanning_uses_original_input_current_plan_and_feedback_without_mutating_them() {
    let mut next = plan_json();
    next["version"] = json!(9);
    next["nodes"]["b"]["objective"] = json!("补充来源后总结");
    let model = Model::new(vec![complete(text(Role::Assistant, &next.to_string()))]);
    let current: Plan = serde_json::from_value(plan_json()).unwrap();
    let before = current.clone();
    let feedback = evaluation(false).feedback.unwrap();
    let query = input();
    let result = block_on(planner(model.clone()).plan(
        &query,
        Some(&current),
        Some(&feedback),
        &Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(result.version, 9); // No caller/runtime increment or repair.
    assert_eq!(current, before);
    assert_eq!(query, input());
    let request = &model.requests.lock().unwrap()[0];
    let mut message_metadata = query.metadata.clone();
    for key in ["chat_id", "node_id", "exec_id"] {
        message_metadata.remove(key);
    }
    assert_eq!(request.messages.0[1].metadata, message_metadata);
    assert_eq!(
        serde_json::Value::Object(request.messages.0[1].content[0].data.clone()),
        query.payload[0].data
    );
    let context: Value = serde_json::from_str(
        request.messages.0[2].content[0].data["value"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(context["current_plan"], plan_json());
    assert_eq!(context["feedback"], serde_json::to_value(feedback).unwrap());
    assert!(context.get("completed_results").is_none());
}

#[test]
fn complete_values_are_accepted_as_the_full_plan() {
    let mut message = Message::new(Role::Assistant);
    message.values = plan_json().as_object().unwrap().clone();
    let result = block_on(planner(Model::new(vec![complete(message)])).plan(
        &input(),
        None,
        None,
        &Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(serde_json::to_value(result).unwrap(), plan_json());
}

#[test]
fn complete_text_parts_join_in_order_and_delta_think_are_not_results() {
    let value = plan_json().to_string();
    let mut message = text(Role::Assistant, &value[..10]);
    message.content.insert(
        0,
        Part {
            r#type: "think".into(),
            data: json!({"value":"not JSON"}).as_object().unwrap().clone(),
        },
    );
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":&value[10..]}).as_object().unwrap().clone(),
    });
    let model = Model::new(vec![
        Ok(Event::Delta {
            message: text(Role::Assistant, "wrong data"),
        }),
        complete(message),
    ]);
    let result = block_on(planner(model).plan(&input(), None, None, &Cancellation::new())).unwrap();
    assert_eq!(serde_json::to_value(result).unwrap(), plan_json());
}

#[test]
fn planner_accepts_task_names_but_rejects_dag_failures_without_changing_snapshot() {
    let current: Plan = serde_json::from_value(plan_json()).unwrap();
    let before = current.clone();
    let mut unknown = plan_json();
    unknown["nodes"]["b"]["name"] = json!("billing");
    let mut dangling = plan_json();
    dangling["edges"][0]["to"] = json!("missing");
    let mut self_edge = plan_json();
    self_edge["edges"][0]["to"] = json!("a");
    let mut repeated = plan_json();
    repeated["edges"]
        .as_array_mut()
        .unwrap()
        .push(json!({"from":"a","to":"b"}));
    let mut cycle = plan_json();
    cycle["edges"]
        .as_array_mut()
        .unwrap()
        .push(json!({"from":"b","to":"a"}));
    let mut zero_version = plan_json();
    zero_version["version"] = json!(0);
    let model = Model::new(vec![complete(text(Role::Assistant, &unknown.to_string()))]);
    let accepted =
        block_on(planner(model.clone()).plan(&input(), Some(&current), None, &Cancellation::new()))
            .unwrap();
    assert_eq!(accepted.nodes["b"].name, "billing");

    for value in [dangling, self_edge, repeated, cycle, zero_version] {
        let model = Model::new(vec![complete(text(Role::Assistant, &value.to_string()))]);
        let error = block_on(planner(model.clone()).plan(
            &input(),
            Some(&current),
            None,
            &Cancellation::new(),
        ))
        .unwrap_err();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
        assert_eq!(model.requests.lock().unwrap().len(), 1);
        assert_eq!(current, before);
    }
}

#[test]
fn planner_uses_current_plan_as_task_data_without_toolkit_validation() {
    let current: Plan = serde_json::from_value(plan_json()).unwrap();
    let model = Model::new(vec![complete(text(
        Role::Assistant,
        &plan_json().to_string(),
    ))]);
    let planner = core::plan::Planner::new(model.clone(), chat::Request::default()).unwrap();
    let result =
        block_on(planner.plan(&input(), Some(&current), None, &Cancellation::new())).unwrap();
    assert_eq!(result, current);
    assert_eq!(model.requests.lock().unwrap().len(), 1);
}

#[test]
fn planner_rejects_patches_unknown_fields_and_duplicate_node_ids() {
    let mut unknown = plan_json();
    unknown["nodes"]["a"]["kind"] = json!("tool");
    let mut runtime = plan_json();
    runtime["state"] = json!({});
    let mut edge_extra = plan_json();
    edge_extra["edges"][0]["exec_id"] = json!(1);
    for value in [json!({"nodes":{}}), unknown, runtime, edge_extra] {
        let model = Model::new(vec![complete(text(Role::Assistant, &value.to_string()))]);
        assert_eq!(
            block_on(planner(model).plan(&input(), None, None, &Cancellation::new()))
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
    }
    let duplicate = r#"{"version":1,"nodes":{"a":{"name":"search","objective":"first"},"a":{"name":"summarize","objective":"second"}},"edges":[]}"#;
    let model = Model::new(vec![complete(text(Role::Assistant, duplicate))]);
    assert_eq!(
        block_on(planner(model).plan(&input(), None, None, &Cancellation::new()))
            .unwrap_err()
            .code,
        "INVALID_RESPONSE"
    );
}

#[test]
fn rejected_structured_output_does_not_silently_fallback_or_execute_tools() {
    let mut values = text(Role::Assistant, &plan_json().to_string());
    values.values.insert("patch".into(), json!({}));
    let mut call = Message::new(Role::Assistant);
    call.content.push(Part {
        r#type: "tool_call".into(),
        data: json!({"name":"search"}).as_object().unwrap().clone(),
    });
    let mut bad_text = Message::new(Role::Assistant);
    bad_text.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":123}).as_object().unwrap().clone(),
    });
    for message in [
        values,
        call,
        bad_text,
        text(Role::Tool, &plan_json().to_string()),
        text(Role::Assistant, "```json\n{}\n```"),
    ] {
        let model = Model::new(vec![complete(message)]);
        assert_eq!(
            block_on(planner(model).plan(&input(), None, None, &Cancellation::new()))
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
    }
}

#[test]
fn delta_then_transport_failure_returns_original_error_without_retry() {
    let error = Error::new("TRANSPORT", "network interrupted").with_details("connection");
    let model = Model::new(vec![
        Ok(Event::Delta {
            message: text(Role::Assistant, &plan_json().to_string()),
        }),
        Err(error.clone()),
    ]);
    assert_eq!(
        block_on(planner(model.clone()).plan(&input(), None, None, &Cancellation::new()))
            .unwrap_err(),
        error
    );
    assert_eq!(model.requests.lock().unwrap().len(), 1);
}

#[test]
fn model_creation_errors_are_preserved_without_retry() {
    let error = Error::new("RATE_LIMITED", "wait").with_details("provider");
    let mut model = Model::new(vec![]);
    model.create_error = Some(error.clone());
    let evaluator =
        core::evaluation::Evaluator::new(model.clone(), chat::Request::default()).unwrap();
    assert_eq!(
        block_on(evaluator.evaluate(&input(), &Messages::default(), &Cancellation::new()))
            .unwrap_err(),
        error
    );
    assert_eq!(model.requests.lock().unwrap().len(), 1);
}

#[test]
fn model_stream_without_complete_returns_no_partial_plan() {
    struct End;
    impl Chat for End {
        type Stream = futures_util::stream::Empty<Result<Event, Error>>;
        async fn stream(&self, _: chat::Request) -> Result<Self::Stream, Error> {
            Ok(futures_util::stream::empty())
        }
    }
    let planner = core::plan::Planner::new(End, chat::Request::default()).unwrap();
    assert_eq!(
        block_on(planner.plan(&input(), None, None, &Cancellation::new()))
            .unwrap_err()
            .code,
        "INVALID_RESPONSE"
    );
}

#[test]
fn model_evaluator_preserves_business_rejection_feedback_and_final_result() {
    let expected = evaluation(false);
    let model = Model::new(vec![complete(text(
        Role::Assistant,
        &serde_json::to_string(&expected).unwrap(),
    ))]);
    let evaluator =
        core::evaluation::Evaluator::new(model.clone(), chat::Request::default()).unwrap();
    let result = Messages::new([text(Role::Assistant, "北京天气晴")]);
    let before = result.clone();
    assert_eq!(
        block_on(evaluator.evaluate(&input(), &result, &Cancellation::new())).unwrap(),
        expected
    );
    assert_eq!(result, before);
    let request = &model.requests.lock().unwrap()[0];
    let context: Value = serde_json::from_str(
        request.messages.0.last().unwrap().content[0].data["value"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(context, json!({"result":result}));
}

#[test]
fn evaluator_accepts_complete_values_without_a_new_result_wrapper() {
    let mut message = Message::new(Role::Assistant);
    message.values = serde_json::from_value(json!({"passed":true,"feedback":null})).unwrap();
    let evaluator = core::evaluation::Evaluator::new(
        Model::new(vec![complete(message)]),
        chat::Request::default(),
    )
    .unwrap();
    assert_eq!(
        block_on(evaluator.evaluate(&input(), &Messages::default(), &Cancellation::new())).unwrap(),
        evaluation(true)
    );
}

#[test]
fn evaluator_rejects_node_targeting_and_wrong_verdict_types() {
    for value in [
        json!({"passed":true,"affected":["a"]}),
        json!({"passed":"true"}),
        json!({"feedback":null}),
        json!({"passed":false,"feedback":"retry"}),
    ] {
        let evaluator = core::evaluation::Evaluator::new(
            Model::new(vec![complete(text(Role::Assistant, &value.to_string()))]),
            chat::Request::default(),
        )
        .unwrap();
        assert_eq!(
            block_on(evaluator.evaluate(&input(), &Messages::default(), &Cancellation::new()))
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
    }
}

#[test]
fn user_function_evaluator_supports_native_non_send_futures_and_feedback() {
    let evaluate = |query: Input, result: Messages, _: Cancellation| async move {
        let marker = Rc::new(1);
        std::future::ready(()).await;
        assert_eq!(*marker, 1);
        assert_eq!(query, input());
        assert_eq!(result.0[0].content[0].data["value"], "answer");
        Ok::<_, Error>(evaluation(false))
    };
    let result = Messages::new([text(Role::Assistant, "answer")]);
    assert_eq!(
        block_on(evaluate.evaluate(&input(), &result, &Cancellation::new())).unwrap(),
        evaluation(false)
    );
}

#[test]
fn user_function_evaluator_returns_original_technical_error() {
    let expected =
        Error::new("UNAVAILABLE", "evaluation backend unavailable").with_details("backend");
    let error = expected.clone();
    let evaluate = move |_: Input, _: Messages, _: Cancellation| {
        let error = error.clone();
        async move { Err::<Evaluation, _>(error) }
    };
    assert_eq!(
        block_on(evaluate.evaluate(&input(), &Messages::default(), &Cancellation::new()))
            .unwrap_err(),
        expected
    );
}

#[derive(Default)]
struct WakeCount(AtomicUsize);
impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn cancel_pending<T: std::fmt::Debug>(
    future: impl Future<Output = Result<T, Error>>,
    token: &Cancellation,
) {
    let mut future = std::pin::pin!(future);
    let counter = Arc::new(WakeCount::default());
    let waker = Waker::from(counter.clone());
    let mut cx = Context::from_waker(&waker);
    assert!(future.as_mut().poll(&mut cx).is_pending());
    token.cancel();
    assert!(counter.0.load(Ordering::SeqCst) > 0);
    let Poll::Ready(result) = future.as_mut().poll(&mut cx) else {
        panic!("cancellation did not wake pending operation")
    };
    assert_eq!(result.unwrap_err().code, "CANCELLED");
}

#[test]
fn cancellation_wakes_pending_model_creation_and_drops_it() {
    let mut model = Model::new(vec![]);
    model.create_pending = true;
    let planner = planner(model.clone());
    let token = Cancellation::new();
    cancel_pending(planner.plan(&input(), None, None, &token), &token);
    assert!(model.create_dropped.load(Ordering::SeqCst));
    assert!(!model.dropped.load(Ordering::SeqCst));
}

#[test]
fn cancellation_wakes_pending_complete_read_and_drops_stream() {
    let mut model = Model::new(vec![]);
    model.stream_pending = true;
    let evaluator =
        core::evaluation::Evaluator::new(model.clone(), chat::Request::default()).unwrap();
    let token = Cancellation::new();
    cancel_pending(
        evaluator.evaluate(&input(), &Messages::default(), &token),
        &token,
    );
    assert!(model.dropped.load(Ordering::SeqCst));
}

#[test]
fn cancellation_stops_pending_function_evaluation_and_drops_future() {
    let dropped = Arc::new(AtomicBool::new(false));
    let flag = dropped.clone();
    let evaluate = move |_: Input, _: Messages, _: Cancellation| {
        let flag = flag.clone();
        async move {
            let _guard = Guard(flag);
            std::future::pending::<Result<Evaluation, Error>>().await
        }
    };
    let token = Cancellation::new();
    cancel_pending(
        evaluate.evaluate(&input(), &Messages::default(), &token),
        &token,
    );
    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn pre_cancelled_calls_do_not_start_model_or_function() {
    let token = Cancellation::new();
    token.cancel();
    let model = Model::new(vec![]);
    let planner = planner(model.clone());
    assert_eq!(
        block_on(planner.plan(&input(), None, None, &token))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert!(model.requests.lock().unwrap().is_empty());
    let evaluate = |_: Input, _: Messages, _: Cancellation| {
        panic!("cancelled evaluation must not start");
        #[allow(unreachable_code)]
        std::future::ready(Ok::<_, Error>(evaluation(true)))
    };
    assert_eq!(
        block_on(evaluate.evaluate(&input(), &Messages::default(), &token))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
}

#[test]
fn synchronous_cancellation_during_model_creation_cannot_return_success() {
    let token = Cancellation::new();
    let mut model = Model::new(vec![complete(text(
        Role::Assistant,
        &plan_json().to_string(),
    ))]);
    model.cancellation = Some(token.clone());
    assert_eq!(
        block_on(planner(model).plan(&input(), None, None, &token))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
}

#[test]
fn cancelled_complete_is_not_a_successful_plan() {
    let event = Ok(Event::Complete {
        message: text(Role::Assistant, &plan_json().to_string()),
        usage: None,
        finish_reason: Some("cancelled".into()),
    });
    assert_eq!(
        block_on(planner(Model::new(vec![event])).plan(&input(), None, None, &Cancellation::new()))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
}
