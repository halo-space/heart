use futures_executor::block_on;
use futures_util::stream;
use halo_agents::{
    Cancellation, Error, Message, Messages, core,
    event::Event,
    memory::{self, extraction::Extractor as _},
    message::Role,
    model::{
        chat::{self, Chat},
        content::part::Part,
    },
};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    rc::Rc,
    sync::{Arc, Mutex},
};

#[derive(Clone, Default)]
struct Model {
    requests: Arc<Mutex<Vec<chat::Request>>>,
    replies: Replies,
}
type Replies = Arc<Mutex<VecDeque<Vec<Result<Event, Error>>>>>;
impl Model {
    fn output(values: Value) -> Self {
        let model = Self::default();
        model.push(values);
        model
    }
    fn push(&self, values: Value) {
        let mut message = Message::new(Role::Assistant);
        message.values = values.as_object().unwrap().clone();
        if message.values.is_empty() {
            message.content.push(Part {
                r#type: "text".into(),
                data: json!({"value":"{}"}).as_object().unwrap().clone(),
            });
        }
        self.replies
            .lock()
            .unwrap()
            .push_back(vec![Ok(Event::Complete {
                message,
                usage: None,
                finish_reason: None,
            })]);
    }
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: chat::Request) -> Result<Self::Stream, Error> {
        let local = Rc::new(1);
        futures_util::future::ready(()).await;
        assert_eq!(*local, 1);
        self.requests.lock().unwrap().push(request);
        Ok(stream::iter(
            self.replies.lock().unwrap().pop_front().unwrap(),
        ))
    }
}
fn input() -> Messages {
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"我喜欢简洁的回答"})
            .as_object()
            .unwrap()
            .clone(),
    });
    Messages::new([message])
}
fn item() -> memory::Item {
    memory::Item {
        id: 1,
        r#type: memory::Type::Preference,
        user_id: 2,
        agent_id: None,
        session_id: None,
        from: 10,
        to: 20,
        content: "中文回答".into(),
        version: 1,
        created_time: 0,
        updated_time: 0,
        metadata: None,
    }
}
#[test]
fn construction_has_no_io_and_preserves_explicit_options() {
    let model = Model::output(json!({"user":{"preference":"简洁回答"}}));
    let request = chat::Request {
        temperature: Some(0.2),
        options: json!({"thinking":{"type":"enabled"}})
            .as_object()
            .unwrap()
            .clone(),
        ..Default::default()
    };
    let extractor = core::memory::extraction::Extractor::new(model.clone(), request).unwrap();
    assert!(model.requests.lock().unwrap().is_empty());
    let result = block_on(extractor.extract(&input(), &Cancellation::new())).unwrap();
    assert_eq!(
        result.0[0].values,
        json!({"user":{"preference":"简洁回答"}})
            .as_object()
            .unwrap()
            .clone()
    );
    let requests = model.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].temperature, Some(0.2));
    assert_eq!(
        requests[0].options.get("thinking"),
        Some(&json!({"type":"enabled"}))
    );
    assert_eq!(
        requests[0].response_format,
        Some(chat::ResponseFormat::JsonObject)
    );
}
#[test]
fn strict_ownership_types_and_plain_text() {
    for invalid in [
        json!({"session":{"summary":"no"}}),
        json!({"user":{"fact":{"data":"no"}}}),
        json!({"user":{"id":1}}),
        json!({"user":{"fact":" "}}),
    ] {
        let model = Model::output(invalid);
        let extractor =
            core::memory::extraction::Extractor::new(model, chat::Request::default()).unwrap();
        assert_eq!(
            block_on(extractor.extract(&input(), &Cancellation::new()))
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
    }
}
#[test]
fn no_new_information_is_an_empty_result_not_an_error() {
    let extractor = core::memory::extraction::Extractor::new(
        Model::output(json!({})),
        chat::Request::default(),
    )
    .unwrap();
    let output = block_on(extractor.extract(&input(), &Cancellation::new())).unwrap();
    assert!(output.0[0].values.is_empty());
}
#[test]
fn merge_uses_existing_content_but_no_database_or_ids() {
    let model = Model::output(json!({"content":"中文、简洁回答"}));
    let extractor =
        core::memory::extraction::Extractor::new(model.clone(), chat::Request::default()).unwrap();
    assert_eq!(
        block_on(extractor.merge(&item(), "简洁回答", &Cancellation::new())).unwrap(),
        "中文、简洁回答"
    );
    let requests = model.requests.lock().unwrap();
    let data: Value = serde_json::from_str(
        requests[0].messages.0.last().unwrap().content[0].data["value"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        data,
        json!({"type":"preference","current":"中文回答","new":"简洁回答"})
    );
}
#[test]
fn model_errors_are_direct_and_not_retried() {
    let model = Model::default();
    let error = Error::new("TRANSPORT", "offline").with_details("safe detail");
    model
        .replies
        .lock()
        .unwrap()
        .push_back(vec![Err(error.clone())]);
    let extractor =
        core::memory::extraction::Extractor::new(model.clone(), chat::Request::default()).unwrap();
    assert_eq!(
        block_on(extractor.extract(&input(), &Cancellation::new())).unwrap_err(),
        error
    );
    assert_eq!(model.requests.lock().unwrap().len(), 1);
}
#[test]
fn delta_never_substitutes_for_complete() {
    let model = Model::default();
    let mut message = Message::new(Role::Assistant);
    message.values = json!({"user":{"fact":"delta"}})
        .as_object()
        .unwrap()
        .clone();
    model
        .replies
        .lock()
        .unwrap()
        .push_back(vec![Ok(Event::Delta { message })]);
    let extractor =
        core::memory::extraction::Extractor::new(model, chat::Request::default()).unwrap();
    assert_eq!(
        block_on(extractor.extract(&input(), &Cancellation::new()))
            .unwrap_err()
            .code,
        "INVALID_RESPONSE"
    );
}
#[test]
fn cancelled_before_model_creation_does_no_io() {
    let model = Model::default();
    let extractor =
        core::memory::extraction::Extractor::new(model.clone(), chat::Request::default()).unwrap();
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        block_on(extractor.extract(&input(), &cancellation))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert!(model.requests.lock().unwrap().is_empty());
}
#[test]
fn extraction_without_trigger_is_rejected_without_allocating_ids() {
    let extractor =
        core::memory::extraction::Extractor::new(Model::default(), chat::Request::default())
            .unwrap();
    let config = memory::Config::new(
        core::memory::short::session::InMemory::new(),
        core::memory::short::chat::InMemory::new(),
        core::memory::short::trace::InMemory::new(),
        core::memory::InMemory::new(),
    )
    .with_extractor(extractor)
    .with_id_fn(|| panic!("build must not invoke the ID function"));
    assert!(
        halo_agents::agent::chat()
            .model(Model::default())
            .memory(config)
            .build()
            .is_err()
    );
}

fn memory_config() -> memory::Config<
    core::memory::short::session::InMemory,
    core::memory::short::chat::InMemory,
    core::memory::short::trace::InMemory,
    core::memory::InMemory,
> {
    memory::Config::new(
        core::memory::short::session::InMemory::new(),
        core::memory::short::chat::InMemory::new(),
        core::memory::short::trace::InMemory::new(),
        core::memory::InMemory::new(),
    )
}
fn unexpected_id() -> i64 {
    panic!("configuration must not invoke the ID function")
}

#[test]
fn missing_id_function_is_a_configuration_error() {
    let extractor =
        core::memory::extraction::Extractor::new(Model::default(), chat::Request::default())
            .unwrap();
    let compressor = core::compression::Compressor::new(Model::default(), chat::Request::default())
        .unwrap()
        .with_size(100);
    let config = memory_config()
        .with_compression(compressor)
        .with_extractor(extractor);
    assert_eq!(
        halo_agents::agent::chat()
            .model(Model::default())
            .memory(config)
            .build()
            .err()
            .unwrap()
            .code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn build_and_clone_accept_a_function_pointer_without_calling_it() {
    let extractor =
        core::memory::extraction::Extractor::new(Model::default(), chat::Request::default())
            .unwrap();
    let compressor = core::compression::Compressor::new(Model::default(), chat::Request::default())
        .unwrap()
        .with_size(100);
    let config = memory_config()
        .with_id_fn(unexpected_id as fn() -> i64)
        .with_extractor(extractor)
        .with_compression(compressor);
    assert!(
        halo_agents::agent::chat()
            .model(Model::default())
            .memory(config.clone())
            .build()
            .is_ok()
    );
    assert!(
        halo_agents::agent::react()
            .model(Model::default())
            .memory(config)
            .build()
            .is_ok()
    );
}

#[test]
fn id_configuration_alone_does_not_enable_memory_transforms() {
    assert!(
        halo_agents::agent::chat()
            .model(Model::default())
            .memory(memory_config().with_id_fn(unexpected_id))
            .build()
            .is_ok()
    );
}

#[test]
fn extraction_preserves_complete_usage_without_aggregating_it() {
    let model = Model::output(json!({"user":{"fact":"new fact"}}));
    let usage = halo_agents::model::token::Usage {
        input: Some(7),
        output: Some(3),
        total: Some(10),
        ..Default::default()
    };
    if let Ok(Event::Complete { usage: value, .. }) = &mut model.replies.lock().unwrap()[0][0] {
        *value = Some(usage.clone());
    }
    let extractor =
        core::memory::extraction::Extractor::new(model, chat::Request::default()).unwrap();
    let output = block_on(extractor.extract(&input(), &Cancellation::new())).unwrap();
    assert_eq!(output.0[0].usage, Some(usage));
}
