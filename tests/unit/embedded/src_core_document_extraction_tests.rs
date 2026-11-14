use futures_core::Stream;
use futures_executor::block_on;
use serde_json::json;
use std::future::Future;
use std::{
    collections::VecDeque,
    pin::Pin,
    task::{Context, Poll},
};

use super::{
    Document, LocalLoader,
    graph::{EntityType, Evidence, Method, decode_output},
    raptor::Raptor,
};
use crate::Cancellation;
use crate::components::{
    document::{
        Loader, Source,
        location::{Location, Range},
    },
    event::Event,
    message::{Message, Role},
    model::{self, chat::Chat, embed::Embed},
};

fn uncalled_provider_chat() -> impl Chat {
    crate::core::provider::vllm::Vllm::builder()
        .api_key("compile-only")
        .build()
        .expect("construct Provider without sending a request")
        .chat(crate::model::chat::Config::new("compile-only"))
        .expect("construct Provider Chat")
}

struct FakeStream(VecDeque<Result<Event, model::Error>>);
impl Stream for FakeStream {
    type Item = Result<Event, model::Error>;
    fn poll_next(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.0.pop_front())
    }
}

struct CompleteThenPending(Option<Result<Event, model::Error>>);
impl Stream for CompleteThenPending {
    type Item = Result<Event, model::Error>;
    fn poll_next(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.0
            .take()
            .map_or(Poll::Pending, |event| Poll::Ready(Some(event)))
    }
}

#[test]
fn raptor_rejects_malformed_options_instead_of_using_defaults() {
    for options in [
        json!({"threshold": "wrong"}),
        json!({"threshold": null}),
        json!({"threshold": -0.1}),
        json!({"threshold": 1.1}),
        json!({"cluster_size": -1}),
        json!({"cluster_size": 0}),
        json!({"levels": -1}),
        json!({"levels": 1.5}),
        json!({"levels": "2"}),
    ] {
        let result = Raptor::new(
            serde_json::from_value(options).unwrap(),
            uncalled_provider_chat(),
            PendingEmbed,
        );
        assert_eq!(result.err().unwrap().code, "INVALID_ARGUMENTS");
    }
    assert!(Raptor::new(Default::default(), uncalled_provider_chat(), PendingEmbed).is_ok());
}

struct PendingEmbed;
impl Embed for PendingEmbed {
    async fn embed(&self, _request: model::embed::Request) -> Result<Message, model::Error> {
        std::future::pending().await
    }
}

#[test]
fn cancellation_unblocks_raptor_while_embed_is_pending() {
    let cancellation = Cancellation::new();
    let raptor = Raptor::new(Default::default(), uncalled_provider_chat(), PendingEmbed).unwrap();
    let documents = [Document {
        id: Some(1),
        content: "text".into(),
        ..Document::default()
    }];
    let mut generated = 0;
    let mut next_id = || {
        generated += 1;
        generated
    };
    let mut future = Box::pin(raptor.run(&documents, &cancellation, &mut next_id));
    let mut context = Context::from_waker(std::task::Waker::noop());
    assert!(future.as_mut().poll(&mut context).is_pending());
    cancellation.cancel();
    let Poll::Ready(result) = future.as_mut().poll(&mut context) else {
        panic!("Embed wait must stop on cancellation");
    };
    assert_eq!(result.unwrap_err().code, "CANCELLED");
    drop(future);
    assert_eq!(generated, 0);
    assert_eq!(
        block_on(raptor.run(&[], &cancellation, &mut || 100))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
}

#[test]
fn graph_decodes_complete_text_and_ignores_think_content() {
    let text =
        json!({"entities": [{"id": "p1", "type": "person", "name": "张三"}], "relations": []})
            .to_string();
    let mut response = Message::new(Role::Assistant);
    response.content.push(model::content::part::Part {
        r#type: "think".into(),
        data: crate::components::model::content::part::object(json!({"value": "not JSON"})),
    });
    // A complete text response can contain more than one text part.
    response.content.push(model::content::part::Part {
        r#type: "text".into(),
        data: crate::components::model::content::part::object(json!({"value": &text[..1]})),
    });
    response.content.push(model::content::part::Part {
        r#type: "text".into(),
        data: crate::components::model::content::part::object(json!({"value": &text[1..]})),
    });
    let entity_types = [EntityType {
        name: "person".into(),
        description: "人物".into(),
    }];
    let output = decode_output(&response, &entity_types, &[]).unwrap();
    assert_eq!(output.entities[0].name, "张三");
    for text in [
        "not JSON",
        "[]",
        "{\"entities\": []}",
        "{\"entities\": [], \"relations\": 1}",
    ] {
        let mut response = Message::new(Role::Assistant);
        response.content.push(model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(json!({"value": text})),
        });
        assert_eq!(
            decode_output(&response, &entity_types, &[])
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
    }
}

#[test]
fn graph_preserves_declared_type_names_without_trimming() {
    let mut response = Message::new(Role::Assistant);
    response.values.insert(
        "entities".into(),
        json!([{"id": "p1", "type": " person ", "name": "张三"}]),
    );
    response.values.insert("relations".into(), json!([]));
    let entity_types = [EntityType {
        name: " person ".into(),
        description: "人物".into(),
    }];
    assert_eq!(
        decode_output(&response, &entity_types, &[])
            .unwrap()
            .entities[0]
            .r#type,
        " person "
    );
}

#[test]
fn location_validates_grouped_ranges_and_coordinates() {
    let location = Location {
        page: Some(Range { from: 2, to: 3 }),
        paragraph: Some(Range { from: 8, to: 2 }),
        r#char: Some(Range { from: 12, to: 18 }),
        x: Some(Range { from: 0.1, to: 0.8 }),
        y: Some(Range { from: 0.2, to: 0.9 }),
    };
    assert!(location.validate().is_ok());
}

#[test]
fn evidence_requires_content_for_method_and_valid_location() {
    let evidence = Evidence {
        content: Some("张三在甲公司工作".into()),
        method: Some(Method::Text),
        location: Some(Location {
            page: Some(Range { from: 3, to: 3 }),
            paragraph: Some(Range { from: 2, to: 2 }),
            ..Location::default()
        }),
        ..Evidence::default()
    };
    assert!(evidence.validate().is_ok());
    let invalid = Evidence {
        method: Some(Method::Ocr),
        ..Evidence::default()
    };
    assert_eq!(invalid.validate().unwrap_err().code, "INVALID_ARGUMENTS");
}

#[test]
fn evidence_reports_the_invalid_coordinate_endpoint() {
    let evidence = Evidence {
        content: Some("依据".into()),
        location: Some(Location {
            x: Some(Range { from: 0.1, to: 1.2 }),
            y: Some(Range { from: 0.2, to: 0.8 }),
            ..Location::default()
        }),
        ..Evidence::default()
    };
    assert_eq!(
        evidence.validate().unwrap_err().details.as_deref(),
        Some("location.x.to")
    );
}

#[test]
fn document_keeps_optional_ids_and_metadata() {
    let document = Document {
        content: "hello".into(),
        metadata: serde_json::Map::from_iter([("layout".into(), json!({"role": "paragraph"}))]),
        ..Document::default()
    };
    let value = serde_json::to_value(document).unwrap();
    assert!(value.get("id").is_none());
    assert_eq!(value["metadata"]["layout"]["role"], "paragraph");
}

#[tokio::test]
async fn local_loader_reads_one_complete_file_and_honors_cancellation() {
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let result = LocalLoader
        .load(
            &Source {
                uri: "/does/not/exist".into(),
            },
            &cancellation,
        )
        .await;
    assert_eq!(result.unwrap_err().code, "CANCELLED");
}

#[test]
fn document_model_call_stops_on_complete_without_waiting_for_stream_end() {
    let response = Message::new(Role::Assistant);
    let events = CompleteThenPending(Some(Ok(Event::Complete {
        message: response.clone(),
        usage: None,
        finish_reason: None,
    })));
    let result = block_on(super::complete_stream(events, &Cancellation::new())).unwrap();
    assert_eq!(result, response);
}

#[test]
fn document_complete_preserves_event_usage_without_estimating_or_aggregating() {
    let message = block_on(super::complete_stream(
        FakeStream(VecDeque::from([Ok(Event::Complete {
            message: Message::new(Role::Assistant),
            usage: Some(model::token::Usage {
                input: Some(7),
                output: Some(3),
                total: Some(10),
                ..Default::default()
            }),
            finish_reason: None,
        })])),
        &Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(message.usage.unwrap().total, Some(10));
    // No second usage or Runtime state is introduced by this conversion.
}

#[test]
fn cancellation_unblocks_a_pending_document_model_stream() {
    let cancellation = Cancellation::new();
    let mut future = Box::pin(super::complete_stream(
        CompleteThenPending(None),
        &cancellation,
    ));
    let waker = std::task::Waker::noop();
    let mut context = Context::from_waker(waker);
    assert!(future.as_mut().poll(&mut context).is_pending());
    cancellation.cancel();
    let Poll::Ready(result) = future.as_mut().poll(&mut context) else {
        panic!("cancelled document stream should be ready");
    };
    assert_eq!(result.unwrap_err().code, "CANCELLED");
}

#[test]
fn graph_validates_config_and_returns_complete_entities_and_relations() {
    let mut response = Message::new(Role::Assistant);
    response.values.insert(
        "entities".into(),
        json!([
            {"id": "person-1", "type": "person", "name": "张三"},
            {"id": "company-1", "type": "company", "name": "甲公司"}
        ]),
    );
    response.values.insert(
        "relations".into(),
        json!([
            {"from": "person-1", "to": "company-1", "kind": "works_for"}
        ]),
    );
    let entity_types = [
        EntityType {
            name: "person".into(),
            description: "人物".into(),
        },
        EntityType {
            name: "company".into(),
            description: "组织".into(),
        },
    ];
    let relation_types = [super::graph::RelationType {
        name: "works_for".into(),
        description: "任职".into(),
        from: vec!["person".into()],
        to: vec!["company".into()],
    }];
    let output = decode_output(&response, &entity_types, &relation_types).unwrap();
    assert_eq!(output.entities.len(), 2);
    assert_eq!(output.relations[0].kind, "works_for");
}
