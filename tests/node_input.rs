//! Public Workflow coverage for `attrs.input` field binding.
//!
//! The field adapter is intentionally private. These tests exercise it through
//! the same compiled Workflow path used by applications instead of exposing a
//! second production API only for tests.

#[path = "support/workflow.rs"]
mod workflow_support;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use futures_executor::block_on;
use halo::agent::workflow::{Builder, FieldSource, InputBinding, NodeAttrs, Runtime, graph};
use halo::agent::{Cancellation, Error, Message, Messages};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, JsonSchema)]
struct Cities {
    first: String,
    second: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Latest {
    x: Option<i64>,
    items: Vec<i64>,
    nested: Nested,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Nested {
    x: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RawInput {
    value: Value,
}

fn binding(target: &str, node_id: &str, field: &str) -> (String, InputBinding) {
    (
        target.into(),
        InputBinding {
            from: FieldSource {
                node_id: node_id.into(),
                field: field.into(),
            },
        },
    )
}

fn run(builder: Builder, chat_id: i64) -> Result<Messages, Error> {
    let workflow = builder
        .compile()
        .map_err(|error| Error::new(error.code, error.message))?;
    let mut runtime = Runtime::new(chat_id, workflow)?;
    let next = Arc::new(AtomicI64::new(chat_id * 100));
    let next_id = {
        let next = next.clone();
        Arc::new(move || next.fetch_add(1, Ordering::SeqCst))
    };
    block_on(workflow_support::run_workflow(
        &mut runtime,
        Messages::default(),
        next_id,
        Cancellation::new(),
    ))
}

fn add_message_source(
    builder: &mut Builder,
    node_id: &str,
    messages: Messages,
) -> Result<(), Error> {
    let value = messages.clone();
    builder
        .add_node(node_id, move |_: Messages, _: Cancellation| {
            let value = value.clone();
            async move { Ok::<_, Error>(value) }
        })
        .map(|_| ())
        .map_err(|error| Error::new(error.code, error.message))?;
    builder
        .add_edge(graph::START, node_id)
        .map(|_| ())
        .map_err(|error| Error::new(error.code, error.message))
}

#[test]
fn sources_bind_to_typed_input_without_merging_or_mutation() {
    let a = Messages::new([Message::function(json!({"city": "北京"}))]);
    let b = Messages::new([Message::function(json!({"city": "上海"}))]);
    let mut builder = Builder::new("input-bindings");
    add_message_source(&mut builder, "a", a.clone()).unwrap();
    add_message_source(&mut builder, "b", b.clone()).unwrap();
    builder
        .add_node("target", |input: Cities, _: Cancellation| async move {
            Ok::<_, Error>(Messages::new([Message::function(json!({
                "first": input.first,
                "second": input.second,
            }))]))
        })
        .unwrap();
    builder
        .set_attrs(
            "target",
            NodeAttrs {
                input: BTreeMap::from([
                    binding("first", "a", "city"),
                    binding("second", "b", "city"),
                ]),
            },
        )
        .unwrap();
    builder.add_edge("a", "target").unwrap();
    builder.add_edge("b", "target").unwrap();
    builder.add_edge("target", graph::END).unwrap();

    let result = run(builder, 10).unwrap();
    assert_eq!(result.as_slice()[0].values["first"], "北京");
    assert_eq!(result.as_slice()[0].values["second"], "上海");
    assert_eq!(a.as_slice()[0].values["city"], "北京");
    assert_eq!(b.as_slice()[0].values["city"], "上海");
}

#[test]
fn missing_source_is_rejected_without_inference_from_another_node() {
    let mut builder = Builder::new("missing-source");
    add_message_source(
        &mut builder,
        "a",
        Messages::new([Message::function(json!({"city": "北京"}))]),
    )
    .unwrap();
    builder
        .add_node("target", |_: Cities, _: Cancellation| async {
            Ok::<_, Error>(Messages::default())
        })
        .unwrap();
    builder
        .set_attrs(
            "target",
            NodeAttrs {
                input: BTreeMap::from([binding("first", "missing", "city")]),
            },
        )
        .unwrap();
    builder.add_edge(graph::START, "target").unwrap();
    builder.add_edge("target", graph::END).unwrap();

    let error = builder
        .compile()
        .expect_err("unknown source must fail compilation");
    assert_eq!(error.code, "INVALID_BINDING");
}

#[test]
fn missing_field_is_not_inferred_from_metadata() {
    let mut message = Message::function(json!({"a.b": 7, "a": {"b": 8}}));
    message.metadata.insert("secret".into(), json!(9));

    let mut literal_builder = Builder::new("literal-field");
    add_message_source(
        &mut literal_builder,
        "source",
        Messages::new([message.clone()]),
    )
    .unwrap();
    literal_builder
        .add_node("target", |input: RawInput, _: Cancellation| async move {
            Ok::<_, Error>(Messages::new([Message::function(input.value)]))
        })
        .unwrap();
    literal_builder
        .set_attrs(
            "target",
            NodeAttrs {
                input: BTreeMap::from([binding("value", "source", "a.b")]),
            },
        )
        .unwrap();
    literal_builder.add_edge("source", "target").unwrap();
    literal_builder.add_edge("target", graph::END).unwrap();
    let literal = run(literal_builder, 110).unwrap();
    assert_eq!(literal.as_slice()[0].values["value"], 7);

    let mut builder = Builder::new("literal-field");
    add_message_source(&mut builder, "source", Messages::new([message])).unwrap();
    builder
        .add_node("target", |input: RawInput, _: Cancellation| async move {
            Ok::<_, Error>(Messages::new([Message::function(input.value)]))
        })
        .unwrap();
    builder
        .set_attrs(
            "target",
            NodeAttrs {
                input: BTreeMap::from([binding("value", "source", "secret")]),
            },
        )
        .unwrap();
    builder.add_edge("source", "target").unwrap();
    builder.add_edge("target", graph::END).unwrap();

    let error = run(builder, 11).expect_err("metadata must not satisfy a field binding");
    assert_eq!(error.code, "INVALID_ARGUMENTS");
}

#[test]
fn latest_values_preserve_null_arrays_and_objects() {
    let messages = Messages::new([
        Message::function(json!({"x": 1})),
        Message::function(json!({"x": null, "items": [1, 2], "nested": {"x": 3}})),
    ]);
    let mut builder = Builder::new("latest-values");
    add_message_source(&mut builder, "source", messages).unwrap();
    builder
        .add_node("target", |input: Latest, _: Cancellation| async move {
            Ok::<_, Error>(Messages::new([Message::function(json!({
                "x": input.x,
                "items": input.items,
                "nested": {"x": input.nested.x},
            }))]))
        })
        .unwrap();
    builder
        .set_attrs(
            "target",
            NodeAttrs {
                input: BTreeMap::from([
                    binding("x", "source", "x"),
                    binding("items", "source", "items"),
                    binding("nested", "source", "nested"),
                ]),
            },
        )
        .unwrap();
    builder.add_edge("source", "target").unwrap();
    builder.add_edge("target", graph::END).unwrap();

    let result = run(builder, 12).unwrap();
    assert!(result.as_slice()[0].values["x"].is_null());
    assert_eq!(result.as_slice()[0].values["items"], json!([1, 2]));
    assert_eq!(result.as_slice()[0].values["nested"], json!({"x": 3}));
}

#[test]
fn empty_binding_field_is_rejected_at_runtime() {
    let mut builder = Builder::new("empty-binding");
    add_message_source(
        &mut builder,
        "source",
        Messages::new([Message::function(json!({"x": 1}))]),
    )
    .unwrap();
    builder
        .add_node("target", |input: RawInput, _: Cancellation| async move {
            Ok::<_, Error>(Messages::new([Message::function(input.value)]))
        })
        .unwrap();
    builder
        .set_attrs(
            "target",
            NodeAttrs {
                input: BTreeMap::from([binding("value", "source", "")]),
            },
        )
        .unwrap();
    builder.add_edge("source", "target").unwrap();
    builder.add_edge("target", graph::END).unwrap();

    let error = run(builder, 13).expect_err("empty source field must fail");
    assert_eq!(error.code, "INVALID_ARGUMENTS");
}
