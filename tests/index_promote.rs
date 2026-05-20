//! Integration tests: metadata promotion.

use rag::index::pagewiki;
use rag::index::pagewiki::Base;
use rag::index::source::{Item, Scenario};
use rag::index::{Builder, NoopTokenizer};
use serde_json::json;
use std::collections::HashMap;

fn make_item_with_metadata(
    doc_id: &str,
    text: &str,
    metadata: serde_json::Map<String, serde_json::Value>,
) -> Item {
    let mut meta = metadata;
    meta.insert("doc_id".into(), json!(doc_id));
    Item {
        text: text.to_string(),
        scenario: Scenario::General,
        metadata: meta,
    }
}

#[tokio::test]
async fn promote_keywords_from_metadata() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(pagewiki::Fixed::new(200)));
    let builder = Builder::new(
        pw_map,
        vec!["keywords".into()],
        Box::new(NoopTokenizer),
        None,
    )
    .unwrap();

    let mut meta = serde_json::Map::new();
    meta.insert("keywords".into(), json!(["rust", "rag"]));
    let items = vec![make_item_with_metadata("doc_001", "text", meta)];
    let pages = builder.build(items).await.unwrap();

    assert_eq!(pages[0].keywords, vec!["rust", "rag"]);
    assert!(!pages[0].metadata.contains_key("keywords"));
}

#[tokio::test]
async fn promote_questions_from_metadata() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(pagewiki::Fixed::new(200)));
    let builder = Builder::new(
        pw_map,
        vec!["questions".into()],
        Box::new(NoopTokenizer),
        None,
    )
    .unwrap();

    let mut meta = serde_json::Map::new();
    meta.insert("questions".into(), json!(["What is RAG?"]));
    let items = vec![make_item_with_metadata("doc_002", "text", meta)];
    let pages = builder.build(items).await.unwrap();

    assert_eq!(pages[0].questions, vec!["What is RAG?"]);
    assert!(!pages[0].metadata.contains_key("questions"));
}

#[tokio::test]
async fn promote_tags_from_metadata() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(pagewiki::Fixed::new(200)));
    let builder = Builder::new(pw_map, vec!["tags".into()], Box::new(NoopTokenizer), None).unwrap();

    let mut meta = serde_json::Map::new();
    meta.insert("tags".into(), json!(["ai", "ml"]));
    let items = vec![make_item_with_metadata("doc_003", "text", meta)];
    let pages = builder.build(items).await.unwrap();

    assert_eq!(pages[0].tags, vec!["ai", "ml"]);
    assert!(!pages[0].metadata.contains_key("tags"));
}

#[tokio::test]
async fn promote_multiple_fields() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(pagewiki::Fixed::new(200)));
    let builder = Builder::new(
        pw_map,
        vec!["keywords".into(), "tags".into()],
        Box::new(NoopTokenizer),
        None,
    )
    .unwrap();

    let mut meta = serde_json::Map::new();
    meta.insert("keywords".into(), json!(["k1"]));
    meta.insert("tags".into(), json!(["t1"]));
    let items = vec![make_item_with_metadata("doc_004", "text", meta)];
    let pages = builder.build(items).await.unwrap();

    assert_eq!(pages[0].keywords, vec!["k1"]);
    assert_eq!(pages[0].tags, vec!["t1"]);
    assert!(!pages[0].metadata.contains_key("keywords"));
    assert!(!pages[0].metadata.contains_key("tags"));
}

#[tokio::test]
async fn promote_skips_absent_field() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(pagewiki::Fixed::new(200)));
    let builder = Builder::new(
        pw_map,
        vec!["keywords".into()],
        Box::new(NoopTokenizer),
        None,
    )
    .unwrap();

    let meta = serde_json::Map::new(); // no keywords
    let items = vec![make_item_with_metadata("doc_005", "text", meta)];
    let pages = builder.build(items).await.unwrap();

    assert!(pages[0].keywords.is_empty());
}

#[tokio::test]
async fn invalid_promote_field_rejected_at_construction() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(pagewiki::Fixed::new(200)));
    let err = Builder::new(
        pw_map,
        vec!["id".into()], // system field, not allowed
        Box::new(NoopTokenizer),
        None,
    )
    .unwrap_err();
    assert!(matches!(err, rag::index::Error::InvalidPromoteField(_)));
}

#[tokio::test]
async fn promote_type_mismatch_returns_error() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(pagewiki::Fixed::new(200)));
    let builder = Builder::new(
        pw_map,
        vec!["keywords".into()],
        Box::new(NoopTokenizer),
        None,
    )
    .unwrap();

    let mut meta = serde_json::Map::new();
    meta.insert("keywords".into(), json!("not-an-array")); // wrong type
    let items = vec![make_item_with_metadata("doc_006", "text", meta)];
    let err = builder.build(items).await.unwrap_err();
    assert!(matches!(err, rag::index::Error::PromoteDeserialize { .. }));
}
