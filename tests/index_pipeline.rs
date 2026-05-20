//! Integration tests: full 7-step pipeline end-to-end.

use rag::index::pagewiki;
use rag::index::pagewiki::Base;
use rag::index::source::{Item, Scenario};
use rag::index::{Builder, NoopTokenizer};
use serde_json::json;
use std::collections::HashMap;

fn make_item(doc_id: &str, text: &str) -> Item {
    let mut metadata = serde_json::Map::new();
    metadata.insert("doc_id".into(), json!(doc_id));
    Item {
        text: text.to_string(),
        scenario: Scenario::General,
        metadata,
    }
}

fn make_builder() -> Builder {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(pagewiki::Fixed::new(200)));
    Builder::new(pw_map, vec![], Box::new(NoopTokenizer), None).unwrap()
}

#[tokio::test]
async fn pipeline_produces_pagewikis() {
    let builder = make_builder();
    let items = vec![make_item(
        "doc_001",
        "Hello world. This is a test document.",
    )];
    let pages = builder.build(items).await.unwrap();
    assert!(!pages.is_empty());
    let p = &pages[0];
    assert!(p.id.is_some());
    assert_eq!(p.doc_id.as_deref(), Some("doc_001"));
    assert!(p.version.is_some());
    assert_eq!(p.scenario, Some(Scenario::General));
    assert_eq!(p.idx, Some(0));
}

#[tokio::test]
async fn missing_doc_id_returns_error() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(pagewiki::Fixed::new(200)));
    let builder = Builder::new(pw_map, vec![], Box::new(NoopTokenizer), None).unwrap();

    let item = Item {
        text: "some text".into(),
        scenario: Scenario::General,
        metadata: serde_json::Map::new(), // no doc_id
    };
    let err = builder.build(vec![item]).await.unwrap_err();
    assert!(matches!(err, rag::index::Error::MissingDocId));
}

#[tokio::test]
async fn missing_scenario_returns_error() {
    let pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new(); // empty
    let builder = Builder::new(pw_map, vec![], Box::new(NoopTokenizer), None).unwrap();
    let items = vec![make_item("doc_x", "text")];
    let err = builder.build(items).await.unwrap_err();
    assert!(matches!(err, rag::index::Error::MissingScenario(_)));
}

#[tokio::test]
async fn noop_tokenizer_sets_empty_tokens() {
    let builder = make_builder();
    let items = vec![make_item("doc_002", "Hello world")];
    let pages = builder.build(items).await.unwrap();
    assert_eq!(pages[0].content_tokens, "");
    assert_eq!(pages[0].keyword_tokens, "");
    assert_eq!(pages[0].question_tokens, "");
}

#[tokio::test]
async fn no_embedder_leaves_embedding_none() {
    let builder = make_builder();
    let items = vec![make_item("doc_003", "text")];
    let pages = builder.build(items).await.unwrap();
    assert!(pages[0].embedding.is_none());
}

#[tokio::test]
async fn embedder_sets_embedding() {
    use rag::index::NoopEmbedder;
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(pagewiki::Fixed::new(200)));
    let builder = Builder::new(
        pw_map,
        vec![],
        Box::new(NoopTokenizer),
        Some(Box::new(NoopEmbedder)),
    )
    .unwrap();
    let items = vec![make_item("doc_004", "embed me")];
    let pages = builder.build(items).await.unwrap();
    assert_eq!(pages[0].embedding, Some(vec![]));
}
