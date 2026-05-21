//! engine.text_search end-to-end：mock storage → Response.

mod common;

use serde_json::{Map, json};

use rag::index::engine::{Context, Engine, FieldsConfig};

use common::mock_storage::{MockStorage, make_es_response};

#[tokio::test]
async fn text_search_returns_total_and_sorts_desc() {
    let resp = make_es_response(vec![
        ("a".into(), "x".into(), 0.5),
        ("b".into(), "y".into(), 0.9),
        ("c".into(), "z".into(), 0.7),
    ]);
    let storage = Box::new(MockStorage::new(resp));
    let engine = Engine::new(storage, None, None, 100, 0.0, 60, Map::new()).unwrap();

    let r = engine
        .text_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            Map::new(),
            100,
            false,
        )
        .await
        .unwrap();

    assert_eq!(r.total, 3);
    assert_eq!(r.hits.len(), 3);
    assert_eq!(r.hits[0].pagewiki.id.as_deref(), Some("b"));
    assert_eq!(r.hits[1].pagewiki.id.as_deref(), Some("c"));
    assert_eq!(r.hits[2].pagewiki.id.as_deref(), Some("a"));
    // stage scores annotated
    for h in &r.hits {
        assert!(h.scores.contains_key("stage"));
    }
}

#[tokio::test]
async fn text_search_threshold_filters_low_scores() {
    let resp = make_es_response(vec![
        ("a".into(), "x".into(), 0.05),
        ("b".into(), "y".into(), 0.9),
    ]);
    let storage = Box::new(MockStorage::new(resp));
    let engine = Engine::new(storage, None, None, 100, 0.5, 60, Map::new()).unwrap();

    let r = engine
        .text_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            Map::new(),
            100,
            false,
        )
        .await
        .unwrap();

    assert_eq!(r.total, 1);
    assert_eq!(r.hits[0].pagewiki.id.as_deref(), Some("b"));
}

#[tokio::test]
async fn text_search_propagates_bad_response_when_source_missing() {
    // hits.hits[0] missing _source → BadResponse.
    let resp = json!({"hits":{"hits":[{"_score":1.0}]}});
    let storage = Box::new(MockStorage::new(resp));
    let engine = Engine::new(storage, None, None, 100, 0.0, 60, Map::new()).unwrap();

    let err = engine
        .text_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            Map::new(),
            100,
            false,
        )
        .await
        .unwrap_err();

    assert!(
        matches!(err, rag::index::engine::Error::BadResponse(_)),
        "expected BadResponse, got {err:?}"
    );
}
