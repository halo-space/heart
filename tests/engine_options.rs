//! engine options 白名单校验 + deep_merge 语义。

mod common;

use serde_json::{Map, json};

use rag::index::engine::{Context, Engine, FieldsConfig};

use common::mock_storage::{MockStorage, make_es_response};

#[tokio::test]
async fn unknown_option_key_rejected() {
    let storage = Box::new(MockStorage::new(make_es_response(vec![])));
    let engine = Engine::new(storage, None, None, 100, 0.0, 60, Map::new()).unwrap();

    // "top" 是保留字但不是合法 options key → UnknownOption
    let mut opts = Map::new();
    opts.insert("top".into(), json!(10));

    let err = engine
        .text_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            opts,
            100,
            false,
        )
        .await
        .unwrap_err();

    assert!(
        matches!(err, rag::index::engine::Error::UnknownOption(_)),
        "expected UnknownOption, got {err:?}"
    );
}

#[tokio::test]
async fn unknown_key_foo_rejected() {
    let storage = Box::new(MockStorage::new(make_es_response(vec![])));
    let engine = Engine::new(storage, None, None, 100, 0.0, 60, Map::new()).unwrap();

    let mut opts = Map::new();
    opts.insert("foo".into(), json!("bar"));

    let err = engine
        .text_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            opts,
            100,
            false,
        )
        .await
        .unwrap_err();

    assert!(
        matches!(err, rag::index::engine::Error::UnknownOption(_)),
        "expected UnknownOption for 'foo', got {err:?}"
    );
}

#[tokio::test]
async fn known_option_fusion_rrf_k_accepted() {
    let storage = Box::new(MockStorage::new(make_es_response(vec![(
        "a".into(),
        "x".into(),
        0.9,
    )])));
    let engine = Engine::new(storage, None, None, 100, 0.0, 60, Map::new()).unwrap();

    // fusion.rrf_k は合法キー → Ok
    let mut opts = Map::new();
    opts.insert("fusion".into(), json!({"rrf_k": 80}));

    let r = engine
        .text_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            opts,
            100,
            false,
        )
        .await
        .unwrap();

    assert_eq!(r.total, 1);
}

#[tokio::test]
async fn default_options_merged_with_per_call_options() {
    // Engine default_options で score_threshold=0.0, call-time で filter.score_threshold=0.8
    let storage = Box::new(MockStorage::new(make_es_response(vec![
        ("a".into(), "x".into(), 0.5),
        ("b".into(), "y".into(), 0.9),
    ])));
    let engine = Engine::new(storage, None, None, 100, 0.0, 60, Map::new()).unwrap();

    let mut opts = Map::new();
    opts.insert("filter".into(), json!({"score_threshold": 0.8}));

    let r = engine
        .text_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            opts,
            100,
            false,
        )
        .await
        .unwrap();

    // 只有 score=0.9 的 b 通过 threshold=0.8
    assert_eq!(r.total, 1);
    assert_eq!(r.hits[0].pagewiki.id.as_deref(), Some("b"));
}
