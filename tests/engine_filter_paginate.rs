//! engine threshold + top + page + doc_aggs.

mod common;

use serde_json::{Map, json};

use rag::index::engine::{Context, Engine, FieldsConfig};

use common::mock_storage::{MockStorage, make_es_response, make_es_response_with_doc};

#[tokio::test]
async fn threshold_filters_low_scores() {
    let resp = make_es_response(vec![
        ("a".into(), "x".into(), 0.05),
        ("b".into(), "y".into(), 0.9),
        ("c".into(), "z".into(), 0.4),
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
async fn top_caps_returned_hits() {
    let resp = make_es_response(vec![
        ("a".into(), "x".into(), 0.9),
        ("b".into(), "y".into(), 0.8),
        ("c".into(), "z".into(), 0.7),
        ("d".into(), "w".into(), 0.6),
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
            2,
            false,
        )
        .await
        .unwrap();

    // top 在 filter 阶段截断；total 反映 top 截断后的数量。
    assert_eq!(r.total, 2);
    assert_eq!(r.hits.len(), 2);
}

#[tokio::test]
async fn page_out_of_bounds_returns_empty_hits_but_total_preserved() {
    let resp = make_es_response(vec![
        ("a".into(), "x".into(), 0.9),
        ("b".into(), "y".into(), 0.8),
    ]);
    let storage = Box::new(MockStorage::new(resp));
    let engine = Engine::new(storage, None, None, 100, 0.0, 60, Map::new()).unwrap();

    let mut opts = Map::new();
    opts.insert("pagination".into(), json!({"page_num": 5, "page_size": 10}));

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

    assert_eq!(r.total, 2);
    assert!(r.hits.is_empty());
}

#[tokio::test]
async fn doc_aggs_count_by_doc_id() {
    // doc_id 序列 [A, A, B, A, C] → A=3, B=1, C=1
    let resp = make_es_response_with_doc(vec![
        ("h1".into(), "A".into(), "x".into(), 0.9),
        ("h2".into(), "A".into(), "x".into(), 0.85),
        ("h3".into(), "B".into(), "y".into(), 0.8),
        ("h4".into(), "A".into(), "x".into(), 0.7),
        ("h5".into(), "C".into(), "z".into(), 0.6),
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

    assert_eq!(r.total, 5);
    let counts: std::collections::HashMap<String, usize> = r
        .doc_aggs
        .iter()
        .map(|d| (d.doc_id.clone(), d.count))
        .collect();
    assert_eq!(counts.get("A").copied(), Some(3));
    assert_eq!(counts.get("B").copied(), Some(1));
    assert_eq!(counts.get("C").copied(), Some(1));
}
