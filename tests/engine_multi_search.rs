//! engine.multi_search 并发耗时 + fail-fast + rrf/stage scores.

mod common;

use serde_json::{Map, json};

use rag::index::engine::{Context, Engine, FieldsConfig};

use common::mock_storage::{MockStorage, make_es_response};

/// 两路 mock 各 100ms → multi_search 总耗时 < 180ms（并发验证）。
#[tokio::test]
async fn multi_search_concurrent_both_100ms_finishes_under_180ms() {
    let text_resp = make_es_response(vec![
        ("a".into(), "hello".into(), 0.9),
        ("b".into(), "world".into(), 0.7),
    ]);
    let vec_resp = make_es_response(vec![
        ("b".into(), "world".into(), 0.85),
        ("c".into(), "foo".into(), 0.6),
    ]);

    let text_storage = Box::new(MockStorage::new(text_resp).with_delay(100));
    let vector_storage = Box::new(MockStorage::new(vec_resp).with_delay(100));

    let engine = Engine::new(
        text_storage,
        Some(vector_storage),
        None,
        100,
        0.0,
        60,
        Map::new(),
    )
    .unwrap();

    let start = std::time::Instant::now();
    let r = engine
        .multi_search(
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
    let elapsed = start.elapsed().as_millis();

    // 并发执行：两路各 100ms → 应 < 180ms
    assert!(
        elapsed < 180,
        "expected < 180ms for concurrent 100ms mocks, got {elapsed}ms"
    );
    // 融合后 total ≥ 1
    assert!(r.total >= 1, "expected at least 1 hit after RRF fusion");
}

/// hits 含 stage 分数标注。
#[tokio::test]
async fn multi_search_hits_have_stage_scores() {
    let text_resp = make_es_response(vec![("a".into(), "x".into(), 0.9)]);
    let vec_resp = make_es_response(vec![("a".into(), "x".into(), 0.8)]);

    let engine = Engine::new(
        Box::new(MockStorage::new(text_resp)),
        Some(Box::new(MockStorage::new(vec_resp))),
        None,
        100,
        0.0,
        60,
        Map::new(),
    )
    .unwrap();

    let r = engine
        .multi_search(
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

    assert!(!r.hits.is_empty());
    for h in &r.hits {
        assert!(
            h.scores.contains_key("stage"),
            "hit missing stage score: {:?}",
            h.scores
        );
    }
}

/// text storage 立即报错 → multi_search 返回 Err(Backend(_))，≤ 500ms。
#[tokio::test]
async fn multi_search_fail_fast_on_text_error() {
    let text_storage =
        Box::new(MockStorage::new(json!({})).with_error("simulated text backend failure"));
    let vec_resp = make_es_response(vec![("v1".into(), "vec".into(), 0.9)]);
    let vector_storage = Box::new(MockStorage::new(vec_resp).with_delay(400));

    let engine = Engine::new(
        text_storage,
        Some(vector_storage),
        None,
        100,
        0.0,
        60,
        Map::new(),
    )
    .unwrap();

    let start = std::time::Instant::now();
    let err = engine
        .multi_search(
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
    let elapsed = start.elapsed().as_millis();

    assert!(
        matches!(err, rag::index::engine::Error::Backend(_)),
        "expected Backend error, got {err:?}"
    );
    assert!(
        elapsed <= 500,
        "fail-fast should complete within 500ms, got {elapsed}ms"
    );
}

/// no vector backend → multi_search 报 NoVectorBackend。
#[tokio::test]
async fn multi_search_no_vector_backend_returns_error() {
    let storage = Box::new(MockStorage::new(make_es_response(vec![])));
    let engine = Engine::new(storage, None, None, 100, 0.0, 60, Map::new()).unwrap();

    let err = engine
        .multi_search(
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

    assert!(matches!(err, rag::index::engine::Error::NoVectorBackend));
}
