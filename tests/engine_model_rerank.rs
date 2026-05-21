//! engine + ModelReranker end-to-end：wiremock HTTP server 验证 body 顺序与重排结果。

mod common;

use serde_json::{Map, Value, json};
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use rag::index::engine::{Context, Engine, FieldsConfig, ModelReranker};

use common::mock_storage::MockStorage;

fn es_resp(rows: Vec<(&str, &str, f32)>) -> Value {
    let arr: Vec<Value> = rows
        .into_iter()
        .map(|(id, content, score)| {
            json!({
                "_score": score,
                "_source": { "id": id, "content": content }
            })
        })
        .collect();
    json!({ "hits": { "total": { "value": arr.len() }, "hits": arr } })
}

fn rerank_enabled_options() -> Map<String, Value> {
    let mut rerank = serde_json::Map::new();
    rerank.insert("enabled".into(), json!(true));
    let mut opts = Map::new();
    opts.insert("rerank".into(), Value::Object(rerank));
    opts
}

#[tokio::test]
async fn model_rerank_e2e_reorders_hits_by_model_score() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .and(body_json(json!({
            "query": "",
            "docs": ["a-content", "b-content", "c-content"]
        })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "scores": [0.1, 0.9, 0.5] })),
        )
        .mount(&server)
        .await;

    let resp = es_resp(vec![
        ("a", "a-content", 0.5),
        ("b", "b-content", 0.5),
        ("c", "c-content", 0.5),
    ]);
    let storage = Box::new(MockStorage::new(resp));
    let model = ModelReranker::new(server.uri());

    let engine = Engine::new(
        storage,
        None,
        Some(Box::new(model)),
        100,
        0.0,
        60,
        Map::new(),
    )
    .unwrap();

    let r = engine
        .text_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            rerank_enabled_options(),
            100,
            false,
        )
        .await
        .unwrap();

    assert_eq!(r.hits.len(), 3);
    assert_eq!(r.hits[0].pagewiki.id.as_deref(), Some("b"));
    assert_eq!(r.hits[1].pagewiki.id.as_deref(), Some("c"));
    assert_eq!(r.hits[2].pagewiki.id.as_deref(), Some("a"));
    assert!((r.hits[0].scores["model"] - 0.9).abs() < 1e-6);
    assert!((r.hits[0].score - 0.9).abs() < 1e-6);
}

#[tokio::test]
async fn model_rerank_http_500_propagates_reranker_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;

    let resp = es_resp(vec![("a", "x", 1.0)]);
    let storage = Box::new(MockStorage::new(resp));
    let model = ModelReranker::new(server.uri());

    let engine = Engine::new(
        storage,
        None,
        Some(Box::new(model)),
        100,
        0.0,
        60,
        Map::new(),
    )
    .unwrap();

    let err = engine
        .text_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            rerank_enabled_options(),
            100,
            false,
        )
        .await
        .unwrap_err();

    assert!(matches!(err, rag::index::engine::Error::Reranker(_)));
}

#[tokio::test]
async fn model_rerank_scores_length_mismatch_propagates_reranker_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "scores": [0.1] })))
        .mount(&server)
        .await;

    let resp = es_resp(vec![("a", "x", 1.0), ("b", "y", 1.0)]);
    let storage = Box::new(MockStorage::new(resp));
    let model = ModelReranker::new(server.uri());

    let engine = Engine::new(
        storage,
        None,
        Some(Box::new(model)),
        100,
        0.0,
        60,
        Map::new(),
    )
    .unwrap();

    let err = engine
        .text_search(
            "idx",
            json!({}),
            &Context::default(),
            &FieldsConfig::default(),
            rerank_enabled_options(),
            100,
            false,
        )
        .await
        .unwrap_err();

    match err {
        rag::index::engine::Error::Reranker(msg) => {
            assert!(msg.contains("got 1"));
            assert!(msg.contains("want 2"));
        }
        other => panic!("expected Reranker, got {other:?}"),
    }
}
