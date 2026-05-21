//! engine + LocalReranker end-to-end：覆盖率 / 向量退化 / rerank 写入 scores。

mod common;

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use rag::index::engine::{
    Context, Engine, FieldsConfig, KeywordGroup, LocalRerankConfig, LocalReranker, TextField,
};

use common::mock_storage::MockStorage;

fn es_resp_with_tokens(rows: Vec<(&str, &str, f32)>) -> Value {
    let arr: Vec<Value> = rows
        .into_iter()
        .map(|(id, tokens, score)| {
            json!({
                "_score": score,
                "_source": { "id": id, "content": "", "content_tokens": tokens }
            })
        })
        .collect();
    json!({ "hits": { "total": { "value": arr.len() }, "hits": arr } })
}

fn rerank_options() -> Map<String, Value> {
    let mut weights = serde_json::Map::new();
    weights.insert("text".into(), json!(1.0));
    let mut rerank = serde_json::Map::new();
    rerank.insert("enabled".into(), json!(true));
    rerank.insert("weights".into(), Value::Object(weights));
    let mut opts = Map::new();
    opts.insert("rerank".into(), Value::Object(rerank));
    opts
}

#[tokio::test]
async fn local_rerank_text_coverage_reorders_hits() {
    // 两条 hits：a 命中 1/2，b 命中 2/2 → b 应排到前面。
    let resp = es_resp_with_tokens(vec![("a", "rust foo", 0.9), ("b", "rust async", 0.1)]);
    let storage = Box::new(MockStorage::new(resp));

    let local = LocalReranker::new(LocalRerankConfig {
        rerank_tokens: vec![TextField {
            field: "content_tokens".into(),
            weight: 1.0,
        }],
        query_keywords: vec![KeywordGroup {
            name: "main".into(),
            terms: vec!["rust".into(), "async".into()],
        }],
        query_vector: None,
    });

    let engine = Engine::new(
        storage,
        None,
        Some(Box::new(local)),
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
            rerank_options(),
            100,
            false,
        )
        .await
        .unwrap();

    assert_eq!(r.hits.len(), 2);
    assert_eq!(r.hits[0].pagewiki.id.as_deref(), Some("b"));
    assert_eq!(r.hits[1].pagewiki.id.as_deref(), Some("a"));
    assert!((r.hits[0].scores["text"] - 1.0).abs() < 1e-6);
    assert!((r.hits[1].scores["text"] - 0.5).abs() < 1e-6);
    // rerank 字段写入
    assert!(r.hits[0].scores.contains_key("rerank"));
}

#[tokio::test]
async fn local_rerank_vector_missing_falls_back_to_text() {
    // 全部 embedding 为 None → 退化路径：score = text + rank_feature
    let resp = es_resp_with_tokens(vec![("a", "rust", 0.7)]);
    let storage = Box::new(MockStorage::new(resp));

    let local = LocalReranker::new(LocalRerankConfig {
        rerank_tokens: vec![TextField {
            field: "content_tokens".into(),
            weight: 1.0,
        }],
        query_keywords: vec![KeywordGroup {
            name: "main".into(),
            terms: vec!["rust".into()],
        }],
        // 给定 query_vector，但 hits 无 embedding → 仍走退化
        query_vector: Some(vec![1.0, 0.0]),
    });

    let engine = Engine::new(
        storage,
        None,
        Some(Box::new(local)),
        100,
        0.0,
        60,
        Map::new(),
    )
    .unwrap();

    let mut weights = serde_json::Map::new();
    weights.insert("text".into(), json!(0.5));
    weights.insert("vector".into(), json!(0.5));
    let mut rerank = serde_json::Map::new();
    rerank.insert("enabled".into(), json!(true));
    rerank.insert("weights".into(), Value::Object(weights));
    let mut opts = Map::new();
    opts.insert("rerank".into(), Value::Object(rerank));

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

    assert_eq!(r.hits.len(), 1);
    assert_eq!(r.hits[0].scores["vector"], 0.0);
    // 退化：text(1.0) + rank_feature(0)
    assert!((r.hits[0].score - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn local_rerank_disabled_when_options_enabled_false() {
    // rerank.enabled = 默认 false → 不会调用 reranker；hits 顺序仍按原始 _score.
    let resp = es_resp_with_tokens(vec![("a", "rust foo", 0.1), ("b", "rust async", 0.9)]);
    let storage = Box::new(MockStorage::new(resp));

    let local = LocalReranker::new(LocalRerankConfig {
        rerank_tokens: vec![TextField {
            field: "content_tokens".into(),
            weight: 1.0,
        }],
        query_keywords: vec![KeywordGroup {
            name: "main".into(),
            terms: vec!["rust".into(), "async".into()],
        }],
        query_vector: None,
    });

    let engine = Engine::new(
        storage,
        None,
        Some(Box::new(local)),
        100,
        0.0,
        60,
        Map::new(),
    )
    .unwrap();

    // 不设 rerank.enabled
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

    assert_eq!(r.hits[0].pagewiki.id.as_deref(), Some("b"));
    assert_eq!(r.hits[1].pagewiki.id.as_deref(), Some("a"));
    // 没有 rerank 字段（reranker 未被调用）
    for h in &r.hits {
        assert!(!h.scores.contains_key("rerank"));
        assert!(!h.scores.contains_key("text"));
    }
    // 用 _ 触发 unused 提醒消除
    let _: HashMap<String, f32> = HashMap::new();
}
