//! `ModelReranker`：通过外部 HTTP 服务做二阶段 rerank。
//!
//! 行为（与 spec 严格对齐）：
//!
//! - POST `{url}/rerank`，body 形态严格 `{"query": <q>, "docs": [<hit.pagewiki.content>, ...]}`
//! - 期望响应严格 `{"scores": [<f32>, ...]}`
//! - 长度不匹配 → `Err(Error::Reranker("scores length mismatch: got N, want M"))`
//! - HTTP 失败 / JSON 解析失败 / 缺 `scores` 字段 → `Err(Error::Reranker(<reason>))`
//! - 组合分公式：
//!     ```text
//!     hit.score = w_text   * (hit.scores.text        || 0)
//!               + w_vector * (hit.scores.vector      || 0)
//!               + w_model  * model_score
//!               + w_rf     * (hit.scores.rank_feature|| 0)
//!     ```
//!   `w_model` 缺省 1.0，其余三项缺省 0.0。
//! - 写 `scores.model = model_score` / `scores.rerank = combined` / `hit.score = combined`；
//!   按 `score` 倒序返回。

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use serde_json::{Value, json};

use crate::index::engine::types::{Error, Hit};

use super::Reranker;

#[derive(Debug)]
pub struct ModelReranker {
    url: String,
    http: reqwest::Client,
}

impl ModelReranker {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            http: reqwest::Client::new(),
        }
    }

    pub fn with_client(url: impl Into<String>, http: reqwest::Client) -> Self {
        Self {
            url: url.into(),
            http,
        }
    }
}

impl Reranker for ModelReranker {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        hits: Vec<Hit>,
        weights: HashMap<String, f32>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Hit>, Error>> + Send + 'a>> {
        Box::pin(async move {
            let docs: Vec<&str> = hits.iter().map(|h| h.pagewiki.content.as_str()).collect();
            let want = docs.len();

            let body = json!({ "query": query, "docs": docs });
            let endpoint = format!("{}/rerank", self.url.trim_end_matches('/'));
            let resp = self
                .http
                .post(&endpoint)
                .json(&body)
                .send()
                .await
                .map_err(|e| Error::Reranker(format!("http: {e}")))?;
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                let err = Error::Reranker(format!("http status {status}: {text}"));
                tracing::warn!(stage = "rerank", error = %err, "engine.rerank.failed");
                return Err(err);
            }
            let value: Value = resp
                .json()
                .await
                .map_err(|e| Error::Reranker(format!("json parse: {e}")))?;
            let scores = value
                .get("scores")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Reranker("missing 'scores' field".into()))?
                .clone();
            if scores.len() != want {
                let got = scores.len();
                let err =
                    Error::Reranker(format!("scores length mismatch: got {got}, want {want}"));
                tracing::warn!(stage = "rerank", error = %err, "engine.rerank.failed");
                return Err(err);
            }

            let w_text = weights.get("text").copied().unwrap_or(0.0);
            let w_vec = weights.get("vector").copied().unwrap_or(0.0);
            let w_rf = weights.get("rank_feature").copied().unwrap_or(0.0);
            let w_model = weights.get("model").copied().unwrap_or(1.0);

            let mut out = hits;
            for (i, score_v) in scores.into_iter().enumerate() {
                let model_score = score_v.as_f64().unwrap_or(0.0) as f32;
                let prev_text = out[i].scores.get("text").copied().unwrap_or(0.0);
                let prev_vector = out[i].scores.get("vector").copied().unwrap_or(0.0);
                let prev_rf = out[i].scores.get("rank_feature").copied().unwrap_or(0.0);
                let combined = w_text * prev_text
                    + w_vec * prev_vector
                    + w_model * model_score
                    + w_rf * prev_rf;
                out[i].scores.insert("model".into(), model_score);
                out[i].scores.insert("rerank".into(), combined);
                out[i].score = combined;
            }
            out.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            tracing::debug!(
                op = "rerank",
                mode = "model",
                hit_count = out.len(),
                "engine.rerank.done"
            );
            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::pagewiki::PageWiki;
    use std::collections::HashMap;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn ph(id: &str, content: &str) -> Hit {
        Hit {
            pagewiki: PageWiki {
                id: Some(id.into()),
                content: content.into(),
                ..Default::default()
            },
            score: 0.0,
            scores: HashMap::new(),
            highlight: None,
        }
    }

    #[tokio::test]
    async fn body_order_matches_hit_order_and_scores_aligned() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .and(body_json(serde_json::json!({
                "query": "q",
                "docs": ["a-content", "b-content", "c-content"]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "scores": [0.1, 0.9, 0.5]
            })))
            .mount(&server)
            .await;

        let r = ModelReranker::new(server.uri());
        let hits = vec![
            ph("a", "a-content"),
            ph("b", "b-content"),
            ph("c", "c-content"),
        ];
        let mut w = HashMap::new();
        w.insert("model".into(), 1.0);
        let out = r.rerank("q", hits, w).await.unwrap();
        // scores 顺序与 hits 顺序对齐写入；返回按 score 倒序
        assert_eq!(out[0].pagewiki.id.as_deref(), Some("b"));
        assert_eq!(out[1].pagewiki.id.as_deref(), Some("c"));
        assert_eq!(out[2].pagewiki.id.as_deref(), Some("a"));
        assert!((out[0].scores["model"] - 0.9).abs() < 1e-6);
        assert!((out[0].score - 0.9).abs() < 1e-6);
    }

    #[tokio::test]
    async fn scores_length_mismatch_returns_reranker_err() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "scores": [0.1, 0.2] })),
            )
            .mount(&server)
            .await;

        let r = ModelReranker::new(server.uri());
        let hits = vec![ph("a", "x"), ph("b", "y"), ph("c", "z")];
        let err = r.rerank("q", hits, HashMap::new()).await.unwrap_err();
        match err {
            Error::Reranker(msg) => {
                assert!(msg.contains("got 2"));
                assert!(msg.contains("want 3"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[tokio::test]
    async fn missing_scores_field_returns_reranker_err() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "oops": [] })),
            )
            .mount(&server)
            .await;

        let r = ModelReranker::new(server.uri());
        let hits = vec![ph("a", "x")];
        let err = r.rerank("q", hits, HashMap::new()).await.unwrap_err();
        assert!(matches!(err, Error::Reranker(_)));
    }

    #[tokio::test]
    async fn http_500_returns_reranker_err() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;

        let r = ModelReranker::new(server.uri());
        let hits = vec![ph("a", "x")];
        let err = r.rerank("q", hits, HashMap::new()).await.unwrap_err();
        assert!(matches!(err, Error::Reranker(_)));
    }

    #[tokio::test]
    async fn weights_default_model_is_one() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "scores": [0.7] })),
            )
            .mount(&server)
            .await;

        let r = ModelReranker::new(server.uri());
        let hits = vec![ph("a", "x")];
        // 不传任何权重 → w_model 默认 1.0；其他三项默认 0；prev_text/vector/rf 也都是 0
        let out = r.rerank("q", hits, HashMap::new()).await.unwrap();
        // combined = 1.0 * 0.7 = 0.7
        assert!((out[0].score - 0.7).abs() < 1e-6);
    }
}
