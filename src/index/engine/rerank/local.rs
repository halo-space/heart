//! `LocalReranker`：v1 完整本地组合分实现。
//!
//! 算法（与 spec 严格对齐）：
//!
//! - `text_score`：query 关键词覆盖率
//!     - `total_weight = sum(1.0 per term in query_keywords[*].terms)`
//!     - `pagewiki_tokens` 取自 `LocalRerankConfig.rerank_tokens` 列出的 PageWiki 字段，按
//!       whitespace 分词并乘以字段权重组成 token 池
//!     - `hit_weight = sum(weight per term if term ∈ pagewiki_tokens)`
//!     - `total_weight == 0` 或 keywords 空 → 0
//! - `vector_score = cosine(query_vector, hit.pagewiki.embedding)`；任一缺失 → 0
//! - `rank_feature_score = 0`（第一版保留扩展位）
//! - 组合分：`score = w_text * text + w_vector * vector + w_rank_feature * rf`
//! - 退化：所有 hit 的 `vector_score` 求和为 0 → `score = text + rank_feature`

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;

use crate::index::engine::types::{Error, Hit, KeywordGroup, TextField};

use super::Reranker;

/// `LocalReranker` 配置。
///
/// `rerank_tokens` 决定 PageWiki 哪些字段按 whitespace 分词组成 token 池；
/// `query_keywords` 是 query 端要打分的关键词组；
/// `query_vector` 是 query 向量（业务侧已算好）。
#[derive(Debug, Clone, Default)]
pub struct LocalRerankConfig {
    pub rerank_tokens: Vec<TextField>,
    pub query_keywords: Vec<KeywordGroup>,
    pub query_vector: Option<Vec<f32>>,
}

#[derive(Debug, Default)]
pub struct LocalReranker {
    pub config: LocalRerankConfig,
}

impl LocalReranker {
    pub fn new(config: LocalRerankConfig) -> Self {
        Self { config }
    }
}

fn tokens_from_field(page: &crate::index::pagewiki::PageWiki, field: &str) -> Vec<String> {
    let raw: String = match field {
        "content" => page.content.clone(),
        "content_tokens" => page.content_tokens.clone(),
        "header" => page.header.clone(),
        "keyword_tokens" => page.keyword_tokens.clone(),
        "keywords" => page.keywords.join(" "),
        "question_tokens" => page.question_tokens.clone(),
        "questions" => page.questions.join(" "),
        "tags" => page.tags.join(" "),
        _ => String::new(),
    };
    raw.split_whitespace().map(|s| s.to_string()).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

impl LocalReranker {
    fn text_score(&self, hit: &Hit) -> f32 {
        let groups = &self.config.query_keywords;
        if groups.is_empty() {
            return 0.0;
        }
        // 收集 PageWiki 的 token 池（按字段权重；权重 0 不计）。
        // 用 HashSet<String>: 为简化，token 池只看"是否出现"，命中即按 query term 权重计 1.0。
        let mut pool: HashSet<String> = HashSet::new();
        let mut any_field = false;
        for tf in &self.config.rerank_tokens {
            if tf.weight <= 0.0 {
                continue;
            }
            any_field = true;
            for tok in tokens_from_field(&hit.pagewiki, &tf.field) {
                pool.insert(tok);
            }
        }
        if !any_field {
            return 0.0;
        }
        let mut total_weight = 0.0_f32;
        let mut hit_weight = 0.0_f32;
        for g in groups {
            for term in &g.terms {
                total_weight += 1.0;
                if pool.contains(term) {
                    hit_weight += 1.0;
                }
            }
        }
        if total_weight == 0.0 {
            0.0
        } else {
            hit_weight / total_weight
        }
    }

    fn vector_score(&self, hit: &Hit) -> f32 {
        let Some(ref qv) = self.config.query_vector else {
            return 0.0;
        };
        let Some(ref dv) = hit.pagewiki.embedding else {
            return 0.0;
        };
        cosine(qv, dv)
    }
}

impl Reranker for LocalReranker {
    fn rerank<'a>(
        &'a self,
        _query: &'a str,
        hits: Vec<Hit>,
        weights: HashMap<String, f32>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Hit>, Error>> + Send + 'a>> {
        Box::pin(async move {
            let w_text = weights.get("text").copied().unwrap_or(0.0);
            let w_vec = weights.get("vector").copied().unwrap_or(0.0);
            let w_rf = weights.get("rank_feature").copied().unwrap_or(0.0);

            let n = hits.len();
            let mut text_scores = Vec::with_capacity(n);
            let mut vec_scores = Vec::with_capacity(n);
            let mut rf_scores = Vec::with_capacity(n);
            for h in &hits {
                text_scores.push(self.text_score(h));
                vec_scores.push(self.vector_score(h));
                rf_scores.push(0.0_f32);
            }
            let vec_sum: f32 = vec_scores.iter().copied().sum();
            let degraded = vec_sum == 0.0;

            let mut out: Vec<Hit> = hits;
            for i in 0..out.len() {
                let t = text_scores[i];
                let v = vec_scores[i];
                let rf = rf_scores[i];
                out[i].scores.insert("text".into(), t);
                out[i].scores.insert("vector".into(), v);
                out[i].scores.insert("rank_feature".into(), rf);
                let combined = if degraded {
                    t + rf
                } else {
                    w_text * t + w_vec * v + w_rf * rf
                };
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
                mode = "local",
                hit_count = n,
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

    fn ph(id: &str, tokens: &str, embedding: Option<Vec<f32>>) -> Hit {
        Hit {
            pagewiki: PageWiki {
                id: Some(id.into()),
                content_tokens: tokens.into(),
                embedding,
                ..Default::default()
            },
            score: 0.0,
            scores: HashMap::new(),
            highlight: None,
        }
    }

    #[tokio::test]
    async fn text_score_coverage_half() {
        let cfg = LocalRerankConfig {
            rerank_tokens: vec![TextField {
                field: "content_tokens".into(),
                weight: 1.0,
            }],
            query_keywords: vec![KeywordGroup {
                name: "main".into(),
                terms: vec!["rust".into(), "async".into()],
            }],
            query_vector: None,
        };
        let r = LocalReranker::new(cfg);
        let hits = vec![ph("a", "rust foo bar", None)];
        let mut w = HashMap::new();
        w.insert("text".into(), 1.0);
        let out = r.rerank("q", hits, w).await.unwrap();
        // 命中 1/2 → 0.5；vector 全 0 → 退化 score = text + 0
        assert!((out[0].scores["text"] - 0.5).abs() < 1e-6);
        assert!((out[0].score - 0.5).abs() < 1e-6);
        assert_eq!(out[0].scores["rank_feature"], 0.0);
    }

    #[tokio::test]
    async fn vector_missing_falls_back_to_text_plus_rf() {
        let cfg = LocalRerankConfig {
            rerank_tokens: vec![TextField {
                field: "content_tokens".into(),
                weight: 1.0,
            }],
            query_keywords: vec![KeywordGroup {
                name: "main".into(),
                terms: vec!["rust".into()],
            }],
            query_vector: Some(vec![1.0, 0.0]), // 给了 query_vector
        };
        let r = LocalReranker::new(cfg);
        // 全部 hits 的 embedding 都为 None → 向量分总和 0 → 退化
        let hits = vec![ph("a", "rust", None)];
        let mut w = HashMap::new();
        w.insert("text".into(), 0.5);
        w.insert("vector".into(), 0.5);
        let out = r.rerank("q", hits, w).await.unwrap();
        assert_eq!(out[0].scores["vector"], 0.0);
        // 退化路径：score = text(1.0) + rank_feature(0)
        assert!((out[0].score - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn rank_feature_is_zero_v1() {
        let cfg = LocalRerankConfig::default();
        let r = LocalReranker::new(cfg);
        let hits = vec![ph("a", "", None)];
        let out = r.rerank("q", hits, HashMap::new()).await.unwrap();
        assert_eq!(out[0].scores["rank_feature"], 0.0);
    }

    #[tokio::test]
    async fn returns_sorted_by_score_desc() {
        let cfg = LocalRerankConfig {
            rerank_tokens: vec![TextField {
                field: "content_tokens".into(),
                weight: 1.0,
            }],
            query_keywords: vec![KeywordGroup {
                name: "main".into(),
                terms: vec!["rust".into(), "async".into()],
            }],
            query_vector: None,
        };
        let r = LocalReranker::new(cfg);
        let hits = vec![
            ph("low", "foo", None),
            ph("hi", "rust async", None),
            ph("mid", "rust", None),
        ];
        let mut w = HashMap::new();
        w.insert("text".into(), 1.0);
        let out = r.rerank("q", hits, w).await.unwrap();
        let ids: Vec<&str> = out
            .iter()
            .map(|h| h.pagewiki.id.as_deref().unwrap())
            .collect();
        assert_eq!(ids, vec!["hi", "mid", "low"]);
    }

    #[tokio::test]
    async fn cosine_full_path() {
        let cfg = LocalRerankConfig {
            rerank_tokens: vec![TextField {
                field: "content_tokens".into(),
                weight: 1.0,
            }],
            query_keywords: vec![KeywordGroup {
                name: "main".into(),
                terms: vec!["rust".into()],
            }],
            query_vector: Some(vec![1.0, 0.0]),
        };
        let r = LocalReranker::new(cfg);
        // 一条命中 + 有 embedding，向量分非 0 → 不退化
        let hits = vec![ph("a", "rust", Some(vec![1.0, 0.0]))];
        let mut w = HashMap::new();
        w.insert("text".into(), 0.5);
        w.insert("vector".into(), 0.5);
        let out = r.rerank("q", hits, w).await.unwrap();
        // text=1.0, vector=cosine(1)=1.0 → score = 0.5*1 + 0.5*1 = 1.0
        assert!((out[0].score - 1.0).abs() < 1e-6);
    }
}
