//! `engine::core`：`Engine` 结构 + 4 个召回入口 + `resolve_options` + 共享后处理。
//!
//! 4 个召回入口共享同一后处理链：
//!
//! ```text
//! resolve_options
//!   -> (text|vector|hybrid|multi) inner
//!   -> annotate_stage / fuse_by_rrf
//!   -> rerank (optional)
//!   -> filter_hits
//!   -> paginate_hits
//!   -> build_response
//! ```

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::index::storage;

use super::filter::filter_hits;
use super::fusion::{FusionWeights, fuse_by_rrf};
use super::paginate::paginate_hits;
use super::rerank::Reranker;
use super::response::build_response;
use super::types::{Context, Error, FieldsConfig, Hit, ResolvedOptions, Response};

const ALLOWED_OPTION_KEYS: &[&str] = &["fusion", "rerank", "filter", "pagination", "trace"];

/// 校验 options 顶层 key（仅顶层；不递归校验）。
pub(crate) fn validate_option_keys(opts: &Map<String, Value>) -> Result<(), Error> {
    for key in opts.keys() {
        if !ALLOWED_OPTION_KEYS.contains(&key.as_str()) {
            return Err(Error::UnknownOption(key.clone()));
        }
    }
    Ok(())
}

/// `serde_json::Map` 递归 deep merge：双 Map → 同名键递归；其他形态 → options 整体覆盖。
pub(crate) fn deep_merge(
    mut default: Map<String, Value>,
    options: &Map<String, Value>,
) -> Map<String, Value> {
    for (k, v) in options {
        match (default.get_mut(k), v) {
            (Some(Value::Object(d_inner)), Value::Object(o_inner)) => {
                let merged = deep_merge(std::mem::take(d_inner), o_inner);
                *d_inner = merged;
            }
            _ => {
                default.insert(k.clone(), v.clone());
            }
        }
    }
    default
}

/// 解析 + 合并 + 反序列化 + 填默认。
pub(crate) fn resolve_options(
    default_options: &Map<String, Value>,
    options: &Map<String, Value>,
    score_threshold_fallback: f32,
    rrf_k_fallback: u32,
) -> Result<ResolvedOptions, Error> {
    validate_option_keys(default_options)?;
    validate_option_keys(options)?;
    let merged = deep_merge(default_options.clone(), options);
    let mut resolved: ResolvedOptions = serde_json::from_value(Value::Object(merged))?;
    resolved.fill_defaults(score_threshold_fallback, rrf_k_fallback);
    Ok(resolved)
}

/// 查询装配引擎。
///
/// 字段全部私有；MUST NOT 实现 `Clone`、MUST NOT 带泛型。
pub struct Engine {
    storage: Box<dyn storage::Base>,
    vector: Option<Box<dyn storage::Base>>,
    reranker: Option<Box<dyn Reranker>>,
    #[allow(dead_code)]
    top: usize,
    score_threshold: f32,
    rrf_k: u32,
    default_options: Map<String, Value>,
}

impl Engine {
    /// 构造一个 Engine。
    ///
    /// `default_options` 任一非白名单顶层 key → `Err(Error::UnknownOption(<key>))`。
    pub fn new(
        storage: Box<dyn storage::Base>,
        vector: Option<Box<dyn storage::Base>>,
        reranker: Option<Box<dyn Reranker>>,
        top: usize,
        score_threshold: f32,
        rrf_k: u32,
        default_options: Map<String, Value>,
    ) -> Result<Self, Error> {
        validate_option_keys(&default_options)?;
        Ok(Self {
            storage,
            vector,
            reranker,
            top,
            score_threshold,
            rrf_k,
            default_options,
        })
    }

    /// 文本路单路召回。
    #[allow(clippy::too_many_arguments)]
    pub async fn text_search(
        &self,
        index: &str,
        search_query: Value,
        context: &Context,
        fields: &FieldsConfig,
        options: Map<String, Value>,
        top: usize,
        highlight: bool,
    ) -> Result<Response, Error> {
        let span = tracing::debug_span!("engine.text_search", top, highlight);
        let _enter = span.enter();
        let resolved = resolve_options(
            &self.default_options,
            &options,
            self.score_threshold,
            self.rrf_k,
        )?;
        let mut hits =
            match text_search_inner(&*self.storage, index, &search_query, highlight).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(stage = "text_search", error = %e, "engine.failed");
                    return Err(e);
                }
            };
        annotate_stage(&mut hits);
        tracing::debug!(
            stage = "text_search",
            n = hits.len(),
            "engine.text_search.done"
        );
        post_process(self, &resolved, context, fields, hits, top).await
    }

    /// 向量路单路召回。
    pub async fn vector_search(
        &self,
        index: &str,
        search_query: Value,
        context: &Context,
        fields: &FieldsConfig,
        options: Map<String, Value>,
        top: usize,
    ) -> Result<Response, Error> {
        let span = tracing::debug_span!("engine.vector_search", top);
        let _enter = span.enter();
        let resolved = resolve_options(
            &self.default_options,
            &options,
            self.score_threshold,
            self.rrf_k,
        )?;
        let backend = self.vector.as_deref().ok_or(Error::NoVectorBackend)?;
        let mut hits = match vector_search_inner(backend, index, &search_query).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(stage = "vector_search", error = %e, "engine.failed");
                return Err(e);
            }
        };
        annotate_stage(&mut hits);
        tracing::debug!(
            stage = "vector_search",
            n = hits.len(),
            "engine.vector_search.done"
        );
        post_process(self, &resolved, context, fields, hits, top).await
    }

    /// 混合 DSL 单路召回（业务保证 DSL 是混合 DSL）；走 `storage`。
    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_search(
        &self,
        index: &str,
        search_query: Value,
        context: &Context,
        fields: &FieldsConfig,
        options: Map<String, Value>,
        top: usize,
        highlight: bool,
    ) -> Result<Response, Error> {
        let span = tracing::debug_span!("engine.hybrid_search", top, highlight);
        let _enter = span.enter();
        let resolved = resolve_options(
            &self.default_options,
            &options,
            self.score_threshold,
            self.rrf_k,
        )?;
        let mut hits =
            match text_search_inner(&*self.storage, index, &search_query, highlight).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(stage = "hybrid_search", error = %e, "engine.failed");
                    return Err(e);
                }
            };
        annotate_stage(&mut hits);
        tracing::debug!(
            stage = "hybrid_search",
            n = hits.len(),
            "engine.hybrid_search.done"
        );
        post_process(self, &resolved, context, fields, hits, top).await
    }

    /// 文本 + 向量并发跑 + RRF 融合。
    ///
    /// `tokio::try_join!` 语义：任一路失败立即 abort 另一路。
    #[allow(clippy::too_many_arguments)]
    pub async fn multi_search(
        &self,
        index: &str,
        search_query: Value,
        context: &Context,
        fields: &FieldsConfig,
        options: Map<String, Value>,
        top: usize,
        highlight: bool,
    ) -> Result<Response, Error> {
        let span = tracing::debug_span!("engine.multi_search", top, highlight);
        let _enter = span.enter();
        let resolved = resolve_options(
            &self.default_options,
            &options,
            self.score_threshold,
            self.rrf_k,
        )?;
        let vector_backend = self.vector.as_deref().ok_or(Error::NoVectorBackend)?;
        let (text_hits, vector_hits) = match tokio::try_join!(
            text_search_inner(&*self.storage, index, &search_query, highlight),
            vector_search_inner(vector_backend, index, &search_query),
        ) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(stage = "multi_search", error = %e, "engine.failed");
                return Err(e);
            }
        };
        let weights = FusionWeights {
            text: resolved.fusion.weights.text,
            vector: resolved.fusion.weights.vector,
        };
        let hits = fuse_by_rrf(text_hits, vector_hits, &weights, resolved.rrf_k_value());
        tracing::debug!(
            stage = "multi_search",
            has_vector = true,
            rrf_enabled = true,
            n = hits.len(),
            "engine.multi_search.done"
        );
        post_process(self, &resolved, context, fields, hits, top).await
    }
}

// ─── private helpers ─────────────────────────────────────────────────────────

async fn text_search_inner(
    backend: &dyn storage::Base,
    index: &str,
    search_query: &Value,
    _highlight: bool,
) -> Result<Vec<Hit>, Error> {
    let resp = backend.search(index, search_query.clone()).await?;
    parse_es_hits(&resp)
}

async fn vector_search_inner(
    backend: &dyn storage::Base,
    index: &str,
    search_query: &Value,
) -> Result<Vec<Hit>, Error> {
    let resp = backend.search(index, search_query.clone()).await?;
    parse_es_hits(&resp)
}

fn parse_es_hits(resp: &Value) -> Result<Vec<Hit>, Error> {
    let hits_arr = resp
        .get("hits")
        .and_then(|h| h.get("hits"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut out: Vec<Hit> = Vec::with_capacity(hits_arr.len());
    for raw in &hits_arr {
        out.push(parse_one_hit(raw)?);
    }
    Ok(out)
}

fn parse_one_hit(raw: &Value) -> Result<Hit, Error> {
    let source = raw
        .get("_source")
        .ok_or_else(|| Error::BadResponse("missing _source in hits.hits[]".into()))?;
    let pagewiki: crate::index::pagewiki::PageWiki = serde_json::from_value(source.clone())?;
    let score = raw
        .get("_score")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(0.0);
    let highlight = raw.get("highlight").and_then(|v| v.as_object()).cloned();
    Ok(Hit {
        pagewiki,
        score,
        scores: HashMap::new(),
        highlight,
    })
}

fn annotate_stage(hits: &mut [Hit]) {
    for h in hits.iter_mut() {
        let s = h.score;
        h.scores.insert("stage".into(), s);
    }
}

async fn post_process(
    engine: &Engine,
    resolved: &ResolvedOptions,
    context: &Context,
    _fields: &FieldsConfig,
    hits: Vec<Hit>,
    top: usize,
) -> Result<Response, Error> {
    let hits = match (&engine.reranker, resolved.rerank.enabled) {
        (Some(r), true) => {
            let weights = resolved.rerank.weights.clone();
            r.rerank(&context.query, hits, weights).await?
        }
        _ => hits,
    };
    let before = hits.len();
    let threshold = resolved.score_threshold_value();
    let disable = resolved.filter.disable_score_threshold;
    let top_hits = filter_hits(hits, threshold, disable, top);
    let after_top = top_hits.len();
    tracing::debug!(
        stage = "filter",
        before,
        after_top,
        threshold,
        top,
        "engine.filter_hits.done"
    );
    let total = top_hits.len();
    let page_hits = paginate_hits(
        top_hits,
        resolved.pagination.page_num,
        resolved.pagination.page_size,
    );
    let returned = page_hits.len();
    tracing::debug!(
        stage = "paginate",
        total,
        returned,
        "engine.paginate_hits.done"
    );
    Ok(build_response(total, page_hits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validate_keys_accepts_all_five() {
        let mut m = Map::new();
        m.insert("fusion".into(), json!({}));
        m.insert("rerank".into(), json!({}));
        m.insert("filter".into(), json!({}));
        m.insert("pagination".into(), json!({}));
        m.insert("trace".into(), json!({}));
        assert!(validate_option_keys(&m).is_ok());
    }

    #[test]
    fn validate_keys_rejects_top() {
        let mut m = Map::new();
        m.insert("top".into(), json!(100));
        m.insert("fusion".into(), json!({}));
        match validate_option_keys(&m) {
            Err(Error::UnknownOption(k)) => assert_eq!(k, "top"),
            _ => panic!("expected UnknownOption"),
        }
    }

    #[test]
    fn validate_keys_rejects_tenant_id_and_foo() {
        for bad in ["tenant_id", "foo", "bar", "options"] {
            let mut m = Map::new();
            m.insert(bad.into(), json!("x"));
            assert!(validate_option_keys(&m).is_err(), "should reject `{bad}`");
        }
    }

    #[test]
    fn deep_merge_recurses_into_maps() {
        let mut d = Map::new();
        d.insert("fusion".into(), json!({"rrf_k": 60}));
        let mut o = Map::new();
        o.insert("fusion".into(), json!({"weights": {"text": 1}}));
        let merged = deep_merge(d, &o);
        assert_eq!(merged["fusion"]["rrf_k"], json!(60));
        assert_eq!(merged["fusion"]["weights"]["text"], json!(1));
    }

    #[test]
    fn deep_merge_overrides_scalar_with_options() {
        let mut d = Map::new();
        d.insert("rerank".into(), json!({"enabled": false}));
        let mut o = Map::new();
        o.insert("rerank".into(), json!({"enabled": true}));
        let merged = deep_merge(d, &o);
        assert_eq!(merged["rerank"]["enabled"], json!(true));
    }

    #[test]
    fn resolve_fills_default_rrf_k_and_threshold() {
        let d = Map::new();
        let o = Map::new();
        let r = resolve_options(&d, &o, 0.42, 80).unwrap();
        assert_eq!(r.rrf_k_value(), 80);
        assert!((r.score_threshold_value() - 0.42).abs() < 1e-6);
    }

    #[test]
    fn resolve_options_explicit_overrides_fallback() {
        let d = Map::new();
        let mut o = Map::new();
        o.insert("fusion".into(), json!({"rrf_k": 30}));
        o.insert("filter".into(), json!({"score_threshold": 0.9}));
        let r = resolve_options(&d, &o, 0.42, 80).unwrap();
        assert_eq!(r.rrf_k_value(), 30);
        assert!((r.score_threshold_value() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn resolve_propagates_unknown_default_option() {
        let mut d = Map::new();
        d.insert("top".into(), json!(100));
        let o = Map::new();
        let err = resolve_options(&d, &o, 0.0, 60).unwrap_err();
        assert!(matches!(err, Error::UnknownOption(_)));
    }
}
