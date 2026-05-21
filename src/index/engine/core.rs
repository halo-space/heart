//! `engine::core`：`Engine` 结构 + 4 个召回入口 + 共享后处理 +
//! 内联的融合 / 过滤 / 分页 / 响应组装工具。
//!
//! 4 个召回入口共享同一后处理链：
//!
//! ```text
//! Engine::resolve_options
//!   -> (text|vector|hybrid|multi) inner
//!   -> Engine::annotate_stage / Engine::fuse_by_rrf
//!   -> rerank (optional)
//!   -> Engine::filter_hits
//!   -> Engine::paginate_hits
//!   -> Engine::build_response
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::index::storage;

use super::rerank::Reranker;
use super::types::{Context, DocAgg, Error, FieldsConfig, Hit, ResolvedOptions, Response};

const ALLOWED_OPTION_KEYS: &[&str] = &["fusion", "rerank", "filter", "pagination", "trace"];

// ─── fusion weights ──────────────────────────────────────────────────────────

/// 文本路 / 向量路融合权重；缺省 1.0 / 1.0。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionWeights {
    pub text: f32,
    pub vector: f32,
}

impl Default for FusionWeights {
    fn default() -> Self {
        Self {
            text: 1.0,
            vector: 1.0,
        }
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

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
        Self::validate_option_keys(&default_options)?;
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

    // ─── options ─────────────────────────────────────────────────────────

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
                    let merged = Self::deep_merge(std::mem::take(d_inner), o_inner);
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
        Self::validate_option_keys(default_options)?;
        Self::validate_option_keys(options)?;
        let merged = Self::deep_merge(default_options.clone(), options);
        let mut resolved: ResolvedOptions = serde_json::from_value(Value::Object(merged))?;
        resolved.fill_defaults(score_threshold_fallback, rrf_k_fallback);
        Ok(resolved)
    }

    // ─── fusion ──────────────────────────────────────────────────────────

    /// 把两路单路 hits 用 RRF 融合为一路。
    ///
    /// 算法（与 spec 一致）：
    ///
    /// 1. 按 `pagewiki.id` 唯一化两路（同 id 的 hit 文本路 / 向量路只占一份；以文本路 hit 为基础合并）。
    /// 2. 文本路 hits 按入参顺序赋 `text_rank`（1-based），同时 `scores.text = 原 score`。
    /// 3. 向量路同理 `vector_rank` / `scores.vector`。
    /// 4. `rrf = weights.text/(rrf_k + text_rank) + weights.vector/(rrf_k + vector_rank)`；
    ///    缺失路贡献 0。
    /// 5. 写 `scores.rrf = rrf` / `scores.stage = rrf` / `hit.score = rrf`；按 `score` 倒序返回。
    ///
    /// `pagewiki.id == None` 视为程序约束错误：Builder 写入前一律分配 UUID v4，运行期到这里
    /// 不应该再出现 `None`，所以用 `expect` 直接 panic。
    pub fn fuse_by_rrf(
        text_hits: Vec<Hit>,
        vector_hits: Vec<Hit>,
        weights: &FusionWeights,
        rrf_k: u32,
    ) -> Vec<Hit> {
        let rrf_k_f = rrf_k as f32;

        let mut by_id: HashMap<String, Hit> =
            HashMap::with_capacity(text_hits.len() + vector_hits.len());
        let mut order: Vec<String> = Vec::with_capacity(text_hits.len() + vector_hits.len());

        for (idx, mut hit) in text_hits.into_iter().enumerate() {
            let id = hit
                .pagewiki
                .id
                .clone()
                .expect("pagewiki.id must be Some at engine layer (Builder ensures UUID v4)");
            let rank = (idx + 1) as f32;
            let text_score = hit.score;
            hit.scores.insert("text".into(), text_score);
            hit.scores.insert("text_rank".into(), rank);
            by_id.insert(id.clone(), hit);
            order.push(id);
        }

        for (idx, hit) in vector_hits.into_iter().enumerate() {
            let id = hit
                .pagewiki
                .id
                .clone()
                .expect("pagewiki.id must be Some at engine layer (Builder ensures UUID v4)");
            let rank = (idx + 1) as f32;
            let vector_score = hit.score;
            if let Some(existing) = by_id.get_mut(&id) {
                existing.scores.insert("vector".into(), vector_score);
                existing.scores.insert("vector_rank".into(), rank);
            } else {
                let mut h = hit;
                h.scores.insert("vector".into(), vector_score);
                h.scores.insert("vector_rank".into(), rank);
                by_id.insert(id.clone(), h);
                order.push(id);
            }
        }

        let mut out: Vec<Hit> = Vec::with_capacity(order.len());
        for id in order {
            let mut h = by_id.remove(&id).expect("entry inserted above");
            let text_rank = h.scores.get("text_rank").copied();
            let vector_rank = h.scores.get("vector_rank").copied();
            let mut rrf = 0.0_f32;
            if let Some(r) = text_rank {
                rrf += weights.text / (rrf_k_f + r);
            }
            if let Some(r) = vector_rank {
                rrf += weights.vector / (rrf_k_f + r);
            }
            h.scores.insert("rrf".into(), rrf);
            h.scores.insert("stage".into(), rrf);
            h.score = rrf;
            out.push(h);
        }

        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    // ─── filter ──────────────────────────────────────────────────────────

    /// 按 score 倒序排序、按阈值过滤、最后取前 `top` 条。
    ///
    /// 1. 先按 `score` 倒序排序（防御性：rerank / fusion 后可能已倒序，再排一次保证）。
    /// 2. `disable_score_threshold == false` → 仅保留 `score >= score_threshold`；
    ///    `true` → 跳过阈值过滤。
    /// 3. 取前 `top` 条返回；`hits.len() < top` 时返回全部。
    pub fn filter_hits(
        mut hits: Vec<Hit>,
        score_threshold: f32,
        disable_score_threshold: bool,
        top: usize,
    ) -> Vec<Hit> {
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if !disable_score_threshold {
            hits.retain(|h| h.score >= score_threshold);
        }
        if hits.len() > top {
            hits.truncate(top);
        }
        hits
    }

    // ─── paginate ────────────────────────────────────────────────────────

    /// 1-based 分页：`start = (page_num - 1) * page_size`。
    ///
    /// 越界（`page_num == 0` 或 `start >= top_hits.len()`）返回 `Vec::new()`，**不**报错、**不** panic。
    pub fn paginate_hits(top_hits: Vec<Hit>, page_num: usize, page_size: usize) -> Vec<Hit> {
        if page_num == 0 || page_size == 0 {
            return Vec::new();
        }
        let start = (page_num - 1) * page_size;
        if start >= top_hits.len() {
            return Vec::new();
        }
        let end = (start + page_size).min(top_hits.len());
        top_hits[start..end].to_vec()
    }

    // ─── response ────────────────────────────────────────────────────────

    /// 把分页后的 hits 与 filter 后的总数组装成 [`Response`]。
    ///
    /// `total` 由调用方传入（spec：等于 `filter` 后 `paginate` 前的 hits 总数）。
    /// `doc_aggs` 按 `page_hits[*].pagewiki.doc_id`（缺省取空串）group_by count，保持首次出现顺序。
    pub fn build_response(total: usize, page_hits: Vec<Hit>) -> Response {
        let mut order: Vec<String> = Vec::new();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for h in &page_hits {
            let doc_id = h.pagewiki.doc_id.clone().unwrap_or_default();
            if !counts.contains_key(&doc_id) {
                order.push(doc_id.clone());
            }
            *counts.entry(doc_id).or_insert(0) += 1;
        }
        let doc_aggs = order
            .into_iter()
            .map(|doc_id| {
                let count = counts.get(&doc_id).copied().unwrap_or(0);
                DocAgg { doc_id, count }
            })
            .collect();
        Response {
            total,
            hits: page_hits,
            doc_aggs,
        }
    }

    // ─── search entries ──────────────────────────────────────────────────

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
        let resolved = Self::resolve_options(
            &self.default_options,
            &options,
            self.score_threshold,
            self.rrf_k,
        )?;
        let resp = match self.storage.search(index, search_query.clone()).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(stage = "text_search", error = %e, "engine.failed");
                return Err(e.into());
            }
        };
        let mut hits = Self::parse_hits(&resp)?;
        Self::annotate_stage(&mut hits);
        tracing::debug!(
            stage = "text_search",
            n = hits.len(),
            "engine.text_search.done"
        );
        self.process(&resolved, context, fields, hits, top).await
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
        let resolved = Self::resolve_options(
            &self.default_options,
            &options,
            self.score_threshold,
            self.rrf_k,
        )?;
        let backend = self.vector.as_deref().ok_or(Error::NoVectorBackend)?;
        let resp = match backend.search(index, search_query.clone()).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(stage = "vector_search", error = %e, "engine.failed");
                return Err(e.into());
            }
        };
        let mut hits = Self::parse_hits(&resp)?;
        Self::annotate_stage(&mut hits);
        tracing::debug!(
            stage = "vector_search",
            n = hits.len(),
            "engine.vector_search.done"
        );
        self.process(&resolved, context, fields, hits, top).await
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
        let resolved = Self::resolve_options(
            &self.default_options,
            &options,
            self.score_threshold,
            self.rrf_k,
        )?;
        let resp = match self.storage.search(index, search_query.clone()).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(stage = "hybrid_search", error = %e, "engine.failed");
                return Err(e.into());
            }
        };
        let mut hits = Self::parse_hits(&resp)?;
        Self::annotate_stage(&mut hits);
        tracing::debug!(
            stage = "hybrid_search",
            n = hits.len(),
            "engine.hybrid_search.done"
        );
        self.process(&resolved, context, fields, hits, top).await
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
        let resolved = Self::resolve_options(
            &self.default_options,
            &options,
            self.score_threshold,
            self.rrf_k,
        )?;
        let vector_backend = self.vector.as_deref().ok_or(Error::NoVectorBackend)?;
        let (text_resp, vector_resp) = match tokio::try_join!(
            self.storage.search(index, search_query.clone()),
            vector_backend.search(index, search_query.clone()),
        ) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(stage = "multi_search", error = %e, "engine.failed");
                return Err(e.into());
            }
        };
        let text_hits = Self::parse_hits(&text_resp)?;
        let vector_hits = Self::parse_hits(&vector_resp)?;
        let weights = FusionWeights {
            text: resolved.fusion.weights.text,
            vector: resolved.fusion.weights.vector,
        };
        let hits = Self::fuse_by_rrf(text_hits, vector_hits, &weights, resolved.rrf_k_value());
        tracing::debug!(
            stage = "multi_search",
            has_vector = true,
            rrf_enabled = true,
            n = hits.len(),
            "engine.multi_search.done"
        );
        self.process(&resolved, context, fields, hits, top).await
    }

    // ─── shared post-processing ──────────────────────────────────────────

    fn annotate_stage(hits: &mut [Hit]) {
        for h in hits.iter_mut() {
            let s = h.score;
            h.scores.insert("stage".into(), s);
        }
    }

    async fn process(
        &self,
        resolved: &ResolvedOptions,
        context: &Context,
        _fields: &FieldsConfig,
        hits: Vec<Hit>,
        top: usize,
    ) -> Result<Response, Error> {
        let hits = match (&self.reranker, resolved.rerank.enabled) {
            (Some(r), true) => {
                let weights = resolved.rerank.weights.clone();
                r.rerank(&context.query, hits, weights).await?
            }
            _ => hits,
        };
        let before = hits.len();
        let threshold = resolved.score_threshold_value();
        let disable = resolved.filter.disable_score_threshold;
        let top_hits = Self::filter_hits(hits, threshold, disable, top);
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
        let page_hits = Self::paginate_hits(
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
        Ok(Self::build_response(total, page_hits))
    }

    // ─── parse ───────────────────────────────────────────────────────────

    /// 把 backend 返回的 ES 风格响应解析为 `Vec<Hit>`。
    ///
    /// 读取 `resp["hits"]["hits"][*]`，每条要求 `_source` 字段（缺失 → `Error::BadResponse`），
    /// 解析成 `PageWiki`；`_score` 缺省 0.0；`highlight` 可选。
    fn parse_hits(resp: &Value) -> Result<Vec<Hit>, Error> {
        let hits_arr = resp
            .get("hits")
            .and_then(|h| h.get("hits"))
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut out: Vec<Hit> = Vec::with_capacity(hits_arr.len());
        for raw in &hits_arr {
            let source = raw
                .get("_source")
                .ok_or_else(|| Error::BadResponse("missing _source in hits.hits[]".into()))?;
            let pagewiki: crate::index::pagewiki::PageWiki =
                serde_json::from_value(source.clone())?;
            let score = raw
                .get("_score")
                .and_then(|v| v.as_f64())
                .map(|f| f as f32)
                .unwrap_or(0.0);
            let highlight = raw.get("highlight").and_then(|v| v.as_object()).cloned();
            out.push(Hit {
                pagewiki,
                score,
                scores: HashMap::new(),
                highlight,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::pagewiki::PageWiki;
    use serde_json::json;

    fn hit(id: &str, score: f32) -> Hit {
        Hit {
            pagewiki: PageWiki {
                id: Some(id.into()),
                ..Default::default()
            },
            score,
            scores: HashMap::new(),
            highlight: None,
        }
    }

    // ── resolve_options / validate_option_keys / deep_merge ────────────────

    #[test]
    fn validate_keys_accepts_all_five() {
        let mut m = Map::new();
        m.insert("fusion".into(), json!({}));
        m.insert("rerank".into(), json!({}));
        m.insert("filter".into(), json!({}));
        m.insert("pagination".into(), json!({}));
        m.insert("trace".into(), json!({}));
        assert!(Engine::validate_option_keys(&m).is_ok());
    }

    #[test]
    fn validate_keys_rejects_top() {
        let mut m = Map::new();
        m.insert("top".into(), json!(100));
        m.insert("fusion".into(), json!({}));
        match Engine::validate_option_keys(&m) {
            Err(Error::UnknownOption(k)) => assert_eq!(k, "top"),
            _ => panic!("expected UnknownOption"),
        }
    }

    #[test]
    fn validate_keys_rejects_tenant_id_and_foo() {
        for bad in ["tenant_id", "foo", "bar", "options"] {
            let mut m = Map::new();
            m.insert(bad.into(), json!("x"));
            assert!(
                Engine::validate_option_keys(&m).is_err(),
                "should reject `{bad}`"
            );
        }
    }

    #[test]
    fn deep_merge_recurses_into_maps() {
        let mut d = Map::new();
        d.insert("fusion".into(), json!({"rrf_k": 60}));
        let mut o = Map::new();
        o.insert("fusion".into(), json!({"weights": {"text": 1}}));
        let merged = Engine::deep_merge(d, &o);
        assert_eq!(merged["fusion"]["rrf_k"], json!(60));
        assert_eq!(merged["fusion"]["weights"]["text"], json!(1));
    }

    #[test]
    fn deep_merge_overrides_scalar_with_options() {
        let mut d = Map::new();
        d.insert("rerank".into(), json!({"enabled": false}));
        let mut o = Map::new();
        o.insert("rerank".into(), json!({"enabled": true}));
        let merged = Engine::deep_merge(d, &o);
        assert_eq!(merged["rerank"]["enabled"], json!(true));
    }

    #[test]
    fn resolve_fills_default_rrf_k_and_threshold() {
        let d = Map::new();
        let o = Map::new();
        let r = Engine::resolve_options(&d, &o, 0.42, 80).unwrap();
        assert_eq!(r.rrf_k_value(), 80);
        assert!((r.score_threshold_value() - 0.42).abs() < 1e-6);
    }

    #[test]
    fn resolve_options_explicit_overrides_fallback() {
        let d = Map::new();
        let mut o = Map::new();
        o.insert("fusion".into(), json!({"rrf_k": 30}));
        o.insert("filter".into(), json!({"score_threshold": 0.9}));
        let r = Engine::resolve_options(&d, &o, 0.42, 80).unwrap();
        assert_eq!(r.rrf_k_value(), 30);
        assert!((r.score_threshold_value() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn resolve_propagates_unknown_default_option() {
        let mut d = Map::new();
        d.insert("top".into(), json!(100));
        let o = Map::new();
        let err = Engine::resolve_options(&d, &o, 0.0, 60).unwrap_err();
        assert!(matches!(err, Error::UnknownOption(_)));
    }

    // ── fuse_by_rrf ────────────────────────────────────────────────────────

    fn rrf_hit(id: &str, score: f32) -> Hit {
        Hit {
            pagewiki: PageWiki {
                id: Some(id.to_string()),
                content: "x".into(),
                ..Default::default()
            },
            score,
            scores: HashMap::new(),
            highlight: None,
        }
    }

    #[test]
    fn rrf_both_paths_full_overlap_three_each() {
        let t = vec![rrf_hit("a", 1.0), rrf_hit("b", 0.9), rrf_hit("c", 0.8)];
        let v = vec![rrf_hit("a", 0.95), rrf_hit("b", 0.92), rrf_hit("c", 0.7)];
        let rrf_k = 60;
        let out = Engine::fuse_by_rrf(t, v, &FusionWeights::default(), rrf_k);
        assert_eq!(out.len(), 3);
        for h in &out {
            assert!(h.scores.contains_key("rrf"));
            assert!(h.scores.contains_key("stage"));
            assert!(h.scores.contains_key("text"));
            assert!(h.scores.contains_key("vector"));
            assert!(h.scores.contains_key("text_rank"));
            assert!(h.scores.contains_key("vector_rank"));
            let tr = h.scores["text_rank"];
            let vr = h.scores["vector_rank"];
            let expected = 1.0 / (rrf_k as f32 + tr) + 1.0 / (rrf_k as f32 + vr);
            assert!((h.scores["rrf"] - expected).abs() < 1e-6);
            assert!((h.score - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn rrf_missing_route_contributes_zero() {
        let t = vec![rrf_hit("a", 1.0)];
        let v: Vec<Hit> = vec![];
        let out = Engine::fuse_by_rrf(t, v, &FusionWeights::default(), 60);
        assert_eq!(out.len(), 1);
        let h = &out[0];
        assert!(!h.scores.contains_key("vector"));
        assert!(!h.scores.contains_key("vector_rank"));
        let expected = 1.0 / (60.0 + 1.0);
        assert!((h.scores["rrf"] - expected).abs() < 1e-6);
    }

    #[test]
    fn rrf_output_sorted_desc_by_score() {
        let t = vec![rrf_hit("a", 1.0), rrf_hit("b", 0.9)];
        let v = vec![rrf_hit("c", 0.8)];
        let out = Engine::fuse_by_rrf(t, v, &FusionWeights::default(), 60);
        for w in out.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn rrf_disjoint_paths_dedup_correctly() {
        let t = vec![rrf_hit("a", 1.0)];
        let v = vec![rrf_hit("b", 0.5)];
        let out = Engine::fuse_by_rrf(t, v, &FusionWeights::default(), 60);
        let ids: Vec<&str> = out
            .iter()
            .map(|h| h.pagewiki.id.as_deref().unwrap())
            .collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn rrf_weights_apply() {
        let t = vec![rrf_hit("a", 1.0)];
        let v = vec![rrf_hit("a", 1.0)];
        let w = FusionWeights {
            text: 2.0,
            vector: 0.5,
        };
        let out = Engine::fuse_by_rrf(t, v, &w, 60);
        let expected = 2.0 / 61.0 + 0.5 / 61.0;
        assert!((out[0].scores["rrf"] - expected).abs() < 1e-6);
    }

    // ── filter_hits ────────────────────────────────────────────────────────

    #[test]
    fn filter_threshold_filters_below_cutoff() {
        let hits = (0..10)
            .map(|i| hit(&format!("h{i}"), 0.1 * (i as f32 + 1.0)))
            .collect::<Vec<_>>();
        let out = Engine::filter_hits(hits, 0.5, false, 100);
        assert!(out.iter().all(|h| h.score >= 0.5));
    }

    #[test]
    fn filter_top_truncates_after_threshold() {
        let hits = (0..10)
            .map(|i| hit(&format!("h{i}"), 0.5 + 0.05 * i as f32))
            .collect::<Vec<_>>();
        let out = Engine::filter_hits(hits, 0.5, false, 3);
        assert_eq!(out.len(), 3);
        for w in out.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn filter_disable_threshold_skips_filter() {
        let hits = (0..10)
            .map(|i| hit(&format!("h{i}"), 0.1 * i as f32))
            .collect::<Vec<_>>();
        let out = Engine::filter_hits(hits, 0.9, true, 100);
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn filter_fewer_than_top_returns_all() {
        let hits = (0..3)
            .map(|i| hit(&format!("h{i}"), 0.6 + 0.1 * i as f32))
            .collect::<Vec<_>>();
        let out = Engine::filter_hits(hits, 0.5, false, 100);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn filter_sort_is_defensive() {
        let mut h = vec![hit("a", 0.3), hit("b", 0.9), hit("c", 0.6)];
        h.reverse();
        let out = Engine::filter_hits(h, 0.0, false, 10);
        assert_eq!(out[0].score, 0.9);
        assert_eq!(out[1].score, 0.6);
        assert_eq!(out[2].score, 0.3);
    }

    // ── paginate_hits ──────────────────────────────────────────────────────

    fn page_h(id: &str) -> Hit {
        Hit {
            pagewiki: PageWiki {
                id: Some(id.into()),
                ..Default::default()
            },
            score: 0.0,
            scores: HashMap::new(),
            highlight: None,
        }
    }

    #[test]
    fn paginate_within_bounds() {
        let v = (0..25)
            .map(|i| page_h(&format!("h{i}")))
            .collect::<Vec<_>>();
        let out = Engine::paginate_hits(v, 2, 10);
        assert_eq!(out.len(), 10);
        assert_eq!(out[0].pagewiki.id.as_deref(), Some("h10"));
        assert_eq!(out[9].pagewiki.id.as_deref(), Some("h19"));
    }

    #[test]
    fn paginate_last_page_partial_returns_remainder() {
        let v = (0..15)
            .map(|i| page_h(&format!("h{i}")))
            .collect::<Vec<_>>();
        let out = Engine::paginate_hits(v, 2, 10);
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn paginate_out_of_range_returns_empty() {
        let v = (0..5).map(|i| page_h(&format!("h{i}"))).collect::<Vec<_>>();
        let out = Engine::paginate_hits(v, 10, 10);
        assert!(out.is_empty());
    }

    #[test]
    fn paginate_page_num_zero_returns_empty() {
        let v = (0..5).map(|i| page_h(&format!("h{i}"))).collect::<Vec<_>>();
        let out = Engine::paginate_hits(v, 0, 10);
        assert!(out.is_empty());
    }

    #[test]
    fn paginate_page_size_zero_returns_empty() {
        let v = (0..5).map(|i| page_h(&format!("h{i}"))).collect::<Vec<_>>();
        let out = Engine::paginate_hits(v, 1, 0);
        assert!(out.is_empty());
    }

    // ── build_response ─────────────────────────────────────────────────────

    fn doc_h(doc: Option<&str>) -> Hit {
        Hit {
            pagewiki: PageWiki {
                doc_id: doc.map(String::from),
                ..Default::default()
            },
            score: 0.0,
            scores: HashMap::new(),
            highlight: None,
        }
    }

    #[test]
    fn response_doc_aggs_count_in_first_seen_order() {
        let hits = vec![
            doc_h(Some("A")),
            doc_h(Some("A")),
            doc_h(Some("B")),
            doc_h(Some("A")),
            doc_h(Some("C")),
        ];
        let r = Engine::build_response(5, hits);
        assert_eq!(
            r.doc_aggs,
            vec![
                DocAgg {
                    doc_id: "A".into(),
                    count: 3
                },
                DocAgg {
                    doc_id: "B".into(),
                    count: 1
                },
                DocAgg {
                    doc_id: "C".into(),
                    count: 1
                }
            ]
        );
    }

    #[test]
    fn response_doc_id_none_aggregated_under_empty_string() {
        let hits = vec![doc_h(None), doc_h(Some("A"))];
        let r = Engine::build_response(2, hits);
        assert_eq!(r.doc_aggs.len(), 2);
        assert_eq!(r.doc_aggs[0].doc_id, "");
        assert_eq!(r.doc_aggs[0].count, 1);
        assert_eq!(r.doc_aggs[1].doc_id, "A");
    }

    #[test]
    fn response_total_is_independent_of_page_hits_len() {
        let r = Engine::build_response(80, vec![doc_h(Some("A"))]);
        assert_eq!(r.total, 80);
        assert_eq!(r.hits.len(), 1);
    }
}
