//! `engine` 公共类型：错误枚举与数据模型。
//!
//! 与 spec `Hit / Context / FieldsConfig / Response / ResolvedOptions` 对齐。

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::index::pagewiki::PageWiki;
use crate::index::storage;

/// `engine::Error`：本模块统一错误类型。
///
/// 严格 8 个 variant；不含 `Other` / `Internal` / `Unknown` 兜底，
/// 不直接持有 `reqwest::Error` / `elasticsearch::Error`。
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("unknown option key: {0}")]
    UnknownOption(String),

    #[error("invalid option: field={field} reason={reason}")]
    InvalidOption { field: String, reason: String },

    #[error("no vector backend configured")]
    NoVectorBackend,

    #[error("missing context.query")]
    MissingContextQuery,

    #[error("backend error: {0}")]
    Backend(#[from] storage::Error),

    #[error("reranker error: {0}")]
    Reranker(String),

    #[error("serialize error: {0}")]
    Serialize(#[from] serde_json::Error),

    #[error("bad ES response: {0}")]
    BadResponse(String),
}

/// 一条单路 / 融合 / rerank 后的检索命中。
///
/// `Hit` 字段恰好 4 个：`pagewiki / score / scores / highlight`。**不**含 `raw_source` / `source` /
/// `extra` 兜底字段；`PageWiki` 顶层 19 字段已严格枚举，配合 `mapping.dynamic = "false"` 直接
/// 反序列化即可。
#[derive(Debug, Clone)]
pub struct Hit {
    /// 反序列化自 `_source` 的 PageWiki。
    pub pagewiki: PageWiki,
    /// 当前阶段最终用于排序的分数。
    pub score: f32,
    /// 分项分数：`text / text_rank / vector / vector_rank / rrf / stage / rerank / model / rank_feature` 等。
    pub scores: HashMap<String, f32>,
    /// ES 返回的 highlight 段，原样保留；业务展示自取。
    pub highlight: Option<Map<String, Value>>,
}

/// 业务上游传入的查询上下文。
///
/// `Engine` 不会修改 `Context`，只在 rerank 阶段把 `query` 字段传给 [`crate::index::engine::Reranker`]。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Context {
    /// 原始 query（未规范化）。
    #[serde(default)]
    pub raw_query: String,
    /// 业务规范化后的 query；reranker 读取这个字段。
    #[serde(default)]
    pub query: String,
    /// 业务自拼的 expression / DSL 子串。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expr: Option<String>,
    /// 关键词分组（业务自分；`LocalReranker` 取这里做命中覆盖率打分）。
    #[serde(default)]
    pub keywords: Vec<KeywordGroup>,
    /// 意图识别结果。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<Intent>,
    /// 业务自定义扩展。
    #[serde(default)]
    pub extra: Map<String, Value>,
}

/// 关键词组。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeywordGroup {
    /// 组名（业务自由约定）。
    pub name: String,
    /// 关键词清单。
    pub terms: Vec<String>,
}

/// 意图识别结果。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Intent {
    /// 意图类型；用 `kind` 字段名避开 Rust 关键字 `type`。
    #[serde(rename = "type", default)]
    pub kind: String,
    /// 模型给出的置信度。
    #[serde(default)]
    pub confidence: f32,
}

/// 一次基础召回的字段配置。
///
/// 业务用它告诉 `LocalReranker` 哪些 PageWiki 字段拼成 token 池、向量字段是哪个、查询向量是什么。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FieldsConfig {
    /// 文本召回字段及权重（业务侧拼 SearchQuery 时常用，也可供 LocalReranker 使用）。
    #[serde(default)]
    pub text: Vec<TextField>,
    /// 向量字段名。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_field: Option<String>,
    /// LocalReranker 计算 text_score 时取用的 PageWiki token 字段及权重。
    #[serde(default)]
    pub rerank_tokens: Vec<TextField>,
    /// 查询向量；business 层先算好；engine 不做 query embedding。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_vector: Option<Vec<f32>>,
}

/// 字段 + 权重组合。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextField {
    /// PageWiki 顶层字段名（如 `content_tokens` / `keywords` / `question_tokens`）。
    pub field: String,
    /// 权重（默认 1.0）。
    #[serde(default = "one_f32")]
    pub weight: f32,
}

fn one_f32() -> f32 {
    1.0
}

/// 召回入口的最终返回。
///
/// 不 derive `Serialize / Deserialize`：`Hit` 内含 `HashMap<String, f32>` 与 `f32`，
/// 没有自然的反序列化语义；上游若要序列化输出，请在业务层显式投影。
#[derive(Debug, Clone, Default)]
pub struct Response {
    /// `filter_hits` 之后的命中总数（不是底层 raw 召回总数）。
    pub total: usize,
    /// 当前页 hits。
    pub hits: Vec<Hit>,
    /// 当前页 hits 按 doc_id 聚合（保持出现顺序）。
    pub doc_aggs: Vec<DocAgg>,
}

/// 单个 doc 聚合统计。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocAgg {
    /// doc_id；缺省取空串。
    pub doc_id: String,
    /// 当前页内的命中条数。
    pub count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// 因为 Hit 内含 HashMap + f32，没有自然的 PartialEq / Default。我们手写一个简化的
// `Default`（用于 ResolvedOptions 反序列化时 hits 字段并不出现，但 Response 用到）。
// ─────────────────────────────────────────────────────────────────────────────

// `Hit` 不 derive PartialEq；`Response.hits: Vec<Hit>` 因此也不能直接 derive PartialEq。
// 所以 Response 没有 `PartialEq` 派生（与 spec 对齐）。

// ResolvedOptions（pub(crate)，仅 engine 内部使用）

/// 解析后的 `options`（typed）。`pub(crate)` 只在 engine 模块内部流转。
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ResolvedOptions {
    #[serde(default)]
    pub fusion: FusionOpts,
    #[serde(default)]
    pub rerank: RerankOpts,
    #[serde(default)]
    pub filter: FilterOpts,
    #[serde(default)]
    pub pagination: PaginationOpts,
    #[serde(default)]
    #[allow(dead_code)] // reserved: 业务侧透传 trace_id，engine 不消费
    pub trace: TraceOpts,
}

impl ResolvedOptions {
    pub(crate) fn fill_defaults(&mut self, score_threshold_fallback: f32, rrf_k_fallback: u32) {
        if self.fusion.rrf_k.is_none() {
            self.fusion.rrf_k = Some(rrf_k_fallback);
        }
        if self.filter.score_threshold.is_none() {
            self.filter.score_threshold = Some(score_threshold_fallback);
        }
    }

    pub(crate) fn rrf_k_value(&self) -> u32 {
        self.fusion.rrf_k.unwrap_or(60)
    }

    pub(crate) fn score_threshold_value(&self) -> f32 {
        self.filter.score_threshold.unwrap_or(0.0)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FusionOpts {
    /// 缺失时由 `Engine.rrf_k` 填回。
    #[serde(default)]
    pub rrf_k: Option<u32>,
    /// 文本/向量两路权重；缺省 1.0 / 1.0。
    #[serde(default)]
    pub weights: FusionWeightsOpt,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FusionWeightsOpt {
    #[serde(default = "one_f32")]
    pub text: f32,
    #[serde(default = "one_f32")]
    pub vector: f32,
}

impl Default for FusionWeightsOpt {
    fn default() -> Self {
        Self {
            text: 1.0,
            vector: 1.0,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RerankOpts {
    /// 是否调用 reranker（仅 `Engine.reranker.is_some()` 时生效）。
    #[serde(default)]
    pub enabled: bool,
    /// 透传给 reranker 的权重表。
    #[serde(default)]
    pub weights: HashMap<String, f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FilterOpts {
    /// 缺失时由 `Engine.score_threshold` 填回。
    #[serde(default)]
    pub score_threshold: Option<f32>,
    /// `true` → 不做阈值过滤。
    #[serde(default)]
    pub disable_score_threshold: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PaginationOpts {
    /// 1-based。`0` 视为越界（返回空 Vec）。
    #[serde(default = "default_page_num")]
    pub page_num: usize,
    /// 默认 10。
    #[serde(default = "default_page_size")]
    pub page_size: usize,
}

impl Default for PaginationOpts {
    fn default() -> Self {
        Self {
            page_num: 1,
            page_size: 10,
        }
    }
}

fn default_page_num() -> usize {
    1
}

fn default_page_size() -> usize {
    10
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TraceOpts {
    /// 业务自定义 trace_id；engine 不消费、只透传给日志。
    #[serde(default)]
    #[allow(dead_code)]
    pub id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_unknown_option() {
        let e = Error::UnknownOption("top".into());
        assert!(e.to_string().contains("unknown option"));
    }

    #[test]
    fn display_no_vector_backend() {
        let e = Error::NoVectorBackend;
        assert!(e.to_string().contains("no vector backend"));
    }

    #[test]
    fn display_reranker() {
        let e = Error::Reranker("boom".into());
        assert!(e.to_string().contains("reranker error"));
    }

    #[test]
    fn display_bad_response() {
        let e = Error::BadResponse("missing _source".into());
        assert!(e.to_string().contains("bad ES response"));
    }

    #[test]
    fn display_invalid_option_includes_field() {
        let e = Error::InvalidOption {
            field: "fusion.rrf_k".into(),
            reason: "negative".into(),
        };
        assert!(e.to_string().contains("invalid option"));
        assert!(e.to_string().contains("fusion.rrf_k"));
    }

    #[test]
    fn from_storage_error_into_backend_variant() {
        let se = storage::Error::Transport("oops".into());
        let e: Error = se.into();
        assert!(matches!(e, Error::Backend(_)));
    }

    #[test]
    fn from_serde_error_into_serialize_variant() {
        let parse_err = serde_json::from_str::<i32>("not int").unwrap_err();
        let e: Error = parse_err.into();
        assert!(matches!(e, Error::Serialize(_)));
    }

    #[test]
    fn intent_uses_type_alias() {
        let v = serde_json::json!({"type": "qa", "confidence": 0.9});
        let i: Intent = serde_json::from_value(v).unwrap();
        assert_eq!(i.kind, "qa");
        assert!((i.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn fields_config_default_is_all_empty() {
        let f = FieldsConfig::default();
        assert!(f.text.is_empty());
        assert!(f.vector_field.is_none());
        assert!(f.rerank_tokens.is_empty());
        assert!(f.query_vector.is_none());
    }

    #[test]
    fn resolved_options_fill_defaults_round_trip() {
        let mut r = ResolvedOptions::default();
        r.fill_defaults(0.42, 80);
        assert_eq!(r.fusion.rrf_k, Some(80));
        assert_eq!(r.filter.score_threshold, Some(0.42));
        assert_eq!(r.rrf_k_value(), 80);
        assert!((r.score_threshold_value() - 0.42).abs() < 1e-6);
    }

    #[test]
    fn fusion_weights_opt_default_one_one() {
        let w = FusionWeightsOpt::default();
        assert!((w.text - 1.0).abs() < 1e-6);
        assert!((w.vector - 1.0).abs() < 1e-6);
    }
}
