//! `pagewiki` 模块对外数据模型：[`PageWiki`] / [`Span`] / [`Graph`] / [`Evidence`] / [`Error`]。
//!
//! 字段所有权与 `openspec/changes/rag-page-wiki/design.md` 第 3、7 节保持一致：
//! - `Base::cut` 阶段填：`header` / `content` / `keywords` / `questions` / `tags` /
//!   `attributes` / `spans` / `graph`；
//! - IndexBuilder 阶段填：`id` / `doc_id` / `version` / `scenario` / `idx` /
//!   `content_tokens` / `keyword_tokens` / `question_tokens` / `embedding` /
//!   `metadata` / `images`。
//!
//! 没有自然空值的 IndexBuilder 字段 (`id` / `doc_id` / `version` / `scenario` /
//! `idx` / `embedding`) 用 `Option<…>` 显式表达"尚未填"。

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::index::source::Scenario;

/// `pagewiki` 模块统一返回类型别名。
pub type Result<T> = std::result::Result<T, Error>;

/// RAG 链路下游统一检索单元。
///
/// 顶层字段严格限定为 19 个；任何业务自定义字段一律留在 [`PageWiki::metadata`]
/// 子对象，不通过 `#[serde(flatten)]` 暴露到 JSON 顶层。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PageWiki {
    // ── IndexBuilder 填（cut 阶段保持 None） ───────────────────────────────
    /// 全局唯一 ID（UUID v4）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// 源文档 ID（提升自 `Item.metadata.doc_id`）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_id: Option<String>,
    /// 版本（毫秒级时间戳字符串）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// 检索场景（透传 `Item.scenario`）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario: Option<Scenario>,
    /// 同一 doc 内的顺序编号。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idx: Option<usize>,

    // ── 切分阶段填（cut 必须提供） ─────────────────────────────────────────
    /// 段落标题 / LLM 抽取的小节名；规则切分默认空串。
    #[serde(default)]
    pub header: String,
    /// 切分后的正文内容。
    pub content: String,
    /// 关键词。
    #[serde(default)]
    pub keywords: Vec<String>,
    /// 可能命中本段的问题列表。
    #[serde(default)]
    pub questions: Vec<String>,
    /// 标签。
    #[serde(default)]
    pub tags: Vec<String>,
    /// 业务自定义属性。
    #[serde(default)]
    pub attributes: Map<String, Value>,
    /// 切分单元在原文中的字符坐标区间。
    #[serde(default)]
    pub spans: Vec<Span>,
    /// 图结构关联（节点类型、邻居、属性）。
    #[serde(default)]
    pub graph: Graph,

    // ── IndexBuilder 填（cut 阶段使用自然空值） ────────────────────────────
    /// `content` 分词结果（空格连接）。
    #[serde(default)]
    pub content_tokens: String,
    /// `keywords` 分词结果。
    #[serde(default)]
    pub keyword_tokens: String,
    /// `questions` 分词结果。
    #[serde(default)]
    pub question_tokens: String,
    /// 向量表示。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// 继承自 `Item.metadata` 的业务元数据。
    #[serde(default)]
    pub metadata: Map<String, Value>,
    /// 图片引用。
    #[serde(default)]
    pub images: Vec<String>,
}

/// 切分单元在原文中的字符坐标区间。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// 起始字符下标（含）。
    pub start: usize,
    /// 结束字符下标（不含）。
    pub end: usize,
    /// 原文截取（`text[start..end]`，字符口径）。
    pub original_text: String,
    /// LLM 置信度、辅助标注等自由扩展。
    #[serde(default)]
    pub extra: Map<String, Value>,
}

/// 图节点附加信息。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Graph {
    /// 节点类型。
    #[serde(default)]
    pub node_type: String,
    /// 邻居节点列表。
    #[serde(default)]
    pub neighbors: Vec<String>,
    /// 节点属性。
    #[serde(default)]
    pub properties: Map<String, Value>,
}

/// LLM 输出的证据 / 锚点。
///
/// 仅模块内（[`Semantic`](crate::index::pagewiki::Semantic) + [`resolve_spans`](crate::index::pagewiki::resolve_spans)）使用，
/// 不写入 [`PageWiki`]。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// 起始锚点文本。
    pub start_text: String,
    /// 结束锚点文本。
    pub end_text: String,
    /// 起始锚点所在行（1-based，用于缩小搜索范围）。
    pub start_line: usize,
    /// 结束锚点所在行。
    pub end_line: usize,
    /// 透传给 [`Span::extra`]；额外识别 `include_end_text: bool`。
    #[serde(default)]
    pub extra: Map<String, Value>,
}

/// `pagewiki` 模块错误类型。
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// `Semantic` LLM 输出 `content` 字符数不在合法区间。
    #[error("content length out of range: actual={actual}, expected {min}-{max}")]
    ContentLength {
        actual: usize,
        min: usize,
        max: usize,
    },
    /// `spans.rs` 反查 / 反算失败。
    #[error("span resolution failed: {0}")]
    SpanResolve(String),
    /// `Qa` JSONL 解析失败。
    #[error("qa parse error at line {line}: {reason}")]
    QaParse { line: usize, reason: String },
    /// LLM HTTP 请求失败。
    #[error("llm request failed: {0}")]
    LlmRequest(String),
    /// LLM 响应解析失败。
    #[error("llm response parse failed: {0}")]
    LlmParse(String),
    /// I/O 错误。
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// 构造参数或调用入参非法。
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pagewiki_default_has_empty_natural_values() {
        let pw = PageWiki::default();
        assert!(pw.id.is_none());
        assert!(pw.doc_id.is_none());
        assert!(pw.version.is_none());
        assert!(pw.scenario.is_none());
        assert!(pw.idx.is_none());
        assert!(pw.embedding.is_none());
        assert_eq!(pw.header, "");
        assert_eq!(pw.content, "");
        assert!(pw.keywords.is_empty());
        assert!(pw.spans.is_empty());
    }

    #[test]
    fn pagewiki_serializes_without_none_options() {
        let pw = PageWiki {
            content: "hello".into(),
            ..Default::default()
        };
        let s = serde_json::to_string(&pw).unwrap();
        // None 字段被 skip
        assert!(!s.contains("\"id\":"));
        assert!(!s.contains("\"doc_id\":"));
        assert!(!s.contains("\"embedding\":"));
        // 自然空值字段照常出现
        assert!(s.contains("\"content\":\"hello\""));
        assert!(s.contains("\"graph\""));
    }

    #[test]
    fn graph_default_serializes_to_empty_shape() {
        let g = Graph::default();
        let s = serde_json::to_string(&g).unwrap();
        assert!(s.contains("\"node_type\":\"\""));
        assert!(s.contains("\"neighbors\":[]"));
        assert!(s.contains("\"properties\":{}"));
    }

    #[test]
    fn pagewiki_roundtrip_preserves_metadata_order() {
        let mut pw = PageWiki {
            content: "x".into(),
            ..Default::default()
        };
        pw.metadata.insert("a".into(), Value::from(1));
        pw.metadata.insert("b".into(), Value::from(2));
        pw.metadata.insert("c".into(), Value::from(3));
        let s = serde_json::to_string(&pw).unwrap();
        let back: PageWiki = serde_json::from_str(&s).unwrap();
        let keys: Vec<&str> = back.metadata.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn span_extra_defaults_to_empty_map() {
        let s = Span {
            start: 0,
            end: 3,
            original_text: "abc".into(),
            extra: Map::new(),
        };
        let j = serde_json::to_string(&s).unwrap();
        assert!(j.contains("\"extra\":{}"));
        let back: Span =
            serde_json::from_str("{\"start\":0,\"end\":3,\"original_text\":\"abc\"}").unwrap();
        assert!(back.extra.is_empty());
    }

    #[test]
    fn evidence_extra_defaults_to_empty_map() {
        let ev: Evidence = serde_json::from_str(
            "{\"start_text\":\"a\",\"end_text\":\"b\",\"start_line\":1,\"end_line\":2}",
        )
        .unwrap();
        assert!(ev.extra.is_empty());
    }
}
