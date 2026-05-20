//! PageWiki：把原始文本切分成统一检索单元 [`PageWiki`]。
//!
//! # 模块结构
//!
//! - [`base`]：核心 trait [`Base`]，定义 `async fn cut(&self, text: &str) -> Result<Vec<PageWiki>, Error>`。
//! - [`rule`]：规则切分器 [`Fixed`] / [`SlidingWindow`] / [`Delimiter`] / [`Qa`]。
//! - [`semantic`]：基于 LLM 的语义切分器 [`Semantic`]，使用 `async-openai` 调用兼容 OpenAI 协议的服务。
//! - [`spans`]：把 LLM 返回的 [`Evidence`] 反算为原文字符坐标 [`Span`]。
//! - [`types`]：数据模型 [`PageWiki`] / [`Span`] / [`Graph`] / [`Evidence`] / [`Error`]。
//!
//! # 字符坐标统一口径
//!
//! 所有 [`Span::start`] / [`Span::end`] 一律为 Unicode scalar value index
//! （`text.chars().count()` 口径），与 `docs/feature-design/05-wiki-page.md` 保持一致。
//!
//! # 自定义切分器
//!
//! 实现 [`Base`] trait 即可接入下游 IndexBuilder：
//!
//! ```rust,ignore
//! use rag::pagewiki::{Base, Error, PageWiki};
//!
//! struct MyCutter;
//!
//! impl Base for MyCutter {
//!     async fn cut(&self, text: &str) -> Result<Vec<PageWiki>, Error> {
//!         Ok(vec![PageWiki { content: text.to_string(), ..Default::default() }])
//!     }
//! }
//! ```
//!
//! 详见 `openspec/changes/rag-page-wiki/design.md`。

pub mod base;
pub mod rule;
pub mod semantic;
pub mod spans;
pub mod types;

pub use base::Base;
pub use rule::delimiter::Delimiter;
pub use rule::fixed::Fixed;
pub use rule::qa::Qa;
pub use rule::sliding_window::SlidingWindow;
pub use semantic::Semantic;
pub use spans::resolve_spans;
pub use types::{Error, Evidence, Graph, PageWiki, Result, Span};
