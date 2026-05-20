//! `builder` —— 装配层。
//!
//! 把 [`source::Item`](crate::index::source::Item) 跑通 normalize / cut / tokenize /
//! embed / 字段提升 / 系统字段补齐 整条流水线，输出可入库的
//! [`Vec<pagewiki::PageWiki>`](crate::index::pagewiki::PageWiki)。

pub mod builder;
pub mod embed;
pub mod normalize;
pub mod promote;
pub mod tokenize;
pub mod types;

pub use embed::{Embedder, NoopEmbedder};
pub use tokenize::{NoopTokenizer, Tokenizer};
pub use types::{Builder, Error};
