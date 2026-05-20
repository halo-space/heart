//! `source` —— 输入层抽象。
//!
//! 提供 [`Base`] trait、[`Item`] / [`Scenario`] / [`Error`] 数据模型，以及
//! 两个内置实现：
//!
//! - [`Inline`] —— 内存 FIFO 队列。
//! - [`Directory`] —— 从本地目录读取 JSON 文件。
//!
//! 外部数据源通过 `impl Base` 接入。
//!
//! 典型用法：
//!
//! ```no_run
//! use rag::index::source::{self, Base, Inline, Item, Scenario};
//!
//! # async fn run() -> Result<(), source::Error> {
//! let mut src = Inline::new();
//! let mut metadata = serde_json::Map::new();
//! metadata.insert("doc_id".into(), serde_json::json!("doc_001"));
//! src.push(Item {
//!     text: "hello".into(),
//!     scenario: Scenario::General,
//!     metadata,
//! })
//! .await;
//!
//! loop {
//!     let items = src.read(100, Scenario::General).await?;
//!     if items.is_empty() {
//!         break;
//!     }
//!     // 把 items 交给 Builder
//! }
//! # Ok(()) }
//! ```

pub mod base;
pub mod directory;
pub mod inline;
pub mod types;

pub use base::Base;
pub use directory::Directory;
pub use inline::Inline;
pub use types::{Error, Item, Scenario};
