//! `storage`：存储抽象 + ES 8.x 实现。
//!
//! # `Base` trait（9 个方法）
//!
//! - `create / get / update / delete`：单条 CRUD。
//! - `bulk_create / bulk_update / bulk_delete`：批量 CRUD（部分失败 → [`Error::BulkPartialFailure`]）。
//! - `search`：DSL 透传 `serde_json::Value`，返回 ES `_search` body。
//! - `multi_search`：N 个 query 走 `_msearch`，返回值与 ES `responses` 数组一一对应。
//!
//! # 最小用法
//!
//! ```rust,no_run
//! use rag::index::storage::{Base, ElasticStorage};
//! use rag::index::storage::mapping::pagewiki_mapping;
//!
//! # async fn run() -> Result<(), rag::index::storage::Error> {
//! let store = ElasticStorage::from_url("http://localhost:9200")?;
//! let _mapping = pagewiki_mapping(1024);
//! let _resp = store
//!     .search("pagewikis", serde_json::json!({ "query": { "match_all": {} } }))
//!     .await?;
//! # Ok(()) }
//! ```

pub mod base;
pub mod elastic;
pub mod mapping;
pub mod types;

pub use base::Base;
pub use elastic::ElasticStorage;
pub use types::{BulkAction, BulkItemFailure, Error};
