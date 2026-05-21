//! `engine`：查询装配层（v2 落地路径：`rag::index::engine`）。
//!
//! # 4 个召回入口
//!
//! - [`Engine::text_search`]：单路 `storage.search`。
//! - [`Engine::vector_search`]：单路向量后端 `search`。
//! - [`Engine::hybrid_search`]：业务自拼混合 DSL，单路 `storage.search`。
//! - [`Engine::multi_search`]：`tokio::try_join!` 并发跑 text + vector → `fuse_by_rrf`。
//!
//! # Reranker
//!
//! [`Reranker`] 是单一可插拔扩展点；内置 [`LocalReranker`]（token + vector + rank_feature）和
//! [`ModelReranker`]（外部 HTTP）。业务自家算法 `impl Reranker` 即可。
//!
//! # 最小用法
//!
//! ```rust,no_run
//! use serde_json::{Map, json};
//! use rag::index::engine::{Context, Engine, FieldsConfig};
//! use rag::index::storage::ElasticStorage;
//!
//! # async fn run() -> Result<(), rag::index::engine::Error> {
//! let storage = Box::new(ElasticStorage::from_url("http://localhost:9200")?);
//! let engine = Engine::new(
//!     storage,
//!     None,
//!     None,
//!     /* top */ 1024,
//!     /* score_threshold */ 0.0,
//!     /* rrf_k */ 60,
//!     Map::new(),
//! )?;
//! let resp = engine
//!     .text_search(
//!         "pagewikis",
//!         json!({ "query": { "match_all": {} } }),
//!         &Context::default(),
//!         &FieldsConfig::default(),
//!         Map::new(),
//!         100,
//!         false,
//!     )
//!     .await?;
//! let _ = resp.total;
//! # Ok(()) }
//! ```

pub mod core;
pub mod rerank;
pub mod types;

pub use core::{Engine, FusionWeights};
pub use rerank::{LocalRerankConfig, LocalReranker, ModelReranker, Reranker};
pub use types::{
    Context, DocAgg, Error, FieldsConfig, Hit, Intent, KeywordGroup, Response, TextField,
};
