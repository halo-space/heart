//! `engine::rerank`：Reranker trait + 内置 `LocalReranker` / `ModelReranker`。
//!
//! Trait 用 `Pin<Box<dyn Future>>` 形式保证 dyn-compat，可直接持 `Box<dyn Reranker>`；
//! 不依赖 `#[async_trait]`、不依赖 `async-trait` crate。

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use super::types::{Error, Hit};

pub mod local;
pub mod model;

pub use local::{LocalRerankConfig, LocalReranker};
pub use model::ModelReranker;

/// 二阶段 rerank 单一扩展点。
///
/// 实现侧职责：
/// - 计算每个 hit 的"组合分"，写入 `hit.score`；
/// - 把分项分（如 `text` / `vector` / `model` / `rank_feature`）写入 `hit.scores`，并把组合分
///   单独再写一份到 `scores["rerank"]`；
/// - 返回的 `Vec<Hit>` 按 `score` 倒序。
///
/// `weights` 是 typed map，键集合开放（业务自由约定）；常见 key：`text / vector / model /
/// rank_feature`。
pub trait Reranker: Send + Sync {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        hits: Vec<Hit>,
        weights: HashMap<String, f32>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Hit>, Error>> + Send + 'a>>;
}

/// 编译期确认 `Reranker` 是 dyn-compat。
#[cfg(test)]
fn _assert_object_safe(_: &dyn Reranker) {}
