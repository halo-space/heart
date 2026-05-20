//! Embedder trait 与 NoopEmbedder。

use crate::index::builder::types::Error;
use std::future::Future;
use std::pin::Pin;

/// 为文本内容生成稠密向量表示。
///
/// 实现必须是 `Send + Sync`。
pub trait Embedder: Send + Sync {
    fn embed<'a>(
        &'a self,
        content: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, Error>> + Send + 'a>>;
}

/// 空操作 embedder——始终返回空向量。
///
/// **注意**：[`Builder`](crate::index::builder::types::Builder) 的 `embedder: None`
/// 表示"不生成 embedding，`pw.embedding` 保持 `None`"；
/// `NoopEmbedder` 会被调用并产出 `pw.embedding = Some(vec![])`，语义不同。
pub struct NoopEmbedder;

impl Embedder for NoopEmbedder {
    fn embed<'a>(
        &'a self,
        _content: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, Error>> + Send + 'a>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_返回空向量() {
        let e = NoopEmbedder;
        assert_eq!(e.embed("hello").await.unwrap(), Vec::<f32>::new());
    }
}
