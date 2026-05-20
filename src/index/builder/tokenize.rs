//! Tokenizer trait 与 NoopTokenizer。

use std::future::Future;
use std::pin::Pin;
use crate::index::builder::types::Error;

/// 把文本分词并拼接为空格分隔的 token 字符串，用于全文检索。
///
/// 实现必须是 `Send + Sync`。
///
/// # 对象安全性
/// ```
/// # use rag::index::Tokenizer;
/// fn _assert_object_safe(_: &dyn Tokenizer) {}
/// ```
pub trait Tokenizer: Send + Sync {
    fn tokenize<'a>(&'a self, text: &'a str) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + 'a>>;
}

/// 空操作 tokenizer——始终返回空字符串。
pub struct NoopTokenizer;

impl Tokenizer for NoopTokenizer {
    fn tokenize<'a>(&'a self, _text: &'a str) -> Pin<Box<dyn Future<Output = Result<String, Error>> + Send + 'a>> {
        Box::pin(async { Ok(String::new()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_返回空字符串() {
        let t = NoopTokenizer;
        assert_eq!(t.tokenize("hello").await.unwrap(), "");
    }
}
