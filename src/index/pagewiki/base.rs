//! `pagewiki::Base` —— PageWiki 切分 trait。
//!
//! 使用 `Pin<Box<dyn Future>>` 形式保证 dyn-compat，可直接用 `Box<dyn Base>`。
//! 任何实现了 `async fn cut` 的类型只需通过 blanket impl 自动获得 `Base`。

use crate::index::pagewiki::types::{Error, PageWiki};
use std::future::Future;
use std::pin::Pin;

/// PageWiki 切分器统一接口（dyn-compat）。
pub trait Base: Send + Sync {
    /// 把整段文本切分为若干 [`PageWiki`]。
    ///
    /// 返回的 `PageWiki` 中，`id` / `doc_id` / `version` / `scenario` / `idx`
    /// **必须**保持 `None`；这些字段由下游 Builder 阶段填入。
    fn cut<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PageWiki>, Error>> + Send + 'a>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Noop;

    impl Base for Noop {
        fn cut<'a>(
            &'a self,
            _text: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<PageWiki>, Error>> + Send + 'a>> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    #[tokio::test]
    async fn impl_compiles_and_runs() {
        let n = Noop;
        let v = n.cut("hello").await.unwrap();
        assert!(v.is_empty());
    }

    /// `Box<dyn Base>` 可作为 trait object 持有
    fn _assert_object_safe(_: &dyn Base) {}
}
