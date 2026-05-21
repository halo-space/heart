//! `storage::Base` 存储抽象 trait（dyn-compat）。
//!
//! 使用 `Pin<Box<dyn Future>>` 形式保证 dyn-compat，可直接用 `Box<dyn Base>`。
//! 与 [`crate::index::pagewiki::Base`] 风格保持一致；不引入 `async-trait` 宏，
//! 也不使用 `#[async_trait]`，所有 future 都是手写 `Box::pin(async move { .. })`。

use std::future::Future;
use std::pin::Pin;

use crate::index::pagewiki::PageWiki;

use super::types::Error;

pub trait Base: Send + Sync {
    fn create<'a>(
        &'a self,
        index: &'a str,
        page: PageWiki,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>>;

    fn get<'a>(
        &'a self,
        index: &'a str,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<PageWiki, Error>> + Send + 'a>>;

    fn update<'a>(
        &'a self,
        index: &'a str,
        id: &'a str,
        page: PageWiki,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>>;

    fn delete<'a>(
        &'a self,
        index: &'a str,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>>;

    fn bulk_create<'a>(
        &'a self,
        index: &'a str,
        pages: Vec<PageWiki>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>>;

    fn bulk_update<'a>(
        &'a self,
        index: &'a str,
        items: Vec<(String, PageWiki)>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>>;

    fn bulk_delete<'a>(
        &'a self,
        index: &'a str,
        ids: Vec<String>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>>;

    fn search<'a>(
        &'a self,
        index: &'a str,
        query: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, Error>> + Send + 'a>>;

    fn multi_search<'a>(
        &'a self,
        index: &'a str,
        queries: Vec<serde_json::Value>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<serde_json::Value>, Error>> + Send + 'a>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Box<dyn Base>` 可作为 trait object 持有。
    fn _assert_object_safe(_: &dyn Base) {}
}
