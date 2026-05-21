//! 集成测试用 mock `storage::Base` 实现。

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use rag::index::pagewiki::PageWiki;
use rag::index::storage::{Base, Error};

/// 简单 MockStorage：每次 `search` 返回预置好的 `Value`，并记录调用次数。
pub struct MockStorage {
    pub response: Value,
    pub call_count: Arc<Mutex<usize>>,
    /// 人工延迟（ms），用于并发耗时测试。
    pub delay_ms: Option<u64>,
    /// 若为 Some，则 search 返回 Err。
    pub error: Option<String>,
}

impl MockStorage {
    pub fn new(response: Value) -> Self {
        Self {
            response,
            call_count: Arc::new(Mutex::new(0)),
            delay_ms: None,
            error: None,
        }
    }

    pub fn with_delay(mut self, ms: u64) -> Self {
        self.delay_ms = Some(ms);
        self
    }

    pub fn with_error(mut self, msg: impl Into<String>) -> Self {
        self.error = Some(msg.into());
        self
    }
}

/// 构造一个 ES hits 格式的 Value，内含若干 PageWiki hit。
pub fn make_es_response(hits: Vec<(String, String, f32)>) -> Value {
    // hits: (id, content, score)
    let hit_arr: Vec<Value> = hits
        .into_iter()
        .map(|(id, content, score)| {
            json!({
                "_score": score,
                "_source": {
                    "id": id,
                    "content": content
                }
            })
        })
        .collect();
    json!({ "hits": { "total": { "value": hit_arr.len() }, "hits": hit_arr } })
}

/// 构造带 doc_id 的 ES hits 响应。
pub fn make_es_response_with_doc(hits: Vec<(String, String, String, f32)>) -> Value {
    // hits: (id, doc_id, content, score)
    let hit_arr: Vec<Value> = hits
        .into_iter()
        .map(|(id, doc_id, content, score)| {
            json!({
                "_score": score,
                "_source": {
                    "id": id,
                    "doc_id": doc_id,
                    "content": content
                }
            })
        })
        .collect();
    json!({ "hits": { "total": { "value": hit_arr.len() }, "hits": hit_arr } })
}

impl Base for MockStorage {
    fn create<'a>(
        &'a self,
        _index: &'a str,
        _page: PageWiki,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn get<'a>(
        &'a self,
        _index: &'a str,
        _id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<PageWiki, Error>> + Send + 'a>> {
        Box::pin(async { Ok(PageWiki::default()) })
    }

    fn update<'a>(
        &'a self,
        _index: &'a str,
        _id: &'a str,
        _page: PageWiki,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn delete<'a>(
        &'a self,
        _index: &'a str,
        _id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn bulk_create<'a>(
        &'a self,
        _index: &'a str,
        _pages: Vec<PageWiki>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn bulk_update<'a>(
        &'a self,
        _index: &'a str,
        _items: Vec<(String, PageWiki)>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn bulk_delete<'a>(
        &'a self,
        _index: &'a str,
        _ids: Vec<String>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn search<'a>(
        &'a self,
        _index: &'a str,
        _query: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, Error>> + Send + 'a>> {
        let count = Arc::clone(&self.call_count);
        let delay = self.delay_ms;
        let err = self.error.clone();
        let resp = self.response.clone();
        Box::pin(async move {
            *count.lock().unwrap() += 1;
            if let Some(ms) = delay {
                tokio::time::sleep(tokio::time::Duration::from_millis(ms)).await;
            }
            if let Some(msg) = err {
                return Err(Error::Transport(msg));
            }
            Ok(resp)
        })
    }

    fn multi_search<'a>(
        &'a self,
        _index: &'a str,
        queries: Vec<serde_json::Value>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<serde_json::Value>, Error>> + Send + 'a>> {
        let resp = self.response.clone();
        let n = queries.len();
        Box::pin(async move { Ok(vec![resp; n]) })
    }
}
