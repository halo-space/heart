//! `ElasticStorage`：`storage::Base` 的 ES 8.x 实现。
//!
//! Trait 形态是 `fn ... -> Pin<Box<dyn Future<...> + Send + 'a>>`（dyn-compat），
//! 每个方法用 `Box::pin(async move { ... })` 包住原本的 async 逻辑；
//! 不依赖 `#[async_trait]`、不依赖 `async-trait` crate。

use std::future::Future;
use std::pin::Pin;

use elasticsearch::{
    BulkOperation, BulkParts, DeleteParts, Elasticsearch, GetParts, IndexParts, MsearchParts,
    SearchParts, UpdateParts,
    http::request::JsonBody,
    http::transport::{SingleNodeConnectionPool, TransportBuilder},
    params::OpType,
};
use serde_json::{Value, json};

use crate::index::pagewiki::PageWiki;

use super::base::Base;
use super::types::{BulkAction, BulkItemFailure, Error};

pub struct ElasticStorage {
    client: Elasticsearch,
}

impl ElasticStorage {
    pub fn new(client: Elasticsearch) -> Self {
        Self { client }
    }

    pub fn from_url(url: &str) -> Result<Self, Error> {
        let parsed =
            url::Url::parse(url).map_err(|e| Error::Transport(format!("invalid url: {e}")))?;
        let pool = SingleNodeConnectionPool::new(parsed);
        let transport = TransportBuilder::new(pool)
            .build()
            .map_err(|e| Error::Transport(e.to_string()))?;
        Ok(Self::new(Elasticsearch::new(transport)))
    }
}

async fn map_response(resp: elasticsearch::http::response::Response) -> Result<Value, Error> {
    let status = resp.status_code().as_u16();
    if (200..300).contains(&status) {
        return resp
            .json::<Value>()
            .await
            .map_err(|e| Error::Transport(e.to_string()));
    }
    let body = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
    Err(Error::Es { status, body })
}

#[derive(Clone, Copy)]
enum Op {
    Create,
    Get,
    Update,
    Delete,
}

fn interpret_es_error(err: Error, index: &str, id: &str, op: Op) -> Error {
    let Error::Es { status, body } = err else {
        return err;
    };
    let err_type = body
        .get("error")
        .and_then(|e| e.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let reason = body
        .get("error")
        .and_then(|e| e.get("reason"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if err_type == "index_not_found_exception" {
        return Error::IndexNotFound(index.to_string());
    }

    match status {
        404 => Error::NotFound {
            index: index.to_string(),
            id: id.to_string(),
        },
        409 => {
            let r = if reason.is_empty() {
                match op {
                    Op::Create => "document already exists".to_string(),
                    _ => "version conflict".to_string(),
                }
            } else {
                reason
            };
            Error::Conflict {
                index: index.to_string(),
                id: id.to_string(),
                reason: r,
            }
        }
        _ => Error::Es { status, body },
    }
}

fn extract_id(page: &PageWiki) -> Result<String, Error> {
    use serde::de::Error as _;
    page.id
        .clone()
        .ok_or_else(|| Error::Serialize(serde_json::Error::custom("PageWiki.id is None")))
}

fn interpret_bulk_response(value: &Value, action: BulkAction) -> Result<(), Error> {
    let errors = value
        .get("errors")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !errors {
        return Ok(());
    }
    let items = value
        .get("items")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut failures: Vec<BulkItemFailure> = Vec::new();
    for item in items {
        let Some(op_val) = item.as_object().and_then(|m| m.values().next().cloned()) else {
            continue;
        };
        let status = op_val.get("status").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
        if (200..300).contains(&status) {
            continue;
        }
        let id = op_val
            .get("_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let reason = op_val
            .get("error")
            .and_then(|e| e.get("reason"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        failures.push(BulkItemFailure {
            action,
            id,
            status,
            reason,
        });
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::BulkPartialFailure { failures })
    }
}

fn approx_body_size(v: &Value) -> usize {
    serde_json::to_vec(v).map(|b| b.len()).unwrap_or(0)
}

impl Base for ElasticStorage {
    fn create<'a>(
        &'a self,
        index: &'a str,
        page: PageWiki,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
            let span = tracing::debug_span!("storage.create", index = %index);
            let _enter = span.enter();
            let id = extract_id(&page).inspect_err(|e| {
                tracing::warn!(op = "create", error = %e, "storage.failed");
            })?;
            let body = serde_json::to_value(&page).inspect_err(|e| {
                tracing::warn!(op = "create", error = %e, "storage.failed");
            })?;
            let resp = self
                .client
                .index(IndexParts::IndexId(index, &id))
                .body(body)
                .op_type(OpType::Create)
                .send()
                .await
                .map_err(|e| {
                    let err = Error::Transport(e.to_string());
                    tracing::warn!(op = "create", error = %err, "storage.failed");
                    err
                })?;
            match map_response(resp).await {
                Ok(_) => Ok(()),
                Err(e) => {
                    let mapped = interpret_es_error(e, index, &id, Op::Create);
                    tracing::warn!(op = "create", error = %mapped, "storage.failed");
                    Err(mapped)
                }
            }
        })
    }

    fn get<'a>(
        &'a self,
        index: &'a str,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<PageWiki, Error>> + Send + 'a>> {
        Box::pin(async move {
            let span = tracing::debug_span!("storage.get", index = %index, id = %id);
            let _enter = span.enter();
            let resp = self
                .client
                .get(GetParts::IndexId(index, id))
                .send()
                .await
                .map_err(|e| {
                    let err = Error::Transport(e.to_string());
                    tracing::warn!(op = "get", error = %err, "storage.failed");
                    err
                })?;
            let value = match map_response(resp).await {
                Ok(v) => v,
                Err(e) => {
                    let mapped = interpret_es_error(e, index, id, Op::Get);
                    tracing::warn!(op = "get", error = %mapped, "storage.failed");
                    return Err(mapped);
                }
            };
            let source = value
                .get("_source")
                .cloned()
                .ok_or_else(|| Error::NotFound {
                    index: index.to_string(),
                    id: id.to_string(),
                })
                .inspect_err(|e| {
                    tracing::warn!(op = "get", error = %e, "storage.failed");
                })?;
            let page: PageWiki = serde_json::from_value(source).inspect_err(|e| {
                tracing::warn!(op = "get", error = %e, "storage.failed");
            })?;
            Ok(page)
        })
    }

    fn update<'a>(
        &'a self,
        index: &'a str,
        id: &'a str,
        page: PageWiki,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
            let span = tracing::debug_span!("storage.update", index = %index, id = %id);
            let _enter = span.enter();
            let body = json!({ "doc": page });
            let resp = self
                .client
                .update(UpdateParts::IndexId(index, id))
                .body(body)
                .send()
                .await
                .map_err(|e| {
                    let err = Error::Transport(e.to_string());
                    tracing::warn!(op = "update", error = %err, "storage.failed");
                    err
                })?;
            match map_response(resp).await {
                Ok(_) => Ok(()),
                Err(e) => {
                    let mapped = interpret_es_error(e, index, id, Op::Update);
                    tracing::warn!(op = "update", error = %mapped, "storage.failed");
                    Err(mapped)
                }
            }
        })
    }

    fn delete<'a>(
        &'a self,
        index: &'a str,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
            let span = tracing::debug_span!("storage.delete", index = %index, id = %id);
            let _enter = span.enter();
            let resp = self
                .client
                .delete(DeleteParts::IndexId(index, id))
                .send()
                .await
                .map_err(|e| {
                    let err = Error::Transport(e.to_string());
                    tracing::warn!(op = "delete", error = %err, "storage.failed");
                    err
                })?;
            match map_response(resp).await {
                Ok(_) => Ok(()),
                Err(e) => {
                    let mapped = interpret_es_error(e, index, id, Op::Delete);
                    tracing::warn!(op = "delete", error = %mapped, "storage.failed");
                    Err(mapped)
                }
            }
        })
    }

    fn bulk_create<'a>(
        &'a self,
        index: &'a str,
        pages: Vec<PageWiki>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
            let span = tracing::debug_span!("storage.bulk_create", index = %index, n = pages.len());
            let _enter = span.enter();
            if pages.is_empty() {
                return Ok(());
            }
            let mut ops: Vec<BulkOperation<Value>> = Vec::with_capacity(pages.len());
            for page in pages {
                let id = extract_id(&page).inspect_err(|e| {
                    tracing::warn!(op = "bulk_create", error = %e, "storage.failed");
                })?;
                let body = serde_json::to_value(&page).inspect_err(|e| {
                    tracing::warn!(op = "bulk_create", error = %e, "storage.failed");
                })?;
                ops.push(BulkOperation::create(body).id(id).into());
            }
            let resp = self
                .client
                .bulk(BulkParts::Index(index))
                .body(ops)
                .send()
                .await
                .map_err(|e| {
                    let err = Error::Transport(e.to_string());
                    tracing::warn!(op = "bulk_create", error = %err, "storage.failed");
                    err
                })?;
            let value = map_response(resp).await.inspect_err(|e| {
                tracing::warn!(op = "bulk_create", error = %e, "storage.failed");
            })?;
            match interpret_bulk_response(&value, BulkAction::Create) {
                Ok(()) => Ok(()),
                Err(Error::BulkPartialFailure { failures }) => {
                    tracing::warn!(
                        op = "bulk_create",
                        n_failures = failures.len(),
                        "storage.bulk.partial"
                    );
                    Err(Error::BulkPartialFailure { failures })
                }
                Err(e) => {
                    tracing::warn!(op = "bulk_create", error = %e, "storage.failed");
                    Err(e)
                }
            }
        })
    }

    fn bulk_update<'a>(
        &'a self,
        index: &'a str,
        items: Vec<(String, PageWiki)>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
            let span = tracing::debug_span!("storage.bulk_update", index = %index, n = items.len());
            let _enter = span.enter();
            if items.is_empty() {
                return Ok(());
            }
            let mut ops: Vec<BulkOperation<Value>> = Vec::with_capacity(items.len());
            for (id, page) in items {
                let body = json!({ "doc": page });
                ops.push(BulkOperation::update(id, body).into());
            }
            let resp = self
                .client
                .bulk(BulkParts::Index(index))
                .body(ops)
                .send()
                .await
                .map_err(|e| {
                    let err = Error::Transport(e.to_string());
                    tracing::warn!(op = "bulk_update", error = %err, "storage.failed");
                    err
                })?;
            let value = map_response(resp).await.inspect_err(|e| {
                tracing::warn!(op = "bulk_update", error = %e, "storage.failed");
            })?;
            match interpret_bulk_response(&value, BulkAction::Update) {
                Ok(()) => Ok(()),
                Err(Error::BulkPartialFailure { failures }) => {
                    tracing::warn!(
                        op = "bulk_update",
                        n_failures = failures.len(),
                        "storage.bulk.partial"
                    );
                    Err(Error::BulkPartialFailure { failures })
                }
                Err(e) => {
                    tracing::warn!(op = "bulk_update", error = %e, "storage.failed");
                    Err(e)
                }
            }
        })
    }

    fn bulk_delete<'a>(
        &'a self,
        index: &'a str,
        ids: Vec<String>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
            let span = tracing::debug_span!("storage.bulk_delete", index = %index, n = ids.len());
            let _enter = span.enter();
            if ids.is_empty() {
                return Ok(());
            }
            let mut ops: Vec<BulkOperation<Value>> = Vec::with_capacity(ids.len());
            for id in ids {
                ops.push(BulkOperation::<Value>::delete(id).into());
            }
            let resp = self
                .client
                .bulk(BulkParts::Index(index))
                .body(ops)
                .send()
                .await
                .map_err(|e| {
                    let err = Error::Transport(e.to_string());
                    tracing::warn!(op = "bulk_delete", error = %err, "storage.failed");
                    err
                })?;
            let value = map_response(resp).await.inspect_err(|e| {
                tracing::warn!(op = "bulk_delete", error = %e, "storage.failed");
            })?;
            match interpret_bulk_response(&value, BulkAction::Delete) {
                Ok(()) => Ok(()),
                Err(Error::BulkPartialFailure { failures }) => {
                    tracing::warn!(
                        op = "bulk_delete",
                        n_failures = failures.len(),
                        "storage.bulk.partial"
                    );
                    Err(Error::BulkPartialFailure { failures })
                }
                Err(e) => {
                    tracing::warn!(op = "bulk_delete", error = %e, "storage.failed");
                    Err(e)
                }
            }
        })
    }

    fn search<'a>(
        &'a self,
        index: &'a str,
        query: Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value, Error>> + Send + 'a>> {
        Box::pin(async move {
            let body_size_bytes = approx_body_size(&query);
            let span = tracing::debug_span!(
                "storage.search",
                index = %index,
                body_size_bytes,
            );
            let _enter = span.enter();
            let resp = self
                .client
                .search(SearchParts::Index(&[index]))
                .body(query)
                .send()
                .await
                .map_err(|e| {
                    let err = Error::Transport(e.to_string());
                    tracing::warn!(op = "search", error = %err, "storage.failed");
                    err
                })?;
            map_response(resp).await.inspect_err(|e| {
                tracing::warn!(op = "search", error = %e, "storage.failed");
            })
        })
    }

    fn multi_search<'a>(
        &'a self,
        index: &'a str,
        queries: Vec<Value>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Value>, Error>> + Send + 'a>> {
        Box::pin(async move {
            let n_queries = queries.len();
            let span = tracing::debug_span!(
                "storage.multi_search",
                index = %index,
                n_queries,
            );
            let _enter = span.enter();
            if queries.is_empty() {
                return Ok(Vec::new());
            }
            let mut body: Vec<JsonBody<Value>> = Vec::with_capacity(queries.len() * 2);
            for q in queries {
                body.push(JsonBody::new(json!({})));
                body.push(JsonBody::new(q));
            }
            let resp = self
                .client
                .msearch(MsearchParts::Index(&[index]))
                .body(body)
                .send()
                .await
                .map_err(|e| {
                    let err = Error::Transport(e.to_string());
                    tracing::warn!(op = "multi_search", error = %err, "storage.failed");
                    err
                })?;
            let value = map_response(resp).await.inspect_err(|e| {
                tracing::warn!(op = "multi_search", error = %e, "storage.failed");
            })?;
            let responses = value
                .get("responses")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            Ok(responses)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_url_ok() {
        assert!(ElasticStorage::from_url("http://localhost:9200").is_ok());
    }

    #[test]
    fn from_url_invalid_returns_transport() {
        match ElasticStorage::from_url("not a url") {
            Err(Error::Transport(_)) => {}
            Err(e) => panic!("wrong error variant: {e:?}"),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn interpret_bulk_no_errors_is_ok() {
        let v = json!({ "errors": false, "items": [] });
        assert!(interpret_bulk_response(&v, BulkAction::Create).is_ok());
    }

    #[test]
    fn interpret_bulk_create_409_yields_partial_failure() {
        let v = json!({
            "errors": true,
            "items": [{
                "create": {
                    "_id": "x",
                    "status": 409,
                    "error": { "reason": "already exists" }
                }
            }]
        });
        let err = interpret_bulk_response(&v, BulkAction::Create).unwrap_err();
        match err {
            Error::BulkPartialFailure { failures } => {
                assert_eq!(failures.len(), 1);
                assert_eq!(failures[0].action, BulkAction::Create);
                assert_eq!(failures[0].id, "x");
                assert_eq!(failures[0].status, 409);
                assert!(failures[0].reason.contains("already exists"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn interpret_bulk_skips_2xx_items() {
        let v = json!({
            "errors": true,
            "items": [
                { "create": { "_id": "ok", "status": 201 } },
                { "create": { "_id": "bad", "status": 409, "error": { "reason": "dup" } } }
            ]
        });
        let err = interpret_bulk_response(&v, BulkAction::Create).unwrap_err();
        match err {
            Error::BulkPartialFailure { failures } => {
                assert_eq!(failures.len(), 1);
                assert_eq!(failures[0].id, "bad");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn interpret_es_404_to_not_found() {
        let e = Error::Es {
            status: 404,
            body: json!({ "error": { "type": "x", "reason": "missing" } }),
        };
        let mapped = interpret_es_error(e, "i", "x", Op::Get);
        assert!(matches!(mapped, Error::NotFound { .. }));
    }

    #[test]
    fn interpret_es_index_not_found() {
        let e = Error::Es {
            status: 404,
            body: json!({ "error": { "type": "index_not_found_exception", "reason": "no" } }),
        };
        let mapped = interpret_es_error(e, "i", "x", Op::Get);
        assert!(matches!(mapped, Error::IndexNotFound(_)));
    }

    #[test]
    fn interpret_es_409_to_conflict() {
        let e = Error::Es {
            status: 409,
            body: json!({ "error": { "type": "version_conflict_engine_exception", "reason": "vc" } }),
        };
        let mapped = interpret_es_error(e, "i", "x", Op::Update);
        match mapped {
            Error::Conflict { reason, .. } => assert!(reason.contains("vc")),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn interpret_es_other_status_passthrough() {
        let e = Error::Es {
            status: 500,
            body: json!({ "error": { "reason": "boom" } }),
        };
        let mapped = interpret_es_error(e, "i", "x", Op::Get);
        assert!(matches!(mapped, Error::Es { status: 500, .. }));
    }

    #[test]
    fn extract_id_missing_returns_serialize_error() {
        let p = PageWiki::default();
        let err = extract_id(&p).unwrap_err();
        assert!(matches!(err, Error::Serialize(_)));
    }
}
