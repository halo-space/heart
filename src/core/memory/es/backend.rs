//! Shared Elasticsearch transport and record primitives for memory stores.
//!
//! Long and short memory expose separate contracts, but both use this backend
//! for optimistic writes, searches, and response decoding.

use elasticsearch::{
    CreateParts, Elasticsearch, GetParts, IndexParts, SearchParts, indices::IndicesExistsParts,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use super::{
    decode_error, invalid_response, status_error, success, transport_error, validate_index,
};
use crate::components::memory::{Error, Filter, Item, Type};

#[derive(Clone, Debug, serde::Deserialize)]
pub(crate) struct StoredMemory {
    #[serde(flatten)]
    pub(crate) item: Item,
}

pub(crate) struct Raw<T> {
    pub(crate) source: T,
    pub(crate) seq_no: i64,
    pub(crate) primary_term: i64,
}

pub(crate) struct Backend {
    pub(crate) client: Elasticsearch,
    pub(crate) index: String,
}

impl Backend {
    pub(crate) fn new(client: Elasticsearch, index: impl Into<String>) -> Result<Self, Error> {
        let index = index.into();
        validate_index(&index)?;
        Ok(Self { client, index })
    }

    pub(crate) async fn ensure_index(&self) -> Result<(), Error> {
        let response = self
            .client
            .indices()
            .exists(IndicesExistsParts::Index(&[&self.index]))
            .send()
            .await
            .map_err(transport_error)?;
        if response.status_code().as_u16() == 200 {
            Ok(())
        } else {
            Err(status_error(response.status_code().as_u16()))
        }
    }

    pub(crate) async fn create(&self, id: i64, source: Value) -> Result<(), Error> {
        let response = self
            .client
            .create(CreateParts::IndexId(&self.index, &id.to_string()))
            .body(source)
            .send()
            .await
            .map_err(transport_error)?;
        let status = response.status_code().as_u16();
        if status == 409 {
            return Err(Error::new("CONFLICT", "record already exists"));
        }
        success(status)
    }

    pub(crate) async fn get<T: DeserializeOwned>(&self, id: i64) -> Result<Option<Raw<T>>, Error> {
        let response = self
            .client
            .get(GetParts::IndexId(&self.index, &id.to_string()))
            .send()
            .await
            .map_err(transport_error)?;
        let status = response.status_code().as_u16();
        let body = response
            .json::<Value>()
            .await
            .map_err(|_| status_error(status))?;
        if status == 404 || body.get("found") == Some(&Value::Bool(false)) {
            return Ok(None);
        }
        success(status)?;
        let source = body.get("_source").cloned().ok_or_else(invalid_response)?;
        Ok(Some(Raw {
            source: serde_json::from_value(source).map_err(decode_error)?,
            seq_no: body
                .get("_seq_no")
                .and_then(Value::as_i64)
                .unwrap_or_default(),
            primary_term: body
                .get("_primary_term")
                .and_then(Value::as_i64)
                .unwrap_or_default(),
        }))
    }

    pub(crate) async fn replace(
        &self,
        id: i64,
        source: Value,
        concurrency: Option<(i64, i64)>,
    ) -> Result<(), Error> {
        let id = id.to_string();
        let mut request = self
            .client
            .index(IndexParts::IndexId(&self.index, &id))
            .body(source);
        if let Some((seq_no, primary_term)) = concurrency {
            request = request.if_seq_no(seq_no).if_primary_term(primary_term);
        }
        let response = request.send().await.map_err(transport_error)?;
        let status = response.status_code().as_u16();
        if status == 409 {
            return Err(Error::new("CONFLICT", "record snapshot is stale"));
        }
        if status == 404 {
            return Err(Error::new("NOT_FOUND", "record not found"));
        }
        success(status)
    }

    pub(crate) async fn delete(&self, id: i64) -> Result<(), Error> {
        let response = self
            .client
            .delete(elasticsearch::DeleteParts::IndexId(
                &self.index,
                &id.to_string(),
            ))
            .send()
            .await
            .map_err(transport_error)?;
        let status = response.status_code().as_u16();
        if status == 404 {
            return Err(Error::new("NOT_FOUND", "record not found"));
        }
        success(status)
    }

    pub(crate) async fn search_scope(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        r#type: &Type,
        size: usize,
    ) -> Result<Vec<Raw<StoredMemory>>, Error> {
        let body = json!({"size": size, "query": {"bool": {"filter": scope_filters(user_id, agent_id, session_id, r#type)}}});
        self.search(body).await
    }

    pub(crate) async fn find_scope(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        r#type: &Type,
    ) -> Result<Option<StoredMemory>, Error> {
        Ok(self
            .search_scope(user_id, agent_id, session_id, r#type, 1)
            .await?
            .into_iter()
            .next()
            .map(|raw| raw.source))
    }

    pub(crate) async fn search_text(
        &self,
        query: &str,
        filter: &Filter,
        offset: usize,
        size: usize,
    ) -> Result<Vec<Raw<StoredMemory>>, Error> {
        let filters = scope_filters(
            filter.user_id,
            filter.agent_id,
            filter.session_id,
            &filter.r#type,
        );
        let body = json!({"from": offset, "size": size, "query": {"bool": {"filter": filters, "must": [{"match": {"content": query}}]}}});
        self.search(body).await
    }

    pub(crate) async fn search_vector(
        &self,
        vector: Vec<f32>,
        filter: &Filter,
        offset: usize,
        size: usize,
    ) -> Result<Vec<Raw<StoredMemory>>, Error> {
        let body = json!({"from": offset, "size": size, "knn": {"field": "vector", "query_vector": vector, "k": size.max(1), "num_candidates": size.saturating_mul(10).max(10), "filter": {"bool": {"filter": scope_filters(filter.user_id, filter.agent_id, filter.session_id, &filter.r#type)}}}});
        self.search(body).await
    }

    pub(crate) async fn search_hybrid(
        &self,
        query: &str,
        vector: Vec<f32>,
        filter: &Filter,
        offset: usize,
        size: usize,
    ) -> Result<Vec<Raw<StoredMemory>>, Error> {
        let filters = scope_filters(
            filter.user_id,
            filter.agent_id,
            filter.session_id,
            &filter.r#type,
        );
        let body = json!({
            "from": offset,
            "size": size,
            "query": {"bool": {"filter": filters.clone(), "should": [{"match": {"content": query}}]}},
            "knn": {"field": "vector", "query_vector": vector, "k": size.max(1), "num_candidates": size.saturating_mul(10).max(10), "filter": {"bool": {"filter": filters}}}
        });
        self.search(body).await
    }

    pub(crate) async fn search<T: DeserializeOwned>(
        &self,
        body: Value,
    ) -> Result<Vec<Raw<T>>, Error> {
        let response = self
            .client
            .search(SearchParts::Index(&[&self.index]))
            .body(body)
            .send()
            .await
            .map_err(transport_error)?;
        let status = response.status_code().as_u16();
        let body = response
            .json::<Value>()
            .await
            .map_err(|_| status_error(status))?;
        success(status)?;
        let hits = body
            .pointer("/hits/hits")
            .and_then(Value::as_array)
            .ok_or_else(invalid_response)?;
        hits.iter()
            .map(|hit| {
                let source = hit.get("_source").cloned().ok_or_else(invalid_response)?;
                Ok(Raw {
                    source: serde_json::from_value(source).map_err(decode_error)?,
                    seq_no: hit
                        .get("_seq_no")
                        .and_then(Value::as_i64)
                        .unwrap_or_default(),
                    primary_term: hit
                        .get("_primary_term")
                        .and_then(Value::as_i64)
                        .unwrap_or_default(),
                })
            })
            .collect()
    }
}

pub(crate) fn source_with_vector(item: &Item, vector: Option<Vec<f32>>) -> Value {
    let mut source = serde_json::to_value(item).expect("memory item is serializable");
    source
        .as_object_mut()
        .expect("memory item is an object")
        .insert(
            "vector".into(),
            vector.map_or(Value::Null, |value| json!(value)),
        );
    source
}

pub(crate) fn scope_filters(
    user_id: i64,
    agent_id: Option<i64>,
    session_id: Option<i64>,
    r#type: &Type,
) -> Vec<Value> {
    let mut filters = vec![
        json!({"term": {"user_id": user_id}}),
        json!({"term": {"type": r#type}}),
    ];
    for (field, value) in [("agent_id", agent_id), ("session_id", session_id)] {
        filters.push(match value {
            Some(value) => json!({"term": {field: value}}),
            None => json!({"bool": {"must_not": {"exists": {"field": field}}}}),
        });
    }
    filters
}
