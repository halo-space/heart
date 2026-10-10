use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::components::memory::{
    Error, Filter, HybridRequest, Item, KeywordRequest, Memory, Request, Type, VectorRequest,
};
use crate::runtime::cancellation::Cancellation;

pub mod extraction;
pub mod short;

#[derive(Clone)]
struct Stored {
    item: Item,
    vector: Option<Vec<f32>>,
}

/// Zero-configuration in-memory implementation. It has no vector capability
/// unless a fixed dimension is explicitly selected at construction time.
pub struct InMemory {
    records: RwLock<BTreeMap<i64, Stored>>,
    vector_dimension: Option<usize>,
}

impl Default for InMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemory {
    pub fn new() -> Self {
        Self {
            records: RwLock::new(BTreeMap::new()),
            vector_dimension: None,
        }
    }

    pub fn with_vector_dimension(dimension: usize) -> Result<Self, Error> {
        if dimension == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "vector dimension must be positive",
            ));
        }
        Ok(Self {
            records: RwLock::new(BTreeMap::new()),
            vector_dimension: Some(dimension),
        })
    }

    fn ensure_active(cancellation: &Cancellation) -> Result<(), Error> {
        if cancellation.is_cancelled() {
            Err(Error::new("CANCELLED", "memory operation cancelled"))
        } else {
            Ok(())
        }
    }

    fn validate_request(&self, request: &Request) -> Result<(), Error> {
        if request.id <= 0 || request.user_id <= 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "id and user_id must be positive",
            ));
        }
        if request.from < 0 || request.to <= request.from {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "memory range must be [from, to)",
            ));
        }
        if request.content.trim().is_empty() {
            return Err(Error::new("INVALID_ARGUMENTS", "memory content is empty"));
        }
        if request.session_id.is_some() && request.agent_id.is_none() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "session scope requires agent_id",
            ));
        }
        if request.agent_id.is_some_and(|id| id <= 0)
            || request.session_id.is_some_and(|id| id <= 0)
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "scope ids must be positive",
            ));
        }
        match (self.vector_dimension, request.vector.as_ref()) {
            (None, Some(_)) => {
                return Err(Error::new("UNSUPPORTED", "vector capability is disabled"));
            }
            (Some(_), None) => return Err(Error::new("INVALID_ARGUMENTS", "vector is required")),
            (Some(dimension), Some(vector)) if vector.len() != dimension => {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "vector dimension does not match storage",
                ));
            }
            (Some(_), Some(vector)) if vector.iter().any(|value| !value.is_finite()) => {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "vector contains a non-finite value",
                ));
            }
            _ => {}
        }
        Ok(())
    }

    fn scope_matches(item: &Item, filter: &Filter) -> bool {
        item.user_id == filter.user_id
            && item.agent_id == filter.agent_id
            && item.session_id == filter.session_id
            && item.r#type == filter.r#type
    }

    fn parse_filter(options: &serde_json::Map<String, Value>) -> Result<Filter, Error> {
        let filter = options.get("filter").ok_or_else(|| {
            Error::new("INVALID_ARGUMENTS", "memory search requires options.filter")
        })?;
        for field in ["user_id", "agent_id", "session_id", "type"] {
            if filter.get(field).is_none() {
                return Err(
                    Error::new("INVALID_ARGUMENTS", "memory filter field is missing")
                        .with_details(format!("filter.{field}")),
                );
            }
        }
        serde_json::from_value(filter.clone()).map_err(|error| {
            Error::new(
                "INVALID_ARGUMENTS",
                format!("invalid memory filter: {error}"),
            )
        })
    }

    fn paginate_and_sort(
        mut records: Vec<Stored>,
        options: &serde_json::Map<String, Value>,
    ) -> Result<Vec<Item>, Error> {
        let (field, descending) = match options.get("sort") {
            None => ("updated_time", true),
            Some(Value::Object(sort)) => {
                let field = match sort.get("field") {
                    None => "updated_time",
                    Some(Value::String(field)) => field.as_str(),
                    Some(_) => {
                        return Err(Error::new("INVALID_ARGUMENTS", "invalid memory sort"));
                    }
                };
                let order = match sort.get("order") {
                    None => "desc",
                    Some(Value::String(order)) => order.as_str(),
                    Some(_) => {
                        return Err(Error::new("INVALID_ARGUMENTS", "invalid memory sort"));
                    }
                };
                if !matches!(field, "created_time" | "updated_time" | "from" | "to")
                    || !matches!(order, "asc" | "desc")
                {
                    return Err(Error::new("INVALID_ARGUMENTS", "invalid memory sort"));
                }
                (field, order == "desc")
            }
            Some(_) => return Err(Error::new("INVALID_ARGUMENTS", "sort must be an object")),
        };
        records.sort_by(|left, right| {
            let ordering = match field {
                "created_time" => left.item.created_time.cmp(&right.item.created_time),
                "from" => left.item.from.cmp(&right.item.from),
                "to" => left.item.to.cmp(&right.item.to),
                _ => left.item.updated_time.cmp(&right.item.updated_time),
            };
            if descending {
                ordering.reverse()
            } else {
                ordering
            }
        });
        let (offset, limit) = pagination(options)?;
        Ok(records
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|stored| stored.item)
            .collect())
    }

    fn exact_scope(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        r#type: Type,
    ) -> Filter {
        Filter {
            user_id,
            agent_id,
            session_id,
            r#type,
        }
    }
}

impl Memory for InMemory {
    async fn write(&self, request: Request, cancellation: &Cancellation) -> Result<Item, Error> {
        Self::ensure_active(cancellation)?;
        self.validate_request(&request)?;
        let now = now_ms();
        let item = Item {
            id: request.id,
            r#type: request.r#type.clone(),
            user_id: request.user_id,
            agent_id: request.agent_id,
            session_id: request.session_id,
            from: request.from,
            to: request.to,
            content: request.content,
            version: 1,
            created_time: now,
            updated_time: now,
            metadata: request.metadata,
        };
        let stored = Stored {
            item: item.clone(),
            vector: request.vector,
        };
        let mut records = self.records.write().expect("memory lock poisoned");
        Self::ensure_active(cancellation)?;
        if records.contains_key(&item.id) {
            return Err(Error::new("CONFLICT", "memory id already exists"));
        }
        if records
            .values()
            .any(|record| same_scope_and_type(&record.item, &item))
        {
            return Err(Error::new(
                "CONFLICT",
                "scope and type already have a memory",
            ));
        }
        records.insert(item.id, stored);
        Ok(item)
    }

    async fn update(
        &self,
        request: Request,
        version: u64,
        cancellation: &Cancellation,
    ) -> Result<Item, Error> {
        Self::ensure_active(cancellation)?;
        self.validate_request(&request)?;
        let mut records = self.records.write().expect("memory lock poisoned");
        Self::ensure_active(cancellation)?;
        let stored = records
            .get_mut(&request.id)
            .ok_or_else(|| Error::new("NOT_FOUND", "memory not found"))?;
        if stored.item.version != version {
            return Err(Error::new("CONFLICT", "memory version does not match"));
        }
        if !same_scope_and_type(
            &stored.item,
            &request_as_item(
                &request,
                stored.item.version,
                stored.item.created_time,
                stored.item.updated_time,
            ),
        ) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "memory scope and type cannot change",
            ));
        }
        let next_version = version
            .checked_add(1)
            .ok_or_else(|| Error::new("BACKEND", "memory version exhausted"))?;
        let updated = Item {
            id: request.id,
            r#type: stored.item.r#type.clone(),
            user_id: stored.item.user_id,
            agent_id: stored.item.agent_id,
            session_id: stored.item.session_id,
            from: request.from,
            to: request.to,
            content: request.content,
            version: next_version,
            created_time: stored.item.created_time,
            updated_time: now_ms(),
            metadata: request.metadata,
        };
        stored.item = updated.clone();
        stored.vector = request.vector;
        Ok(updated)
    }

    async fn delete(
        &self,
        id: i64,
        version: u64,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        Self::ensure_active(cancellation)?;
        let mut records = self.records.write().expect("memory lock poisoned");
        Self::ensure_active(cancellation)?;
        let stored = records
            .get(&id)
            .ok_or_else(|| Error::new("NOT_FOUND", "memory not found"))?;
        if stored.item.version != version {
            return Err(Error::new("CONFLICT", "memory version does not match"));
        }
        records.remove(&id);
        Ok(())
    }

    async fn get(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        r#type: Type,
        cancellation: &Cancellation,
    ) -> Result<Option<Item>, Error> {
        Self::ensure_active(cancellation)?;
        let filter = self.exact_scope(user_id, agent_id, session_id, r#type);
        let records = self.records.read().expect("memory lock poisoned");
        Ok(records
            .values()
            .find(|record| Self::scope_matches(&record.item, &filter))
            .map(|record| record.item.clone()))
    }

    async fn list(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        r#type: Type,
        cancellation: &Cancellation,
    ) -> Result<Vec<Item>, Error> {
        Self::ensure_active(cancellation)?;
        let filter = self.exact_scope(user_id, agent_id, session_id, r#type);
        let records = self.records.read().expect("memory lock poisoned");
        let found = records
            .values()
            .filter(|record| Self::scope_matches(&record.item, &filter))
            .cloned()
            .collect::<Vec<_>>();
        Ok(found.into_iter().map(|record| record.item).collect())
    }

    async fn keyword(
        &self,
        request: KeywordRequest,
        cancellation: &Cancellation,
    ) -> Result<Vec<Item>, Error> {
        Self::ensure_active(cancellation)?;
        let filter = Self::parse_filter(&request.options)?;
        let query = request.query.to_lowercase();
        let records = self.records.read().expect("memory lock poisoned");
        let found = records
            .values()
            .filter(|record| {
                Self::scope_matches(&record.item, &filter)
                    && record.item.content.to_lowercase().contains(&query)
            })
            .cloned()
            .collect::<Vec<_>>();
        Self::paginate_and_sort(found, &request.options)
    }

    async fn vector(
        &self,
        request: VectorRequest,
        cancellation: &Cancellation,
    ) -> Result<Vec<Item>, Error> {
        self.search_vector(&request.vector, &request.options, cancellation, None)
            .await
    }

    async fn hybrid(
        &self,
        request: HybridRequest,
        cancellation: &Cancellation,
    ) -> Result<Vec<Item>, Error> {
        self.search_vector(
            &request.vector,
            &request.options,
            cancellation,
            Some(request.query),
        )
        .await
    }
}

impl InMemory {
    async fn search_vector(
        &self,
        query: &[f32],
        options: &serde_json::Map<String, Value>,
        cancellation: &Cancellation,
        keyword: Option<String>,
    ) -> Result<Vec<Item>, Error> {
        Self::ensure_active(cancellation)?;
        let dimension = self
            .vector_dimension
            .ok_or_else(|| Error::new("UNSUPPORTED", "vector capability is disabled"))?;
        if query.len() != dimension || query.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "query vector dimension is invalid",
            ));
        }
        let filter = Self::parse_filter(options)?;
        let query_text = keyword.map(|value| value.to_lowercase());
        let records = self.records.read().expect("memory lock poisoned");
        let mut scored = records
            .values()
            .filter_map(|record| {
                if !Self::scope_matches(&record.item, &filter) {
                    return None;
                }
                if let Some(text) = &query_text
                    && !record.item.content.to_lowercase().contains(text)
                {
                    return None;
                }
                let vector = record.vector.as_ref()?;
                Some((cosine(query, vector), record.clone()))
            })
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(Ordering::Equal)
                .reverse()
        });
        let (offset, limit) = pagination(options)?;
        Ok(scored
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(_, record)| record.item)
            .collect())
    }
}

fn pagination(options: &serde_json::Map<String, Value>) -> Result<(usize, usize), Error> {
    let invalid = || Error::new("INVALID_ARGUMENTS", "invalid memory pagination");
    let pagination = match options.get("pagination") {
        None => return Ok((0, 20)),
        Some(Value::Object(pagination)) => pagination,
        Some(_) => return Err(invalid()),
    };
    let read = |field, default| match pagination.get(field) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(invalid),
    };
    let offset = read("offset", 0)?;
    let limit = read("limit", 20)?;
    if limit == 0 {
        return Err(Error::new("INVALID_ARGUMENTS", "invalid memory pagination"));
    }
    Ok((offset, limit))
}

fn same_scope_and_type(left: &Item, right: &Item) -> bool {
    left.user_id == right.user_id
        && left.agent_id == right.agent_id
        && left.session_id == right.session_id
        && left.r#type == right.r#type
}

fn request_as_item(request: &Request, version: u64, created_time: i64, updated_time: i64) -> Item {
    Item {
        id: request.id,
        r#type: request.r#type.clone(),
        user_id: request.user_id,
        agent_id: request.agent_id,
        session_id: request.session_id,
        from: request.from,
        to: request.to,
        content: request.content.clone(),
        version,
        created_time,
        updated_time,
        metadata: request.metadata.clone(),
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    // Finite f32 inputs may overflow or underflow when multiplied in f32.
    // Widen before arithmetic so legal vectors keep meaningful scores.
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        (dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(-1.0, 1.0)
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_executor::block_on;
    use serde_json::json;

    fn memory_request(id: i64, content: &str) -> Request {
        Request {
            id,
            r#type: Type::Fact,
            user_id: 7,
            agent_id: Some(9),
            session_id: None,
            from: 0,
            to: 10,
            content: content.into(),
            metadata: None,
            vector: None,
        }
    }

    fn options() -> serde_json::Map<String, Value> {
        serde_json::from_value(json!({
            "filter": {"user_id": 7, "agent_id": 9, "session_id": null, "type": "fact"}
        }))
        .unwrap()
    }

    #[test]
    fn writes_updates_and_deletes_with_version_checks() {
        let memory = InMemory::new();
        let cancellation = Cancellation::new();
        let item = block_on(memory.write(memory_request(1, "北京天气"), &cancellation)).unwrap();
        assert_eq!(item.version, 1);
        assert_eq!(
            block_on(memory.write(memory_request(2, "另一条"), &cancellation))
                .unwrap_err()
                .code,
            "CONFLICT"
        );

        let mut changed = memory_request(1, "北京天气晴");
        changed.metadata = Some(serde_json::Map::from_iter([(
            String::from("source"),
            json!("chat"),
        )]));
        let updated = block_on(memory.update(changed, item.version, &cancellation)).unwrap();
        assert_eq!(updated.version, 2);
        assert_eq!(
            block_on(memory.update(memory_request(1, "过期"), 1, &cancellation))
                .unwrap_err()
                .code,
            "CONFLICT"
        );
        block_on(memory.delete(1, updated.version, &cancellation)).unwrap();
        assert!(
            block_on(memory.get(7, Some(9), None, Type::Fact, &cancellation))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn keyword_requires_exact_scope_and_supports_pagination() {
        let memory = InMemory::new();
        let cancellation = Cancellation::new();
        block_on(memory.write(memory_request(1, "北京天气晴"), &cancellation)).unwrap();
        let mut different = memory_request(2, "上海天气");
        different.r#type = Type::Preference;
        block_on(memory.write(different, &cancellation)).unwrap();
        let result = block_on(memory.keyword(
            KeywordRequest {
                query: "天气".into(),
                options: options(),
            },
            &cancellation,
        ))
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 1);
    }

    #[test]
    fn vector_capability_is_explicit_and_dimension_is_fixed() {
        let cancellation = Cancellation::new();
        let memory = InMemory::new();
        let request = VectorRequest {
            vector: vec![1.0],
            options: options(),
        };
        assert_eq!(
            block_on(memory.vector(request, &cancellation))
                .unwrap_err()
                .code,
            "UNSUPPORTED"
        );

        let memory = InMemory::with_vector_dimension(2).unwrap();
        let mut item = memory_request(3, "北京天气");
        item.vector = Some(vec![1.0, 0.0]);
        block_on(memory.write(item, &cancellation)).unwrap();
        let result = block_on(memory.vector(
            VectorRequest {
                vector: vec![1.0, 0.0],
                options: options(),
            },
            &cancellation,
        ))
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 3);
    }

    #[test]
    fn search_requires_every_scope_field_including_explicit_nulls() {
        let memory = InMemory::with_vector_dimension(2).unwrap();
        let cancellation = Cancellation::new();
        for field in ["user_id", "agent_id", "session_id", "type"] {
            let mut options = options();
            options["filter"].as_object_mut().unwrap().remove(field);
            let error = block_on(memory.keyword(
                KeywordRequest {
                    query: "天气".into(),
                    options: options.clone(),
                },
                &cancellation,
            ))
            .unwrap_err();
            assert_eq!(error.code, "INVALID_ARGUMENTS");
            assert_eq!(error.details, Some(format!("filter.{field}")));
            assert_eq!(
                block_on(memory.vector(
                    VectorRequest {
                        vector: vec![1.0, 0.0],
                        options: options.clone(),
                    },
                    &cancellation
                ))
                .unwrap_err()
                .code,
                "INVALID_ARGUMENTS"
            );
            assert_eq!(
                block_on(memory.hybrid(
                    HybridRequest {
                        query: "天气".into(),
                        vector: vec![1.0, 0.0],
                        options,
                    },
                    &cancellation
                ))
                .unwrap_err()
                .code,
                "INVALID_ARGUMENTS"
            );
        }
        let options = serde_json::from_value(json!({
            "filter": {"user_id": 7, "agent_id": null, "session_id": null, "type": "fact"}
        }))
        .unwrap();
        assert!(
            block_on(memory.keyword(
                KeywordRequest {
                    query: "天气".into(),
                    options
                },
                &cancellation
            ))
            .unwrap()
            .is_empty()
        );
    }

    #[test]
    fn cancellation_does_not_mutate_memory() {
        let memory = InMemory::new();
        let cancellation = Cancellation::new();
        cancellation.cancel();
        assert_eq!(
            block_on(memory.write(memory_request(1, "ignored"), &cancellation))
                .unwrap_err()
                .code,
            "CANCELLED"
        );
    }

    fn malformed_paginations() -> Vec<Value> {
        vec![
            Value::Null,
            json!("20"),
            json!([]),
            json!(true),
            json!({"offset": -1}),
            json!({"offset": "1"}),
            json!({"offset": 1.5}),
            json!({"offset": null}),
            json!({"limit": -1}),
            json!({"limit": "20"}),
            json!({"limit": true}),
            json!({"limit": null}),
            json!({"limit": 0}),
        ]
    }

    #[test]
    fn keyword_rejects_malformed_pagination_instead_of_defaulting() {
        let memory = InMemory::new();
        let cancellation = Cancellation::new();
        for pagination in malformed_paginations() {
            let mut options = options();
            options.insert("pagination".into(), pagination);
            let error = block_on(memory.keyword(
                KeywordRequest {
                    query: "天气".into(),
                    options,
                },
                &cancellation,
            ))
            .unwrap_err();
            assert_eq!(error.code, "INVALID_ARGUMENTS");
        }
    }

    #[test]
    fn vector_rejects_malformed_pagination_instead_of_defaulting() {
        let memory = InMemory::with_vector_dimension(2).unwrap();
        let cancellation = Cancellation::new();
        for pagination in malformed_paginations() {
            let mut options = options();
            options.insert("pagination".into(), pagination);
            let error = block_on(memory.vector(
                VectorRequest {
                    vector: vec![1.0, 0.0],
                    options,
                },
                &cancellation,
            ))
            .unwrap_err();
            assert_eq!(error.code, "INVALID_ARGUMENTS");
        }
    }

    #[test]
    fn hybrid_rejects_malformed_pagination_instead_of_defaulting() {
        let memory = InMemory::with_vector_dimension(2).unwrap();
        let cancellation = Cancellation::new();
        for pagination in malformed_paginations() {
            let mut options = options();
            options.insert("pagination".into(), pagination);
            let error = block_on(memory.hybrid(
                HybridRequest {
                    query: "天气".into(),
                    vector: vec![1.0, 0.0],
                    options,
                },
                &cancellation,
            ))
            .unwrap_err();
            assert_eq!(error.code, "INVALID_ARGUMENTS");
        }
    }

    #[test]
    fn pagination_defaults_only_apply_to_absent_fields() {
        assert_eq!(pagination(&options()).unwrap(), (0, 20));
        for (value, expected) in [
            (json!({}), (0, 20)),
            (json!({"offset": 3}), (3, 20)),
            (json!({"limit": 5}), (0, 5)),
            (json!({"offset": 0, "limit": 5}), (0, 5)),
        ] {
            let mut options = options();
            options.insert("pagination".into(), value);
            assert_eq!(pagination(&options).unwrap(), expected);
        }
    }

    #[test]
    fn keyword_rejects_wrong_sort_types() {
        let memory = InMemory::new();
        let cancellation = Cancellation::new();
        for sort in [
            json!({"field": null}),
            json!({"field": 7}),
            json!({"order": false}),
            json!({"order": ["desc"]}),
        ] {
            let mut options = options();
            options.insert("sort".into(), sort);
            let error = block_on(memory.keyword(
                KeywordRequest {
                    query: "天气".into(),
                    options,
                },
                &cancellation,
            ))
            .unwrap_err();
            assert_eq!(error.code, "INVALID_ARGUMENTS");
        }
    }

    #[test]
    fn exhausted_version_leaves_entire_record_and_vector_unchanged() {
        let memory = InMemory::with_vector_dimension(2).unwrap();
        let cancellation = Cancellation::new();
        let mut original = memory_request(1, "original");
        original.vector = Some(vec![1.0, 0.0]);
        block_on(memory.write(original, &cancellation)).unwrap();
        let before = {
            let mut records = memory.records.write().unwrap();
            let stored = records.get_mut(&1).unwrap();
            stored.item.version = u64::MAX;
            stored.clone()
        };
        let mut replacement = memory_request(1, "replacement");
        replacement.from = 10;
        replacement.to = 20;
        replacement.vector = Some(vec![0.0, 1.0]);
        replacement.metadata = Some(serde_json::Map::from_iter([(
            "changed".into(),
            json!(true),
        )]));
        let error = block_on(memory.update(replacement, u64::MAX, &cancellation)).unwrap_err();
        assert_eq!(error.code, "BACKEND");
        let records = memory.records.read().unwrap();
        assert_eq!(records[&1].item, before.item);
        assert_eq!(records[&1].vector, before.vector);
    }

    #[test]
    fn cosine_remains_finite_for_extreme_finite_vectors() {
        for magnitude in [f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)] {
            let forward = [magnitude, magnitude];
            let opposite = [-magnitude, -magnitude];
            let orthogonal = [magnitude, -magnitude];
            for (other, expected) in [(forward, 1.0), (opposite, -1.0), (orthogonal, 0.0)] {
                let score = cosine(&forward, &other);
                assert!(score.is_finite());
                assert!((score - expected).abs() < 1e-12);
            }
        }
        assert_eq!(cosine(&[0.0, 0.0], &[f32::MAX, f32::MAX]), 0.0);
    }
}
