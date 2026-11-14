use std::collections::BTreeMap;

use tokio::sync::RwLock;

use serde_json::Value;

use crate::components::memory::{
    Error, Filter, HybridRequest, Item, KeywordRequest, Memory, Request, Type, VectorRequest,
};
use crate::{Cancellation, cancellation::check};

// The Elasticsearch adapter is kept private at the implementation root. It is
// re-exported below through the semantic `long` and `short` memory paths so
// callers always choose the memory kind before the backend.
mod es;
pub mod extraction;
mod helpers;
pub mod short;

use helpers::*;

/// Long-term memory implementations grouped by backend.
pub mod long {
    /// Elasticsearch implementation of the long-term Memory contract.
    pub mod es {
        use elasticsearch::Elasticsearch;

        /// Construct the long-memory ES backend without exposing an additional
        /// public type in the application-facing path.
        pub fn new(
            client: Elasticsearch,
            index: impl Into<String>,
        ) -> Result<impl crate::components::memory::Memory, crate::Error> {
            super::super::es::Memory::new(client, index)
        }

        /// Construct a long-memory ES backend with an explicit fixed vector dimension.
        pub fn with_vector_dimension(
            client: Elasticsearch,
            index: impl Into<String>,
            dimension: usize,
        ) -> Result<impl crate::components::memory::Memory, crate::Error> {
            super::super::es::Memory::with_vector_dimension(client, index, dimension)
        }
    }
}

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
        check(cancellation)?;
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
        let mut records = self.records.write().await;
        check(cancellation)?;
        if records.contains_key(&item.id) {
            return Err(Error::new("CONFLICT", "memory id already exists"));
        }
        if records
            .values()
            .any(|record| identity_matches(&record.item, &item))
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
        check(cancellation)?;
        self.validate_request(&request)?;
        let mut records = self.records.write().await;
        check(cancellation)?;
        let stored = records
            .get_mut(&request.id)
            .ok_or_else(|| Error::new("NOT_FOUND", "memory not found"))?;
        if stored.item.version != version {
            return Err(Error::new("CONFLICT", "memory version does not match"));
        }
        if !identity_matches(
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

    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        check(cancellation)?;
        let mut records = self.records.write().await;
        check(cancellation)?;
        if records.remove(&id).is_none() {
            return Err(Error::new("NOT_FOUND", "memory not found"));
        }
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
        check(cancellation)?;
        let filter = self.exact_scope(user_id, agent_id, session_id, r#type);
        let records = self.records.read().await;
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
        check(cancellation)?;
        let filter = self.exact_scope(user_id, agent_id, session_id, r#type);
        let records = self.records.read().await;
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
        check(cancellation)?;
        let filter = Self::parse_filter(&request.options)?;
        let query = request.query.to_lowercase();
        let records = self.records.read().await;
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
        check(cancellation)?;
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
        let records = self.records.read().await;
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
                .unwrap_or(std::cmp::Ordering::Equal)
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

#[cfg(test)]
#[path = "../../tests/unit/embedded/src_core_memory_tests.rs"]
mod tests;
