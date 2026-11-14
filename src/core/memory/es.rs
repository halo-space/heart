//! Elasticsearch implementations for the existing Memory contracts.
//!
//! The application owns the initialized `elasticsearch::Elasticsearch` client,
//! index mappings, credentials and index lifecycle. These adapters only map
//! the public Memory/short-memory contracts to one prepared index; they do not
//! create indices, call Embed, retry requests or manage Runtime state.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use elasticsearch::Elasticsearch;
use serde_json::{Map, Value, json};

use crate::components::memory::{self, Error, Filter, Item, Request, Type};
use crate::{Cancellation, cancellation::check};

mod backend;
pub(crate) use backend::{Backend, StoredMemory, source_with_vector};

/// Long-term Memory stored in one application-prepared Elasticsearch index.
pub struct Memory {
    backend: Arc<Backend>,
    vector_dimension: Option<usize>,
}

impl Memory {
    pub fn new(client: Elasticsearch, index: impl Into<String>) -> Result<Self, Error> {
        Self::with_dimension(client, index, None)
    }

    pub fn with_vector_dimension(
        client: Elasticsearch,
        index: impl Into<String>,
        dimension: usize,
    ) -> Result<Self, Error> {
        if dimension == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "vector dimension must be positive",
            ));
        }
        Self::with_dimension(client, index, Some(dimension))
    }

    fn with_dimension(
        client: Elasticsearch,
        index: impl Into<String>,
        vector_dimension: Option<usize>,
    ) -> Result<Self, Error> {
        Ok(Self {
            backend: Arc::new(Backend::new(client, index)?),
            vector_dimension,
        })
    }
}

impl crate::components::memory::Memory for Memory {
    async fn write(&self, request: Request, cancellation: &Cancellation) -> Result<Item, Error> {
        check(cancellation)?;
        validate_request(&request, self.vector_dimension)?;
        self.backend.ensure_index().await?;
        let item = Item {
            id: request.id,
            r#type: request.r#type,
            user_id: request.user_id,
            agent_id: request.agent_id,
            session_id: request.session_id,
            from: request.from,
            to: request.to,
            content: request.content,
            version: 1,
            created_time: now_ms(),
            updated_time: now_ms(),
            metadata: request.metadata,
        };
        if self
            .backend
            .find_scope(item.user_id, item.agent_id, item.session_id, &item.r#type)
            .await?
            .is_some()
        {
            return Err(Error::new(
                "CONFLICT",
                "memory scope and type already exists",
            ));
        }
        let source = source_with_vector(&item, request.vector);
        self.backend.create(item.id, source).await?;
        Ok(item)
    }

    async fn update(
        &self,
        request: Request,
        version: u64,
        cancellation: &Cancellation,
    ) -> Result<Item, Error> {
        check(cancellation)?;
        validate_request(&request, self.vector_dimension)?;
        let current = self
            .backend
            .get::<StoredMemory>(request.id)
            .await?
            .ok_or_else(|| Error::new("NOT_FOUND", "memory not found"))?;
        if current.source.item.version != version {
            return Err(Error::new("CONFLICT", "memory version is stale"));
        }
        if current.source.item.user_id != request.user_id
            || current.source.item.agent_id != request.agent_id
            || current.source.item.session_id != request.session_id
            || current.source.item.r#type != request.r#type
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "memory ownership and type cannot change",
            ));
        }
        let next_version = version
            .checked_add(1)
            .ok_or_else(|| Error::new("BACKEND", "memory version exhausted"))?;
        let item = Item {
            id: request.id,
            r#type: request.r#type,
            user_id: request.user_id,
            agent_id: request.agent_id,
            session_id: request.session_id,
            from: request.from,
            to: request.to,
            content: request.content,
            version: next_version,
            created_time: current.source.item.created_time,
            updated_time: now_ms(),
            metadata: request.metadata,
        };
        self.backend
            .replace(
                request.id,
                source_with_vector(&item, request.vector),
                Some((current.seq_no, current.primary_term)),
            )
            .await?;
        Ok(item)
    }

    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        check(cancellation)?;
        self.backend.delete(id).await
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
        Ok(self
            .backend
            .find_scope(user_id, agent_id, session_id, &r#type)
            .await?
            .map(|record| record.item))
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
        Ok(self
            .backend
            .search_scope(user_id, agent_id, session_id, &r#type, 1000)
            .await?
            .into_iter()
            .map(|record| record.source.item)
            .collect())
    }

    async fn keyword(
        &self,
        request: memory::keyword::Request,
        cancellation: &Cancellation,
    ) -> Result<Vec<Item>, Error> {
        check(cancellation)?;
        let filter = parse_filter(&request.options)?;
        let (offset, size) = parse_pagination(&request.options)?;
        Ok(self
            .backend
            .search_text(&request.query, &filter, offset, size)
            .await?
            .into_iter()
            .map(|record| record.source.item)
            .collect())
    }

    async fn vector(
        &self,
        request: memory::vector::Request,
        cancellation: &Cancellation,
    ) -> Result<Vec<Item>, Error> {
        check(cancellation)?;
        let dimension = self
            .vector_dimension
            .ok_or_else(|| Error::new("UNSUPPORTED", "vector capability is disabled"))?;
        if request.vector.len() != dimension {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "vector dimension does not match storage",
            ));
        }
        let filter = parse_filter(&request.options)?;
        let (offset, size) = parse_pagination(&request.options)?;
        Ok(self
            .backend
            .search_vector(request.vector, &filter, offset, size)
            .await?
            .into_iter()
            .map(|record| record.source.item)
            .collect())
    }

    async fn hybrid(
        &self,
        request: memory::hybrid::Request,
        cancellation: &Cancellation,
    ) -> Result<Vec<Item>, Error> {
        check(cancellation)?;
        let dimension = self
            .vector_dimension
            .ok_or_else(|| Error::new("UNSUPPORTED", "vector capability is disabled"))?;
        if request.vector.len() != dimension {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "vector dimension does not match storage",
            ));
        }
        let filter = parse_filter(&request.options)?;
        let (offset, size) = parse_pagination(&request.options)?;
        Ok(self
            .backend
            .search_hybrid(&request.query, request.vector, &filter, offset, size)
            .await?
            .into_iter()
            .map(|record| record.source.item)
            .collect())
    }
}

/// Short-memory ES stores. Session and Chat remain independent component
/// contracts while one backend owns their construction.
pub mod short;

fn parse_filter(options: &Map<String, Value>) -> Result<Filter, Error> {
    let filter = options
        .get("filter")
        .cloned()
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "memory search requires options.filter"))?;
    serde_json::from_value(filter)
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid memory filter"))
}

pub(crate) fn parse_chat_options(
    options: &Map<String, Value>,
) -> Result<(Vec<Value>, bool, usize, usize), Error> {
    let invalid = || Error::new("INVALID_ARGUMENTS", "invalid chat history options");
    if options
        .keys()
        .any(|key| !matches!(key.as_str(), "filter" | "sort" | "pagination"))
    {
        return Err(invalid());
    }
    let filter = options
        .get("filter")
        .and_then(Value::as_object)
        .ok_or_else(invalid)?;
    let mut filters = Vec::new();
    for field in ["tenant_id", "user_id", "agent_id", "session_id"] {
        let id = filter
            .get(field)
            .and_then(Value::as_i64)
            .filter(|id| *id > 0)
            .ok_or_else(invalid)?;
        filters.push(json!({"term": {field: id}}));
    }
    let r#type: crate::components::memory::short::Type =
        serde_json::from_value(filter.get("type").cloned().ok_or_else(invalid)?)
            .map_err(|_| invalid())?;
    filters.push(json!({"term": {"type": r#type}}));
    if let Some(range) = filter.get("created_time") {
        let range = range.as_object().ok_or_else(invalid)?;
        if range.len() != 2 {
            return Err(invalid());
        }
        let from = range
            .get("from")
            .and_then(Value::as_i64)
            .ok_or_else(invalid)?;
        let to = range
            .get("to")
            .and_then(Value::as_i64)
            .ok_or_else(invalid)?;
        if from >= to {
            return Err(invalid());
        }
        filters.push(json!({"range": {"created_time": {"gte": from, "lt": to}}}));
    }
    if filter.keys().any(|key| {
        !matches!(
            key.as_str(),
            "tenant_id" | "user_id" | "agent_id" | "session_id" | "type" | "created_time"
        )
    }) {
        return Err(invalid());
    }
    let descending = match options.get("sort") {
        None => false,
        Some(Value::Object(sort))
            if sort.len() == 2
                && sort.get("field").and_then(Value::as_str) == Some("created_time") =>
        {
            match sort.get("order").and_then(Value::as_str) {
                Some("asc") => false,
                Some("desc") => true,
                _ => return Err(invalid()),
            }
        }
        Some(_) => return Err(invalid()),
    };
    let (offset, limit) = match options.get("pagination") {
        None => (0, 10_000),
        Some(Value::Object(page))
            if page
                .keys()
                .all(|key| matches!(key.as_str(), "offset" | "limit")) =>
        {
            let number = |key: &str, default: usize| -> Result<usize, Error> {
                match page.get(key) {
                    None => Ok(default),
                    Some(value) => value
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(invalid),
                }
            };
            let limit = number("limit", 10_000)?;
            if limit == 0 {
                return Err(invalid());
            }
            (number("offset", 0)?, limit)
        }
        Some(_) => return Err(invalid()),
    };
    Ok((filters, descending, offset, limit))
}

fn parse_pagination(options: &Map<String, Value>) -> Result<(usize, usize), Error> {
    let pagination = match options.get("pagination") {
        None => return Ok((0, 20)),
        Some(Value::Object(pagination)) => pagination,
        Some(_) => return Err(Error::new("INVALID_ARGUMENTS", "invalid memory pagination")),
    };
    let offset = pagination
        .get("offset")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let value = pagination
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(20);
    if value == 0 {
        return Err(Error::new("INVALID_ARGUMENTS", "invalid memory pagination"));
    }
    Ok((
        usize::try_from(offset)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid memory pagination offset"))?,
        usize::try_from(value)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid memory pagination limit"))?,
    ))
}

fn validate_request(request: &Request, dimension: Option<usize>) -> Result<(), Error> {
    if request.id <= 0
        || request.user_id <= 0
        || request.from < 0
        || request.to <= request.from
        || request.content.trim().is_empty()
    {
        return Err(Error::new("INVALID_ARGUMENTS", "invalid memory request"));
    }
    if request.session_id.is_some() && request.agent_id.is_none() {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "session scope requires agent_id",
        ));
    }
    match (dimension, request.vector.as_ref()) {
        (None, Some(_)) => Err(Error::new("UNSUPPORTED", "vector capability is disabled")),
        (Some(_), None) => Err(Error::new("INVALID_ARGUMENTS", "vector is required")),
        (Some(expected), Some(vector)) if vector.len() != expected => Err(Error::new(
            "INVALID_ARGUMENTS",
            "vector dimension does not match storage",
        )),
        (Some(_), Some(vector)) if vector.iter().any(|value| !value.is_finite()) => Err(
            Error::new("INVALID_ARGUMENTS", "vector contains a non-finite value"),
        ),
        _ => Ok(()),
    }
}

fn validate_session(
    value: &crate::components::memory::short::session::Session,
) -> Result<(), Error> {
    if value.id <= 0 || value.tenant_id <= 0 || value.user_id <= 0 {
        Err(Error::new(
            "INVALID_ARGUMENTS",
            "session identifiers must be positive",
        ))
    } else {
        Ok(())
    }
}

fn validate_chat(value: &crate::components::memory::short::chat::Chat) -> Result<(), Error> {
    if value.id <= 0
        || value.tenant_id <= 0
        || value.user_id <= 0
        || value.agent_id <= 0
        || value.session_id <= 0
    {
        Err(Error::new(
            "INVALID_ARGUMENTS",
            "chat identifiers must be positive",
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_index(index: &str) -> Result<(), Error> {
    if index.is_empty()
        || index.len() > 255
        || index.starts_with(['-', '_', '+'])
        || index.chars().any(|c| {
            c.is_whitespace() || c.is_control() || c.is_uppercase() || "\\/*?\"<>|,#:%".contains(c)
        })
    {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "memory requires one concrete Elasticsearch index",
        ));
    }
    Ok(())
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn encode_error(error: serde_json::Error) -> Error {
    Error::new(
        "INVALID_ARGUMENTS",
        format!("cannot encode memory record: {error}"),
    )
}
fn decode_error(error: serde_json::Error) -> Error {
    Error::new(
        "INVALID_RESPONSE",
        format!("invalid memory record: {error}"),
    )
}
fn invalid_response() -> Error {
    Error::new("INVALID_RESPONSE", "invalid Elasticsearch response")
}
fn transport_error(error: elasticsearch::Error) -> Error {
    Error::new(
        "UNAVAILABLE",
        format!("Elasticsearch request failed: {error}"),
    )
}
fn success(status: u16) -> Result<(), Error> {
    if (200..300).contains(&status) {
        Ok(())
    } else {
        Err(status_error(status))
    }
}
fn status_error(status: u16) -> Error {
    let code = match status {
        400 | 422 => "INVALID_ARGUMENTS",
        401 => "AUTHENTICATION",
        403 => "PERMISSION_DENIED",
        404 => "NOT_FOUND",
        408 | 504 => "TIMEOUT",
        409 => "CONFLICT",
        429 => "RATE_LIMITED",
        500..=599 => "UNAVAILABLE",
        _ => "INVALID_RESPONSE",
    };
    Error::new(code, "Elasticsearch request failed").with_details(format!("http_status={status}"))
}
