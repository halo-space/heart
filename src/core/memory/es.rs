//! Elasticsearch implementations for the existing Memory contracts.
//!
//! The application owns the initialized `elasticsearch::Elasticsearch` client,
//! index mappings, credentials and index lifecycle. These adapters only map
//! the public Memory/short-memory contracts to one prepared index; they do not
//! create indices, call Embed, retry requests or manage Runtime state.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use elasticsearch::{
    CreateParts, Elasticsearch, GetParts, IndexParts, SearchParts, indices::IndicesExistsParts,
};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::Cancellation;
use crate::components::memory::short as short_contracts;
use crate::components::memory::{
    self, Error, Filter, Item, Memory as MemoryContract, Request, Type,
};

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

impl MemoryContract for Memory {
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
pub mod short {
    use super::*;

    /// Index names used by the two short-memory records.
    ///
    /// Keeping these names together lets an application construct one
    /// short-memory backend without exposing how the stores are split
    /// internally.
    pub struct Config {
        pub session_index: String,
        pub chat_index: String,
    }

    impl Config {
        pub fn new(session_index: impl Into<String>, chat_index: impl Into<String>) -> Self {
            Self {
                session_index: session_index.into(),
                chat_index: chat_index.into(),
            }
        }
    }

    pub struct Session {
        backend: Arc<Backend>,
    }

    pub struct Chat {
        backend: Arc<Backend>,
    }

    /// One short-memory backend containing Session and Chat stores.
    ///
    /// The two stores keep their independent component contracts, while the
    /// application only constructs this backend once.
    pub struct Memory {
        session: Session,
        chat: Chat,
    }

    impl Memory {
        pub fn new(client: Elasticsearch, config: Config) -> Result<Self, Error> {
            let session = Session::new(client.clone(), config.session_index)?;
            let chat = Chat::new(client, config.chat_index)?;
            Ok(Self { session, chat })
        }
    }

    impl short_contracts::Store for Memory {
        type Session = Session;
        type Chat = Chat;

        fn session(&self) -> &Self::Session {
            &self.session
        }

        fn chat(&self) -> &Self::Chat {
            &self.chat
        }
    }

    impl Session {
        pub fn new(client: Elasticsearch, index: impl Into<String>) -> Result<Self, Error> {
            Ok(Self {
                backend: Arc::new(Backend::new(client, index)?),
            })
        }
    }

    impl Chat {
        pub fn new(client: Elasticsearch, index: impl Into<String>) -> Result<Self, Error> {
            Ok(Self {
                backend: Arc::new(Backend::new(client, index)?),
            })
        }
    }

    impl short_contracts::session::Store for Session {
        async fn create(
            &self,
            mut value: short_contracts::session::Session,
            cancellation: &Cancellation,
        ) -> Result<short_contracts::session::Session, Error> {
            check(cancellation)?;
            validate_session(&value)?;
            self.backend.ensure_index().await?;
            value.created_time = now_ms();
            value.updated_time = value.created_time;
            self.backend
                .create(
                    value.id,
                    serde_json::to_value(&value).map_err(encode_error)?,
                )
                .await?;
            Ok(value)
        }

        async fn read(
            &self,
            id: i64,
            cancellation: &Cancellation,
        ) -> Result<short_contracts::session::Session, Error> {
            check(cancellation)?;
            self.backend
                .get(id)
                .await?
                .map(|raw| raw.source)
                .ok_or_else(|| Error::new("NOT_FOUND", "session not found"))
        }

        async fn update(
            &self,
            mut value: short_contracts::session::Session,
            cancellation: &Cancellation,
        ) -> Result<short_contracts::session::Session, Error> {
            check(cancellation)?;
            validate_session(&value)?;
            let current = self
                .backend
                .get::<short_contracts::session::Session>(value.id)
                .await?
                .ok_or_else(|| Error::new("NOT_FOUND", "session not found"))?;
            if current.source.tenant_id != value.tenant_id
                || current.source.user_id != value.user_id
                || current.source.r#type != value.r#type
            {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "session ownership and type cannot change",
                ));
            }
            if current.source.updated_time != value.updated_time {
                return Err(Error::new("CONFLICT", "session snapshot is stale"));
            }
            value.created_time = current.source.created_time;
            value.updated_time = now_ms();
            self.backend
                .replace(
                    value.id,
                    serde_json::to_value(&value).map_err(encode_error)?,
                    Some((current.seq_no, current.primary_term)),
                )
                .await?;
            Ok(value)
        }

        async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
            check(cancellation)?;
            self.backend.delete(id).await
        }
    }

    impl short_contracts::chat::Store for Chat {
        async fn create(
            &self,
            mut value: short_contracts::chat::Chat,
            cancellation: &Cancellation,
        ) -> Result<short_contracts::chat::Chat, Error> {
            check(cancellation)?;
            validate_chat(&value)?;
            self.backend.ensure_index().await?;
            value.created_time = now_ms();
            value.updated_time = value.created_time;
            self.backend
                .create(
                    value.id,
                    serde_json::to_value(&value).map_err(encode_error)?,
                )
                .await?;
            Ok(value)
        }

        async fn read(
            &self,
            id: i64,
            cancellation: &Cancellation,
        ) -> Result<short_contracts::chat::Chat, Error> {
            check(cancellation)?;
            self.backend
                .get(id)
                .await?
                .map(|raw| raw.source)
                .ok_or_else(|| Error::new("NOT_FOUND", "chat not found"))
        }

        async fn update(
            &self,
            mut value: short_contracts::chat::Chat,
            cancellation: &Cancellation,
        ) -> Result<short_contracts::chat::Chat, Error> {
            check(cancellation)?;
            validate_chat(&value)?;
            let current = self
                .backend
                .get::<short_contracts::chat::Chat>(value.id)
                .await?
                .ok_or_else(|| Error::new("NOT_FOUND", "chat not found"))?;
            if current.source.tenant_id != value.tenant_id
                || current.source.user_id != value.user_id
                || current.source.agent_id != value.agent_id
                || current.source.session_id != value.session_id
                || current.source.r#type != value.r#type
            {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "chat ownership and type cannot change",
                ));
            }
            if current.source.input != value.input {
                return Err(Error::new("INVALID_ARGUMENTS", "chat input cannot change"));
            }
            value.created_time = current.source.created_time;
            value.updated_time = now_ms();
            self.backend
                .replace(
                    value.id,
                    serde_json::to_value(&value).map_err(encode_error)?,
                    Some((current.seq_no, current.primary_term)),
                )
                .await?;
            Ok(value)
        }

        async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
            check(cancellation)?;
            self.backend.delete(id).await
        }

        async fn list(
            &self,
            options: Map<String, Value>,
            cancellation: &Cancellation,
        ) -> Result<Vec<short_contracts::chat::Chat>, Error> {
            check(cancellation)?;
            let (filters, descending, offset, limit) = parse_chat_options(&options)?;
            let order = if descending { "desc" } else { "asc" };
            let body = json!({
                "from": offset,
                "size": limit,
                "query": {"bool": {"filter": filters}},
                "sort": [
                    {"created_time": {"order": order}},
                    {"id": {"order": order}}
                ]
            });
            Ok(self
                .backend
                .search::<short_contracts::chat::Chat>(body)
                .await?
                .into_iter()
                .map(|record| record.source)
                .collect())
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
struct StoredMemory {
    #[serde(flatten)]
    item: Item,
}

struct Raw<T> {
    source: T,
    seq_no: i64,
    primary_term: i64,
}

struct Backend {
    client: Elasticsearch,
    index: String,
}

impl Backend {
    fn new(client: Elasticsearch, index: impl Into<String>) -> Result<Self, Error> {
        let index = index.into();
        validate_index(&index)?;
        Ok(Self { client, index })
    }

    async fn ensure_index(&self) -> Result<(), Error> {
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

    async fn create(&self, id: i64, source: Value) -> Result<(), Error> {
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

    async fn get<T: DeserializeOwned>(&self, id: i64) -> Result<Option<Raw<T>>, Error> {
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

    async fn replace(
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

    async fn delete(&self, id: i64) -> Result<(), Error> {
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

    async fn search_scope(
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

    async fn find_scope(
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

    async fn search_text(
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

    async fn search_vector(
        &self,
        vector: Vec<f32>,
        filter: &Filter,
        offset: usize,
        size: usize,
    ) -> Result<Vec<Raw<StoredMemory>>, Error> {
        let body = json!({"from": offset, "size": size, "knn": {"field": "vector", "query_vector": vector, "k": size.max(1), "num_candidates": size.saturating_mul(10).max(10), "filter": {"bool": {"filter": scope_filters(filter.user_id, filter.agent_id, filter.session_id, &filter.r#type)}}}});
        self.search(body).await
    }

    async fn search_hybrid(
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

    async fn search<T: DeserializeOwned>(&self, body: Value) -> Result<Vec<Raw<T>>, Error> {
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

fn source_with_vector(item: &Item, vector: Option<Vec<f32>>) -> Value {
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

fn scope_filters(
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

fn parse_filter(options: &Map<String, Value>) -> Result<Filter, Error> {
    let filter = options
        .get("filter")
        .cloned()
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "memory search requires options.filter"))?;
    serde_json::from_value(filter)
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid memory filter"))
}

fn parse_chat_options(
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
    let r#type: short_contracts::Type =
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

fn validate_session(value: &short_contracts::session::Session) -> Result<(), Error> {
    if value.id <= 0 || value.tenant_id <= 0 || value.user_id <= 0 {
        Err(Error::new(
            "INVALID_ARGUMENTS",
            "session identifiers must be positive",
        ))
    } else {
        Ok(())
    }
}

fn validate_chat(value: &short_contracts::chat::Chat) -> Result<(), Error> {
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

fn validate_index(index: &str) -> Result<(), Error> {
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

fn check(cancellation: &Cancellation) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        Err(Error::new("CANCELLED", "memory operation cancelled"))
    } else {
        Ok(())
    }
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
