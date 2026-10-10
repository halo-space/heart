use std::collections::BTreeMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

use crate::components::memory::short::{Error, chat};
use crate::runtime::cancellation::Cancellation;

pub struct InMemory {
    records: RwLock<BTreeMap<i64, chat::Chat>>,
}

impl InMemory {
    pub fn new() -> Self {
        Self {
            records: RwLock::new(BTreeMap::new()),
        }
    }
}
impl Default for InMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl chat::Store for InMemory {
    async fn create(
        &self,
        mut value: chat::Chat,
        cancellation: &Cancellation,
    ) -> Result<chat::Chat, Error> {
        check(cancellation)?;
        validate(&value)?;
        let now = now_ms();
        value.created_time = now;
        value.updated_time = now;
        let mut records = self.records.write().expect("chat lock poisoned");
        check(cancellation)?;
        if records.contains_key(&value.id) {
            return Err(Error::new("CONFLICT", "chat id already exists"));
        }
        records.insert(value.id, value.clone());
        Ok(value)
    }

    async fn read(&self, id: i64, cancellation: &Cancellation) -> Result<chat::Chat, Error> {
        check(cancellation)?;
        self.records
            .read()
            .expect("chat lock poisoned")
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::new("NOT_FOUND", "chat not found"))
    }

    async fn update(
        &self,
        mut value: chat::Chat,
        cancellation: &Cancellation,
    ) -> Result<chat::Chat, Error> {
        check(cancellation)?;
        validate(&value)?;
        let mut records = self.records.write().expect("chat lock poisoned");
        check(cancellation)?;
        let old = records
            .get(&value.id)
            .ok_or_else(|| Error::new("NOT_FOUND", "chat not found"))?;
        if old.tenant_id != value.tenant_id
            || old.user_id != value.user_id
            || old.agent_id != value.agent_id
            || old.session_id != value.session_id
            || old.r#type != value.r#type
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "chat ownership and type cannot change",
            ));
        }
        if old.input != value.input {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "chat input cannot change; a new request requires a new chat",
            ));
        }
        value.created_time = old.created_time;
        value.updated_time = now_ms();
        records.insert(value.id, value.clone());
        Ok(value)
    }

    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        check(cancellation)?;
        let mut records = self.records.write().expect("chat lock poisoned");
        check(cancellation)?;
        if records.remove(&id).is_none() {
            return Err(Error::new("NOT_FOUND", "chat not found"));
        }
        Ok(())
    }

    async fn list(
        &self,
        options: Map<String, Value>,
        cancellation: &Cancellation,
    ) -> Result<Vec<chat::Chat>, Error> {
        check(cancellation)?;
        let query = Query::parse(&options)?;
        let records = self.records.read().expect("chat lock poisoned");
        let mut matches = Vec::new();
        for record in records.values() {
            check(cancellation)?;
            if query.matches(record) {
                matches.push(record);
            }
        }
        matches.sort_by_key(|record| (record.created_time, record.id));
        if query.descending {
            matches.reverse();
        }
        let result = matches
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .cloned()
            .collect();
        check(cancellation)?;
        Ok(result)
    }
}

struct Query {
    tenant_id: i64,
    user_id: i64,
    agent_id: i64,
    session_id: i64,
    r#type: crate::memory::short::Type,
    start: Option<i64>,
    end: Option<i64>,
    descending: bool,
    offset: usize,
    limit: usize,
}

impl Query {
    fn parse(options: &Map<String, Value>) -> Result<Self, Error> {
        fn invalid() -> Error {
            Error::new("INVALID_ARGUMENTS", "invalid chat history options")
        }
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
        if filter.keys().any(|key| {
            !matches!(
                key.as_str(),
                "tenant_id" | "user_id" | "agent_id" | "session_id" | "type" | "created_time"
            )
        }) {
            return Err(invalid());
        }
        let id = |field: &str| {
            filter
                .get(field)
                .and_then(Value::as_i64)
                .filter(|id| *id > 0)
                .ok_or_else(invalid)
        };
        let r#type = serde_json::from_value(filter.get("type").cloned().ok_or_else(invalid)?)
            .map_err(|_| invalid())?;
        let (start, end) = match filter.get("created_time") {
            None => (None, None),
            Some(Value::Object(range)) if range.len() == 2 => {
                let start = range
                    .get("from")
                    .and_then(Value::as_i64)
                    .ok_or_else(invalid)?;
                let end = range
                    .get("to")
                    .and_then(Value::as_i64)
                    .ok_or_else(invalid)?;
                if start >= end {
                    return Err(invalid());
                }
                (Some(start), Some(end))
            }
            Some(_) => return Err(invalid()),
        };
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
            None => (0, usize::MAX),
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
                let limit = number("limit", usize::MAX)?;
                if limit == 0 {
                    return Err(invalid());
                }
                (number("offset", 0)?, limit)
            }
            Some(_) => return Err(invalid()),
        };
        Ok(Self {
            tenant_id: id("tenant_id")?,
            user_id: id("user_id")?,
            agent_id: id("agent_id")?,
            session_id: id("session_id")?,
            r#type,
            start,
            end,
            descending,
            offset,
            limit,
        })
    }

    fn matches(&self, record: &chat::Chat) -> bool {
        record.tenant_id == self.tenant_id
            && record.user_id == self.user_id
            && record.agent_id == self.agent_id
            && record.session_id == self.session_id
            && record.r#type == self.r#type
            && self.start.is_none_or(|start| record.created_time >= start)
            && self.end.is_none_or(|end| record.created_time < end)
    }
}

fn check(cancellation: &Cancellation) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        Err(Error::new("CANCELLED", "chat operation cancelled"))
    } else {
        Ok(())
    }
}
fn validate(value: &chat::Chat) -> Result<(), Error> {
    if value.id <= 0 || value.tenant_id <= 0 || value.user_id <= 0 {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "chat identifiers must be positive",
        ));
    }
    if value.agent_id <= 0 || value.session_id <= 0 {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "chat identifiers must be positive",
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
