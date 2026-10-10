use std::collections::BTreeMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::components::memory::short::{Error, trace};
use crate::runtime::cancellation::Cancellation;

pub struct InMemory {
    records: RwLock<BTreeMap<i64, trace::Trace>>,
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

impl trace::Store for InMemory {
    async fn create(
        &self,
        mut value: trace::Trace,
        cancellation: &Cancellation,
    ) -> Result<trace::Trace, Error> {
        check(cancellation)?;
        validate(&value)?;
        let now = now_ms();
        value.created_time = now;
        value.updated_time = now;
        let mut records = self.records.write().expect("trace lock poisoned");
        check(cancellation)?;
        if records.contains_key(&value.chat_id) {
            return Err(Error::new("CONFLICT", "trace already exists for chat"));
        }
        records.insert(value.chat_id, value.clone());
        Ok(value)
    }

    async fn read(&self, chat_id: i64, cancellation: &Cancellation) -> Result<trace::Trace, Error> {
        check(cancellation)?;
        self.records
            .read()
            .expect("trace lock poisoned")
            .get(&chat_id)
            .cloned()
            .ok_or_else(|| Error::new("NOT_FOUND", "trace not found"))
    }

    async fn update(
        &self,
        mut value: trace::Trace,
        cancellation: &Cancellation,
    ) -> Result<trace::Trace, Error> {
        check(cancellation)?;
        validate(&value)?;
        let mut records = self.records.write().expect("trace lock poisoned");
        check(cancellation)?;
        let old = records
            .get(&value.chat_id)
            .ok_or_else(|| Error::new("NOT_FOUND", "trace not found"))?;
        // Optional projection fields may be filled later but an established
        // owner cannot be moved or erased by a subsequent projection update.
        if old.user_id.is_some() && old.user_id != value.user_id
            || old.session_id.is_some() && old.session_id != value.session_id
            || old.agent_id.is_some() && old.agent_id != value.agent_id
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "trace ownership cannot change",
            ));
        }
        value.created_time = old.created_time;
        value.updated_time = now_ms();
        records.insert(value.chat_id, value.clone());
        Ok(value)
    }

    async fn delete(&self, chat_id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        check(cancellation)?;
        let mut records = self.records.write().expect("trace lock poisoned");
        check(cancellation)?;
        if records.remove(&chat_id).is_none() {
            return Err(Error::new("NOT_FOUND", "trace not found"));
        }
        Ok(())
    }
}

fn check(cancellation: &Cancellation) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        Err(Error::new("CANCELLED", "trace operation cancelled"))
    } else {
        Ok(())
    }
}

fn validate(value: &trace::Trace) -> Result<(), Error> {
    if value.chat_id <= 0 {
        Err(Error::new("INVALID_ARGUMENTS", "chat_id must be positive"))
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
