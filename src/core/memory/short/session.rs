use std::collections::BTreeMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::components::memory::short::{Error, session};
use crate::runtime::cancellation::Cancellation;

/// Independent in-memory Session CRUD. `update` uses the submitted snapshot's
/// `updated_time` for optimistic concurrency; read latest after `CONFLICT` and
/// recompute changes. Summary, coverage metadata and current Usage replace as
/// one record. This backend does not schedule compression or create a Runtime.
pub struct InMemory {
    records: RwLock<BTreeMap<i64, session::Session>>,
    // Private clock marker only, not a public version or memory scope index.
    // Retained across deletes so recreating an id cannot revive a stale read.
    last_time: AtomicI64,
}

impl InMemory {
    pub fn new() -> Self {
        Self {
            records: RwLock::new(BTreeMap::new()),
            last_time: AtomicI64::new(0),
        }
    }

    fn check(cancellation: &Cancellation) -> Result<(), Error> {
        if cancellation.is_cancelled() {
            Err(Error::new("CANCELLED", "session operation cancelled"))
        } else {
            Ok(())
        }
    }

    fn next_time(&self) -> Result<i64, Error> {
        let now = now_ms();
        let mut previous = self.last_time.load(Ordering::Relaxed);
        loop {
            let next = previous
                .checked_add(1)
                .map(|next| now.max(next))
                .ok_or_else(|| Error::new("BACKEND", "session update timestamp exhausted"))?;
            match self.last_time.compare_exchange_weak(
                previous,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(next),
                Err(current) => previous = current,
            }
        }
    }
}

impl Default for InMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl session::Store for InMemory {
    async fn create(
        &self,
        mut value: session::Session,
        cancellation: &Cancellation,
    ) -> Result<session::Session, Error> {
        Self::check(cancellation)?;
        validate(&value)?;
        let mut records = self.records.write().expect("session lock poisoned");
        Self::check(cancellation)?;
        if records.contains_key(&value.id) {
            return Err(Error::new("CONFLICT", "session id already exists"));
        }
        value.created_time = now_ms();
        value.updated_time = self.next_time()?;
        records.insert(value.id, value.clone());
        Ok(value)
    }

    async fn read(&self, id: i64, cancellation: &Cancellation) -> Result<session::Session, Error> {
        Self::check(cancellation)?;
        self.records
            .read()
            .expect("session lock poisoned")
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::new("NOT_FOUND", "session not found"))
    }

    async fn update(
        &self,
        mut value: session::Session,
        cancellation: &Cancellation,
    ) -> Result<session::Session, Error> {
        Self::check(cancellation)?;
        validate(&value)?;
        let mut records = self.records.write().expect("session lock poisoned");
        Self::check(cancellation)?;
        let old = records
            .get(&value.id)
            .ok_or_else(|| Error::new("NOT_FOUND", "session not found"))?;
        if old.tenant_id != value.tenant_id
            || old.user_id != value.user_id
            || old.r#type != value.r#type
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "session ownership and type cannot change",
            ));
        }
        // The caller submits the updated_time from its read/create/update
        // snapshot. Compare inside the same critical section as replacement,
        // so a delayed compression task cannot overwrite a newer summary.
        if old.updated_time != value.updated_time {
            return Err(Error::new("CONFLICT", "session snapshot is stale"));
        }
        value.created_time = old.created_time;
        // Two updates within one clock millisecond must still have distinct
        // snapshot markers, including when the system clock moves backwards.
        value.updated_time = self.next_time()?;
        records.insert(value.id, value.clone());
        Ok(value)
    }

    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        Self::check(cancellation)?;
        let mut records = self.records.write().expect("session lock poisoned");
        Self::check(cancellation)?;
        if records.remove(&id).is_none() {
            return Err(Error::new("NOT_FOUND", "session not found"));
        }
        Ok(())
    }
}

fn validate(value: &session::Session) -> Result<(), Error> {
    if value.id <= 0 || value.tenant_id <= 0 || value.user_id <= 0 {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "session identifiers must be positive",
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
