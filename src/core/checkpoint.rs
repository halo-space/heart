use std::collections::BTreeMap;
use tokio::sync::RwLock;

use serde_json::Value;

use crate::components::checkpoint::{Checkpoint, Error};
use crate::{Cancellation, cancellation::check};

/// Zero-configuration checkpoint implementation. Each `id` has one
/// current state and a later save atomically replaces that state.
#[derive(Default)]
pub struct InMemory {
    states: RwLock<BTreeMap<i64, Value>>,
}

impl InMemory {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Checkpoint for InMemory {
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        check(cancellation)?;
        self.validate_state(&state)?;
        self.states.write().await.insert(id, state);
        Ok(())
    }

    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        check(cancellation)?;
        self.states
            .read()
            .await
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::new("NOT_FOUND", "checkpoint not found"))
    }

    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        check(cancellation)?;
        if self.states.write().await.remove(&id).is_none() {
            return Err(Error::new("NOT_FOUND", "checkpoint not found"));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/embedded/src_core_checkpoint_tests.rs"]
mod tests;
