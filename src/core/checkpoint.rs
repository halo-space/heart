use std::collections::BTreeMap;
use std::sync::RwLock;

use serde_json::Value;

use crate::components::checkpoint::{Checkpoint, Error};
use crate::runtime::cancellation::Cancellation;

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

    fn ensure_active(cancellation: &Cancellation) -> Result<(), Error> {
        if cancellation.is_cancelled() {
            Err(Error::new("CANCELLED", "checkpoint operation cancelled"))
        } else {
            Ok(())
        }
    }
}

impl Checkpoint for InMemory {
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        Self::ensure_active(cancellation)?;
        self.validate_state(&state)?;
        self.states
            .write()
            .expect("checkpoint lock poisoned")
            .insert(id, state);
        Ok(())
    }

    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        Self::ensure_active(cancellation)?;
        self.states
            .read()
            .expect("checkpoint lock poisoned")
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::new("NOT_FOUND", "checkpoint not found"))
    }

    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        Self::ensure_active(cancellation)?;
        if self
            .states
            .write()
            .expect("checkpoint lock poisoned")
            .remove(&id)
            .is_none()
        {
            return Err(Error::new("NOT_FOUND", "checkpoint not found"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use futures_executor::block_on;

    use super::*;

    #[test]
    fn save_replaces_one_current_state_and_load_clones_it() {
        let checkpoint = InMemory::new();
        let cancellation = Cancellation::new();
        block_on(checkpoint.save(7, serde_json::json!({"step": 1}), &cancellation)).unwrap();
        block_on(checkpoint.save(7, serde_json::json!({"step": 2}), &cancellation)).unwrap();
        assert_eq!(
            block_on(checkpoint.load(7, &cancellation)).unwrap(),
            serde_json::json!({"step": 2})
        );
        block_on(checkpoint.delete(7, &cancellation)).unwrap();
        assert_eq!(
            block_on(checkpoint.load(7, &cancellation))
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
    }

    #[test]
    fn cancellation_does_not_mutate_checkpoint() {
        let checkpoint = InMemory::new();
        let cancellation = Cancellation::new();
        block_on(checkpoint.save(7, serde_json::json!({"step": 1}), &cancellation)).unwrap();
        cancellation.cancel();
        assert_eq!(
            block_on(checkpoint.save(7, serde_json::json!({"step": 2}), &cancellation))
                .unwrap_err()
                .code,
            "CANCELLED"
        );
        assert_eq!(
            block_on(checkpoint.load(7, &Cancellation::new())).unwrap(),
            serde_json::json!({"step": 1})
        );
    }
}
