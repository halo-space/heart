use serde_json::Value;
use std::sync::Arc;

use crate::runtime::cancellation::Cancellation;

pub type Error = crate::Error;

/// Storage contract for the one current checkpoint of an application `id`.
/// The JSON value is the Runtime state as-is; the component does not define a
/// second field hierarchy or interpret node/execution fields.
#[allow(async_fn_in_trait)]
pub trait Checkpoint: Send + Sync {
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error>;

    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error>;

    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error>;

    /// Validates the serialized Runtime state before it is persisted.
    /// Backends may override this when their storage contract adds stricter
    /// requirements, while the default contract only accepts JSON objects.
    fn validate_state(&self, state: &Value) -> Result<(), Error> {
        if state.is_object() {
            Ok(())
        } else {
            Err(Error::new(
                "INVALID_ARGUMENTS",
                "checkpoint state must be a JSON object",
            ))
        }
    }
}

/// Shared ownership convenience for configured Runtime backends. This keeps
/// the native async trait contract while allowing one backend handle to be
/// retained by the caller and the Runtime at the same time.
impl<T: Checkpoint + ?Sized> Checkpoint for Arc<T> {
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        self.as_ref().save(id, state, cancellation).await
    }

    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        self.as_ref().load(id, cancellation).await
    }

    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        self.as_ref().delete(id, cancellation).await
    }

    fn validate_state(&self, state: &Value) -> Result<(), Error> {
        self.as_ref().validate_state(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Probe;

    impl Checkpoint for Probe {
        async fn save(
            &self,
            _id: i64,
            _state: Value,
            _cancellation: &Cancellation,
        ) -> Result<(), Error> {
            Ok(())
        }

        async fn load(&self, _id: i64, _cancellation: &Cancellation) -> Result<Value, Error> {
            Err(Error::new("NOT_FOUND", "probe has no state"))
        }

        async fn delete(&self, _id: i64, _cancellation: &Cancellation) -> Result<(), Error> {
            Ok(())
        }
    }

    #[test]
    fn validates_runtime_state_shape_through_the_contract() {
        let checkpoint = Probe;
        assert_eq!(
            checkpoint
                .validate_state(&Value::String("state".into()))
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        assert!(
            checkpoint
                .validate_state(&serde_json::json!({"status": "running"}))
                .is_ok()
        );
    }
}
