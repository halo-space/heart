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
