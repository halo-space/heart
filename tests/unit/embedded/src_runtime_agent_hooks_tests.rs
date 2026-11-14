use super::{Hooks, Stage};
use crate::{Cancellation, Error, middleware::Middleware};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct Transform {
    key: &'static str,
    calls: Arc<AtomicUsize>,
    fail: Arc<AtomicBool>,
}

impl Transform {
    fn apply(&self, value: &mut Value) -> Result<(), Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        value["values"][self.key] = json!(true);
        if self.fail.load(Ordering::SeqCst) {
            return Err(Error::new("HOOK_FAILED", "controlled hook failure"));
        }
        Ok(())
    }
}

impl Middleware for Transform {
    async fn after_system_prompt(&self, value: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.apply(value)
    }
}

#[tokio::test]
async fn only_acknowledged_hook_transformations_advance_the_cursor() {
    let first_calls = Arc::new(AtomicUsize::new(0));
    let second_calls = Arc::new(AtomicUsize::new(0));
    let fail_second = Arc::new(AtomicBool::new(true));
    let mut hooks = Hooks::default();
    hooks.add(Transform {
        key: "first",
        calls: Arc::clone(&first_calls),
        fail: Arc::new(AtomicBool::new(false)),
    });
    hooks.add(Transform {
        key: "second",
        calls: Arc::clone(&second_calls),
        fail: Arc::clone(&fail_second),
    });

    let mut acknowledged = Vec::new();
    let error = hooks
        .apply_from(
            Stage::SystemPrompt,
            false,
            json!({"values": {}, "usage": {"total": 15}}),
            0,
            |idx, value| {
                acknowledged.push((idx, value));
                Ok(())
            },
            &Cancellation::new(),
        )
        .await
        .expect_err("second hook is configured to fail");
    assert_eq!(error.code, "HOOK_FAILED");
    assert_eq!(acknowledged.len(), 1);
    assert_eq!(acknowledged[0].0, 1);
    assert_eq!(acknowledged[0].1["values"]["first"], true);
    assert!(acknowledged[0].1["values"].get("second").is_none());
    assert_eq!(acknowledged[0].1["usage"]["total"], 15);
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert_eq!(second_calls.load(Ordering::SeqCst), 1);

    fail_second.store(false, Ordering::SeqCst);
    let (cursor, saved_value) = acknowledged.pop().unwrap();
    let mut restored_cursor = 0;
    let mut restored_value = Value::Null;
    let final_value = hooks
        .apply_from(
            Stage::SystemPrompt,
            false,
            saved_value,
            cursor,
            |idx, value| {
                restored_cursor = idx;
                restored_value = value;
                Ok(())
            },
            &Cancellation::new(),
        )
        .await
        .expect("resume from the acknowledged cursor");
    assert_eq!(restored_cursor, 2);
    assert_eq!(restored_value, final_value);
    assert_eq!(final_value["values"]["first"], true);
    assert_eq!(final_value["values"]["second"], true);
    assert_eq!(final_value["usage"]["total"], 15);
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert_eq!(second_calls.load(Ordering::SeqCst), 2);
}
