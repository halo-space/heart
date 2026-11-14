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
