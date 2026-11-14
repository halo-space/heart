use futures_executor::block_on;

use super::*;

#[test]
fn closure_can_implement_function_without_runtime_state() {
    let function = |input: u64, cancellation: Cancellation| async move {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "function cancelled"));
        }
        Ok(input + 1)
    };
    assert_eq!(
        block_on(Function::call(&function, 41, &Cancellation::new())).unwrap(),
        42
    );
}

#[test]
fn function_can_observe_cancellation() {
    let function = |input: u64, _cancellation: Cancellation| async move { Ok(input) };
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        block_on(Function::call(&function, 41, &cancellation)).unwrap(),
        41
    );
}
