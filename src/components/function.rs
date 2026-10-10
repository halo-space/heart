use std::future::Future;

use crate::runtime::cancellation::Cancellation;

pub type Error = crate::Error;

/// A directly orchestrated business computation. Function does not know about
/// Graph, Runtime, Messages, node IDs, or execution IDs.
#[allow(async_fn_in_trait)]
pub trait Function<I, O>: Send + Sync {
    async fn call(&self, input: I, cancellation: &Cancellation) -> Result<O, Error>;
}

impl<I, O, F, Fut> Function<I, O> for F
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(I, Cancellation) -> Fut + Send + Sync,
    Fut: Future<Output = Result<O, Error>> + Send + 'static,
{
    async fn call(&self, input: I, cancellation: &Cancellation) -> Result<O, Error> {
        (self)(input, cancellation.clone()).await
    }
}

#[cfg(test)]
mod tests {
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
}
