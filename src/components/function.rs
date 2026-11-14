use std::future::Future;

use crate::Cancellation;

pub type Error = crate::Error;

/// A directly orchestrated business computation. Function does not know about
/// Graph, Runtime, Messages, node IDs, or execution IDs.
///
/// The trait method receives `&Cancellation`. The blanket implementation for
/// ordinary closures deliberately accepts an owned `Cancellation` and clones
/// the shared signal before calling the closure; this keeps the returned future
/// independent of the borrowed method frame while preserving one cancellation
/// scope. Explicit implementations can use the borrowed trait signature.
pub trait Function<I, O>: Send + Sync {
    fn call(
        &self,
        input: I,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<O, Error>> + Send;
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
#[path = "../../tests/unit/embedded/src_components_function_tests.rs"]
mod tests;
