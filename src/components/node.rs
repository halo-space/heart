use std::future::Future;

use crate::Cancellation;

/// Node callbacks use the same read-only projection owned by the Agent
/// contract. The definition lives in `components::agent`, so Agent no longer
/// depends on Node.
pub use crate::components::agent::{Status, View};

pub(crate) mod input;
pub(crate) mod schema;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wait {
    pub wait_id: String,
}

#[derive(Debug, Eq, PartialEq)]
pub enum Signal<O> {
    Complete(O),
    Wait(Wait),
}

pub trait Node<I, O, E>: Send + Sync + 'static {
    /// Execute this node as part of one run's cancellation scope.
    fn call(
        &self,
        input: I,
        cancellation: &Cancellation,
    ) -> impl Future<Output = Result<Signal<O>, E>> + Send;

    /// Execute with a read-only local runtime view.
    ///
    /// Existing Nodes only need `call`; the default keeps that contract
    /// unchanged. A Node that needs diagnostics can override this method and
    /// inspect `view` without receiving or mutating the global runtime state.
    fn call_with_view(
        &self,
        input: I,
        _view: &View,
        cancellation: &Cancellation,
    ) -> impl Future<Output = Result<Signal<O>, E>> + Send {
        self.call(input, cancellation)
    }
}

/// A normal async function or closure is a Node without an adapter object.
/// Its successful value is promoted to `Signal::Complete` by the blanket
/// implementation; functions that need `Wait` can implement `Node` directly.
///
/// The explicit Node method receives `&Cancellation`; the ordinary free
/// function/closure form accepts an owned `Cancellation` so its returned
/// future can be moved into Runtime execution. Inherent methods marked with
/// `#[node(call = ...)]` use the borrowed form.
impl<I, O, E, F, Fut> Node<I, O, E> for F
where
    I: Send + 'static,
    F: Fn(I, Cancellation) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, E>> + Send + 'static,
{
    async fn call(&self, input: I, cancellation: &Cancellation) -> Result<Signal<O>, E> {
        (self)(input, cancellation.clone())
            .await
            .map(Signal::Complete)
    }
}
