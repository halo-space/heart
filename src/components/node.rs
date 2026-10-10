use std::future::Future;

use crate::components::message::Messages;
use crate::runtime::cancellation::Cancellation;

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
    fn call(
        &self,
        input: I,
        cancellation: Cancellation,
    ) -> impl Future<Output = Result<Signal<O>, E>> + Send;
}

/// A normal async function or closure is a Node without an adapter object.
/// Its successful value is promoted to `Signal::Complete` by the blanket
/// implementation; functions that need `Wait` can implement `Node` directly.
impl<I, O, E, F, Fut> Node<I, O, E> for F
where
    I: Send + 'static,
    F: Fn(I, Cancellation) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, E>> + Send + 'static,
{
    async fn call(&self, input: I, cancellation: Cancellation) -> Result<Signal<O>, E> {
        (self)(input, cancellation).await.map(Signal::Complete)
    }
}

/// Small first-slice smoke node used to verify the contract before the full
/// Workflow Runtime is introduced.
pub struct Echo;

impl Node<Messages, Messages, crate::Error> for Echo {
    async fn call(
        &self,
        input: Messages,
        _cancellation: Cancellation,
    ) -> Result<Signal<Messages>, crate::Error> {
        Ok(Signal::Complete(input))
    }
}
