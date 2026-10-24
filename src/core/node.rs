//! Small concrete Nodes that exercise the shared Node contract.

use crate::{
    Cancellation, Error, Messages,
    components::node::{Node, Signal},
};

/// Pass input through unchanged. Useful as a lightweight Workflow node.
pub struct Echo;

impl Node<Messages, Messages, Error> for Echo {
    async fn call(
        &self,
        input: Messages,
        _cancellation: &Cancellation,
    ) -> Result<Signal<Messages>, Error> {
        Ok(Signal::Complete(input))
    }
}
