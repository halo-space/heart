use crate::components::message::Messages;
use crate::runtime::cancellation::Cancellation;

pub type Error = crate::Error;

/// Optional reasoning component. Callers or Harness decide when to invoke it;
/// runtime::agent owns any surrounding graph and execution state.
#[allow(async_fn_in_trait)]
pub trait Reasoner: Send + Sync {
    async fn reason(&self, input: Messages, cancellation: &Cancellation)
    -> Result<Messages, Error>;
}
