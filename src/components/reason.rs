use crate::components::message::Messages;
use crate::runtime::cancellation::Cancellation;

pub type Error = crate::Error;

/// Optional reasoning component. Callers or Harness decide when to invoke it;
/// runtime::agent owns any surrounding graph and execution state.
pub trait Reasoner: Send + Sync {
    fn reason(
        &self,
        input: Messages,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Messages, Error>> + Send;
}
