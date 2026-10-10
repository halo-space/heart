use crate::components::message::Messages;
use crate::runtime::cancellation::Cancellation;

pub type Error = crate::Error;

/// Context compression is a pure component operation. Harness coordinates
/// triggering and Session updates; runtime::agent owns run accounting.
#[allow(async_fn_in_trait)]
pub trait Compressor: Send + Sync {
    /// Trigger threshold used by an automatic-compression owner. This is not
    /// the target summary length or the model's max_tokens. Direct compress
    /// calls do not require it; automatic assembly must reject None/Some(0).
    fn size(&self) -> Option<u64> {
        None
    }

    async fn compress(
        &self,
        input: Messages,
        cancellation: &Cancellation,
    ) -> Result<Messages, Error>;
}
