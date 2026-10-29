use crate::Cancellation;
use crate::components::message::Messages;

pub type Error = crate::Error;

/// Context compression is a pure component operation. Harness coordinates
/// triggering and Session updates; runtime::agent owns run accounting.
pub trait Compressor: Send + Sync {
    /// Trigger threshold used by an automatic-compression owner. This is not
    /// the target summary length or the model's max_tokens. Direct compress
    /// calls do not require it; automatic assembly must reject None/Some(0).
    fn size(&self) -> Option<u64> {
        None
    }

    /// Maximum estimated Token count sent to the compression component.
    /// `None` keeps the complete uncovered history.
    fn limit(&self) -> Option<u64> {
        None
    }

    /// Estimated Token count kept as the newest uncovered history after a
    /// rolling Session compression. `None` keeps no extra suffix.
    fn retain(&self) -> Option<u64> {
        None
    }

    /// Counts a temporary compression input view. This is separate from
    /// Provider Usage and defaults to a JSON byte estimate.
    fn count(&self, input: &Messages) -> u64 {
        serde_json::to_vec(input)
            .map(|bytes| u64::try_from(bytes.len().saturating_add(3) / 4).unwrap_or(u64::MAX))
            .unwrap_or(u64::MAX)
    }

    fn compress(
        &self,
        input: Messages,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Messages, Error>>;
}
