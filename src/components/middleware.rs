use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Error;
use crate::runtime::cancellation::Cancellation;

/// Lifecycle extension contract shared by Node and Agent owners. Runtime
/// remains responsible for execution identity, attempts and state commits.
#[allow(async_fn_in_trait)]
pub trait Middleware: Send + Sync {
    async fn before_execute(
        &self,
        _input: &mut Value,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn after_execute(
        &self,
        _output: &mut Value,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn before_model(
        &self,
        _input: &mut Value,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn after_model(
        &self,
        _output: &mut Value,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn before_permission(
        &self,
        _input: &mut Value,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn after_permission(
        &self,
        _output: &mut Value,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn before_compress_context(
        &self,
        _input: &mut Value,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn after_compress_context(
        &self,
        _output: &mut Value,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn before_system_prompt(
        &self,
        _input: &mut Value,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn after_system_prompt(
        &self,
        _output: &mut Value,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        Ok(())
    }
}

pub mod retry {
    use super::*;

    #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
    pub struct Policy {
        pub max_retries: u32,
        pub min_delay_ms: u64,
        pub max_delay_ms: u64,
    }

    impl Default for Policy {
        fn default() -> Self {
            Self {
                max_retries: 0,
                min_delay_ms: 500,
                max_delay_ms: 30_000,
            }
        }
    }

    impl Policy {
        pub fn validate(&self) -> Result<(), Error> {
            if self.min_delay_ms > self.max_delay_ms {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "retry delay range is invalid",
                ));
            }
            Ok(())
        }

        /// Whether an error code belongs to the default transient class.
        pub fn should_retry(&self, attempt_idx: u32, error: &Error) -> bool {
            attempt_idx < self.max_retries
                && matches!(
                    error.code.as_str(),
                    "TRANSPORT" | "TIMEOUT" | "RATE_LIMITED" | "UNAVAILABLE"
                )
        }

        /// Returns the deterministic exponential backoff cap for an attempt.
        /// A caller may add jitter before sleeping; this policy never sleeps.
        pub fn delay(&self, attempt_idx: u32) -> Duration {
            let shift = attempt_idx.min(63);
            let factor = 1u64.checked_shl(shift).unwrap_or(u64::MAX);
            let cap = self
                .min_delay_ms
                .saturating_mul(factor)
                .min(self.max_delay_ms);
            Duration::from_millis(cap)
        }
    }
}

pub use retry::Policy as RetryPolicy;

#[cfg(test)]
mod tests {
    use super::retry::Policy;
    use crate::Error;

    #[test]
    fn retry_policy_only_retries_transient_errors_and_caps_delay() {
        let policy = Policy {
            max_retries: 2,
            min_delay_ms: 10,
            max_delay_ms: 15,
        };
        assert!(policy.validate().is_ok());
        assert!(policy.should_retry(0, &Error::new("TIMEOUT", "temporary")));
        assert!(!policy.should_retry(2, &Error::new("TIMEOUT", "temporary")));
        assert!(!policy.should_retry(0, &Error::new("INVALID_ARGUMENTS", "permanent")));
        assert_eq!(policy.delay(0).as_millis(), 10);
        assert_eq!(policy.delay(2).as_millis(), 15);
    }
}
