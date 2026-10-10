use std::{future::Future, time::Duration};

use crate::{Cancellation, Error, middleware::RetryPolicy};

use super::operation::{cancellable, check};

/// Explicit caller-owned retry execution. Does not create Attempts, execution
/// identities or state. The caller chooses the operation (for example eval
/// alone after a successful Tool call). Nonzero backoff uses a Tokio timer.
#[derive(Clone, Debug, Default)]
pub struct Retry {
    policy: RetryPolicy,
}

impl Retry {
    pub fn new(policy: RetryPolicy) -> Result<Self, Error> {
        policy.validate()?;
        Ok(Self { policy })
    }

    pub async fn call<T, F, Fut>(
        &self,
        mut operation: F,
        cancellation: &Cancellation,
    ) -> Result<T, Error>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, Error>>,
    {
        check(cancellation)?;
        let mut idx = 0;
        loop {
            check(cancellation)?;
            match cancellable(operation(), cancellation).await {
                Ok(value) => return Ok(value),
                Err(error) if self.policy.should_retry(idx, &error) => {
                    let cap = self.policy.delay(idx).as_millis() as u64;
                    let min = self.policy.min_delay_ms.min(cap);
                    let delay = rand::random_range(min..=cap);
                    if delay > 0 {
                        // Return a normal error instead of panicking if this
                        // independent component is invoked without Tokio.
                        tokio::runtime::Handle::try_current().map_err(|_| {
                            Error::new("UNAVAILABLE", "retry backoff requires a Tokio runtime")
                        })?;
                        cancellable(
                            async {
                                tokio::time::sleep(Duration::from_millis(delay)).await;
                                Ok(())
                            },
                            cancellation,
                        )
                        .await?;
                    }
                    idx += 1;
                }
                Err(error) => return Err(error),
            }
        }
    }
}
