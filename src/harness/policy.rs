//! Deterministic Harness policies.
//!
//! These limits configure the existing Runtime Agent. They do not create a
//! second loop or become Runtime State.

use crate::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Policy {
    pub max_iters: usize,
    pub max_concurrency: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            max_iters: 20,
            max_concurrency: 100,
        }
    }
}

impl Policy {
    pub fn validate(&self) -> Result<(), Error> {
        if self.max_iters == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "harness max_iters must be positive",
            ));
        }
        if self.max_concurrency == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "harness max_concurrency must be positive",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/embedded/src_harness_policy_tests.rs"]
mod tests;
