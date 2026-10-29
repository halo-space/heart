use super::{Mode, Policy};

/// Allows read-only actions and denies mutations.
pub fn new() -> Policy {
    Policy::new(Mode::Explore)
}
