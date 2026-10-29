use super::{Mode, Policy};

/// Allows every request. Use only when the surrounding environment is trusted.
pub fn new() -> Policy {
    Policy::new(Mode::Bypass)
}
