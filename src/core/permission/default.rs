use super::{Mode, Policy};

/// Allows clearly read-only actions and asks for all other actions.
pub fn new() -> Policy {
    Policy::new(Mode::Default)
}
