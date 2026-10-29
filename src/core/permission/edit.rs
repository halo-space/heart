use super::{Mode, Policy};

/// Allows reads and ordinary edits; destructive or unknown actions ask.
pub fn new() -> Policy {
    Policy::new(Mode::Edit)
}
