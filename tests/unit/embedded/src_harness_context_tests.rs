use super::{DEFAULT_SYSTEM_PROMPT, insert_context, system_prompt};
use crate::components::message::{Message, Messages, Role};

#[test]
fn keeps_default_prompt_and_accepts_explicit_override() {
    assert_eq!(system_prompt(None), DEFAULT_SYSTEM_PROMPT);
    assert_eq!(system_prompt(Some("custom")), "custom");
    assert_eq!(system_prompt(Some("  ")), DEFAULT_SYSTEM_PROMPT);
}

#[test]
fn inserts_context_after_fixed_instructions() {
    let mut messages = Messages::new([
        Message::new(Role::System),
        Message::new(Role::Developer),
        Message::new(Role::User),
    ]);
    insert_context(&mut messages, Message::new(Role::System));
    assert!(matches!(messages.0[0].role, Role::System));
    assert!(matches!(messages.0[1].role, Role::Developer));
    assert!(matches!(messages.0[2].role, Role::System));
    assert!(matches!(messages.0[3].role, Role::User));
}
