//! Harness context defaults.
//!
//! Context construction stays separate from Runtime State. This module only
//! defines the default long-task instruction and the explicit override rule.

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a capable agent for multi-step tasks. Break substantial work into clear steps, use the available tools to gather evidence and make changes, verify important results, and report what is complete and what remains. Do not claim that an action succeeded unless its result confirms it.";

pub fn system_prompt(custom: Option<&str>) -> String {
    custom
        .filter(|prompt| !prompt.trim().is_empty())
        .unwrap_or(DEFAULT_SYSTEM_PROMPT)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_SYSTEM_PROMPT, system_prompt};

    #[test]
    fn keeps_default_prompt_and_accepts_explicit_override() {
        assert_eq!(system_prompt(None), DEFAULT_SYSTEM_PROMPT);
        assert_eq!(system_prompt(Some("custom")), "custom");
        assert_eq!(system_prompt(Some("  ")), DEFAULT_SYSTEM_PROMPT);
    }
}
