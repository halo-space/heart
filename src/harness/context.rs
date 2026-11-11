//! Harness context defaults and the per-request model adapter.
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

use std::sync::{Arc, Mutex};

use crate::components::message::{Message, Messages, Role};
use crate::components::model::Error;
use crate::components::model::chat::{Chat, Request};
use crate::components::model::content::part::{Part, object};
use crate::{Cancellation, harness::state::Store};
use serde_json::Value;

/// Chat adapter that refreshes Harness context immediately before each model
/// request. Runtime continues to use only the ordinary `Chat` contract.
#[doc(hidden)]
pub struct ContextChat<M> {
    model: M,
    store: Arc<Mutex<Option<Arc<Store>>>>,
    subagents: Arc<Mutex<Vec<String>>>,
}

impl<M> ContextChat<M> {
    pub(crate) fn new(
        model: M,
        store: Arc<Mutex<Option<Arc<Store>>>>,
        subagents: Arc<Mutex<Vec<String>>>,
    ) -> Self {
        Self {
            model,
            store,
            subagents,
        }
    }
}

impl<M> Chat for ContextChat<M>
where
    M: Chat,
{
    type Stream = M::Stream;

    fn context_limit(&self) -> Option<u64> {
        self.model.context_limit()
    }

    async fn stream(&self, mut request: Request) -> Result<Self::Stream, Error> {
        let store = self
            .store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        if let Some(store) = store
            && let Some(state) = store.load(&Cancellation::new()).await?
        {
            let registered = self
                .subagents
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            let session_id = request
                .messages
                .as_slice()
                .iter()
                .find_map(|message| message.metadata.get("session_id").and_then(Value::as_i64));
            if let Some(context) = state.context_for(session_id, &registered)? {
                let mut context_message = Message::new(Role::System);
                context_message.content.push(Part {
                    r#type: "text".into(),
                    data: object(serde_json::json!({"value": context})),
                });
                insert_context(&mut request.messages, context_message);
            }
        }
        self.model.stream(request).await
    }
}

/// Keep fixed system/developer instructions first, then place the latest
/// Harness context before history and user input. This prevents durable state
/// from accidentally becoming the highest-priority instruction.
fn insert_context(messages: &mut Messages, context: Message) {
    let index = messages
        .0
        .iter()
        .take_while(|message| matches!(message.role, Role::System | Role::Developer))
        .count();
    messages.0.insert(index, context);
}

#[cfg(test)]
mod tests {
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
}
