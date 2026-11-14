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
use crate::harness::state::Store;
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
        let cancellation = crate::cancellation::current().unwrap_or_default();
        if let Some(store) = store
            && let Some(state) = store.load(&cancellation).await?
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
#[path = "../../tests/unit/embedded/src_harness_context_tests.rs"]
mod tests;
