//! Private Agent checkpoint I/O. Component instances never enter snapshots.
use std::{future::Future, pin::Pin, sync::Arc};

use serde_json::{Map, Value};

use crate::{Cancellation, Error, event::Event};

type Save = Arc<
    dyn Fn(i64, Value, Cancellation) -> Pin<Box<dyn Future<Output = Result<(), Error>>>>
        + Send
        + Sync,
>;
type Load = Arc<
    dyn Fn(i64, Cancellation) -> Pin<Box<dyn Future<Output = Result<Value, Error>>>> + Send + Sync,
>;

#[derive(Clone)]
pub(crate) struct Backend {
    save: Save,
    load: Load,
}

impl Backend {
    pub(crate) fn new<C: crate::checkpoint::Checkpoint + 'static>(checkpoint: C) -> Self {
        let checkpoint = Arc::new(checkpoint);
        let writer = checkpoint.clone();
        Self {
            save: Arc::new(move |id, state, cancellation| {
                let checkpoint = writer.clone();
                Box::pin(async move { checkpoint.save(id, state, &cancellation).await })
            }),
            load: Arc::new(move |id, cancellation| {
                let checkpoint = checkpoint.clone();
                Box::pin(async move { checkpoint.load(id, &cancellation).await })
            }),
        }
    }

    pub(crate) async fn save(
        &self,
        id: i64,
        state: Value,
        cancellation: Cancellation,
    ) -> Result<(), Error> {
        (self.save)(id, state, cancellation).await
    }

    pub(crate) async fn load(
        &self,
        id: i64,
        cancellation: Cancellation,
    ) -> Result<Option<Value>, Error> {
        match (self.load)(id, cancellation).await {
            Ok(state) if state.is_object() && state["type"] == "agent" => {
                if state
                    .get("chat_id")
                    .is_some_and(|value| value.as_i64() != Some(id))
                {
                    return Err(invalid("checkpoint identity does not match this run"));
                }
                Ok(Some(state))
            }
            Ok(_) => Err(invalid("checkpoint is not an Agent state")),
            Err(error) if error.code == "NOT_FOUND" => Ok(None),
            Err(error) => Err(error),
        }
    }
}

pub(crate) fn invalid(message: &str) -> Error {
    Error::new("INVALID_STATE", message)
}

/// Delivery mode is a run fact, not a choice that can change when a waiting
/// invocation is restored. Missing or malformed saved mode is not guessed.
pub(crate) fn stream_mode(state: &Value) -> Result<bool, Error> {
    state
        .get("stream")
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid("Agent checkpoint has no valid stream mode"))
}

/// Validate native waiting positions before any Model/Tool/Memory work. Until
/// the targeted consumer is connected, an explicit human wait must not fall
/// through to generic continuation or silently lose its ordered index.
pub(crate) fn check_waits(state: &Value) -> Result<(), Error> {
    if crate::runtime::agent::wait::validate(state)? {
        return Err(Error::new(
            "WAITING",
            "Agent is waiting for targeted external feedback",
        ));
    }
    Ok(())
}

/// Check ownership from the canonical state and its saved original input.
/// Root metadata can be empty; the original query still owns the run and must
/// never be bypassed when continuation changes payload.
pub(crate) fn identity_matches(state: &Value, metadata: &Map<String, Value>) -> bool {
    let owners = [
        state.get("metadata"),
        state.get("query").and_then(|query| query.get("metadata")),
    ];
    ["tenant_id", "user_id", "agent_id", "session_id", "chat_id"]
        .into_iter()
        .all(|key| {
            owners.iter().all(|owner| {
                owner
                    .and_then(Value::as_object)
                    .and_then(|saved| saved.get(key))
                    .is_none_or(|value| metadata.get(key) == Some(value))
            })
        })
        && state
            .get("chat_id")
            .filter(|value| !value.is_null())
            .is_none_or(|value| metadata.get("chat_id") == Some(value))
}

/// A terminal checkpoint is already settled: return its recorded result,
/// without invoking a model, Tool, hooks, Memory writeback or an evaluator.
pub(crate) fn terminal(state: &Value) -> Result<Option<Result<Event, Error>>, Error> {
    match state["status"].as_str() {
        Some("succeeded" | "stopped") => {
            let message = serde_json::from_value(state["final_message"].clone())
                .map_err(|_| invalid("terminal checkpoint has no valid final message"))?;
            let usage = serde_json::from_value(state["final_usage"].clone())
                .map_err(|_| invalid("terminal checkpoint has invalid final usage"))?;
            let finish_reason = serde_json::from_value(state["finish_reason"].clone())
                .map_err(|_| invalid("terminal checkpoint has invalid finish reason"))?;
            Ok(Some(Ok(Event::Complete {
                message,
                usage,
                finish_reason,
            })))
        }
        Some("failed") => {
            let error = serde_json::from_value(state["error"].clone())
                .map_err(|_| invalid("failed checkpoint has no valid error"))?;
            Ok(Some(Err(error)))
        }
        Some("running" | "waiting" | "paused") => Ok(None),
        _ => Err(invalid("checkpoint has an invalid Agent status")),
    }
}

pub(crate) fn record_result(state: &mut Value, result: &Result<Event, Error>) {
    match result {
        Ok(Event::Complete {
            message,
            usage,
            finish_reason,
        }) => {
            state["final_message"] =
                serde_json::to_value(crate::runtime::observation::original_message(message))
                    .expect("Message serialization");
            state["final_usage"] = serde_json::to_value(usage).expect("Usage serialization");
            state["finish_reason"] =
                serde_json::to_value(finish_reason).expect("finish reason serialization");
        }
        Err(error) => state["error"] = serde_json::to_value(error).expect("Error serialization"),
        Ok(Event::Delta { .. }) => {}
    }
}
