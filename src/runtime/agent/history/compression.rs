//! Agent memory compression over a fixed, request-triggered history snapshot.
//! The configured compression component performs the transformation; this
//! module checks the trigger, manages the job and conditionally writes Session.
use serde_json::Value;
use std::{
    collections::{BTreeSet, HashSet},
    sync::{Arc, Mutex, OnceLock},
};

use crate::{
    Cancellation, Error, Message, Messages,
    memory::{
        self, Memory,
        short::{self, session},
    },
    message::Role,
    model::token::Usage,
};

use super::{extraction, progress};
use crate::components::memory::short::session::Store as _;
use crate::runtime::agent::hooks::Hooks;

type Key = (usize, i64);
static ACTIVE: OnceLock<Mutex<HashSet<Key>>> = OnceLock::new();

struct Claim(Key);
impl Claim {
    fn take(key: Key) -> Option<Self> {
        let mut active = ACTIVE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        active.insert(key).then(|| Self(key))
    }
}
impl Drop for Claim {
    fn drop(&mut self) {
        ACTIVE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.0);
    }
}

/// Recorded Usage only, never a second tokenization of historical payloads.
/// A missing total falls back to available input/output counters; unknown
/// usage contributes no known amount. Saturation still crosses any threshold.
pub(super) fn tokens(usage: Option<&Usage>) -> u64 {
    usage.map_or(0, |usage| {
        usage.total.unwrap_or_else(|| {
            usage
                .input
                .unwrap_or(0)
                .saturating_add(usage.output.unwrap_or(0))
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn start<S, L>(
    config: Arc<memory::Config<S, L>>,
    snapshot: session::Session,
    input: Vec<Message>,
    endpoints: Option<(i64, i64)>,
    total: u64,
    hooks: Hooks,
    extraction: extraction::Work,
) where
    S: short::Store + 'static,
    L: Memory + 'static,
{
    let Some(component) = config.compression.clone() else {
        return;
    };
    let Ok(size) = component.size() else {
        return;
    };
    let pending = snapshot.metadata.contains_key(progress::KEY);
    let compress = endpoints.is_some() && total >= size;
    if !pending && !compress {
        return;
    }
    // Actual shared backend identity, not agent name or a second namespace.
    // Hold its Arc for the entire claim lifetime to prevent pointer reuse.
    let key = (Arc::as_ptr(&config.short) as usize, snapshot.id);
    let Some(claim) = Claim::take(key) else {
        return;
    };
    let spawned = std::thread::Builder::new()
        .name("heart-compression".into())
        .spawn(move || {
            let _claim = claim;
            let work = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|_| Error::new("BACKEND", "cannot start background compression"))?;
                runtime.block_on(async move {
                    // Independent of cancellation of the foreground Chat. Backend
                    // timeouts remain the backend's responsibility.
                    let cancellation = Cancellation::new();
                    let current = config.short.session().read(snapshot.id, &cancellation).await?;
                    if current != snapshot {
                        // Another writer already advanced the snapshot. Only a
                        // subsequent request may start work from the latest state.
                        return Ok(());
                    }
                    let mut journal = progress::Journal::new(config.short.clone(), current);
                    let state = match progress::read(&journal.current)? {
                        Some(state) => state,
                        None if compress => {
                            let (from, to) = endpoints.unwrap();
                            let job = if config.extraction.is_some() {
                                Some(extraction::Job::new(
                                    snapshot.user_id,
                                    extraction.agent_id,
                                    from,
                                    to,
                                    extraction.input,
                                )?)
                            } else {
                                None
                            };
                            let input = truncate_tool_results(safe_messages(input));
                            let input = hooks
                                .before_compress(Messages::new(input), &cancellation)
                                .await?;
                            let output = component.compress(input, &cancellation).await?;
                            let usage = output
                                .as_slice()
                                .iter()
                                .filter_map(|m| m.usage.clone())
                                .reduce(|a, b| {
                                    crate::model::token::merge(Some(a), Some(b)).unwrap()
                                });
                            let output = hooks.after_compress(output, &cancellation).await?;
                            let state = progress::State::Summary {
                                base: Box::new(progress::Base::new(&snapshot)),
                                summary: summary(&output)?,
                                usage: usage.map(Box::new),
                                from,
                                to,
                                extraction: job,
                            };
                            // Save the successfully prepared result before the
                            // summary commit. A lost response resumes this scene.
                            journal.save(Some(&state), &cancellation).await?;
                            state
                        }
                        None => return Ok(()),
                    };
                    let job = match state {
                        progress::State::Summary {
                            base,
                            summary,
                            usage,
                            from,
                            to,
                            extraction,
                        } => {
                            if !base.matches(&journal.current) {
                                // A different summary has superseded this
                                // snapshot; preserve it and drop stale work only.
                                journal.save(None, &cancellation).await?;
                                return Ok(());
                            }
                            let mut updated = journal.current.clone();
                            updated.summary = Some(summary);
                            updated.usage = usage.map(|value| *value); // Replace, never accumulate.
                            let from = updated
                                .metadata
                                .get("from")
                                .and_then(serde_json::Value::as_i64)
                                .unwrap_or(from);
                            updated.metadata.insert("from".into(), from.into());
                            updated.metadata.insert("to".into(), to.into());
                            let next = extraction
                                .as_ref()
                                .map(|job| progress::State::Extraction { job: job.clone() });
                            // Summary, coverage, Usage and extraction handoff
                            // are committed by the same existing conditional CRUD.
                            progress::encode(&mut updated, next.as_ref())?;
                            journal.update(updated, &cancellation).await?;
                            extraction
                        }
                        progress::State::Extraction { job } => Some(job),
                    };
                    if let Some(mut job) = job {
                        let extractor = config.extraction.as_ref().ok_or_else(|| {
                            Error::new(
                                "INVALID_ARGUMENTS",
                                "pending Memory extraction requires its configured component",
                            )
                        })?;
                        extraction::resume_job(
                            &mut job,
                            &mut journal,
                            extractor,
                            config.long.as_ref(),
                            config.id_fn.ok_or_else(|| {
                                Error::new(
                                    "INVALID_ARGUMENTS",
                                    "automatic extraction requires an ID function",
                                )
                            })?,
                            &cancellation,
                        )
                        .await?;
                        journal.save(None, &cancellation).await?;
                    }
                    Ok::<_, Error>(())
                })
            }));
            match work {
                Ok(Ok(())) => (),
                // Do not log private history, model output or backend credentials.
                Ok(Err(_)) => {
                    eprintln!(
                        "heart: background memory processing failed; continue from saved progress on a later request"
                    )
                }
                Err(_) => {
                    eprintln!("heart: background Session compression panicked; claim released")
                }
            }
        });
    if spawned.is_err() {
        eprintln!("heart: cannot launch background Session compression; retry on a later request");
    }
}

fn summary(output: &Messages) -> Result<String, Error> {
    let invalid = || {
        Error::new(
            "INVALID_RESPONSE",
            "compression must return nonempty assistant text",
        )
    };
    let mut messages = Vec::new();
    for message in output.as_slice() {
        if message.role != Role::Assistant {
            return Err(invalid());
        }
        let mut text = String::new();
        for part in &message.content {
            match part.r#type.as_str() {
                "think" => (), // Reasoning is not the stored context summary.
                "text" => text.push_str(
                    part.data
                        .get("value")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(invalid)?,
                ),
                _ => return Err(invalid()),
            }
        }
        if !text.trim().is_empty() {
            messages.push(text);
        }
    }
    if messages.is_empty() {
        return Err(invalid());
    }
    Ok(messages.join("\n"))
}

/// Keep tool-call groups intact when preparing historical messages for a
/// summary. An unfinished call belongs to the live Agent scene and must stay
/// there; it must not be represented by a summary without its result.
fn safe_messages(messages: Vec<Message>) -> Vec<Message> {
    let mut pending = BTreeSet::new();
    let mut group_start = None;
    let mut result = Vec::with_capacity(messages.len());

    for (index, message) in messages.iter().enumerate() {
        let calls = if message.role == Role::Assistant {
            message
                .content
                .iter()
                .filter(|part| part.r#type == "tool_call")
                .filter_map(|part| part.data.get("call_id").and_then(|value| value.as_str()))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if !calls.is_empty() {
            group_start.get_or_insert(index);
            pending.extend(calls.into_iter().map(str::to_owned));
        }

        if matches!(message.role, Role::Tool | Role::Function)
            && let Some(call_id) = message_call_id(message)
        {
            pending.remove(call_id);
        }

        result.push(message.clone());
        if pending.is_empty() {
            group_start = None;
        }
    }

    if let Some(index) = group_start {
        result.truncate(index);
    }
    result
}

const MAX_TOOL_RESULT_CHARS: usize = 50_000;

/// Reduce only the temporary summary input view. The original tool result is
/// retained in Chat/Step state and remains available for normal recovery.
fn truncate_tool_results(mut messages: Vec<Message>) -> Vec<Message> {
    for message in &mut messages {
        if !matches!(message.role, Role::Tool | Role::Function) {
            continue;
        }
        for part in &mut message.content {
            for value in part.data.values_mut() {
                truncate_value(value);
            }
        }
        for value in message.values.values_mut() {
            truncate_value(value);
        }
    }
    messages
}

fn truncate_value(value: &mut Value) {
    match value {
        Value::String(text) if text.chars().count() > MAX_TOOL_RESULT_CHARS => {
            let chars = text.chars().collect::<Vec<_>>();
            let keep = MAX_TOOL_RESULT_CHARS / 2;
            let removed = chars.len().saturating_sub(keep * 2);
            let prefix = chars[..keep].iter().collect::<String>();
            let suffix = chars[chars.len() - keep..].iter().collect::<String>();
            *text = format!(
                "{prefix}\n...[tool result truncated for compression input: {removed} chars omitted]...\n{suffix}"
            );
        }
        Value::Array(values) => {
            for value in values {
                truncate_value(value);
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                truncate_value(value);
            }
        }
        _ => {}
    }
}

fn message_call_id(message: &Message) -> Option<&str> {
    message
        .metadata
        .get("call_id")
        .or_else(|| message.values.get("call_id"))
        .and_then(|value| value.as_str())
}

#[cfg(test)]
mod tests;
