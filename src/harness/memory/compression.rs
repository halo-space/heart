//! Harness compression workflow over a fixed, request-triggered history snapshot.
//! The configured compression component performs the transformation; this
//! module checks the trigger, manages the job and conditionally writes Session.
use std::{
    collections::HashSet,
    sync::{Arc, Mutex, OnceLock},
};

use crate::{
    Cancellation, Error, Message, Messages,
    memory::{
        self, Memory,
        short::{chat, session, trace},
    },
    message::Role,
    model::token::Usage,
};

use super::super::hooks::Hooks;
use super::extraction;

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
pub(super) fn start<S, C, T, L>(
    config: Arc<memory::Config<S, C, T, L>>,
    snapshot: session::Session,
    input: Vec<Message>,
    endpoints: Option<(i64, i64)>,
    total: u64,
    hooks: Hooks,
    extraction: extraction::Work,
) where
    S: session::Store + 'static,
    C: chat::Store + 'static,
    T: trace::Store + 'static,
    L: Memory + 'static,
{
    let Some(component) = config.compression.clone() else {
        return;
    };
    let Ok(size) = component.size() else {
        return;
    };
    let pending = extraction.jobs.contains(snapshot.id);
    let compress = endpoints.is_some() && total >= size;
    if !pending && !compress {
        return;
    }
    // Actual shared backend identity, not agent name or a second namespace.
    // Hold its Arc for the entire claim lifetime to prevent pointer reuse.
    let key = (Arc::as_ptr(&config.session) as usize, snapshot.id);
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
                    // Resume the failed stage first, without recompressing its
                    // already committed Session snapshot. No idle retry loop.
                    if let Some(extractor) = &config.extraction {
                        if extraction.jobs.expected(snapshot.id).is_some() {
                            let current = config.session.read(snapshot.id, &cancellation).await?;
                            if !extraction.jobs.confirm(snapshot.id, &current) {
                                extraction.jobs.discard(snapshot.id);
                            }
                        }
                        extraction::resume(
                            extraction.jobs.clone(),
                            snapshot.id,
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
                    }
                    if !compress {
                        return Ok(());
                    }
                    let (from, to) = endpoints.unwrap();
                    let current = config.session.read(snapshot.id, &cancellation).await?;
                    if current != snapshot {
                        // Another writer already advanced the snapshot. Only a
                        // subsequent request may start work from the latest state.
                        return Ok(());
                    }
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
                    let input = hooks
                        .before_compress(Messages::new(input), &cancellation)
                        .await?;
                    let output = component.compress(input, &cancellation).await?;
                    let usage = output
                        .as_slice()
                        .iter()
                        .filter_map(|m| m.usage.clone())
                        .reduce(|a, b| crate::model::token::merge(Some(a), Some(b)).unwrap());
                    let output = hooks.after_compress(output, &cancellation).await?;
                    let summary = summary(&output)?;
                    let mut updated = snapshot;
                    updated.summary = Some(summary);
                    // Replace the current invocation's Usage, never add old summary
                    // usage or the Chat ledger to this new summary's accounting.
                    updated.usage = usage;
                    let from = updated
                        .metadata
                        .get("from")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(from);
                    updated.metadata.insert("from".into(), from.into());
                    updated.metadata.insert("to".into(), to.into());
                    // Submitted updated_time is the original CAS marker. CONFLICT
                    // never causes a stale result to overwrite a fresh snapshot.
                    let session_id = updated.id;
                    if let Some(job) = job {
                        // Retain the original pairs before the commit call. A
                        // lost acknowledgement is reconciled on a later request,
                        // without repeating a successfully committed summary.
                        extraction
                            .jobs
                            .insert(session_id, job.awaiting_commit(updated.clone()));
                    }
                    let committed = config.session.update(updated, &cancellation).await?;
                    if let Some(extractor) = &config.extraction {
                        if !extraction.jobs.confirm(session_id, &committed) {
                            // Do not discard retained input based on a malformed
                            // acknowledgement; reconcile by reading next time.
                            return Err(Error::new(
                                "INVALID_RESPONSE",
                                "Session update did not acknowledge the expected summary",
                            ));
                        }
                        extraction::resume(
                            extraction.jobs,
                            session_id,
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
                    }
                    Ok::<_, Error>(())
                })
            }));
            match work {
                Ok(Ok(())) => (),
                // Do not log private history, model output or backend credentials.
                Ok(Err(_)) => {
                    eprintln!(
                        "heart: background memory processing failed; completed stages retained"
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

#[cfg(test)]
mod tests;
