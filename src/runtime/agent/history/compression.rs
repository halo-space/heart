//! Agent memory compression over a fixed, request-triggered history snapshot.
//! The configured compression component performs the transformation; this
//! module checks the trigger, manages the job and conditionally writes Session.
use std::sync::Arc;

#[cfg(test)]
#[allow(unused_imports)]
use serde_json::Value;
#[cfg(test)]
#[allow(unused_imports)]
use std::collections::{BTreeSet, HashSet};
#[cfg(test)]
#[allow(unused_imports)]
use std::sync::{Mutex, OnceLock};

#[cfg(test)]
#[allow(unused_imports)]
use crate::message::Role;
#[cfg(test)]
#[allow(unused_imports)]
use crate::model::token::Usage;
use crate::{
    Cancellation, Error, Message, Messages,
    memory::{
        Memory,
        short::{self, session},
    },
};

use super::{extraction, progress};
use crate::components::memory::short::session::Store as _;
use crate::runtime::agent::hooks::Hooks;
use crate::runtime::observation::observe_usage;
use crate::runtime::trace::fastrace::Root;
mod claim;
mod messages;

#[cfg(test)]
pub(super) use claim::ACTIVE;
pub(super) use claim::{Claim, tokens};
#[cfg(test)]
pub(super) use messages::MAX_TOOL_RESULT_CHARS;
pub(super) use messages::{safe_messages, summary, truncate_tool_results};

#[allow(clippy::too_many_arguments)]
pub(super) fn start<S, L>(
    config: Arc<crate::runtime::agent::memory::Config<S, L>>,
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
    tokio::spawn(async move {
        let _claim = claim;
        // Background Memory calls have no public Event stream. Keep their
        // settled model usage visible through the existing trace audit.
        let trace = Root::start("memory.compression");
        let work = async move {
            // Independent of cancellation of the foreground Chat. Backend
            // timeouts remain the backend's responsibility.
            let cancellation = Cancellation::new();
            let current = config
                .short
                .session()
                .read(snapshot.id, &cancellation)
                .await?;
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
                        .reduce(|a, b| crate::model::token::merge(Some(a), Some(b)).unwrap());
                    observe_usage(&trace, "memory.compression.usage", usage.as_ref());
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
                    config.vectorizer(),
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
        }
        .await;
        match work {
            Ok(()) => (),
            // Do not log private history, model output or backend credentials.
            Err(_) => {
                eprintln!(
                    "heart: background memory processing failed; continue from saved progress on a later request"
                )
            }
        }
    });
}

#[cfg(test)]
#[path = "../../../../tests/unit/embedded/src_runtime_agent_history_compression_tests.rs"]
mod tests;
