//! Agent-local composition of Memory contracts.
//!
//! Storage and model interfaces remain in `components`; this module only
//! combines those contracts for the Runtime Agent lifecycle.

use crate::Error;
use crate::components::compression::Compressor;
use crate::components::memory::{Memory, short};
use crate::components::model::embed::Embed;

mod compression;
pub(crate) mod extraction;
mod vectorize;

pub(crate) use vectorize::Configured as Vectorizer;

/// Agent-local composition of existing short and long memory components.
/// This is configuration, not another storage contract or execution state.
pub struct Config<S, L> {
    pub short: std::sync::Arc<S>,
    pub long: std::sync::Arc<L>,
    pub(crate) scopes: Vec<String>,
    pub(crate) compression: Option<compression::Configured>,
    pub(crate) extraction: Option<extraction::Configured>,
    pub(crate) vectorizer: Option<Vectorizer>,
    pub(crate) id_fn: Option<fn() -> i64>,
}

impl<S, L> Clone for Config<S, L> {
    fn clone(&self) -> Self {
        Self {
            short: self.short.clone(),
            long: self.long.clone(),
            scopes: self.scopes.clone(),
            compression: self.compression.clone(),
            extraction: self.extraction.clone(),
            vectorizer: self.vectorizer.clone(),
            id_fn: self.id_fn,
        }
    }
}

impl<S, L> Config<S, L>
where
    S: short::Store,
    L: Memory,
{
    /// Compose two already-initialized memory stores.
    ///
    /// The short backend owns its Session/Chat implementation and the long
    /// backend owns its long-memory implementation. This constructor does not
    /// create, split, or configure either backend.
    pub fn new(short: S, long: L) -> Self {
        Self {
            short: std::sync::Arc::new(short),
            long: std::sync::Arc::new(long),
            scopes: Vec::new(),
            compression: None,
            extraction: None,
            vectorizer: None,
            id_fn: None,
        }
    }

    /// Access the Session store through the unified short-memory backend.
    pub fn session(&self) -> &S::Session {
        self.short.session()
    }

    /// Access the Chat store through the unified short-memory backend.
    pub fn chat(&self) -> &S::Chat {
        self.short.chat()
    }

    /// Access the configured long-memory backend.
    pub fn long(&self) -> &L {
        self.long.as_ref()
    }

    /// Select long-memory recall levels only. Empty means short history only;
    /// Session context is always short memory, not a long-memory recall level.
    /// Configuration is validated by Agent::build, not silently normalized.
    pub fn with_scopes<I, V>(mut self, scopes: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: AsRef<str>,
    {
        self.scopes = scopes
            .into_iter()
            .map(|scope| scope.as_ref().to_owned())
            .collect();
        self
    }

    /// Enable request-triggered background Session compression. The component
    /// owns its trigger size; Agent::build rejects an absent or zero size.
    /// Without this call no automatic compression is performed.
    pub fn with_compression<P: Compressor + 'static>(mut self, compressor: P) -> Self {
        self.compression = Some(compression::Configured::new(compressor));
        self
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        let mut seen = std::collections::BTreeSet::new();
        for scope in &self.scopes {
            if !matches!(scope.as_str(), "user" | "agent") || !seen.insert(scope) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "invalid long memory recall scopes",
                ));
            }
        }
        if let Some(compression) = &self.compression {
            compression.validate()?;
        }
        if self.extraction.is_some() && self.compression.is_none() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "automatic extraction requires a compression trigger",
            ));
        }
        if self.extraction.is_some() && self.id_fn.is_none() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "automatic extraction requires an ID function",
            ));
        }
        Ok(())
    }

    pub(crate) fn compression_limit(&self) -> Option<u64> {
        self.compression
            .as_ref()
            .and_then(compression::Configured::limit)
    }

    pub(crate) fn compression_retain(&self) -> Option<u64> {
        self.compression
            .as_ref()
            .and_then(compression::Configured::retain)
    }

    pub(crate) fn compression_count(&self, input: &crate::Messages) -> u64 {
        self.compression.as_ref().map_or_else(
            || default_compression_count(input),
            |value| value.count(input),
        )
    }

    /// Configure long-memory extraction for the completed Chat snapshot of a
    /// compression job. Recall scopes do not restrict extraction ownership.
    /// Record IDs are configured separately through `with_id_fn`.
    pub fn with_extractor<E>(mut self, extractor: E) -> Self
    where
        E: crate::components::memory::extraction::Extractor + 'static,
    {
        self.extraction = Some(extraction::Configured::new(extractor));
        self
    }

    /// Prepare vectors for automatic long-memory writeback with an application
    /// supplied Embed component. The long-memory backend never owns or calls
    /// Embed; it only receives the resulting vector in `memory::Request`.
    pub fn with_embed<E>(mut self, embed: E) -> Self
    where
        E: Embed + 'static,
    {
        self.vectorizer = Some(Vectorizer::new(embed));
        self
    }

    pub(crate) fn vectorizer(&self) -> Option<Vectorizer> {
        self.vectorizer.clone()
    }

    /// Supply the application's ID function, not a fixed ID. Configuration does
    /// not invoke it. Harness calls it only for a new aggregate; updates and
    /// write retries reuse the already allocated ID.
    /// Different Sessions may call it on concurrent background threads. The
    /// application function must return fresh positive IDs across those calls.
    ///
    /// A fixed value is not a valid configuration:
    /// ```compile_fail
    /// use halo::agent::{core::memory, memory::Config};
    /// let config = Config::new(memory::short::InMemory::new(), memory::InMemory::new());
    /// config.with_id_fn(123_i64);
    /// ```
    /// Capturing closures are not function pointers:
    /// ```compile_fail
    /// use halo::agent::{core::memory, memory::Config};
    /// use std::sync::{Arc, atomic::{AtomicI64, Ordering}};
    /// let config = Config::new(memory::short::InMemory::new(), memory::InMemory::new());
    /// let counter = Arc::new(AtomicI64::new(1));
    /// config.with_id_fn(move || counter.fetch_add(1, Ordering::SeqCst));
    /// ```
    pub fn with_id_fn(mut self, id_fn: fn() -> i64) -> Self {
        self.id_fn = Some(id_fn);
        self
    }
}

fn default_compression_count(input: &crate::Messages) -> u64 {
    serde_json::to_vec(input)
        .map(|bytes| u64::try_from(bytes.len().saturating_add(3) / 4).unwrap_or(u64::MAX))
        .unwrap_or(u64::MAX)
}
