use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Cancellation;

pub type Error = crate::Error;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Type {
    Summary,
    Fact,
    Preference,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Request {
    pub id: i64,
    pub r#type: Type,
    pub user_id: i64,
    pub agent_id: Option<i64>,
    pub session_id: Option<i64>,
    pub from: i64,
    pub to: i64,
    pub content: String,
    pub metadata: Option<Map<String, Value>>,
    pub vector: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Item {
    pub id: i64,
    pub r#type: Type,
    pub user_id: i64,
    pub agent_id: Option<i64>,
    pub session_id: Option<i64>,
    pub from: i64,
    pub to: i64,
    pub content: String,
    pub version: u64,
    pub created_time: i64,
    pub updated_time: i64,
    pub metadata: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Filter {
    pub user_id: i64,
    pub agent_id: Option<i64>,
    pub session_id: Option<i64>,
    pub r#type: Type,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct KeywordRequest {
    pub query: String,
    #[serde(default)]
    pub options: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VectorRequest {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub options: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HybridRequest {
    pub query: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub options: Map<String, Value>,
}

pub mod keyword {
    pub use super::KeywordRequest as Request;
}

pub mod vector {
    pub use super::VectorRequest as Request;
}

pub mod hybrid {
    pub use super::HybridRequest as Request;
}

mod compression;
pub mod extraction;
pub mod short;

/// Agent-local composition of existing short and long memory components.
/// This is configuration, not another storage contract or execution state.
pub struct Config<S, L> {
    pub short: std::sync::Arc<S>,
    pub long: std::sync::Arc<L>,
    pub(crate) scopes: Vec<String>,
    pub(crate) compression: Option<compression::Configured>,
    pub(crate) extraction: Option<extraction::Configured>,
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
    pub fn with_compression<P: crate::compression::Compressor + 'static>(
        mut self,
        compressor: P,
    ) -> Self {
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
        E: extraction::Extractor + 'static,
    {
        self.extraction = Some(extraction::Configured::new(extractor));
        self
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

pub trait Memory: Send + Sync {
    fn write(
        &self,
        request: Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Item, Error>>;

    fn update(
        &self,
        request: Request,
        version: u64,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Item, Error>>;

    fn delete(
        &self,
        id: i64,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<(), Error>>;

    fn get(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        r#type: Type,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Option<Item>, Error>>;

    fn list(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        r#type: Type,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Item>, Error>>;

    fn keyword(
        &self,
        request: keyword::Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Item>, Error>>;

    fn vector(
        &self,
        request: vector::Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Item>, Error>>;

    fn hybrid(
        &self,
        request: hybrid::Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Item>, Error>>;
}
