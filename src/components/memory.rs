use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::runtime::cancellation::Cancellation;

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
pub struct Config<S, C, T, L> {
    pub session: std::sync::Arc<S>,
    pub chat: std::sync::Arc<C>,
    pub trace: std::sync::Arc<T>,
    pub long: std::sync::Arc<L>,
    pub(crate) scopes: Vec<String>,
    pub(crate) compression: Option<compression::Configured>,
    pub(crate) extraction: Option<extraction::Configured>,
    pub(crate) id_fn: Option<fn() -> i64>,
}

impl<S, C, T, L> Clone for Config<S, C, T, L> {
    fn clone(&self) -> Self {
        Self {
            session: self.session.clone(),
            chat: self.chat.clone(),
            trace: self.trace.clone(),
            long: self.long.clone(),
            scopes: self.scopes.clone(),
            compression: self.compression.clone(),
            extraction: self.extraction.clone(),
            id_fn: self.id_fn,
        }
    }
}

impl<S, C, T, L> Config<S, C, T, L>
where
    S: short::session::Store,
    C: short::chat::Store,
    T: short::trace::Store,
    L: Memory,
{
    pub fn new(session: S, chat: C, trace: T, long: L) -> Self {
        Self {
            session: std::sync::Arc::new(session),
            chat: std::sync::Arc::new(chat),
            trace: std::sync::Arc::new(trace),
            long: std::sync::Arc::new(long),
            scopes: Vec::new(),
            compression: None,
            extraction: None,
            id_fn: None,
        }
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
    /// use halo_agents::{core::memory, memory::Config};
    /// let config = Config::new(memory::short::session::InMemory::new(),
    ///     memory::short::chat::InMemory::new(), memory::short::trace::InMemory::new(),
    ///     memory::InMemory::new());
    /// config.with_id_fn(123_i64);
    /// ```
    /// Capturing closures are not function pointers:
    /// ```compile_fail
    /// use halo_agents::{core::memory, memory::Config};
    /// use std::sync::{Arc, atomic::{AtomicI64, Ordering}};
    /// let config = Config::new(memory::short::session::InMemory::new(),
    ///     memory::short::chat::InMemory::new(), memory::short::trace::InMemory::new(),
    ///     memory::InMemory::new());
    /// let counter = Arc::new(AtomicI64::new(1));
    /// config.with_id_fn(move || counter.fetch_add(1, Ordering::SeqCst));
    /// ```
    pub fn with_id_fn(mut self, id_fn: fn() -> i64) -> Self {
        self.id_fn = Some(id_fn);
        self
    }
}

#[allow(async_fn_in_trait)]
pub trait Memory: Send + Sync {
    async fn write(&self, request: Request, cancellation: &Cancellation) -> Result<Item, Error>;

    async fn update(
        &self,
        request: Request,
        version: u64,
        cancellation: &Cancellation,
    ) -> Result<Item, Error>;

    async fn delete(&self, id: i64, version: u64, cancellation: &Cancellation)
    -> Result<(), Error>;

    async fn get(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        r#type: Type,
        cancellation: &Cancellation,
    ) -> Result<Option<Item>, Error>;

    async fn list(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        r#type: Type,
        cancellation: &Cancellation,
    ) -> Result<Vec<Item>, Error>;

    async fn keyword(
        &self,
        request: keyword::Request,
        cancellation: &Cancellation,
    ) -> Result<Vec<Item>, Error>;

    async fn vector(
        &self,
        request: vector::Request,
        cancellation: &Cancellation,
    ) -> Result<Vec<Item>, Error>;

    async fn hybrid(
        &self,
        request: hybrid::Request,
        cancellation: &Cancellation,
    ) -> Result<Vec<Item>, Error>;
}
