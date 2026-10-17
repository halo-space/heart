//! Private execution scope for Agent runs owned by another runtime or Agent.

use std::future::Future;
use std::sync::{Arc, Mutex};

use serde_json::Value;

#[derive(Clone)]
pub(crate) struct Scope {
    restore: Arc<Mutex<Option<Value>>>,
    latest: Arc<Mutex<Option<Arc<Value>>>>,
    id_fn: Option<fn() -> i64>,
    interaction: Option<crate::runtime::agent::reply::Run>,
    feedback: bool,
}

impl Scope {
    pub(crate) fn new(restore: Option<Value>) -> Self {
        Self {
            restore: Arc::new(Mutex::new(restore)),
            latest: Arc::new(Mutex::new(None)),
            id_fn: id_fn(),
            interaction: None,
            feedback: false,
        }
    }

    pub(crate) fn with_id_fn(mut self, id_fn: Option<fn() -> i64>) -> Self {
        self.id_fn = self.id_fn.or(id_fn);
        self
    }
    pub(crate) fn with_feedback(mut self, enabled: bool) -> Self {
        self.feedback = enabled;
        self
    }
    pub(crate) fn with_interaction(
        mut self,
        interaction: Option<crate::runtime::agent::reply::Run>,
    ) -> Self {
        self.interaction = interaction;
        self
    }

    pub(crate) fn take_restore(&self) -> Option<Value> {
        self.restore.lock().ok()?.take()
    }

    pub(crate) fn capture(&self, state: Value) {
        if let Ok(mut latest) = self.latest.lock() {
            *latest = Some(Arc::new(state));
        }
    }

    pub(crate) fn latest(&self) -> Option<Value> {
        self.latest_shared().map(|state| (*state).clone())
    }

    pub(crate) fn latest_shared(&self) -> Option<Arc<Value>> {
        self.latest.lock().ok()?.clone()
    }
}

tokio::task_local! {
    static CURRENT: Scope;
}

pub(crate) async fn run<F, T>(scope: Scope, future: F) -> T
where
    F: Future<Output = T>,
{
    CURRENT.scope(scope, future).await
}

pub(crate) fn restore() -> Option<Value> {
    CURRENT.try_with(Scope::take_restore).ok().flatten()
}

/// Run configuration inherited by children, never a serialized generator.
pub(crate) fn id_fn() -> Option<fn() -> i64> {
    CURRENT.try_with(|scope| scope.id_fn).ok().flatten()
}

pub(crate) fn interaction() -> Option<crate::runtime::agent::reply::Run> {
    CURRENT
        .try_with(|scope| scope.interaction.clone())
        .ok()
        .flatten()
}

pub(crate) fn feedback() -> bool {
    CURRENT.try_with(|scope| scope.feedback).unwrap_or(false)
}

/// An embedded run has an enclosing owner, even on its first call when there
/// is no Agent scene to restore yet. It must not open another top-level
/// Memory or Checkpoint lifecycle.
pub(crate) fn active() -> bool {
    CURRENT.try_with(|_| ()).is_ok()
}

pub(crate) fn capture(state: Value) -> bool {
    CURRENT.try_with(|scope| scope.capture(state)).is_ok()
}
