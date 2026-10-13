//! Private scene ownership for Workflow-embedded Agents and nested SubAgents.

use std::future::Future;
use std::sync::{Arc, Mutex};

use serde_json::Value;

#[derive(Clone)]
pub(crate) struct Scope {
    restore: Arc<Mutex<Option<Value>>>,
    latest: Arc<Mutex<Option<Value>>>,
}

impl Scope {
    pub(crate) fn new(restore: Option<Value>) -> Self {
        Self {
            restore: Arc::new(Mutex::new(restore)),
            latest: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) fn take_restore(&self) -> Option<Value> {
        self.restore.lock().ok()?.take()
    }

    pub(crate) fn capture(&self, state: Value) {
        if let Ok(mut latest) = self.latest.lock() {
            *latest = Some(state);
        }
    }

    pub(crate) fn latest(&self) -> Option<Value> {
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

/// An embedded run has an enclosing owner, even on its first call when there
/// is no Agent scene to restore yet. It must not open another top-level
/// Memory or Checkpoint lifecycle.
pub(crate) fn active() -> bool {
    CURRENT.try_with(|_| ()).is_ok()
}

pub(crate) fn capture(state: Value) -> bool {
    CURRENT.try_with(|scope| scope.capture(state)).is_ok()
}
