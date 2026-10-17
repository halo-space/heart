//! Transient leaf admission for one outer Workflow invocation.
//! No semaphore, permit or queue is serialized into Runtime state.
use std::{future::Future, sync::Arc};

use tokio::sync::Semaphore;

tokio::task_local! {
    static CURRENT: Arc<Semaphore>;
}

/// A public root run always owns an independent budget, even if a component
/// happens to invoke it while polling a different Workflow.
pub(super) async fn run<F: Future>(limit: usize, future: F) -> F::Output {
    CURRENT.scope(Arc::new(Semaphore::new(limit)), future).await
}

/// Private Graph invocations inherit the outer budget; direct engine tests
/// use the same entry without needing a second public execution method.
pub(super) async fn inherit<F: Future>(limit: usize, future: F) -> F::Output {
    if CURRENT.try_with(|_| ()).is_ok() {
        future.await
    } else {
        run(limit, future).await
    }
}

pub(super) fn slots() -> Arc<Semaphore> {
    CURRENT.with(Arc::clone)
}
