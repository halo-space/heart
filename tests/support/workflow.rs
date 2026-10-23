//! Test-only collection of the new run stream followed by reading saved results.

use std::sync::Arc;

use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Messages};
use halo::runtime::workflow::{Runtime, graph, node};

pub async fn run_workflow(
    runtime: &mut Runtime,
    initial: Messages,
    next_id: Arc<dyn Fn() -> i64 + Send + Sync>,
    cancellation: Cancellation,
) -> Result<Messages, Error> {
    {
        let mut events = runtime.run(initial, next_id, cancellation).await?;
        while let Some(event) = events.next().await {
            event?;
        }
    }
    Ok(Messages::new(
        runtime
            .state()
            .nodes
            .values()
            .flat_map(|node| node.executions.values())
            .filter(|execution| {
                execution.status == node::ExecutionStatus::Succeeded
                    && execution.to.iter().any(|to| to == graph::END)
            })
            .flat_map(|execution| execution.messages.as_slice().iter().cloned()),
    ))
}
