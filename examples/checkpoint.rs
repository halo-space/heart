//! Save a complete Workflow Runtime snapshot and reconstruct it.
use std::sync::{
    Arc,
    atomic::{AtomicI64, Ordering},
};

use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Messages};
use halo::agent::{
    checkpoint::Checkpoint as _,
    core,
    workflow::{
        GraphError, Runtime, WorkflowBuilder,
        graph::{END, START},
    },
};
use schemars::JsonSchema;
use serde::Serialize;

#[derive(JsonSchema, Serialize)]
struct Answer {
    value: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let mut builder = WorkflowBuilder::new("checkpoint-example");
    builder
        .add_node("answer", |_: Messages, _: Cancellation| async {
            Ok::<_, Error>(Answer {
                value: "checkpointed".into(),
            })
        })
        .map_err(graph_error)?;
    builder.add_edge(START, "answer").map_err(graph_error)?;
    builder.add_edge("answer", END).map_err(graph_error)?;
    let workflow = builder.compile().map_err(graph_error)?;

    // InMemory is enough for this runnable example. Replace it with a
    // persistent Checkpoint implementation to survive process restarts.
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let runtime = Runtime::new(9001, workflow.clone())?.with_checkpoint(checkpoint.clone());
    let next_id = Arc::new(AtomicI64::new(1));
    let mut events = runtime
        .run(
            Messages::default(),
            Arc::new(move || next_id.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        )
        .await?;
    while let Some(event) = events.next().await {
        println!("event = {:?}", event?);
    }
    drop(events);

    // The backend contains the complete Runtime State, not only the final text.
    let saved = checkpoint.load(9001, &Cancellation::new()).await?;
    println!("saved checkpoint status = {}", saved["status"]);

    // A new Runtime can be reconstructed from the saved scene and the same
    // compiled definition. A persistent backend is used in a multi-process app.
    let mut restored = Runtime::from_snapshot(9001, workflow, saved)?;
    println!("restored status = {:?}", restored.state().status);
    Ok(())
}

fn graph_error(error: GraphError) -> Error {
    Error {
        code: error.code,
        message: error.message,
        details: error.details,
    }
}
