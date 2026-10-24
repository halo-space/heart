//! Workflow.run returns the shared observation stream; values stay in state.
use std::sync::{
    Arc,
    atomic::{AtomicI64, Ordering},
};

use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Messages};
use halo::{
    agent::event::Event,
    runtime::workflow::{
        GraphError, Runtime, WorkflowBuilder,
        graph::{END, START},
    },
};
use schemars::JsonSchema;
use serde::Serialize;

#[derive(Serialize, JsonSchema)]
struct Answer {
    answer: i64,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let mut builder = WorkflowBuilder::new("example");
    builder
        .add_node("answer", |_: Messages, _: Cancellation| async {
            Ok::<_, Error>(Answer { answer: 42 })
        })
        .map_err(graph_error)?;
    builder.add_edge(START, "answer").map_err(graph_error)?;
    builder.add_edge("answer", END).map_err(graph_error)?;
    let mut runtime = Runtime::new(1, builder.compile().map_err(graph_error)?)?;
    let ids = AtomicI64::new(100);
    {
        let mut events = runtime
            .run(
                Messages::default(),
                Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
                Cancellation::new(),
            )
            .await?;
        while let Some(event) = events.next().await {
            match event? {
                Event::Delta { message } => println!("增量：{:?}", message.content),
                Event::Complete { message, .. } => println!("完成：{:?}", message.content),
            }
        }
    }
    println!(
        "业务结果：{:?}",
        runtime.state().nodes["answer"].executions[&100].messages.0[0].values
    );
    Ok(())
}

fn graph_error(error: GraphError) -> Error {
    Error {
        code: error.code,
        message: error.message,
        details: error.details,
    }
}
