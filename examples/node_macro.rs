//! Native function and component Node declarations; no provider calls.
use futures_executor::block_on;
use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::workflow::{
        GraphError, Runtime, WorkflowBuilder,
        graph::{END, START},
    },
    node,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

#[derive(Deserialize, JsonSchema)]
struct Input {
    value: i64,
}
#[derive(Serialize, JsonSchema)]
struct Output {
    answer: i64,
}

struct Compute;
#[node(call = calculate)]
impl Compute {
    async fn calculate(&self, input: Input, _: &Cancellation) -> Result<Output, Error> {
        Ok(Output {
            answer: input.value + 1,
        })
    }
}
fn graph_error(error: GraphError) -> Error {
    Error {
        code: error.code,
        message: error.message,
        details: error.details,
    }
}
fn main() -> Result<(), Error> {
    block_on(async {
        let mut builder = WorkflowBuilder::new("node-macro");
        builder
            .add_node("calculate", Compute)
            .map_err(graph_error)?;
        builder.add_edge(START, "calculate").map_err(graph_error)?;
        builder.add_edge("calculate", END).map_err(graph_error)?;
        let mut runtime = Runtime::new(8301, builder.compile().map_err(graph_error)?)?;
        {
            let mut events = runtime
                .run(
                    Messages::new([Message::function(json!({"value":41}))]),
                    Arc::new(|| 701),
                    Cancellation::new(),
                )
                .await?;
            while let Some(event) = events.next().await {
                println!("事件：{:?}", event?);
            }
        }
        println!(
            "业务结果：{:?}",
            runtime.state().nodes["calculate"].executions[&701]
                .messages
                .0[0]
                .values
        );
        Ok(())
    })
}
