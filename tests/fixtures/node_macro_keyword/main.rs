use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use r#type::{Cancellation, Error};

#[derive(Deserialize, JsonSchema)]
struct Input {
    value: i64,
}
#[derive(Serialize, JsonSchema)]
struct Output {
    value: i64,
}

#[r#type::node]
async fn calculate(input: Input, _: Cancellation) -> Result<Output, Error> {
    Ok(Output { value: input.value })
}

fn main() {
    let mut builder = r#type::runtime::workflow::WorkflowBuilder::new("keyword");
    builder.add_node("calculate", calculate).unwrap();
}
