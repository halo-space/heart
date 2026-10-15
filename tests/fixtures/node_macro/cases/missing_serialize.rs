include!("shared.rs");
#[derive(JsonSchema)]
struct NoSerialize {
    value: i64,
}
#[heart::node]
async fn wrong(input: Input, _: Cancellation) -> Result<NoSerialize, Error> {
    Ok(NoSerialize { value: input.value })
}
fn main() {}
