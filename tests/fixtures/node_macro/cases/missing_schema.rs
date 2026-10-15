include!("shared.rs");
#[derive(Deserialize)]
struct NoSchema {
    value: i64,
}
#[heart::node]
async fn wrong(input: NoSchema, _: Cancellation) -> Result<Output, Error> {
    Ok(Output { value: input.value })
}
fn main() {}
