include!("shared.rs");
struct OtherError;
#[heart::node]
async fn wrong(input: Input, _: Cancellation) -> Result<Output, OtherError> {
    Ok(Output { value: input.value })
}
fn main() {}
