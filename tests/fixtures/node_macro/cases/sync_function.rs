include!("shared.rs");
#[heart::node]
fn wrong(input: Input, _: Cancellation) -> Result<Output, Error> {
    Ok(Output { value: input.value })
}
fn main() {}
