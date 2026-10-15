include!("shared.rs");
#[heart::node]
async fn wrong(input: Input, _: Cancellation) -> Result<Output, Error> {
    let value = std::rc::Rc::new(input.value);
    std::future::ready(()).await;
    Ok(Output { value: *value })
}
fn main() {}
