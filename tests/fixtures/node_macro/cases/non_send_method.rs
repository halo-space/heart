include!("shared.rs");
struct Component;
#[heart::node(call = compute)]
impl Component {
    async fn compute(&self, input: Input, _: &Cancellation) -> Result<Output, Error> {
        let value = std::rc::Rc::new(input.value);
        std::future::ready(()).await;
        Ok(Output { value: *value })
    }
}
fn main() {}
