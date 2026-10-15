include!("shared.rs");
struct Component;
#[heart::node(call = compute)]
impl Component {
    async fn compute(&mut self, _: Input, _: &Cancellation) -> Result<Output, Error> {
        unimplemented!()
    }
}
fn main() {}
