include!("shared.rs");
struct Component;
#[heart::node(call = compute)]
impl Component {
    async fn compute(
        &self,
        _: Input,
        _: &Cancellation,
    ) -> Result<heart::node::Signal<Output>, Error> {
        unimplemented!()
    }
}
fn main() {}
