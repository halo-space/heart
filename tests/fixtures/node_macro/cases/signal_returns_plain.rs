include!("shared.rs");
struct Component;
#[heart::node(signal = compute)]
impl Component {
    async fn compute(&self, _: Input, _: &Cancellation) -> Result<Output, Error> {
        unimplemented!()
    }
}
fn main() {}
