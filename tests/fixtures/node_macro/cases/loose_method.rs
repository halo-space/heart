include!("shared.rs");
struct Component;
impl Component {
    #[heart::node]
    async fn compute(&self, _: Input, _: &Cancellation) -> Result<Output, Error> {
        unimplemented!()
    }
}
fn main() {}
