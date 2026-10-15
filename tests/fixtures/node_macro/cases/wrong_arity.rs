include!("shared.rs");
#[heart::node]
async fn wrong(_: Input, _: Input, _: Cancellation) -> Result<Output, Error> {
    unimplemented!()
}
fn main() {}
