include!("shared.rs");
#[heart::node]
async fn wrong(_: Input, _: Cancellation) -> Result<&'static Output, Error> {
    unimplemented!()
}
fn main() {}
