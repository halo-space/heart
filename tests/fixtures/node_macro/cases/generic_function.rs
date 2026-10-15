include!("shared.rs");
#[heart::node]
async fn wrong<T>(_: T, _: Cancellation) -> Result<Output, Error> {
    unimplemented!()
}
fn main() {}
