include!("shared.rs");
#[heart::node]
#[cfg(any())]
async fn excluded(_: MissingInput, _: MissingCancellation) -> MissingResult {
    unimplemented!()
}
#[heart::node(call = excluded)]
#[cfg(any())]
impl MissingComponent {
    async fn excluded(&self, _: MissingInput) -> MissingResult {
        unimplemented!()
    }
}
fn main() {}
