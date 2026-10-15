include!("shared.rs");
struct Component;
#[heart::node(call = compute)]
impl std::fmt::Display for Component {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}
fn main() {}
