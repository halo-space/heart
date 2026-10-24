include!("shared.rs");

#[heart::node]
async fn ordinary(input: Input, _: Cancellation) -> Result<Output, Error> {
    Ok(Output { value: input.value })
}
struct Component;
#[heart::node(call = compute)]
impl Component {
    async fn compute(&self, input: Input, _: &Cancellation) -> Result<Output, Error> {
        Ok(Output { value: input.value })
    }
}
// Names used inside macro assertions must not shadow the original function.
#[heart::node]
async fn verify(input: Input, _: Cancellation) -> Result<Output, Error> {
    Ok(Output { value: input.value })
}
#[heart::node]
async fn callable(input: Input, _: Cancellation) -> Result<Output, Error> {
    Ok(Output { value: input.value })
}
struct SameNameCall;
#[heart::node(call = call)]
impl SameNameCall {
    async fn call(&self, input: Input, _: &Cancellation) -> Result<Output, Error> {
        Ok(Output { value: input.value })
    }
}
type N = Input;
type __HeartNodeDeclaration0 = Output;
#[heart::node]
async fn collision(input: N, _: Cancellation) -> Result<__HeartNodeDeclaration0, Error> {
    Ok(Output { value: input.value })
}
struct Review;
#[heart::node(signal = execute)]
impl Review {
    async fn execute(
        &self,
        _: Input,
        _: &Cancellation,
    ) -> Result<heart::agent::node::Signal<Output>, Error> {
        Ok(heart::agent::node::Signal::Wait(heart::agent::node::Wait {
            wait_id: "review".into(),
        }))
    }
}
fn main() {
    let mut builder = heart::runtime::workflow::WorkflowBuilder::new("renamed");
    builder.add_node("ordinary", ordinary).unwrap();
    builder.add_node("component", Component).unwrap();
    builder.add_node("review", Review).unwrap();
    builder.add_node("verify", verify).unwrap();
    builder.add_node("callable", callable).unwrap();
    builder.add_node("same_name_call", SameNameCall).unwrap();
    builder.add_node("collision", collision).unwrap();
}
