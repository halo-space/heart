use halo::agent::workflow::{Builder, Runtime, graph};
use halo::agent::{Cancellation, Messages};
use std::sync::Arc;

#[test]
fn workflow_run_future_stays_bounded() {
    let mut builder = Builder::new("future-boundary");
    builder.add_node("echo", halo::agent::node::Echo).unwrap();
    builder.add_edge(graph::START, "echo").unwrap();
    builder.add_edge("echo", graph::END).unwrap();
    let runtime = Runtime::new(1001, builder.compile().unwrap()).unwrap();
    let ids: Arc<dyn Fn() -> i64 + Send + Sync> = Arc::new(|| 1);
    let future = runtime.run(Messages::default(), ids, Cancellation::new());
    assert!(
        std::mem::size_of_val(&future) < 16 * 1024,
        "Workflow public run Future: {} bytes",
        std::mem::size_of_val(&future)
    );
}
