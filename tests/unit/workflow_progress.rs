use super::*;

#[test]
fn workflow_run_does_not_inline_the_restore_machine() {
    let mut builder = WorkflowBuilder::new("future-boundary");
    builder.add_node("echo", crate::node::Echo).unwrap();
    builder.add_edge(graph::START, "echo").unwrap();
    builder.add_edge("echo", graph::END).unwrap();
    let mut engine = Engine::new(1001, builder.compile().unwrap()).unwrap();
    let ids: Arc<dyn Fn() -> i64 + Send + Sync> = Arc::new(|| 1);
    let future = engine.run_inner(Messages::default(), ids, crate::Cancellation::new());
    assert!(
        std::mem::size_of_val(&future) < 16 * 1024,
        "Workflow main Future: {} bytes",
        std::mem::size_of_val(&future)
    );
}
