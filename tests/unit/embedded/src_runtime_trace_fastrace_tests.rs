use super::Root;

#[test]
fn builds_explicit_parent_child_spans() {
    let root = Root::start("heart.test.run");
    root.property("chat_id", "1");
    root.event("run.started");

    let node = root.child("heart.test.node");
    node.property("node_id", "answer");
    node.event("node.completed");

    let model = node.child("heart.test.model");
    model.property("provider", "test");
    model.event("model.completed");
}
