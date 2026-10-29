use halo::agent::runtime::agent::{graph::Graph, node::Status};
use halo::agent::{self, Agent, Cancellation, Input};

fn configured_chat() -> halo::agent::provider::vllm::chat::Chat {
    let provider = halo::agent::provider::vllm::Vllm::builder()
        .api_key("compile-only")
        .build()
        .unwrap();
    provider
        .chat(halo::agent::model::chat::Config::new("compile-only"))
        .unwrap()
}

fn run_future_needs_no_caller_pinning<A: Agent>(agent: &A) {
    let future = agent.run(Input::default(), true, Cancellation::new());
    drop(future);
}

#[test]
fn chat_uses_the_agent_public_entry_without_a_runtime_owner() {
    let agent = agent::chat().model(configured_chat()).build().unwrap();
    run_future_needs_no_caller_pinning(&agent);
}

#[test]
fn react_and_configured_tools_are_agent_assembly_not_runtime_kinds() {
    let agent: halo::agent::runtime::agent::react::Agent<_> =
        agent::react().model(configured_chat()).build().unwrap();
    run_future_needs_no_caller_pinning(&agent);
}

#[test]
fn runtime_graph_keeps_execution_state_and_its_existing_json_shape() {
    use halo::agent::plan::{Plan, PlanNode};
    let mut plan = Plan::new(1).unwrap();
    plan.add(
        PlanNode {
            node_id: "a".into(),
            name: "search".into(),
            objective: "find information".into(),
        },
        vec![],
        vec![],
    )
    .unwrap();
    let graph = Graph::new(plan).unwrap();
    assert_eq!(graph.state("a").unwrap().status, Status::Ready);
    let json = serde_json::to_value(&graph).unwrap();
    assert_eq!(json["plan"]["nodes"]["a"]["name"], "search");
    assert_eq!(json["nodes"]["a"]["status"], "ready");
    assert_eq!(json["nodes"]["a"]["execution"], serde_json::Value::Null);
    assert_eq!(serde_json::from_value::<Graph>(json).unwrap(), graph);
}

#[test]
fn workflow_public_entry_uses_agent_namespace() {
    use halo::agent::core::node::Echo;
    use halo::agent::workflow::{
        WorkflowBuilder,
        graph::{END, START},
    };

    let mut builder = WorkflowBuilder::new("public-workflow-entry");
    builder.add_node("echo", Echo).unwrap();
    builder.add_edge(START, "echo").unwrap();
    builder.add_edge("echo", END).unwrap();
    let workflow = builder.compile().unwrap();

    assert_eq!(workflow.definition().definition_id, "public-workflow-entry");
}

#[test]
fn provider_public_entry_uses_agent_namespace() {
    let _ = halo::agent::provider::vllm::Config::default();
}
