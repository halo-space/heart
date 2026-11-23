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
    let agent: halo::agent::react::Agent<_> =
        agent::react().model(configured_chat()).build().unwrap();
    run_future_needs_no_caller_pinning(&agent);
}

#[test]
fn agent_graph_projection_keeps_the_static_plan_json_shape() {
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
    let graph = halo::agent::graph::State { plan };
    let json = serde_json::to_value(&graph).unwrap();
    assert_eq!(json["plan"]["nodes"]["a"]["name"], "search");
    assert_eq!(
        serde_json::from_value::<halo::agent::graph::State>(json).unwrap(),
        graph
    );
}

#[test]
fn workflow_public_entry_uses_agent_namespace() {
    use halo::agent::core::node::Echo;
    use halo::agent::workflow::{
        Builder,
        graph::{END, START},
    };

    let mut builder = Builder::new("public-workflow-entry");
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
