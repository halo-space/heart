use futures_executor::block_on;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
struct Input {
    query: String,
}
#[derive(JsonSchema, Serialize)]
struct Output {
    answer: String,
}

async fn answer(input: Input, _: Cancellation) -> Result<Output, Error> {
    Ok(Output {
        answer: input.query,
    })
}

#[derive(Deserialize, JsonSchema)]
struct LimitedInput {
    #[schemars(length(min = 3))]
    query: String,
}

async fn limited(input: LimitedInput, _: Cancellation) -> Result<Output, Error> {
    Ok(Output {
        answer: input.query,
    })
}

fn single() -> Builder {
    let mut builder = Builder::new("example");
    builder.add_node("answer", answer).unwrap();
    builder.add_edge(START, "answer").unwrap();
    builder.add_edge("answer", END).unwrap();
    builder
}

#[test]
fn compile_preserves_schema_executor_and_stable_version() {
    let first = single().compile().unwrap();
    let second = single().compile().unwrap();
    assert_eq!(first.definition().version, second.definition().version);
    assert_eq!(
        first.definition().nodes["answer"].schema.input["type"],
        "object"
    );
    assert_eq!(
        serde_json::to_value(&first.definition().graph.ordinary_edges[0]).unwrap(),
        serde_json::json!({"from": "START", "to": "answer"})
    );
    let result = block_on(
        first
            .executor("answer")
            .unwrap()
            .call(serde_json::json!({"query": "hello"}), Cancellation::new()),
    )
    .unwrap();
    assert_eq!(
        result,
        Signal::Complete(serde_json::json!({"answer": "hello"}))
    );
}

#[test]
fn executor_enforces_json_schema_before_calling_node() {
    let mut builder = Builder::new("schema");
    builder.add_node("limited", limited).unwrap();
    builder.add_edge(START, "limited").unwrap();
    builder.add_edge("limited", END).unwrap();
    let workflow = builder.compile().unwrap();

    let invalid = block_on(
        workflow
            .executor("limited")
            .unwrap()
            .call(serde_json::json!({"query": "x"}), Cancellation::new()),
    )
    .unwrap_err();
    assert_eq!(invalid.code, "INVALID_ARGUMENTS");

    let valid = block_on(
        workflow
            .executor("limited")
            .unwrap()
            .call(serde_json::json!({"query": "abc"}), Cancellation::new()),
    )
    .unwrap();
    assert_eq!(
        valid,
        Signal::Complete(serde_json::json!({"answer": "abc"}))
    );
}

#[test]
fn rejects_scalar_node_and_invalid_graph() {
    let mut builder = Builder::new("scalar");
    let scalar = async |_: String, _: Cancellation| -> Result<String, Error> { Ok(String::new()) };
    assert_eq!(
        builder.add_node("bad", scalar).err().unwrap().code,
        "INVALID_SCHEMA"
    );
    let mut builder = Builder::new("missing");
    builder.add_node("answer", answer).unwrap();
    assert_eq!(builder.compile().unwrap_err().code, "MISSING_BOUNDARY");
}

#[test]
fn state_usage_is_recomputed_from_attempt_details() {
    let builder = single();
    let workflow = builder.compile().unwrap();
    let mut state = State::new(1, workflow.definition());
    state.nodes.get_mut("answer").unwrap().executions.insert(
        11,
        halo::agent::workflow::node::Execution {
            exec_id: 11,
            attempts: vec![halo::agent::workflow::node::Attempt {
                usage: Some(Usage {
                    input: Some(12),
                    output: Some(5),
                    total: Some(17),
                    ..Usage::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    state.recompute_usage();
    assert_eq!(state.usage.as_ref().and_then(|usage| usage.input), Some(12));
    assert_eq!(state.usage.as_ref().and_then(|usage| usage.total), Some(17));
}

#[test]
fn usage_recomputation_preserves_retry_details_without_double_counting_child() {
    use halo::agent::model::token::{InputDetails, OutputDetails};
    use halo::agent::workflow::node::{Attempt, Execution};

    let workflow = single().compile().unwrap();
    let mut child = State::new(1, workflow.definition());
    let make_usage = |total, private| Usage {
        input: Some(total - 2),
        output: Some(2),
        total: Some(total),
        input_details: Some(InputDetails { cached: Some(3) }),
        output_details: Some(OutputDetails { think: Some(1) }),
        details: serde_json::Map::from_iter([("provider".into(), serde_json::json!(private))]),
    };
    child.nodes.get_mut("answer").unwrap().executions.insert(
        12,
        Execution {
            exec_id: 12,
            attempts: vec![
                Attempt {
                    idx: 0,
                    status: halo::agent::workflow::node::status::attempt::Status::Failed,
                    usage: Some(make_usage(10, "failed")),
                    ..Default::default()
                },
                Attempt {
                    idx: 1,
                    status: halo::agent::workflow::node::status::attempt::Status::Succeeded,
                    usage: Some(make_usage(20, "succeeded")),
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
    );
    // Deliberately stale persisted aggregates must not become another
    // billable record after restore or another consistency commit.
    child.usage = Some(make_usage(99, "stale"));
    let mut parent = State::new(1, workflow.definition());
    parent.nodes.get_mut("answer").unwrap().executions.insert(
        11,
        Execution {
            exec_id: 11,
            attempts: vec![Attempt {
                usage: Some(make_usage(99, "stale")),
                graph: Some(Box::new(child)),
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    for _ in 0..3 {
        parent.recompute_usage();
        let total = parent.usage.as_ref().unwrap();
        assert_eq!(total.total, Some(30));
        assert_eq!(total.input_details.as_ref().unwrap().cached, Some(6));
        assert_eq!(total.output_details.as_ref().unwrap().think, Some(2));
        assert!(total.details.is_empty());
        let execution = &parent.nodes["answer"].executions[&11];
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(30));
        assert_eq!(
            execution.attempts[0].usage.as_ref().unwrap().total,
            Some(30)
        );
        let original = &execution.attempts[0].graph.as_ref().unwrap().nodes["answer"].executions
            [&12]
            .attempts[0];
        assert_eq!(
            original.usage.as_ref().unwrap().details["provider"],
            "failed"
        );
    }
}

#[test]
fn join_conflicts_with_ordinary_activation() {
    let mut builder = Builder::new("join");
    for id in ["a", "b", "merge"] {
        builder.add_node(id, halo::agent::node::Echo).unwrap();
    }
    builder.add_edge(START, "a").unwrap();
    builder.add_edge(START, "b").unwrap();
    builder.add_join_edge(["a", "b"], "merge").unwrap();
    assert_eq!(
        builder.add_edge("a", "merge").err().unwrap().code,
        "CONFLICTING_ACTIVATION"
    );
    builder.add_edge("merge", END).unwrap();
    assert_eq!(
        builder
            .compile()
            .unwrap()
            .definition()
            .graph
            .join_edges
            .len(),
        1
    );
}

#[test]
fn state_initializes_start_nodes_and_keeps_pause_facts_on_execution() {
    let compiled = single().compile().unwrap();
    let mut state = State::new(1, compiled.definition());
    assert_eq!(state.step, 0);
    assert_eq!(state.status, Status::Running);
    assert_eq!(
        state.nodes["answer"].status,
        halo::agent::workflow::node::status::Status::Idle
    );

    state
        .start_execution(compiled.definition(), "answer", 1, vec![])
        .unwrap();
    state
        .wait_execution("answer", 1, "wait_001".into())
        .unwrap();
    assert!(state.waits.is_empty());
    assert_eq!(state.pause("answer").unwrap().unwrap().wait_id, "wait_001");
    assert_eq!(
        state.nodes["answer"].executions[&1].waits["wait_001"].status,
        halo::agent::workflow::node::status::wait::Status::Waiting
    );
}

#[test]
fn state_rejects_pause_for_unknown_node_and_duplicate_node_pause() {
    let compiled = single().compile().unwrap();
    let mut state = State::new(1, compiled.definition());
    assert_eq!(
        state
            .wait_execution("missing", 1, "wait_001".into())
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    state
        .start_execution(compiled.definition(), "answer", 1, vec![])
        .unwrap();
    state
        .wait_execution("answer", 1, "wait_001".into())
        .unwrap();
    assert_eq!(
        state
            .wait_execution("answer", 1, "wait_002".into())
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    assert_eq!(state.pause("answer").unwrap().unwrap().wait_id, "wait_001");
}

#[test]
fn resume_only_changes_the_current_attempt_nested_wait() {
    use halo::agent::workflow::node::{Attempt, Execution};

    let compiled = single().compile().unwrap();
    let mut state = State::new(1, compiled.definition());

    let waiting_child = || {
        let mut child = State::new(1, compiled.definition());
        child.nodes.get_mut("answer").unwrap().executions.insert(
            10,
            Execution {
                exec_id: 10,
                status: halo::agent::workflow::node::status::execution::Status::Waiting,
                attempts: vec![Attempt {
                    status: halo::agent::workflow::node::status::attempt::Status::Waiting,
                    ..Default::default()
                }],
                waits: [(
                    "child_wait".into(),
                    halo::agent::workflow::node::Wait {
                        kind: "external".into(),
                        status: halo::agent::workflow::node::status::wait::Status::Waiting,
                        created_time: 0,
                        resolved_time: None,
                        response: None,
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            },
        );
        child
    };

    let mut historical = waiting_child();
    let mut current = waiting_child();
    let execution = Execution {
        exec_id: 1,
        status: halo::agent::workflow::node::status::execution::Status::Waiting,
        attempts: vec![
            Attempt {
                idx: 0,
                status: halo::agent::workflow::node::status::attempt::Status::Succeeded,
                graph: Some(Box::new(historical.clone())),
                ..Default::default()
            },
            Attempt {
                idx: 1,
                status: halo::agent::workflow::node::status::attempt::Status::Waiting,
                graph: Some(Box::new(current.clone())),
                ..Default::default()
            },
        ],
        waits: [(
            "wait_001".into(),
            halo::agent::workflow::node::Wait {
                kind: "external".into(),
                status: halo::agent::workflow::node::status::wait::Status::Waiting,
                created_time: 0,
                resolved_time: None,
                response: None,
            },
        )]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    state
        .nodes
        .get_mut("answer")
        .unwrap()
        .executions
        .insert(1, execution);

    // A valid child wait must not be resolved when the parent record is
    // invalid: the entire failed operation leaves the live graph intact.
    let mut invalid = state.clone();
    invalid
        .nodes
        .get_mut("answer")
        .unwrap()
        .executions
        .get_mut(&1)
        .unwrap()
        .waits
        .clear();
    let before = invalid.clone();
    assert_eq!(
        invalid
            .resume("answer", Messages::default())
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert_eq!(invalid, before);

    state
        .resume(
            "answer",
            Messages::new([halo::agent::Message::function(serde_json::json!({
                "ok": true
            }))]),
        )
        .unwrap();

    historical = state.nodes["answer"].executions[&1].attempts[0]
        .graph
        .as_ref()
        .unwrap()
        .as_ref()
        .clone();
    current = state.nodes["answer"].executions[&1].attempts[1]
        .graph
        .as_ref()
        .unwrap()
        .as_ref()
        .clone();
    assert_eq!(
        historical.pause("answer").unwrap().unwrap().wait_id,
        "child_wait"
    );
    assert_eq!(
        historical.nodes["answer"].executions[&10].status,
        halo::agent::workflow::node::status::execution::Status::Waiting
    );
    assert!(current.waits.is_empty());
    assert_eq!(
        current.nodes["answer"].executions[&10].status,
        halo::agent::workflow::node::status::execution::Status::Running
    );
}

#[test]
fn add_graph_node_preserves_child_definition_boundary() {
    let child = single().compile().unwrap();
    let mut parent = Builder::new("parent");
    parent.add_graph_node("child", &child).unwrap();
    parent.add_edge(START, "child").unwrap();
    parent.add_edge("child", END).unwrap();
    let parent = parent.compile().unwrap();

    let node = &parent.definition().nodes["child"];
    assert_eq!(node.workflow.as_ref().unwrap().definition_id, "example");
    assert_eq!(
        parent.graph_node("child").unwrap().definition(),
        child.definition()
    );
}

#[test]
fn add_graph_node_rejects_recursive_definition() {
    let child = single().compile().unwrap();
    let mut builder = Builder::new("example");
    assert_eq!(
        builder.add_graph_node("nested", &child).err().unwrap().code,
        "RECURSIVE_GRAPH"
    );
}

#[test]
fn state_runs_one_barrier_and_activates_ordinary_successor() {
    let mut builder = Builder::new("barrier");
    builder.add_node("a", halo::agent::node::Echo).unwrap();
    builder.add_node("b", halo::agent::node::Echo).unwrap();
    builder.add_edge(START, "a").unwrap();
    builder.add_edge("a", "b").unwrap();
    builder.add_edge("b", END).unwrap();
    let compiled = builder.compile().unwrap();
    let mut state = State::new(1, compiled.definition());

    assert_eq!(state.ready_nodes(compiled.definition()), vec!["a"]);
    state
        .start_execution(compiled.definition(), "a", 1, Vec::new())
        .unwrap();
    assert!(!state.advance_step());
    state
        .complete_execution("a", 1, Messages::default(), vec!["b".into()])
        .unwrap();
    assert!(state.advance_step());
    assert_eq!(state.step, 1);
    assert_eq!(state.ready_nodes(compiled.definition()), vec!["b"]);
}

#[test]
fn join_waits_until_all_sources_succeed() {
    let mut builder = Builder::new("join-state");
    for id in ["a", "b", "merge"] {
        builder.add_node(id, halo::agent::node::Echo).unwrap();
    }
    builder.add_edge(START, "a").unwrap();
    builder.add_edge(START, "b").unwrap();
    builder.add_join_edge(["a", "b"], "merge").unwrap();
    builder.add_edge("merge", END).unwrap();
    let compiled = builder.compile().unwrap();
    let mut state = State::new(1, compiled.definition());
    assert_eq!(state.ready_nodes(compiled.definition()), vec!["a", "b"]);
    state
        .start_batch(
            compiled.definition(),
            vec![("a".into(), 1, Vec::new()), ("b".into(), 2, Vec::new())],
        )
        .unwrap();
    state
        .complete_execution("a", 1, Messages::default(), vec!["merge".into()])
        .unwrap();
    assert!(
        !state
            .ready_nodes(compiled.definition())
            .iter()
            .any(|id| id == "merge")
    );
    state
        .complete_execution("b", 2, Messages::default(), vec!["merge".into()])
        .unwrap();
    assert!(state.advance_step());
    assert!(
        state
            .ready_nodes(compiled.definition())
            .iter()
            .any(|id| id == "merge")
    );
    state
        .start_execution(
            compiled.definition(),
            "merge",
            3,
            vec![
                halo::agent::workflow::node::ExecutionRef {
                    node_id: "a".into(),
                    exec_id: 1,
                },
                halo::agent::workflow::node::ExecutionRef {
                    node_id: "b".into(),
                    exec_id: 2,
                },
            ],
        )
        .unwrap();
}

#[test]
fn start_batch_is_atomic_when_one_entry_is_invalid() {
    let mut builder = Builder::new("batch-atomic");
    builder.add_node("a", halo::agent::node::Echo).unwrap();
    builder.add_node("b", halo::agent::node::Echo).unwrap();
    builder.add_edge(START, "a").unwrap();
    builder.add_edge(START, "b").unwrap();
    builder.add_edge("a", END).unwrap();
    builder.add_edge("b", END).unwrap();
    let compiled = builder.compile().unwrap();
    let mut state = State::new(1, compiled.definition());
    let error = state
        .start_batch(
            compiled.definition(),
            vec![
                ("a".into(), 1, Vec::new()),
                ("missing".into(), 2, Vec::new()),
            ],
        )
        .unwrap_err();
    assert_eq!(error.code, "INVALID_STATE");
    assert!(state.nodes["a"].executions.is_empty());
    assert!(state.nodes["b"].executions.is_empty());
}
