use super::*;
use crate::components::plan::{Edge, PlanNode};

fn plan(edges: &[(&str, &str)]) -> Plan {
    let mut nodes = BTreeMap::new();
    for id in ["a", "b", "c"] {
        nodes.insert(
            id.to_owned(),
            PlanNode {
                node_id: id.to_owned(),
                name: id.to_owned(),
                objective: format!("run {id}"),
            },
        );
    }
    Plan {
        version: 1,
        nodes,
        edges: edges
            .iter()
            .map(|(from, to)| Edge {
                from: (*from).to_owned(),
                to: (*to).to_owned(),
            })
            .collect(),
    }
}

#[test]
fn roots_are_ready_and_join_waits_for_all_predecessors() {
    let mut graph = Graph::new(plan(&[("a", "c"), ("b", "c")])).unwrap();
    assert_eq!(graph.ready_nodes(), vec!["a", "b"]);
    assert_eq!(graph.state("c").unwrap().status, Status::Pending);

    graph.start("a", 101).unwrap();
    assert!(
        graph
            .succeed("a", Messages::default(), None)
            .unwrap()
            .is_empty()
    );
    assert_eq!(graph.state("c").unwrap().status, Status::Pending);

    graph.start("b", 102).unwrap();
    assert_eq!(
        graph.succeed("b", Messages::default(), None).unwrap(),
        vec!["c"]
    );
    assert_eq!(graph.state("c").unwrap().status, Status::Ready);
}

#[test]
fn failure_blocks_only_its_downstream_branch() {
    let mut graph = Graph::new(plan(&[("a", "c")])).unwrap();
    graph.start("a", 101).unwrap();
    graph.fail("a").unwrap();
    assert!(graph.ready_nodes().contains(&"b".to_owned()));
    assert_eq!(graph.state("c").unwrap().status, Status::Pending);
}

#[test]
fn waiting_resumes_without_activating_downstream() {
    let mut graph = Graph::new(plan(&[("a", "c")])).unwrap();
    graph.start("a", 101).unwrap();
    graph.wait("a").unwrap();
    assert_eq!(graph.state("c").unwrap().status, Status::Pending);
    graph.resume("a").unwrap();
    assert_eq!(graph.state("a").unwrap().status, Status::Running);
}

#[test]
fn invalid_transition_does_not_mutate_state() {
    let mut graph = Graph::new(plan(&[])).unwrap();
    graph.start("a", 101).unwrap();
    graph.succeed("a", Messages::default(), None).unwrap();
    let error = graph.resume("a").unwrap_err();
    assert_eq!(error.code, "INVALID_STATE");
    assert_eq!(graph.state("a").unwrap().status, Status::Succeeded);
}

#[test]
fn completion_requires_successful_leaves() {
    let mut graph = Graph::new(plan(&[("a", "c"), ("b", "c")])).unwrap();
    assert!(!graph.is_complete());
    graph.start("a", 101).unwrap();
    graph.succeed("a", Messages::default(), None).unwrap();
    graph.start("b", 102).unwrap();
    graph.succeed("b", Messages::default(), None).unwrap();
    graph.start("c", 103).unwrap();
    graph.succeed("c", Messages::default(), None).unwrap();
    assert!(graph.is_complete());
}

#[test]
fn retry_reuses_exec_id_and_keeps_attempt_history() {
    let mut graph = Graph::new(plan(&[])).unwrap();
    graph.start("a", 9001).unwrap();
    let first_error = Error::new("TEMPORARY", "provider unavailable");
    graph
        .fail_with_usage(
            "a",
            Some(first_error.clone()),
            Some(Usage {
                input: Some(2),
                output: Some(3),
                total: Some(5),
                input_details: None,
                output_details: None,
                details: Default::default(),
            }),
        )
        .unwrap();
    graph.retry("a").unwrap();
    graph
        .succeed(
            "a",
            Messages::default(),
            Some(Usage {
                input: Some(4),
                output: Some(6),
                total: Some(10),
                input_details: None,
                output_details: None,
                details: Default::default(),
            }),
        )
        .unwrap();

    let execution = graph.state("a").unwrap().execution.as_ref().unwrap();
    assert_eq!(execution.exec_id, 9001);
    assert_eq!(execution.attempts.len(), 2);
    assert_eq!(execution.attempts[0].error, Some(first_error));
    assert_eq!(execution.usage.as_ref().unwrap().total, Some(15));
    assert_eq!(
        graph.state("a").unwrap().usage.as_ref().unwrap().total,
        Some(15)
    );
}

#[test]
fn successful_execution_keeps_messages_and_finish_reason_at_each_runtime_level() {
    let mut graph = Graph::new(plan(&[])).unwrap();
    graph.start("a", 901).unwrap();
    let messages = Messages::new([crate::Message::function(serde_json::json!({
        "answer": "ok"
    }))]);
    graph
        .succeed("a", messages.clone(), None)
        .expect("success should commit");
    graph
        .set_finish_reason("a", Some("stop".into()))
        .expect("finish reason should commit");

    let state = graph.state("a").unwrap();
    let execution = state.execution.as_ref().unwrap();
    assert_eq!(state.messages.as_ref(), Some(&messages));
    assert_eq!(execution.messages, messages);
    assert_eq!(execution.attempts[0].messages, messages);
    assert_eq!(execution.attempts[0].finish_reason.as_deref(), Some("stop"));
    assert!(execution.attempts[0].finished_time.is_some());
}

#[test]
fn batch_start_is_atomic_and_enforces_concurrency() {
    let mut graph = Graph::new(plan(&[])).unwrap();
    let starts = vec![
        Start {
            node_id: "a".into(),
            exec_id: 1,
        },
        Start {
            node_id: "b".into(),
            exec_id: 2,
        },
    ];
    let error = graph.start_batch(1, &starts).unwrap_err();
    assert_eq!(error.code, "CONCURRENCY_LIMIT");
    assert_eq!(graph.state("a").unwrap().status, Status::Ready);
    assert_eq!(graph.state("b").unwrap().status, Status::Ready);

    assert_eq!(graph.start_batch(2, &starts).unwrap(), vec!["a", "b"]);
    assert_eq!(
        graph
            .state("a")
            .unwrap()
            .execution
            .as_ref()
            .unwrap()
            .exec_id,
        1
    );
}

#[test]
fn waiting_releases_a_batch_slot() {
    let mut graph = Graph::new(plan(&[])).unwrap();
    graph
        .start_batch(
            1,
            &[Start {
                node_id: "a".into(),
                exec_id: 1,
            }],
        )
        .unwrap();
    graph.wait("a").unwrap();
    assert_eq!(
        graph
            .start_batch(
                1,
                &[Start {
                    node_id: "b".into(),
                    exec_id: 2,
                }],
            )
            .unwrap(),
        vec!["b"]
    );
}

#[test]
fn replacing_plan_keeps_old_execution_history_without_reuse() {
    let mut graph = Graph::new(plan(&[("a", "c")])).unwrap();
    graph.start("a", 101).unwrap();
    graph.succeed("a", Messages::default(), None).unwrap();

    let replacement = Plan {
        version: 2,
        nodes: BTreeMap::from([(
            "new".into(),
            PlanNode {
                node_id: "new".into(),
                name: "a".into(),
                objective: "new objective".into(),
            },
        )]),
        edges: Vec::new(),
    };
    graph.update(replacement).unwrap();

    assert_eq!(graph.history().len(), 1);
    assert_eq!(graph.history()[0].nodes["a"].status, Status::Succeeded);
    assert_eq!(graph.state("new").unwrap().status, Status::Ready);
    assert!(graph.state("new").unwrap().execution.is_none());
}

#[test]
fn plan_update_reuses_independent_success_and_invalidates_downstream_closure() {
    let mut graph = Graph::new(plan(&[("a", "c")])).unwrap();
    graph.start("a", 101).unwrap();
    graph.succeed("a", Messages::default(), None).unwrap();
    graph.start("b", 102).unwrap();
    graph.succeed("b", Messages::default(), None).unwrap();

    let mut replacement = plan(&[("a", "c")]);
    replacement.version = 2;
    replacement.nodes.get_mut("a").unwrap().objective = "changed".into();
    graph.update(replacement).unwrap();

    assert_eq!(graph.state("b").unwrap().status, Status::Succeeded);
    assert_eq!(
        graph.state("b").unwrap().messages,
        Some(Messages::default())
    );
    assert_eq!(graph.state("a").unwrap().status, Status::Ready);
    assert_eq!(graph.state("c").unwrap().status, Status::Pending);
    assert!(graph.state("c").unwrap().execution.is_none());
}

#[test]
fn restored_graph_accepts_task_names_without_mounted_tools_validation() {
    let graph = Graph::new(plan(&[])).unwrap();
    let snapshot = serde_json::to_value(&graph).unwrap();
    let mut restored: Graph = serde_json::from_value(snapshot).unwrap();
    let replacement = plan(&[]);
    restored.update(replacement).unwrap();

    let snapshot = serde_json::to_value(&restored).unwrap();
    let mut restored: Graph = serde_json::from_value(snapshot).unwrap();
    let before = restored.clone();
    let replacement = Plan {
        version: 3,
        nodes: BTreeMap::from([(
            "billing".into(),
            PlanNode {
                node_id: "billing".into(),
                name: "billing".into(),
                objective: "charge the account".into(),
            },
        )]),
        edges: Vec::new(),
    };
    restored.update(replacement).unwrap();
    assert_ne!(restored, before);
}

#[test]
fn source_messages_are_read_by_node_id_without_merging() {
    let mut graph = Graph::new(plan(&[("a", "c"), ("b", "c")])).unwrap();
    graph.start("a", 101).unwrap();
    graph
        .succeed(
            "a",
            Messages::new([crate::Message::function(serde_json::json!({
                "source": "a"
            }))]),
            None,
        )
        .unwrap();
    graph.start("b", 102).unwrap();
    graph
        .succeed(
            "b",
            Messages::new([crate::Message::function(serde_json::json!({
                "source": "b"
            }))]),
            None,
        )
        .unwrap();

    let sources = graph.source_messages("c").unwrap();
    assert_eq!(sources.len(), 2);
    assert_eq!(sources["a"].as_slice()[0].values["source"], "a");
    assert_eq!(sources["b"].as_slice()[0].values["source"], "b");
}

#[test]
fn source_messages_require_all_sources_to_have_committed_success() {
    let mut graph = Graph::new(plan(&[("a", "c"), ("b", "c")])).unwrap();
    graph.start("a", 101).unwrap();
    graph.succeed("a", Messages::default(), None).unwrap();
    let error = graph.source_messages("c").unwrap_err();
    assert_eq!(error.code, "INVALID_STATE");
}

#[test]
fn invalid_plan_replacement_is_atomic() {
    let mut graph = Graph::new(plan(&[])).unwrap();
    let invalid = Plan {
        version: 2,
        nodes: BTreeMap::from([(
            "a".into(),
            PlanNode {
                node_id: "a".into(),
                name: "a".into(),
                objective: "a".into(),
            },
        )]),
        edges: vec![Edge {
            from: "a".into(),
            to: "missing".into(),
        }],
    };
    let error = graph.update(invalid).unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert_eq!(graph.plan.version, 1);
    assert!(graph.history().is_empty());
}

#[test]
fn execution_identity_cannot_be_shared_by_different_nodes() {
    let mut graph = Graph::new(plan(&[])).unwrap();
    graph.start("a", 101).unwrap();
    let before = graph.clone();
    assert_eq!(graph.start("b", 101).unwrap_err().code, "CONFLICT");
    assert_eq!(graph, before);
    graph.wait("a").unwrap();
    let before = graph.clone();
    assert_eq!(graph.start("b", 101).unwrap_err().code, "CONFLICT");
    assert_eq!(graph, before);
}

#[test]
fn duplicate_execution_identity_rejects_the_whole_batch() {
    let mut graph = Graph::new(plan(&[])).unwrap();
    let before = graph.clone();
    let starts = [
        Start {
            node_id: "a".into(),
            exec_id: 101,
        },
        Start {
            node_id: "b".into(),
            exec_id: 101,
        },
    ];
    assert_eq!(graph.start_batch(2, &starts).unwrap_err().code, "CONFLICT");
    assert_eq!(graph, before);
}

#[test]
fn late_invalid_batch_item_does_not_start_earlier_nodes() {
    for invalid in ["a", "missing", "c"] {
        let mut graph = Graph::new(plan(&[("b", "c")])).unwrap();
        let before = graph.clone();
        let starts = [
            Start {
                node_id: "a".into(),
                exec_id: 101,
            },
            Start {
                node_id: invalid.into(),
                exec_id: 102,
            },
        ];
        assert!(graph.start_batch(2, &starts).is_err());
        assert_eq!(graph, before);
    }
}

#[test]
fn historical_identity_remains_reserved_after_replan_and_restore() {
    let mut graph = Graph::from_plan(plan(&[])).unwrap();
    graph.start("a", 101).unwrap();
    graph.succeed("a", Messages::default(), None).unwrap();
    let mut replacement = plan(&[]);
    replacement.version = 2;
    replacement.nodes.get_mut("a").unwrap().objective = "changed".into();
    graph.update(replacement).unwrap();
    let mut graph: Graph = serde_json::from_value(serde_json::to_value(graph).unwrap()).unwrap();
    let before = graph.clone();
    assert_eq!(graph.start("a", 101).unwrap_err().code, "CONFLICT");
    assert_eq!(graph, before);
    let starts = [
        Start {
            node_id: "a".into(),
            exec_id: 102,
        },
        Start {
            node_id: "b".into(),
            exec_id: 101,
        },
    ];
    assert_eq!(graph.start_batch(2, &starts).unwrap_err().code, "CONFLICT");
    assert_eq!(graph, before);
    graph.start("a", 103).unwrap();
    assert_eq!(
        graph.history[0].nodes["a"]
            .execution
            .as_ref()
            .unwrap()
            .exec_id,
        101
    );
}

#[test]
fn plan_update_never_replaces_running_or_waiting_nodes() {
    for waiting in [false, true] {
        let mut graph = Graph::from_plan(plan(&[])).unwrap();
        graph.start("a", 101).unwrap();
        if waiting {
            graph.wait("a").unwrap();
        }
        let before = graph.clone();
        assert_eq!(graph.update(plan(&[])).unwrap_err().code, "INVALID_STATE");
        assert_eq!(graph, before);
    }
}

#[test]
fn source_success_requires_a_successful_current_attempt() {
    let mut graph = Graph::new(plan(&[("a", "c")])).unwrap();
    let state = graph.state_mut("a").unwrap();
    state.status = Status::Succeeded;
    state.messages = Some(Messages::new([crate::Message::function(
        serde_json::json!({"value": "old"}),
    )]));
    let before = graph.clone();
    assert_eq!(
        graph.source_messages("c").unwrap_err().code,
        "INVALID_STATE"
    );
    assert_eq!(graph, before);

    let state = graph.state_mut("a").unwrap();
    state.begin_execution(101).unwrap();
    state.execution.as_mut().unwrap().attempts[0].status = Status::Failed;
    let before = graph.clone();
    assert_eq!(
        graph.source_messages("c").unwrap_err().code,
        "INVALID_STATE"
    );
    assert_eq!(graph, before);
}

#[test]
fn same_capability_keeps_results_separate_and_sources_are_owned_snapshots() {
    let mut input = plan(&[("a", "c"), ("b", "c")]);
    input.nodes.get_mut("a").unwrap().name = "search".into();
    input.nodes.get_mut("b").unwrap().name = "search".into();
    let mut graph = Graph::new(input).unwrap();
    for (node_id, exec_id, city) in [("a", 101, "北京"), ("b", 102, "上海")] {
        graph.start(node_id, exec_id).unwrap();
        graph
            .succeed(
                node_id,
                Messages::new([crate::Message::function(serde_json::json!({"city": city}))]),
                None,
            )
            .unwrap();
    }
    let before = graph.clone();
    let mut sources = graph.source_messages("c").unwrap();
    assert_eq!(sources["a"].as_slice()[0].values["city"], "北京");
    assert_eq!(sources["b"].as_slice()[0].values["city"], "上海");
    sources.get_mut("a").unwrap().0[0]
        .values
        .insert("city".into(), serde_json::json!("changed"));
    assert_eq!(graph, before);
}

#[test]
fn failed_or_waiting_sources_cannot_publish_saved_messages() {
    for waiting in [false, true] {
        let mut graph = Graph::new(plan(&[("a", "c")])).unwrap();
        graph.start("a", 101).unwrap();
        if waiting {
            graph.wait("a").unwrap();
        } else {
            graph.fail("a").unwrap();
        }
        graph.state_mut("a").unwrap().messages = Some(Messages::new([crate::Message::function(
            serde_json::json!({"value": "uncommitted"}),
        )]));
        let before = graph.clone();
        assert_eq!(
            graph.source_messages("c").unwrap_err().code,
            "INVALID_STATE"
        );
        assert_eq!(graph, before);
    }
}
