//! Generic batch semantics without special Node kinds or candidate executions.
//! The Node owns its private successful-item cache; Runtime owns the one
//! execution, retry Attempts and all-or-nothing downstream submission.
#[path = "support/workflow.rs"]
mod workflow_support;
use futures_executor::block_on;
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::runtime::workflow::{
    CompiledWorkflow, FieldSource, InputBinding, NodeAttrs, RetryPolicy, Runtime, WorkflowBuilder,
    graph,
    node::{self, ExecutionRef},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicI64, AtomicUsize, Ordering},
};
use workflow_support::run_workflow;

#[derive(Clone, Deserialize, Serialize, JsonSchema)]
struct Tasks {
    tasks: Vec<i64>,
}

#[derive(Deserialize, Serialize, JsonSchema)]
struct Results {
    results: Vec<i64>,
}

fn binding(field: &str, source: &str) -> NodeAttrs {
    NodeAttrs {
        input: BTreeMap::from([(
            field.into(),
            InputBinding {
                from: FieldSource {
                    node_id: source.into(),
                    field: field.into(),
                },
            },
        )]),
    }
}

#[test]
fn ordinary_batch_node_retries_only_failed_items_and_submits_once() {
    for max_retries in [0, 1] {
        let cached = Arc::new(Mutex::new(BTreeMap::<i64, i64>::new()));
        let calls = Arc::new(Mutex::new(BTreeMap::<i64, usize>::new()));
        let downstream = Arc::new(AtomicUsize::new(0));
        let mut builder = WorkflowBuilder::new("generic-batch-subset");
        builder
            .add_node("source", |_: Messages, _: Cancellation| async {
                Ok::<_, Error>(Tasks {
                    tasks: vec![30, 10, 20],
                })
            })
            .unwrap();
        builder
            .add_node("batch", {
                let cached = cached.clone();
                let calls = calls.clone();
                move |input: Tasks, cancellation: Cancellation| {
                    let cached = cached.clone();
                    let calls = calls.clone();
                    async move {
                        let tasks = input.tasks;
                        let results =
                            stream::iter(tasks.iter().copied().enumerate().map(|(idx, id)| {
                                let cached = cached.clone();
                                let calls = calls.clone();
                                let cancellation = cancellation.clone();
                                async move {
                                    if cancellation.is_cancelled() {
                                        return (
                                            idx,
                                            Err(Error::new("CANCELLED", "batch cancelled")),
                                        );
                                    }
                                    if let Some(result) = cached.lock().unwrap().get(&id).copied() {
                                        return (idx, Ok(result));
                                    }
                                    let attempt = {
                                        let mut calls = calls.lock().unwrap();
                                        let count = calls.entry(id).or_default();
                                        let attempt = *count;
                                        *count += 1;
                                        attempt
                                    };
                                    if id == 20 && attempt == 0 {
                                        eprintln!("business batch item failed: TIMEOUT");
                                        return (
                                            idx,
                                            Err(Error::new("TIMEOUT", "temporary item failure")),
                                        );
                                    }
                                    let result = id * 10;
                                    cached.lock().unwrap().insert(id, result);
                                    (idx, Ok(result))
                                }
                            }))
                            .buffer_unordered(3)
                            .collect::<Vec<_>>()
                            .await;
                        let mut results = results;
                        results.sort_by_key(|(idx, _)| *idx);
                        let output = results
                            .into_iter()
                            .map(|(_, result)| result)
                            .collect::<Result<Vec<_>, _>>()?;
                        // Final delivery releases this Node's own retry state.
                        for id in tasks {
                            cached.lock().unwrap().remove(&id);
                        }
                        Ok::<_, Error>(Results { results: output })
                    }
                }
            })
            .unwrap();
        builder
            .add_node("summary", {
                let downstream = downstream.clone();
                move |input: Results, _: Cancellation| {
                    let downstream = downstream.clone();
                    async move {
                        downstream.fetch_add(1, Ordering::SeqCst);
                        assert_eq!(input.results, [300, 100, 200]);
                        Ok::<_, Error>(Messages::new([Message::function(
                            serde_json::json!({"results":input.results}),
                        )]))
                    }
                }
            })
            .unwrap();
        builder.add_edge(graph::START, "source").unwrap();
        builder.add_fan_out_edges("source", "batch").unwrap();
        builder.add_edge("batch", "summary").unwrap();
        builder.add_edge("summary", graph::END).unwrap();
        builder
            .set_attrs("batch", binding("tasks", "source"))
            .unwrap();
        builder
            .set_attrs("summary", binding("results", "batch"))
            .unwrap();
        builder
            .set_retry(
                "batch",
                RetryPolicy {
                    max_retries,
                    min_delay_ms: 0,
                    max_delay_ms: 0,
                },
            )
            .unwrap();
        let mut runtime = Runtime::new(901, builder.compile().unwrap()).unwrap();
        let ids = Arc::new(AtomicI64::new(1000));
        let ids_for_run = ids.clone();
        let result = block_on(run_workflow(
            &mut runtime,
            Messages::default(),
            Arc::new(move || ids_for_run.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        ));
        let state = runtime.state();
        let executions = &state.nodes["batch"].executions;
        assert_eq!(executions.len(), 1);
        let execution = &executions[&1001];
        assert_eq!(
            execution.from,
            [ExecutionRef {
                node_id: "source".into(),
                exec_id: 1000
            }]
        );
        assert_eq!(execution.attempts.len(), max_retries as usize + 1);
        assert!(
            execution.attempts[0].messages.is_empty(),
            "partial batches are not normal output"
        );
        assert_eq!(
            execution.attempts[0].error.as_ref().unwrap().code,
            "TIMEOUT"
        );
        assert!(
            execution
                .attempts
                .iter()
                .all(|attempt| attempt.steps.is_empty()),
            "items must not acquire Runtime Attempts or Steps"
        );
        let calls = calls.lock().unwrap();
        assert_eq!(calls[&30], 1);
        assert_eq!(calls[&10], 1);
        assert_eq!(calls[&20], max_retries as usize + 1);
        if max_retries == 0 {
            assert_eq!(result.unwrap_err().code, "TIMEOUT");
            assert_eq!(downstream.load(Ordering::SeqCst), 0);
            assert!(execution.messages.is_empty());
            assert!(state.nodes["summary"].executions.is_empty());
            assert_eq!(execution.status, node::ExecutionStatus::Failed);
        } else {
            assert_eq!(
                result.unwrap().0[0].values["results"],
                serde_json::json!([300, 100, 200])
            );
            assert_eq!(downstream.load(Ordering::SeqCst), 1);
            assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
            assert_eq!(execution.attempts[1].idx, 1);
            assert!(execution.error.is_none());
            assert!(cached.lock().unwrap().is_empty());
        }
    }
}

async fn source_tasks(source_calls: &AtomicUsize) -> Tasks {
    source_calls.fetch_add(1, Ordering::SeqCst);
    Tasks {
        tasks: vec![30, 10, 20],
    }
}

fn stateless_batch_workflow(
    source_calls: Arc<AtomicUsize>,
    item_calls: Arc<Mutex<Vec<i64>>>,
) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("batch-snapshot-fallback");
    builder
        .add_node("source", move |_: Messages, _: Cancellation| {
            let source_calls = source_calls.clone();
            async move { Ok::<_, Error>(source_tasks(&source_calls).await) }
        })
        .unwrap();
    builder
        .add_node("batch", move |input: Tasks, _: Cancellation| {
            let item_calls = item_calls.clone();
            async move {
                let mut results = Vec::new();
                for id in input.tasks {
                    item_calls.lock().unwrap().push(id);
                    results.push(id * 10);
                }
                Ok::<_, Error>(Results { results })
            }
        })
        .unwrap();
    builder.add_edge(graph::START, "source").unwrap();
    builder.add_fan_out_edges("source", "batch").unwrap();
    builder.add_edge("batch", graph::END).unwrap();
    builder
        .set_attrs("batch", binding("tasks", "source"))
        .unwrap();
    builder.compile().unwrap()
}

#[test]
fn restoring_full_runtime_reuses_successful_nodes_and_does_not_guess_internal_batch_facts() {
    use halo::agent::checkpoint::Checkpoint as _;
    block_on(async {
        let original_calls = Arc::new(AtomicUsize::new(0));
        let workflow = stateless_batch_workflow(original_calls.clone(), Default::default());
        let mut original = Runtime::new(902, workflow).unwrap();
        original.state_mut().initial = Some(Messages::default());
        original
            .start_batch(vec![("source".into(), 2000, vec![])])
            .unwrap();
        // Build the saved in-flight scene through existing public State
        // transitions, using the same source function registered in the graph.
        let source = source_tasks(&original_calls).await;
        let messages = Messages::new([Message::function(serde_json::to_value(source).unwrap())]);
        original
            .complete("source", 2000, messages, vec!["batch".into()])
            .unwrap();
        assert!(original.advance_step());
        original
            .start_batch(vec![(
                "batch".into(),
                2001,
                vec![ExecutionRef {
                    node_id: "source".into(),
                    exec_id: 2000,
                }],
            )])
            .unwrap();
        assert_eq!(original_calls.load(Ordering::SeqCst), 1);
        let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
        original
            .save_checkpoint(checkpoint.as_ref(), &Cancellation::new())
            .await
            .unwrap();
        let saved = checkpoint.load(902, &Cancellation::new()).await.unwrap();
        let encoded = serde_json::to_vec(&saved).unwrap();
        drop(original);

        // Reconstruct components with no process-local cache. The snapshot
        // contains a successful source and an unfinished batch, but no item
        // results: restoring the graph must not infer missing internal facts.
        let source_calls = Arc::new(AtomicUsize::new(0));
        let item_calls = Arc::new(Mutex::new(Vec::new()));
        let mut restored = Runtime::from_snapshot(
            902,
            stateless_batch_workflow(source_calls.clone(), item_calls.clone()),
            serde_json::from_slice(&encoded).unwrap(),
        )
        .unwrap()
        .with_checkpoint(checkpoint.clone());
        let result = run_workflow(
            &mut restored,
            Messages::default(),
            Arc::new(|| panic!("restore must reuse existing execution identities")),
            Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(source_calls.load(Ordering::SeqCst), 0);
        assert_eq!(*item_calls.lock().unwrap(), [30, 10, 20]);
        assert_eq!(
            result.0[0].values["results"],
            serde_json::json!([300, 100, 200])
        );
        assert_eq!(restored.state().nodes["batch"].executions.len(), 1);
        let execution = &restored.state().nodes["batch"].executions[&2001];
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].idx, 0);
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(
            serde_json::to_value(&restored.state().nodes["source"]).unwrap(),
            saved["nodes"]["source"]
        );
        assert_eq!(
            checkpoint.load(902, &Cancellation::new()).await.unwrap(),
            restored.snapshot().unwrap()
        );
    });
}

fn failing_batch_workflow(
    source_calls: Arc<AtomicUsize>,
    item_calls: Arc<Mutex<Vec<i64>>>,
    downstream: Arc<AtomicUsize>,
) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("batch-recovery-retry-budget");
    builder
        .add_node("source", move |_: Messages, _: Cancellation| {
            let source_calls = source_calls.clone();
            async move { Ok::<_, Error>(source_tasks(&source_calls).await) }
        })
        .unwrap();
    builder
        .add_node("batch", move |input: Tasks, _: Cancellation| {
            let item_calls = item_calls.clone();
            async move {
                // The function deliberately has no successful-item cache.
                // Its private partial computation is not a saved Node result.
                let mut results = Vec::new();
                let mut failed = false;
                for id in input.tasks {
                    item_calls.lock().unwrap().push(id);
                    if id == 20 {
                        failed = true;
                    } else {
                        results.push(id * 10);
                    }
                }
                if failed {
                    Err(Error::new("TIMEOUT", "batch item still unavailable"))
                } else {
                    Ok(Results { results })
                }
            }
        })
        .unwrap();
    builder
        .add_node("summary", move |input: Results, _: Cancellation| {
            let downstream = downstream.clone();
            async move {
                downstream.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Error>(Messages::new([Message::function(
                    serde_json::to_value(input).unwrap(),
                )]))
            }
        })
        .unwrap();
    builder.add_edge(graph::START, "source").unwrap();
    builder.add_fan_out_edges("source", "batch").unwrap();
    builder.add_edge("batch", "summary").unwrap();
    builder.add_edge("summary", graph::END).unwrap();
    builder
        .set_attrs("batch", binding("tasks", "source"))
        .unwrap();
    builder
        .set_attrs("summary", binding("results", "batch"))
        .unwrap();
    builder
        .set_retry(
            "batch",
            RetryPolicy {
                max_retries: 1,
                min_delay_ms: 0,
                max_delay_ms: 0,
            },
        )
        .unwrap();
    builder.compile().unwrap()
}

#[test]
fn failed_batch_checkpoint_is_terminal_and_never_replays_completed_or_failed_nodes() {
    use halo::agent::checkpoint::Checkpoint as _;
    block_on(async {
        let source_calls = Arc::new(AtomicUsize::new(0));
        let item_calls = Arc::new(Mutex::new(Vec::new()));
        let downstream = Arc::new(AtomicUsize::new(0));
        let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
        let workflow =
            failing_batch_workflow(source_calls.clone(), item_calls.clone(), downstream.clone());
        let ids = Arc::new(AtomicI64::new(3000));
        let mut original = Runtime::new(903, workflow)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        let error = run_workflow(
            &mut original,
            Messages::default(),
            Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        )
        .await
        .unwrap_err();
        assert_eq!(error.code, "TIMEOUT");
        assert_eq!(source_calls.load(Ordering::SeqCst), 1);
        assert_eq!(*item_calls.lock().unwrap(), [30, 10, 20, 30, 10, 20]);
        assert_eq!(downstream.load(Ordering::SeqCst), 0);
        let saved = checkpoint.load(903, &Cancellation::new()).await.unwrap();
        assert_eq!(saved, original.snapshot().unwrap());
        drop(original);

        let source_calls = Arc::new(AtomicUsize::new(0));
        let item_calls = Arc::new(Mutex::new(Vec::new()));
        let downstream = Arc::new(AtomicUsize::new(0));
        let mut restored = Runtime::from_snapshot(
            903,
            failing_batch_workflow(source_calls.clone(), item_calls.clone(), downstream.clone()),
            saved.clone(),
        )
        .unwrap()
        .with_checkpoint(checkpoint.clone());
        for _ in 0..2 {
            assert_eq!(
                run_workflow(
                    &mut restored,
                    Messages::default(),
                    Arc::new(|| panic!("terminal restore must not allocate an execution")),
                    Cancellation::new(),
                )
                .await
                .unwrap_err(),
                error
            );
            assert_eq!(restored.snapshot().unwrap(), saved);
        }
        assert_eq!(source_calls.load(Ordering::SeqCst), 0);
        assert!(item_calls.lock().unwrap().is_empty());
        assert_eq!(downstream.load(Ordering::SeqCst), 0);
        let execution = &restored.state().nodes["batch"].executions[&3001];
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(execution.status, node::ExecutionStatus::Failed);
        assert!(execution.messages.is_empty());
        assert!(restored.state().nodes["summary"].executions.is_empty());
        assert_eq!(
            checkpoint.load(903, &Cancellation::new()).await.unwrap(),
            saved
        );
    });
}

#[test]
fn unfinished_batch_restore_preserves_failed_attempt_history_and_remaining_retry_budget() {
    use halo::agent::checkpoint::Checkpoint as _;
    block_on(async {
        let original_calls = Arc::new(AtomicUsize::new(0));
        let mut original = Runtime::new(
            904,
            failing_batch_workflow(
                original_calls.clone(),
                Default::default(),
                Default::default(),
            ),
        )
        .unwrap();
        original.state_mut().initial = Some(Messages::default());
        original
            .start_batch(vec![("source".into(), 4000, vec![])])
            .unwrap();
        original
            .complete(
                "source",
                4000,
                Messages::new([Message::function(
                    serde_json::to_value(source_tasks(&original_calls).await).unwrap(),
                )]),
                vec!["batch".into()],
            )
            .unwrap();
        assert!(original.advance_step());
        original
            .start_batch(vec![(
                "batch".into(),
                4001,
                vec![ExecutionRef {
                    node_id: "source".into(),
                    exec_id: 4000,
                }],
            )])
            .unwrap();
        // Model a saved scene after the first Attempt failed and the next
        // Attempt actually started. Recovery is not a new retry allowance.
        let batch = original.state_mut().nodes.get_mut("batch").unwrap();
        let execution = batch.execution_mut(4001).unwrap();
        execution.attempts[0].status = node::AttemptStatus::Failed;
        execution.attempts[0].error = Some(
            Error::new("TIMEOUT", "original failed attempt").with_details("original diagnosis"),
        );
        execution.attempts[0].created_time = Some(10);
        execution.attempts[0].finished_time = Some(20);
        execution.attempts[0].usage = Some(halo::agent::model::token::Usage {
            input: Some(7),
            output: Some(3),
            total: Some(10),
            ..Default::default()
        });
        let history = execution.attempts[0].clone();
        batch
            .append_attempt(
                4001,
                node::Attempt {
                    idx: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
        original
            .save_checkpoint(checkpoint.as_ref(), &Cancellation::new())
            .await
            .unwrap();
        let saved = checkpoint.load(904, &Cancellation::new()).await.unwrap();
        let encoded = serde_json::to_vec(&saved).unwrap();
        drop(original);

        let source_calls = Arc::new(AtomicUsize::new(0));
        let item_calls = Arc::new(Mutex::new(Vec::new()));
        let downstream = Arc::new(AtomicUsize::new(0));
        let mut restored = Runtime::from_snapshot(
            904,
            failing_batch_workflow(source_calls.clone(), item_calls.clone(), downstream.clone()),
            serde_json::from_slice(&encoded).unwrap(),
        )
        .unwrap()
        .with_checkpoint(checkpoint.clone());
        assert_eq!(
            run_workflow(
                &mut restored,
                Messages::default(),
                Arc::new(|| panic!("restore must keep the original exec_id")),
                Cancellation::new(),
            )
            .await
            .unwrap_err()
            .code,
            "TIMEOUT"
        );
        assert_eq!(source_calls.load(Ordering::SeqCst), 0);
        assert_eq!(*item_calls.lock().unwrap(), [30, 10, 20]);
        assert_eq!(downstream.load(Ordering::SeqCst), 0);
        let execution = &restored.state().nodes["batch"].executions[&4001];
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(execution.attempts[0], history);
        assert_eq!(execution.attempts[1].idx, 1);
        assert_eq!(execution.attempts[1].status, node::AttemptStatus::Failed);
        assert_eq!(execution.status, node::ExecutionStatus::Failed);
        assert!(execution.messages.is_empty());
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(10));
        assert_eq!(restored.state().usage.as_ref().unwrap().total, Some(10));
        assert!(restored.state().nodes["summary"].executions.is_empty());
        assert_eq!(
            serde_json::to_value(&restored.state().nodes["source"]).unwrap(),
            saved["nodes"]["source"]
        );
        assert_eq!(
            checkpoint.load(904, &Cancellation::new()).await.unwrap(),
            restored.snapshot().unwrap()
        );
    });
}
