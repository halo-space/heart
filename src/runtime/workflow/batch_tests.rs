//! Generic batch semantics without special Node kinds or candidate executions.
//! The Node owns its private successful-item cache; Runtime owns the one
//! execution, retry Attempts and all-or-nothing downstream submission.
use super::*;
use crate::Cancellation;
use futures_executor::block_on;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{
    Mutex,
    atomic::{AtomicI64, AtomicUsize, Ordering},
};

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
                        Ok::<_, Error>(Messages::new([crate::Message::function(
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
        let result = block_on(runtime.run(
            Messages::default(),
            Arc::new(move || ids_for_run.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        ));
        let executions = &runtime.state.nodes["batch"].executions;
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
            assert!(runtime.state.nodes["summary"].executions.is_empty());
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
