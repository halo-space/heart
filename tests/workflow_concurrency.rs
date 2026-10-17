//! The outer Workflow budget counts actual leaves, never Graph containers.
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
};

use futures_util::{StreamExt, poll};
use halo_agents::{
    Cancellation, Error, Message, Messages,
    runtime::workflow::{
        CompiledWorkflow, Runtime, WorkflowBuilder,
        graph::{END, START},
    },
};
use serde_json::json;

#[derive(Default)]
struct Probe {
    active: AtomicUsize,
    peak: AtomicUsize,
    calls: AtomicUsize,
}
struct Active(Arc<Probe>);
impl Active {
    fn new(probe: Arc<Probe>) -> Self {
        let active = probe.active.fetch_add(1, Ordering::SeqCst) + 1;
        probe.peak.fetch_max(active, Ordering::SeqCst);
        probe.calls.fetch_add(1, Ordering::SeqCst);
        Self(probe)
    }
}
impl Drop for Active {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

fn compiled(probe: Arc<Probe>, branches: usize, depth: usize) -> CompiledWorkflow {
    gated(probe, branches, depth, None)
}
fn gated(
    probe: Arc<Probe>,
    branches: usize,
    depth: usize,
    gate: Option<Arc<AtomicBool>>,
) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("concurrency-leaves");
    for index in 0..branches {
        let probe = probe.clone();
        let gate = gate.clone();
        let name = format!("leaf_{index}");
        builder
            .add_node(&name, move |_: Messages, cancellation: Cancellation| {
                let probe = probe.clone();
                let gate = gate.clone();
                async move {
                    let _active = Active::new(probe);
                    if let Some(gate) = gate {
                        while !gate.load(Ordering::SeqCst) {
                            if cancellation.is_cancelled() {
                                return Err(Error::new("CANCELLED", "leaf cancelled"));
                            }
                            tokio::task::yield_now().await;
                        }
                    }
                    // Every admitted leaf yields, so simultaneous admission is
                    // observable without a timer or assuming executor threads.
                    tokio::task::yield_now().await;
                    tokio::task::yield_now().await;
                    Ok::<_, Error>(Messages::new([Message::function(json!({"done":true}))]))
                }
            })
            .unwrap();
        builder.add_edge(START, name.as_str()).unwrap();
    }
    builder
        .add_node("merge", |_: Messages, _: Cancellation| async {
            Ok::<_, Error>(Messages::default())
        })
        .unwrap();
    builder
        .add_join((0..branches).map(|index| format!("leaf_{index}")), "merge")
        .unwrap();
    builder.add_edge("merge", END).unwrap();
    let mut graph = builder.compile().unwrap();
    for index in 0..depth {
        let mut parent = WorkflowBuilder::new(format!("concurrency-wrapper-{index}"));
        parent.add_graph_node("child", &graph).unwrap();
        parent.add_edge(START, "child").unwrap();
        parent.add_edge("child", END).unwrap();
        graph = parent.compile().unwrap();
    }
    graph
}
fn ids() -> Arc<dyn Fn() -> i64 + Send + Sync> {
    let next = AtomicI64::new(70_000_000);
    Arc::new(move || next.fetch_add(1, Ordering::SeqCst))
}

#[tokio::test]
async fn nested_graphs_use_the_outer_leaf_budget() {
    for depth in [0, 1, 3] {
        for capacity in [1, 2, 4] {
            let probe = Arc::new(Probe::default());
            let runtime = Runtime::new(1001, compiled(probe.clone(), 8, depth))
                .unwrap()
                .with_max_concurrency(capacity)
                .unwrap();
            let events = runtime
                .run(Messages::default(), ids(), Cancellation::new())
                .await
                .unwrap();
            let output = tokio::time::timeout(
                std::time::Duration::from_secs(3),
                events.collect::<Vec<_>>(),
            )
            .await
            .expect("containers must not hold the only leaf permit");
            assert!(output.iter().all(Result::is_ok), "{output:?}");
            assert_eq!(probe.calls.load(Ordering::SeqCst), 8);
            assert_eq!(probe.active.load(Ordering::SeqCst), 0);
            assert_eq!(probe.peak.load(Ordering::SeqCst), capacity, "depth={depth}");
        }
    }
}

#[tokio::test]
async fn sibling_graphs_and_direct_leaves_share_one_budget() {
    for capacity in [1, 2, 4] {
        let probe = Arc::new(Probe::default());
        let child = compiled(probe.clone(), 8, 2);
        let mut builder = WorkflowBuilder::new("shared-sibling-budget");
        for name in ["left", "right"] {
            builder.add_graph_node(name, &child).unwrap();
            builder.add_edge(START, name).unwrap();
        }
        let direct_probe = probe.clone();
        builder
            .add_node("direct", move |_: Messages, _: Cancellation| {
                let probe = direct_probe.clone();
                async move {
                    let _active = Active::new(probe);
                    tokio::task::yield_now().await;
                    Ok::<_, Error>(Messages::default())
                }
            })
            .unwrap();
        builder.add_edge(START, "direct").unwrap();
        builder
            .add_node("merge", |_: Messages, _: Cancellation| async {
                Ok::<_, Error>(Messages::default())
            })
            .unwrap();
        builder
            .add_join(["left", "right", "direct"], "merge")
            .unwrap();
        builder.add_edge("merge", END).unwrap();
        let runtime = Runtime::new(1001, builder.compile().unwrap())
            .unwrap()
            .with_max_concurrency(capacity)
            .unwrap();
        let events = runtime
            .run(Messages::default(), ids(), Cancellation::new())
            .await
            .unwrap();
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            events.collect::<Vec<_>>(),
        )
        .await
        .unwrap();
        assert!(output.iter().all(Result::is_ok), "{output:?}");
        assert_eq!(probe.calls.load(Ordering::SeqCst), 17);
        assert_eq!(probe.peak.load(Ordering::SeqCst), capacity);
        assert_eq!(probe.active.load(Ordering::SeqCst), 0);
    }
}

fn executions(
    state: &serde_json::Value,
    ids: &mut std::collections::BTreeSet<i64>,
    counts: &mut [usize; 3],
) {
    for node in state["nodes"].as_object().unwrap().values() {
        for execution in node["executions"].as_object().unwrap().values() {
            ids.insert(execution["exec_id"].as_i64().unwrap());
            let attempt = execution["attempts"].as_array().unwrap().last().unwrap();
            assert_eq!(execution["attempts"].as_array().unwrap().len(), 1);
            assert_eq!(attempt["idx"], 0);
            if let Some(child) = attempt.get("graph").filter(|child| child.is_object()) {
                executions(child, ids, counts);
            } else if execution["status"] == "pending" {
                counts[0] += 1;
            } else if execution["status"] == "running" {
                counts[1] += 1;
            } else if execution["status"] == "succeeded" {
                counts[2] += 1;
            }
        }
    }
}

#[tokio::test]
async fn dropped_nested_run_retains_pending_executions_and_reacquires_fresh_capacity() {
    for depth in [1, 3] {
        let probe = Arc::new(Probe::default());
        let gate = Arc::new(AtomicBool::new(false));
        let graph = gated(probe.clone(), 8, depth, Some(gate.clone()));
        let runtime = Runtime::new(1001, graph.clone())
            .unwrap()
            .with_max_concurrency(2)
            .unwrap();
        let mut events = Box::pin(
            runtime
                .run(Messages::default(), ids(), Cancellation::new())
                .await
                .unwrap(),
        );
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while probe.active.load(Ordering::SeqCst) < 2 {
                assert!(poll!(events.next()).is_pending());
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(events);
        assert_eq!(probe.active.load(Ordering::SeqCst), 0);
        let saved = runtime.snapshot().unwrap();
        let mut original_ids = Default::default();
        let mut counts = [0; 3];
        executions(&saved, &mut original_ids, &mut counts);
        assert_eq!(
            counts[0], 6,
            "queued leaves remain pending, not fictitious running calls"
        );
        assert_eq!(original_ids.len(), 8 + depth);
        drop(runtime);
        gate.store(true, Ordering::SeqCst);
        probe.peak.store(0, Ordering::SeqCst);
        let restored = Runtime::from_snapshot(1001, graph, saved)
            .unwrap()
            .with_max_concurrency(1)
            .unwrap();
        let next = AtomicI64::new(80_000_000);
        let events = restored
            .run(
                Messages::default(),
                Arc::new(move || next.fetch_add(1, Ordering::SeqCst)),
                Cancellation::new(),
            )
            .await
            .unwrap();
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            events.collect::<Vec<_>>(),
        )
        .await
        .unwrap();
        assert!(output.iter().all(Result::is_ok), "{output:?}");
        let finished = restored.snapshot().unwrap();
        let mut final_ids = Default::default();
        let mut counts = [0; 3];
        executions(&finished, &mut final_ids, &mut counts);
        assert!(original_ids.is_subset(&final_ids));
        assert_eq!(counts, [0, 0, 9]);
        assert_eq!(probe.peak.load(Ordering::SeqCst), 1);
        assert_eq!(
            probe.calls.load(Ordering::SeqCst),
            10,
            "only the two uncommitted calls reenter; pending calls start once"
        );
        assert_eq!(probe.active.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn cancellation_and_drop_release_nested_admission_without_running_the_queue() {
    for cancel in [false, true] {
        let probe = Arc::new(Probe::default());
        let gate = Arc::new(AtomicBool::new(false));
        let runtime = Runtime::new(1001, gated(probe.clone(), 8, 2, Some(gate)))
            .unwrap()
            .with_max_concurrency(1)
            .unwrap();
        let cancellation = Cancellation::new();
        let mut events = Box::pin(
            runtime
                .run(Messages::default(), ids(), cancellation.clone())
                .await
                .unwrap(),
        );
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while probe.active.load(Ordering::SeqCst) == 0 {
                assert!(poll!(events.next()).is_pending());
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        if cancel {
            cancellation.cancel();
            let output = tokio::time::timeout(
                std::time::Duration::from_secs(3),
                events.by_ref().collect::<Vec<_>>(),
            )
            .await
            .unwrap();
            assert_eq!(
                output
                    .iter()
                    .filter_map(|event| event.as_ref().err())
                    .map(|error| error.code.as_str())
                    .collect::<Vec<_>>(),
                ["CANCELLED"]
            );
        }
        drop(events);
        assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
        assert_eq!(probe.active.load(Ordering::SeqCst), 0);
        if cancel {
            assert_eq!(runtime.snapshot().unwrap()["status"], "cancelled");
        }
        let fresh = Runtime::new(2002, compiled(probe.clone(), 8, 1))
            .unwrap()
            .with_max_concurrency(2)
            .unwrap();
        let events = fresh
            .run(Messages::default(), ids(), Cancellation::new())
            .await
            .unwrap();
        assert!(events.collect::<Vec<_>>().await.iter().all(Result::is_ok));
        assert_eq!(probe.active.load(Ordering::SeqCst), 0);
        assert_eq!(probe.calls.load(Ordering::SeqCst), 9);
    }
}

#[tokio::test]
async fn independent_root_runs_keep_separate_limits_on_the_same_task() {
    let probe = Arc::new(Probe::default());
    let first = Runtime::new(1001, compiled(probe.clone(), 8, 1))
        .unwrap()
        .with_max_concurrency(1)
        .unwrap();
    let second = Runtime::new(2002, compiled(probe.clone(), 8, 1))
        .unwrap()
        .with_max_concurrency(2)
        .unwrap();
    let a = first
        .run(Messages::default(), ids(), Cancellation::new())
        .await
        .unwrap();
    let b = second
        .run(Messages::default(), ids(), Cancellation::new())
        .await
        .unwrap();
    let (a, b) = tokio::time::timeout(std::time::Duration::from_secs(3), async {
        tokio::join!(a.collect::<Vec<_>>(), b.collect::<Vec<_>>())
    })
    .await
    .unwrap();
    assert!(a.iter().chain(&b).all(Result::is_ok));
    assert_eq!(probe.peak.load(Ordering::SeqCst), 3);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 16);
    assert_eq!(probe.active.load(Ordering::SeqCst), 0);
}

#[test]
fn invalid_concurrency_limits_return_errors_instead_of_panicking() {
    let graph = compiled(Default::default(), 2, 0);
    for limit in [0, usize::MAX] {
        let result = Runtime::new(1001, graph.clone())
            .unwrap()
            .with_max_concurrency(limit);
        assert_eq!(
            result.err().expect("invalid limit").code,
            "INVALID_ARGUMENTS"
        );
    }
}
