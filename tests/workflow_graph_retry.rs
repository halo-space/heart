#[path = "support/workflow.rs"]
mod workflow_support;
use workflow_support::run_workflow;

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
};

use futures_executor::block_on;
use futures_util::stream;
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::{Agent, Input},
    checkpoint::Checkpoint as _,
    event::Event,
    message::Role,
    model::token::Usage,
    node::{Node, Signal, Wait},
    runtime::workflow::{
        CompiledWorkflow, RetryPolicy, Runtime, WorkflowBuilder,
        graph::{self, END, START},
        node,
    },
};

struct CountedAgent {
    calls: Arc<AtomicUsize>,
    fail_once: bool,
    tokens: u64,
}

struct Approval(AtomicBool);

impl Node<Messages, Messages, Error> for Approval {
    async fn call(&self, input: Messages, _: Cancellation) -> Result<Signal<Messages>, Error> {
        if self.0.swap(true, Ordering::SeqCst) {
            Ok(Signal::Complete(input))
        } else {
            Ok(Signal::Wait(Wait {
                wait_id: "approval".into(),
            }))
        }
    }
}

fn approval_workflow(approved: bool) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("restored-approval-downstream");
    builder.add_node("source", halo::node::Echo).unwrap();
    builder
        .add_node("approval", Approval(AtomicBool::new(approved)))
        .unwrap();
    builder.add_node("next", halo::node::Echo).unwrap();
    builder.add_edge(START, "source").unwrap();
    builder.add_edge("source", "approval").unwrap();
    builder.add_edge("approval", "next").unwrap();
    builder.add_edge("next", END).unwrap();
    builder.compile().unwrap()
}

#[test]
fn restored_external_wait_advances_the_barrier_and_runs_its_downstream() {
    block_on(async {
        let checkpoint = Arc::new(halo::core::checkpoint::InMemory::new());
        let mut original = Runtime::new(908, approval_workflow(false))
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        let ids = Arc::new(AtomicI64::new(8500));
        assert_eq!(
            run_workflow(
                &mut original,
                Messages::default(),
                Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
                Cancellation::new(),
            )
            .await
            .unwrap_err()
            .code,
            "WAITING"
        );
        let saved = checkpoint.load(908, &Cancellation::new()).await.unwrap();
        assert_eq!(saved["step"], 1);
        drop(original);
        let mut restored = Runtime::from_snapshot(908, approval_workflow(true), saved.clone())
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        let response = Messages::new([Message::function(serde_json::json!({"approved":true}))]);
        restored.resume("approval", response.clone()).unwrap();
        assert_eq!(
            run_workflow(
                &mut restored,
                Messages::default(),
                Arc::new(|| 8502),
                Cancellation::new(),
            )
            .await
            .unwrap(),
            response
        );
        assert_eq!(restored.state().status, graph::Status::Succeeded);
        assert_eq!(restored.state().step, 3);
        assert!(restored.state().waits.is_empty());
        assert_eq!(
            serde_json::to_value(&restored.state().nodes["source"]).unwrap(),
            saved["nodes"]["source"]
        );
        let execution = &restored.state().nodes["approval"].executions[&8501];
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(restored.state().nodes["next"].executions.len(), 1);
        assert_eq!(
            checkpoint.load(908, &Cancellation::new()).await.unwrap(),
            restored.snapshot().unwrap()
        );
    });
}

impl Agent for CountedAgent {
    async fn reply(
        &self,
        _: halo::agent::Input,
        _: halo::agent::Cancellation,
    ) -> Result<(), halo::agent::Error> {
        Err(halo::agent::Error::new(
            "NOT_FOUND",
            "fixture has no live feedback consumer",
        ))
    }
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn run(&self, _: Input, _stream: bool, _: Cancellation) -> Result<Self::Stream, Error> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 && self.fail_once {
            return Err(Error::new("TIMEOUT", "temporary child failure"));
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: Message::new(Role::Assistant),
            usage: Some(Usage {
                input: Some(self.tokens),
                output: Some(0),
                total: Some(self.tokens),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}

fn workflow(
    first: Arc<AtomicUsize>,
    second: Arc<AtomicUsize>,
    downstream: Arc<AtomicUsize>,
) -> CompiledWorkflow {
    workflow_at(first, second, Default::default(), downstream, false, false)
}

fn workflow_at(
    first: Arc<AtomicUsize>,
    second: Arc<AtomicUsize>,
    merge_calls: Arc<AtomicUsize>,
    downstream: Arc<AtomicUsize>,
    fail_at_merge: bool,
    nested: bool,
) -> CompiledWorkflow {
    let mut child = WorkflowBuilder::new("retry-child");
    child
        .add_agent_node(
            "a",
            CountedAgent {
                calls: first,
                fail_once: false,
                tokens: 10,
            },
        )
        .unwrap();
    child
        .add_agent_node(
            "b",
            CountedAgent {
                calls: second,
                fail_once: !fail_at_merge,
                tokens: 20,
            },
        )
        .unwrap();
    if fail_at_merge {
        child
            .add_agent_node(
                "merge",
                CountedAgent {
                    calls: merge_calls,
                    fail_once: true,
                    tokens: 30,
                },
            )
            .unwrap();
    } else {
        child.add_node("merge", halo::node::Echo).unwrap();
    }
    child.add_edge(START, "a").unwrap();
    child.add_edge(START, "b").unwrap();
    child.add_join(["a", "b"], "merge").unwrap();
    child.add_edge("merge", END).unwrap();
    let child = child.compile().unwrap();
    let child = if nested {
        let mut wrapper = WorkflowBuilder::new("retry-nested-child");
        wrapper.add_graph_node("inner", &child).unwrap();
        wrapper.add_edge(START, "inner").unwrap();
        wrapper.add_edge("inner", END).unwrap();
        wrapper.compile().unwrap()
    } else {
        child
    };

    let mut parent = WorkflowBuilder::new("retry-parent");
    parent.add_node("source", halo::node::Echo).unwrap();
    parent.add_graph_node("batch", &child).unwrap();
    parent
        .add_node("next", move |input: Messages, _: Cancellation| {
            let downstream = downstream.clone();
            async move {
                downstream.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Error>(input)
            }
        })
        .unwrap();
    parent.add_edge(START, "source").unwrap();
    parent.add_fan_out_edges("source", "batch").unwrap();
    parent.add_edge("batch", "next").unwrap();
    parent.add_edge("next", END).unwrap();
    parent
        .set_retry(
            "batch",
            RetryPolicy {
                max_retries: 1,
                min_delay_ms: 0,
                max_delay_ms: 0,
            },
        )
        .unwrap();
    parent.compile().unwrap()
}

#[test]
fn graph_retry_reuses_successes_at_multiple_depths_and_at_later_steps() {
    block_on(async {
        for nested in [false, true] {
            for fail_at_merge in [false, true] {
                let first = Arc::new(AtomicUsize::new(0));
                let second = Arc::new(AtomicUsize::new(0));
                let merge_calls = Arc::new(AtomicUsize::new(0));
                let downstream = Arc::new(AtomicUsize::new(0));
                let mut runtime = Runtime::new(
                    906,
                    workflow_at(
                        first.clone(),
                        second.clone(),
                        merge_calls.clone(),
                        downstream.clone(),
                        fail_at_merge,
                        nested,
                    ),
                )
                .unwrap();
                let ids = Arc::new(AtomicI64::new(6000));
                run_workflow(
                    &mut runtime,
                    Messages::default(),
                    Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
                    Cancellation::new(),
                )
                .await
                .unwrap();
                assert_eq!(first.load(Ordering::SeqCst), 1);
                assert_eq!(
                    second.load(Ordering::SeqCst),
                    if fail_at_merge { 1 } else { 2 }
                );
                assert_eq!(
                    merge_calls.load(Ordering::SeqCst),
                    if fail_at_merge { 2 } else { 0 }
                );
                assert_eq!(downstream.load(Ordering::SeqCst), 1);
                let total = if fail_at_merge { 60 } else { 30 };
                assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(total));
                let saved = runtime.snapshot().unwrap();
                let mut restored = Runtime::from_snapshot(
                    906,
                    workflow_at(
                        first.clone(),
                        second.clone(),
                        merge_calls.clone(),
                        downstream.clone(),
                        fail_at_merge,
                        nested,
                    ),
                    saved.clone(),
                )
                .unwrap();
                run_workflow(
                    &mut restored,
                    Messages::default(),
                    Arc::new(|| panic!("terminal nested scene must not allocate identities")),
                    Cancellation::new(),
                )
                .await
                .unwrap();
                assert_eq!(restored.snapshot().unwrap(), saved);
                assert_eq!(restored.state().usage.as_ref().unwrap().total, Some(total));
            }
        }
    });
}

fn unfinished_retry_scene(scene: &mut serde_json::Value, nested: bool) {
    scene["status"] = "running".into();
    scene["step"] = 0.into();
    if nested {
        scene["nodes"]["inner"]["status"] = "active".into();
        let execution = scene["nodes"]["inner"]["executions"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap();
        execution["status"] = "running".into();
        execution["attempts"][0]["status"] = "running".into();
        unfinished_retry_scene(&mut execution["attempts"][0]["graph"], false);
    } else {
        for node in ["b", "merge"] {
            scene["nodes"][node]["status"] = "idle".into();
            scene["nodes"][node]["executions"] = serde_json::json!({});
        }
    }
}

#[test]
fn checkpoint_resumes_an_outer_graph_retry_with_its_original_successes_and_attempts() {
    block_on(async {
        for nested in [false, true] {
            let first = Arc::new(AtomicUsize::new(0));
            let second = Arc::new(AtomicUsize::new(0));
            let mut original = Runtime::new(
                907,
                workflow_at(
                    first,
                    second,
                    Default::default(),
                    Default::default(),
                    false,
                    nested,
                ),
            )
            .unwrap();
            let ids = Arc::new(AtomicI64::new(7000));
            run_workflow(
                &mut original,
                Messages::default(),
                Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
                Cancellation::new(),
            )
            .await
            .unwrap();
            // Explicitly model an earlier save point in the real retry scene:
            // Attempt 0 is archived, Attempt 1 has begun with child a's success,
            // but child b and merge have not run yet. This is not a claim that
            // a hard kill automatically saves an in-flight Runtime.
            let mut saved = original.snapshot().unwrap();
            saved["status"] = "running".into();
            saved["step"] = 1.into();
            saved["nodes"]["next"]["status"] = "idle".into();
            saved["nodes"]["next"]["executions"] = serde_json::json!({});
            saved["nodes"]["batch"]["status"] = "active".into();
            let execution = &mut saved["nodes"]["batch"]["executions"]["7001"];
            execution["status"] = "running".into();
            execution["messages"] = serde_json::json!([]);
            execution["to"] = serde_json::json!([]);
            execution["attempts"][1]["status"] = "running".into();
            unfinished_retry_scene(&mut execution["attempts"][1]["graph"], nested);
            let archived = execution["attempts"][0].clone();
            let bytes = serde_json::to_vec(&saved).unwrap();
            drop(original);

            let first = Arc::new(AtomicUsize::new(0));
            // The rebuilt service is healthy; this counter starts at one to
            // disable this test service's initial transient failure, not to
            // provide a cache or influence Runtime recovery decisions.
            let second = Arc::new(AtomicUsize::new(1));
            let downstream = Arc::new(AtomicUsize::new(0));
            let checkpoint = Arc::new(halo::core::checkpoint::InMemory::new());
            let mut restored = Runtime::from_snapshot(
                907,
                workflow_at(
                    first.clone(),
                    second.clone(),
                    Default::default(),
                    downstream.clone(),
                    false,
                    nested,
                ),
                serde_json::from_slice(&bytes).unwrap(),
            )
            .unwrap()
            .with_checkpoint(checkpoint.clone());
            let ids = Arc::new(AtomicI64::new(9000));
            run_workflow(
                &mut restored,
                Messages::default(),
                Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
                Cancellation::new(),
            )
            .await
            .unwrap();
            assert_eq!(first.load(Ordering::SeqCst), 0);
            assert_eq!(second.load(Ordering::SeqCst), 2);
            assert_eq!(downstream.load(Ordering::SeqCst), 1);
            let execution = &restored.state().nodes["batch"].executions[&7001];
            assert_eq!(execution.attempts.len(), 2);
            assert_eq!(execution.attempts[1].idx, 1);
            assert_eq!(
                serde_json::to_value(&execution.attempts[0]).unwrap(),
                archived
            );
            assert_eq!(restored.state().usage.as_ref().unwrap().total, Some(30));
            assert_eq!(
                checkpoint.load(907, &Cancellation::new()).await.unwrap(),
                restored.snapshot().unwrap(),
            );
        }
    });
}

#[test]
fn graph_node_retry_inherits_successes_without_recounting_their_usage() {
    block_on(async {
        let first = Arc::new(AtomicUsize::new(0));
        let second = Arc::new(AtomicUsize::new(0));
        let downstream = Arc::new(AtomicUsize::new(0));
        let checkpoint = Arc::new(halo::core::checkpoint::InMemory::new());
        let compiled = workflow(first.clone(), second.clone(), downstream.clone());
        let mut runtime = Runtime::new(905, compiled)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        let ids = Arc::new(AtomicI64::new(5000));
        run_workflow(
            &mut runtime,
            Messages::default(),
            Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(
            first.load(Ordering::SeqCst),
            1,
            "successful child must not replay"
        );
        assert_eq!(second.load(Ordering::SeqCst), 2);
        assert_eq!(downstream.load(Ordering::SeqCst), 1);
        let execution = &runtime.state().nodes["batch"].executions[&5001];
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(execution.attempts[0].status, node::AttemptStatus::Failed);
        let before = execution.attempts[0].graph.as_ref().unwrap();
        let after = execution.attempts[1].graph.as_ref().unwrap();
        assert_eq!(before.status, graph::Status::Failed);
        assert_eq!(after.status, graph::Status::Succeeded);
        assert_eq!(before.nodes["a"], after.nodes["a"]);
        assert_eq!(before.nodes["b"].executions.len(), 1);
        assert_eq!(after.nodes["b"].executions.len(), 1);
        assert_ne!(
            before.nodes["b"].executions.keys().next(),
            after.nodes["b"].executions.keys().next()
        );
        assert_eq!(
            execution.attempts[0].usage.as_ref().unwrap().total,
            Some(10)
        );
        assert_eq!(
            execution.attempts[1].usage.as_ref().unwrap().total,
            Some(20)
        );
        assert_eq!(after.usage.as_ref().unwrap().total, Some(30));
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(30));
        assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(30));

        let saved = checkpoint.load(905, &Cancellation::new()).await.unwrap();
        assert_eq!(saved, runtime.snapshot().unwrap());
        let mut restored = Runtime::from_snapshot(
            905,
            workflow(first.clone(), second.clone(), downstream.clone()),
            saved.clone(),
        )
        .unwrap();
        for _ in 0..2 {
            run_workflow(
                &mut restored,
                Messages::default(),
                Arc::new(|| panic!("terminal graph restore must not allocate an id")),
                Cancellation::new(),
            )
            .await
            .unwrap();
            assert_eq!(restored.snapshot().unwrap(), saved);
        }
        assert_eq!(first.load(Ordering::SeqCst), 1);
        assert_eq!(second.load(Ordering::SeqCst), 2);
        assert_eq!(downstream.load(Ordering::SeqCst), 1);
    });
}
