//! Attribute expansion preserves the existing Node and Workflow contracts.
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use futures_executor::block_on;
use futures_util::StreamExt;
use halo_agents::{
    Cancellation, Error, Message, Messages, node,
    node::{Node, Signal, Wait},
    runtime::workflow::{
        Runtime, WorkflowBuilder,
        graph::{END, START},
    },
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
struct Input {
    value: i64,
}

#[derive(Debug, Eq, PartialEq, Serialize, JsonSchema)]
struct Output {
    value: i64,
}

#[node]
async fn increment(input: Input, cancellation: Cancellation) -> Result<Output, Error> {
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "cancelled"));
    }
    Ok(Output {
        value: input.value + 1,
    })
}

async fn plain_increment(input: Input, cancellation: Cancellation) -> Result<Output, Error> {
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "cancelled"));
    }
    Ok(Output {
        value: input.value + 1,
    })
}

#[derive(Debug, Eq, PartialEq)]
struct ComponentError {
    message: &'static str,
}
impl From<ComponentError> for Error {
    fn from(error: ComponentError) -> Self {
        Error::new("COMPONENT_FAILURE", error.message)
    }
}

struct Compute {
    calls: Arc<AtomicUsize>,
}

#[node(call = compute)]
impl Compute {
    async fn compute(
        &self,
        input: Input,
        cancellation: &Cancellation,
    ) -> Result<Output, ComponentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if cancellation.is_cancelled() {
            return Err(ComponentError {
                message: "cancelled",
            });
        }
        if input.value < 0 {
            return Err(ComponentError {
                message: "original component error",
            });
        }
        Ok(Output {
            value: input.value * 2,
        })
    }
}

struct Approval;

#[node(signal = execute)]
impl Approval {
    async fn execute(
        &self,
        input: Input,
        cancellation: &Cancellation,
    ) -> Result<Signal<Output>, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "cancelled"));
        }
        if input.value == 0 {
            return Ok(Signal::Wait(Wait {
                wait_id: "approval".into(),
            }));
        }
        if input.value < 0 {
            return Err(Error::new("DENIED", "original signal error"));
        }
        Ok(Signal::Complete(Output { value: input.value }))
    }
}

#[node]
async fn scalar(input: Input, _: Cancellation) -> Result<i64, Error> {
    Ok(input.value)
}

#[test]
fn original_functions_and_methods_remain_directly_callable() {
    block_on(async {
        assert_eq!(
            increment(Input { value: 2 }, Cancellation::new())
                .await
                .unwrap(),
            Output { value: 3 }
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let component = Compute {
            calls: calls.clone(),
        };
        assert_eq!(
            component
                .compute(Input { value: 3 }, &Cancellation::new())
                .await
                .unwrap(),
            Output { value: 6 }
        );
        assert_eq!(
            Node::call(&component, Input { value: 4 }, Cancellation::new())
                .await
                .unwrap(),
            Signal::Complete(Output { value: 8 })
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn original_component_errors_and_cancellation_are_not_wrapped_or_retried() {
    block_on(async {
        let calls = Arc::new(AtomicUsize::new(0));
        let component = Compute {
            calls: calls.clone(),
        };
        let error = Node::call(&component, Input { value: -1 }, Cancellation::new())
            .await
            .unwrap_err();
        assert_eq!(
            error,
            ComponentError {
                message: "original component error"
            }
        );
        let cancellation = Cancellation::new();
        cancellation.cancel();
        assert_eq!(
            Node::call(&component, Input { value: 1 }, cancellation.clone())
                .await
                .unwrap_err(),
            ComponentError {
                message: "cancelled"
            }
        );
        assert_eq!(
            increment(Input { value: 1 }, cancellation)
                .await
                .unwrap_err()
                .code,
            "CANCELLED"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2, "the macro never retries");
    });
}

#[test]
fn signal_glue_preserves_wait_complete_and_error() {
    block_on(async {
        assert_eq!(
            Node::call(&Approval, Input { value: 0 }, Cancellation::new())
                .await
                .unwrap(),
            Signal::Wait(Wait {
                wait_id: "approval".into()
            })
        );
        assert_eq!(
            Node::call(&Approval, Input { value: 1 }, Cancellation::new())
                .await
                .unwrap(),
            Signal::Complete(Output { value: 1 })
        );
        assert_eq!(
            Node::call(&Approval, Input { value: -1 }, Cancellation::new())
                .await
                .unwrap_err(),
            Error::new("DENIED", "original signal error")
        );
    });
}

#[test]
fn marked_and_unmarked_functions_use_the_same_graph_io_contract() {
    let mut marked = WorkflowBuilder::new("macro-parity");
    let mut plain = WorkflowBuilder::new("macro-parity");
    marked.add_node("compute", increment).unwrap();
    plain.add_node("compute", plain_increment).unwrap();
    for builder in [&mut marked, &mut plain] {
        builder.add_edge(START, "compute").unwrap();
        builder.add_edge("compute", END).unwrap();
    }
    let marked = marked.compile().unwrap();
    let plain = plain.compile().unwrap();
    assert_eq!(marked.definition().version, plain.definition().version);
    for (chat_id, graph) in [(8201, marked), (8202, plain)] {
        block_on(async {
            let mut runtime = Runtime::new(chat_id, graph).unwrap();
            {
                let events = runtime
                    .run(
                        Messages::new([Message::function(json!({"value":41}))]),
                        Arc::new(|| 501),
                        Cancellation::new(),
                    )
                    .await
                    .unwrap();
                let events = events.collect::<Vec<_>>().await;
                assert_eq!(events.len(), 1);
                assert!(events[0].is_ok());
            }
            assert_eq!(
                runtime.state().nodes["compute"].executions[&501].messages.0[0].values["value"],
                42
            );
        });
    }
}

#[test]
fn registration_is_pure_and_schema_roots_are_not_rewrapped() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = WorkflowBuilder::new("pure-registration");
    builder
        .add_node(
            "component",
            Compute {
                calls: calls.clone(),
            },
        )
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        builder.add_node("scalar", scalar).err().unwrap().code,
        "INVALID_SCHEMA"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn a_signal_component_waits_in_the_existing_workflow_scene() {
    block_on(async {
        let mut builder = WorkflowBuilder::new("macro-signal-wait");
        builder.add_node("review", Approval).unwrap();
        builder.add_edge(START, "review").unwrap();
        builder.add_edge("review", END).unwrap();
        let mut runtime = Runtime::new(8203, builder.compile().unwrap()).unwrap();
        {
            let events = runtime
                .run(
                    Messages::new([Message::function(json!({"value":0}))]),
                    Arc::new(|| 502),
                    Cancellation::new(),
                )
                .await
                .unwrap();
            let events = events.collect::<Vec<_>>().await;
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].as_ref().unwrap_err().code, "WAITING");
        }
        assert!(runtime.state().waits.is_empty());
        assert_eq!(
            runtime.state().nodes["review"].executions[&502].waits["approval"].status,
            "waiting"
        );
    });
}
