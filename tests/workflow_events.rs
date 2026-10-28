//! Public Workflow stream tests: no private executor or observation API.
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicI64, AtomicUsize, Ordering},
    },
    task::{Context, Poll},
};

use futures_core::Stream;
use futures_executor::block_on;
use futures_util::{StreamExt, poll, stream};
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::checkpoint::Checkpoint as _,
    agent::event::Event,
    agent::message::Role,
    agent::model::{content::part::Part, token::Usage},
    agent::node::{Node, Signal, Wait},
    agent::{Agent, Input},
    runtime::workflow::{
        CompiledWorkflow, FieldSource, InputBinding, NodeAttrs, Runtime, WorkflowBuilder,
        graph::{self, END, START},
        node,
    },
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::oneshot;

fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":value}).as_object().unwrap().clone(),
    });
    message
}

fn complete(value: &str, answer: i64, tokens: u64) -> Event {
    let mut message = text(Role::Assistant, value);
    message.values.insert("answer".into(), json!(answer));
    Event::Complete {
        message,
        usage: Some(Usage {
            input: Some(tokens),
            output: Some(0),
            total: Some(tokens),
            ..Default::default()
        }),
        finish_reason: Some("stop".into()),
    }
}

fn ids(start: i64) -> Arc<dyn Fn() -> i64 + Send + Sync> {
    let next = AtomicI64::new(start);
    Arc::new(move || next.fetch_add(1, Ordering::SeqCst))
}

struct ReadyAgent {
    events: Vec<Result<Event, Error>>,
    calls: Arc<AtomicUsize>,
}
impl Agent for ReadyAgent {
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(stream::iter(self.events.clone()))
    }
}

#[derive(Deserialize, JsonSchema)]
struct Answer {
    answer: i64,
}
#[derive(Serialize, JsonSchema)]
struct Doubled {
    doubled: i64,
}

fn graph_with_agent<A: Agent + 'static>(
    agent: A,
    depth: usize,
    downstream: Arc<AtomicUsize>,
) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("observed-agent-and-function");
    builder.add_agent_node("agent", agent).unwrap();
    builder
        .add_node("double", move |input: Answer, _: Cancellation| {
            let downstream = downstream.clone();
            async move {
                downstream.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Error>(Doubled {
                    doubled: input.answer * 2,
                })
            }
        })
        .unwrap();
    builder
        .set_attrs(
            "double",
            NodeAttrs {
                input: std::collections::BTreeMap::from([(
                    "answer".into(),
                    InputBinding {
                        from: FieldSource {
                            node_id: "agent".into(),
                            field: "answer".into(),
                        },
                    },
                )]),
            },
        )
        .unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", "double").unwrap();
    builder.add_edge("double", END).unwrap();
    let mut compiled = builder.compile().unwrap();
    for level in 0..depth {
        let mut wrapper = WorkflowBuilder::new(format!("observation-wrapper-{level}"));
        wrapper.add_graph_node("child", &compiled).unwrap();
        wrapper.add_edge(START, "child").unwrap();
        wrapper.add_edge("child", END).unwrap();
        compiled = wrapper.compile().unwrap();
    }
    compiled
}

fn leaf(state: &graph::State) -> &graph::State {
    if let Some(child) = state.nodes.get("child") {
        leaf(
            child
                .executions
                .values()
                .next()
                .unwrap()
                .attempts
                .last()
                .unwrap()
                .graph
                .as_ref()
                .unwrap(),
        )
    } else {
        state
    }
}

fn message(event: &Event) -> &Message {
    match event {
        Event::Delta { message } | Event::Complete { message, .. } => message,
    }
}

#[test]
fn nested_graphs_forward_each_real_event_once_without_leaking_values() {
    block_on(async {
        let calls = Arc::new(AtomicUsize::new(0));
        let downstream = Arc::new(AtomicUsize::new(0));
        let mut tool = text(Role::Tool, "weather result");
        tool.values.insert("private".into(), json!("tool data"));
        tool.metadata.insert("call_id".into(), json!("weather_1"));
        let sequence = vec![
            Ok(Event::Delta {
                message: text(Role::Assistant, "delta-only-marker"),
            }),
            Ok(complete("model call one", 0, 10)),
            Ok(Event::Complete {
                message: tool,
                usage: None,
                finish_reason: None,
            }),
            Ok(complete("final model answer", 21, 20)),
        ];
        let compiled = graph_with_agent(
            ReadyAgent {
                events: sequence.clone(),
                calls: calls.clone(),
            },
            2,
            downstream.clone(),
        );
        let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
        let mut runtime = Runtime::new(1001, compiled)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        let observed = {
            let mut events = runtime
                .run(Messages::default(), ids(10000), Cancellation::new())
                .await
                .unwrap();
            let mut observed = Vec::new();
            while let Some(event) = events.next().await {
                observed.push(event.unwrap());
            }
            observed
        };
        assert_eq!(observed.len(), 5, "no container aggregate Complete");
        assert!(matches!(observed[0], Event::Delta { .. }));
        assert!(
            observed
                .iter()
                .all(|event| message(event).values.is_empty())
        );
        for (actual, original) in observed[..4].iter().zip(sequence) {
            assert_eq!(message(actual).content, message(&original.unwrap()).content);
        }
        let model = &observed[1];
        let Event::Complete {
            message: model_message,
            usage,
            finish_reason,
        } = model
        else {
            panic!("model Complete")
        };
        assert_eq!(model_message.usage, *usage);
        assert_eq!(finish_reason.as_deref(), Some("stop"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(downstream.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(30));
        let state = leaf(runtime.state());
        let agent = state.nodes["agent"].executions.values().next().unwrap();
        assert_eq!(agent.messages.0[0].values["answer"], 21);
        assert_eq!(agent.attempts[0].messages.0.len(), 3);
        let output = state.nodes["double"].executions.values().next().unwrap();
        assert_eq!(output.messages.0[0].values["doubled"], 42);
        for event in &observed[..4] {
            assert_eq!(message(event).metadata["chat_id"], 1001);
            assert_eq!(message(event).metadata["node_id"], "agent");
            assert_eq!(message(event).metadata["exec_id"], agent.exec_id);
        }
        assert_eq!(message(&observed[4]).metadata["node_id"], "double");
        assert_eq!(message(&observed[4]).metadata["exec_id"], output.exec_id);
        for original in agent
            .messages
            .0
            .iter()
            .chain(agent.attempts[0].messages.0.iter())
            .chain(output.messages.0.iter())
        {
            for key in ["chat_id", "node_id", "exec_id"] {
                assert!(!original.metadata.contains_key(key));
            }
        }
        assert_eq!(
            agent.attempts[0].messages.0[1].metadata["call_id"],
            "weather_1"
        );
        assert!(
            !runtime
                .snapshot()
                .unwrap()
                .to_string()
                .contains("delta-only-marker")
        );
        assert_eq!(
            checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
            runtime.snapshot().unwrap()
        );
    });
}

struct Gated {
    first: Option<Event>,
    gate: oneshot::Receiver<()>,
    done: bool,
}

#[test]
fn invocation_completes_do_not_assert_agent_or_workflow_completion() {
    block_on(async {
        let calls = Arc::new(AtomicUsize::new(0));
        let downstream = Arc::new(AtomicUsize::new(0));
        let workflow = graph_with_agent(
            ReadyAgent {
                events: vec![
                    Ok(complete("first model call", 0, 2)),
                    Ok(Event::Complete {
                        message: text(Role::Tool, "actual tool result"),
                        usage: None,
                        finish_reason: None,
                    }),
                    Ok(complete("final model call", 21, 3)),
                ],
                calls: calls.clone(),
            },
            0,
            downstream.clone(),
        );
        let mut runtime = Runtime::new(1011, workflow).unwrap();
        {
            let mut events = runtime
                .run(Messages::default(), ids(21000), Cancellation::new())
                .await
                .unwrap();
            for content in ["first model call", "actual tool result", "final model call"] {
                let event = events.next().await.unwrap().unwrap();
                assert!(matches!(event, Event::Complete { .. }));
                assert_eq!(message(&event).content[0].data["value"], content);
                assert_eq!(message(&event).metadata["node_id"], "agent");
                assert!(!message(&event).metadata.contains_key("status"));
                assert_eq!(
                    downstream.load(Ordering::SeqCst),
                    0,
                    "an invocation Complete is not the containing Node's commit"
                );
            }
            let committed_function = events.next().await.unwrap().unwrap();
            assert!(matches!(committed_function, Event::Complete { .. }));
            assert_eq!(message(&committed_function).metadata["node_id"], "double");
            assert_eq!(downstream.load(Ordering::SeqCst), 1);
            assert!(
                events.next().await.is_none(),
                "no independent status notifications"
            );
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.state().status, graph::Status::Succeeded);
        let execution = runtime.state().nodes["agent"]
            .executions
            .values()
            .next()
            .unwrap();
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(
            execution.attempts.last().unwrap().status,
            node::AttemptStatus::Succeeded
        );
        assert_eq!(execution.messages.0[0].values["answer"], 21);
    });
}
impl Stream for Gated {
    type Item = Result<Event, Error>;
    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(first) = self.first.take() {
            return Poll::Ready(Some(Ok(first)));
        }
        if self.done {
            return Poll::Ready(None);
        }
        match Pin::new(&mut self.gate).poll(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(_) => {
                self.done = true;
                Poll::Ready(Some(Ok(complete(
                    "provider-complete-not-delta-concatenation",
                    21,
                    7,
                ))))
            }
        }
    }
}

struct GatedAgent(Mutex<Option<oneshot::Receiver<()>>>);
impl Agent for GatedAgent {
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
    type Stream = Gated;
    async fn run(&self, _: Input, _stream: bool, _: Cancellation) -> Result<Self::Stream, Error> {
        Ok(Gated {
            first: Some(Event::Delta {
                message: text(Role::Assistant, "live delta"),
            }),
            gate: self.0.lock().unwrap().take().unwrap(),
            done: false,
        })
    }
}

#[test]
fn delta_arrives_before_agent_and_downstream_completion() {
    block_on(async {
        let (release, gate) = oneshot::channel();
        let downstream = Arc::new(AtomicUsize::new(0));
        let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
        let compiled = graph_with_agent(GatedAgent(Mutex::new(Some(gate))), 1, downstream.clone());
        let mut runtime = Runtime::new(1002, compiled)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        {
            let mut events = runtime
                .run(Messages::default(), ids(11000), Cancellation::new())
                .await
                .unwrap();
            assert!(matches!(
                events.next().await.unwrap().unwrap(),
                Event::Delta { .. }
            ));
            assert_eq!(downstream.load(Ordering::SeqCst), 0);
            assert_eq!(
                checkpoint
                    .load(1002, &Cancellation::new())
                    .await
                    .unwrap_err()
                    .code,
                "NOT_FOUND"
            );
            let mut pending = Box::pin(events.next());
            assert!(poll!(pending.as_mut()).is_pending());
            drop(pending);
            release.send(()).unwrap();
            let complete = events.next().await.unwrap().unwrap();
            assert_eq!(
                message(&complete).content[0].data["value"],
                "provider-complete-not-delta-concatenation"
            );
            while let Some(event) = events.next().await {
                event.unwrap();
            }
        }
        assert_eq!(downstream.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.state().status, graph::Status::Succeeded);
        assert_eq!(
            checkpoint.load(1002, &Cancellation::new()).await.unwrap(),
            runtime.snapshot().unwrap()
        );
    });
}

#[test]
fn concurrent_root_runs_do_not_cross_observation_scopes() {
    block_on(async {
        let mut left = Runtime::new(
            1003,
            graph_with_agent(
                ReadyAgent {
                    events: vec![Ok(complete("left", 21, 1))],
                    calls: Default::default(),
                },
                0,
                Default::default(),
            ),
        )
        .unwrap();
        let mut right = Runtime::new(
            1004,
            graph_with_agent(
                ReadyAgent {
                    events: vec![Ok(complete("right", 21, 1))],
                    calls: Default::default(),
                },
                1,
                Default::default(),
            ),
        )
        .unwrap();
        {
            let left = left
                .run(Messages::default(), ids(12000), Cancellation::new())
                .await
                .unwrap();
            let right = right
                .run(Messages::default(), ids(13000), Cancellation::new())
                .await
                .unwrap();
            let (left, right) =
                futures_util::join!(left.collect::<Vec<_>>(), right.collect::<Vec<_>>());
            assert_eq!(left.len(), 2);
            assert_eq!(right.len(), 2);
            assert_eq!(
                message(left[0].as_ref().unwrap()).content[0].data["value"],
                "left"
            );
            assert_eq!(
                message(right[0].as_ref().unwrap()).content[0].data["value"],
                "right"
            );
        }
        assert_eq!(left.state().status, graph::Status::Succeeded);
        assert_eq!(right.state().status, graph::Status::Succeeded);
    });
}

#[test]
fn final_error_is_direct_and_checkpoint_is_saved_before_it_is_delivered() {
    block_on(async {
        let error = Error::new("TIMEOUT", "raw provider error").with_details("provider diagnosis");
        let mut builder = WorkflowBuilder::new("observed-failure");
        builder
            .add_agent_node(
                "agent",
                ReadyAgent {
                    events: vec![
                        Ok(Event::Delta {
                            message: text(Role::Assistant, "partial"),
                        }),
                        Err(error.clone()),
                    ],
                    calls: Default::default(),
                },
            )
            .unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
        let mut runtime = Runtime::new(1005, builder.compile().unwrap())
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        {
            let mut events = runtime
                .run(Messages::default(), ids(14000), Cancellation::new())
                .await
                .unwrap();
            assert!(matches!(
                events.next().await.unwrap().unwrap(),
                Event::Delta { .. }
            ));
            assert_eq!(events.next().await.unwrap().unwrap_err(), error);
            let saved = checkpoint.load(1005, &Cancellation::new()).await.unwrap();
            assert_eq!(saved["status"], "failed");
            assert_eq!(
                saved["nodes"]["agent"]["executions"]["14000"]["attempts"][0]["error"],
                serde_json::to_value(&error).unwrap()
            );
            assert!(events.next().await.is_none());
            assert!(events.next().await.is_none());
        }
        assert_eq!(runtime.state().status, graph::Status::Failed);
        assert!(
            runtime.state().nodes["agent"].executions[&14000]
                .messages
                .is_empty()
        );
    });
}

struct Review;
impl Node<Messages, Messages, Error> for Review {
    async fn call(&self, _: Messages, _: &Cancellation) -> Result<Signal<Messages>, Error> {
        Ok(Signal::Wait(Wait {
            wait_id: "review".into(),
        }))
    }
}

#[test]
fn waiting_stream_ends_after_saving_the_same_waiting_scene() {
    block_on(async {
        let mut builder = WorkflowBuilder::new("observed-wait");
        builder.add_node("review", Review).unwrap();
        builder.add_edge(START, "review").unwrap();
        builder.add_edge("review", END).unwrap();
        let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
        let mut runtime = Runtime::new(1006, builder.compile().unwrap())
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        {
            let mut events = runtime
                .run(Messages::default(), ids(15000), Cancellation::new())
                .await
                .unwrap();
            assert_eq!(events.next().await.unwrap().unwrap_err().code, "WAITING");
            let saved = checkpoint.load(1006, &Cancellation::new()).await.unwrap();
            assert_eq!(saved["status"], "paused");
            assert_eq!(saved["waits"], json!([]));
            assert_eq!(
                saved["nodes"]["review"]["executions"]["15000"]["waits"]["review"]["status"],
                "waiting"
            );
            assert!(events.next().await.is_none());
        }
        assert_eq!(
            runtime.state().nodes["review"].executions[&15000]
                .attempts
                .len(),
            1
        );
        assert_eq!(
            runtime.state().nodes["review"].executions[&15000].status,
            node::ExecutionStatus::Waiting
        );
    });
}

#[test]
fn terminal_run_observes_cached_results_without_calls_or_new_ids() {
    block_on(async {
        let calls = Arc::new(AtomicUsize::new(0));
        let downstream = Arc::new(AtomicUsize::new(0));
        let workflow = graph_with_agent(
            ReadyAgent {
                events: vec![Ok(complete("answer", 21, 9))],
                calls: calls.clone(),
            },
            2,
            downstream.clone(),
        );
        let mut runtime = Runtime::new(1008, workflow.clone()).unwrap();
        {
            let events = runtime
                .run(Messages::default(), ids(17000), Cancellation::new())
                .await
                .unwrap();
            assert_eq!(events.collect::<Vec<_>>().await.len(), 2);
        }
        let snapshot = runtime.snapshot().unwrap();
        let source_id = leaf(runtime.state()).nodes["double"]
            .executions
            .values()
            .next()
            .unwrap()
            .exec_id;
        let restored = Runtime::from_snapshot(1008, workflow, snapshot.clone()).unwrap();
        {
            let observed = restored
                .run(
                    Messages::default(),
                    Arc::new(|| panic!("terminal read allocates no ids")),
                    Cancellation::new(),
                )
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert_eq!(observed.len(), 1);
            assert!(matches!(observed[0], Ok(Event::Complete { .. })));
            assert!(message(observed[0].as_ref().unwrap()).values.is_empty());
            let source = &message(observed[0].as_ref().unwrap()).metadata;
            assert_eq!(source["chat_id"], 1008);
            assert_eq!(source["node_id"], "double");
            assert_eq!(source["exec_id"], source_id);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(downstream.load(Ordering::SeqCst), 1);
        assert_eq!(restored.snapshot().unwrap(), snapshot);
    });
}

struct LocalEvents {
    events: std::vec::IntoIter<Result<Event, Error>>,
    marker: std::rc::Rc<()>,
}
impl Stream for LocalEvents {
    type Item = Result<Event, Error>;
    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        assert_eq!(std::rc::Rc::strong_count(&self.marker), 1);
        Poll::Ready(self.events.next())
    }
}
struct LocalAgent;
impl Agent for LocalAgent {
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
    type Stream = LocalEvents;
    async fn run(&self, _: Input, _stream: bool, _: Cancellation) -> Result<Self::Stream, Error> {
        Ok(LocalEvents {
            events: vec![Ok(complete("local", 21, 4))].into_iter(),
            marker: std::rc::Rc::new(()),
        })
    }
}

#[test]
fn native_non_send_agent_stream_needs_no_spawn_or_extra_executor() {
    block_on(async {
        let mut runtime =
            Runtime::new(1009, graph_with_agent(LocalAgent, 1, Default::default())).unwrap();
        {
            let events = runtime
                .run(Messages::default(), ids(18000), Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert_eq!(events.len(), 2);
            assert!(events.iter().all(Result::is_ok));
        }
        assert_eq!(runtime.state().status, graph::Status::Succeeded);
    });
}

#[test]
fn model_message_usage_is_preserved_when_the_event_usage_is_missing() {
    block_on(async {
        let Event::Complete {
            mut message, usage, ..
        } = complete("message usage", 21, 5)
        else {
            unreachable!()
        };
        message.usage = usage.clone();
        let agent = ReadyAgent {
            events: vec![Ok(Event::Complete {
                message,
                usage: None,
                finish_reason: Some("stop".into()),
            })],
            calls: Default::default(),
        };
        let mut runtime =
            Runtime::new(1011, graph_with_agent(agent, 0, Default::default())).unwrap();
        {
            let observed = runtime
                .run(Messages::default(), ids(20000), Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            let Event::Complete {
                message,
                usage: observed_usage,
                ..
            } = observed[0].as_ref().unwrap()
            else {
                panic!("model Complete")
            };
            assert_eq!(*observed_usage, usage);
            assert_eq!(message.usage, usage);
        }
        assert_eq!(runtime.state().usage, usage);
    });
}

struct BatchTool {
    definition: halo::agent::tool::ToolDefinition,
    calls: Arc<AtomicUsize>,
    fail_first: bool,
}
impl halo::agent::tool::Tool for BatchTool {
    fn definition(&self) -> &halo::agent::tool::ToolDefinition {
        &self.definition
    }
    async fn call(&self, _: &halo::agent::tool::Call, _: &Cancellation) -> Result<Message, Error> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 && self.fail_first {
            Err(Error::new("TIMEOUT", "raw batch tool failure"))
        } else {
            let mut result = text(Role::Tool, &self.definition.name);
            result.values.insert("private".into(), json!(42));
            Ok(result)
        }
    }
}
fn batch_tool(name: &str, calls: Arc<AtomicUsize>, fail_first: bool) -> BatchTool {
    BatchTool {
        definition: halo::agent::tool::ToolDefinition {
            name: name.into(),
            description: name.into(),
            parameters: json!({"type":"object"}).as_object().unwrap().clone(),
        },
        calls,
        fail_first,
    }
}

#[test]
fn native_tool_batch_observes_saved_successes_once_without_error_messages() {
    block_on(async {
        for max_retries in [0, 1] {
            let good = Arc::new(AtomicUsize::new(0));
            let bad = Arc::new(AtomicUsize::new(0));
            let tools = halo::runtime::workflow::tool::ToolNode::new(batch_tool(
                "good",
                good.clone(),
                false,
            ))
            .tool(batch_tool("bad", bad.clone(), true));
            let mut builder = WorkflowBuilder::new("observed-tool-batch");
            builder.add_node("tools", tools).unwrap();
            builder
                .set_retry(
                    "tools",
                    halo::agent::middleware::RetryPolicy {
                        max_retries,
                        min_delay_ms: 0,
                        max_delay_ms: 0,
                    },
                )
                .unwrap();
            builder.add_edge(START, "tools").unwrap();
            builder.add_edge("tools", END).unwrap();
            let mut input = Message::new(Role::Assistant);
            for name in ["good", "bad"] {
                input.content.push(Part {
                    r#type: "tool_call".into(),
                    data: json!({"call_id":name,"name":name,"arguments":{}})
                        .as_object()
                        .unwrap()
                        .clone(),
                });
            }
            let mut runtime = Runtime::new(1010, builder.compile().unwrap()).unwrap();
            let events = {
                runtime
                    .run(Messages::new([input]), ids(19000), Cancellation::new())
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await
            };
            assert_eq!(good.load(Ordering::SeqCst), 1);
            assert_eq!(bad.load(Ordering::SeqCst), max_retries as usize + 1);
            assert_eq!(events.len(), 2);
            let Event::Complete {
                message: good_event,
                usage,
                finish_reason,
            } = events[0].as_ref().unwrap()
            else {
                panic!("tool Complete")
            };
            assert_eq!(good_event.metadata["call_id"], "good");
            assert_eq!(good_event.role, Role::Tool);
            assert!(good_event.values.is_empty());
            assert!(usage.is_none() && finish_reason.is_none());
            let execution = &runtime.state().nodes["tools"].executions[&19000];
            assert_eq!(execution.attempts[0].messages.0[0].values["private"], 42);
            if max_retries == 0 {
                assert_eq!(events[1].as_ref().unwrap_err().code, "TIMEOUT");
                assert!(execution.messages.is_empty());
                assert_eq!(runtime.state().status, graph::Status::Failed);
            } else {
                assert_eq!(
                    message(events[1].as_ref().unwrap()).metadata["call_id"],
                    "bad"
                );
                assert_eq!(execution.messages.0.len(), 2);
                assert_eq!(runtime.state().status, graph::Status::Succeeded);
            }
        }
    });
}
