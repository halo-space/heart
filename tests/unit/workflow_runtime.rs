use super::*;
use crate::Cancellation;
use crate::checkpoint::Checkpoint;
use crate::components::agent::{Agent, Input as AgentInput};
use crate::components::node::{Node, Signal, View, Wait as NodeWait};
use crate::runtime::workflow::graph::{END, START};
use futures_executor::block_on;
use futures_util::stream;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

struct WaitOnce(AtomicBool);

impl Node<Messages, Messages, crate::Error> for WaitOnce {
    async fn call(
        &self,
        input: Messages,
        _cancellation: &Cancellation,
    ) -> Result<Signal<Messages>, crate::Error> {
        if !self.0.swap(true, Ordering::AcqRel) {
            Ok(Signal::Wait(NodeWait {
                wait_id: "approval".into(),
            }))
        } else {
            Ok(Signal::Complete(input))
        }
    }
}

struct Flaky(AtomicBool);

impl Node<Messages, Messages, crate::Error> for Flaky {
    async fn call(
        &self,
        input: Messages,
        _cancellation: &Cancellation,
    ) -> Result<Signal<Messages>, crate::Error> {
        if !self.0.swap(true, Ordering::AcqRel) {
            Err(crate::Error::new("TIMEOUT", "temporary"))
        } else {
            Ok(Signal::Complete(input))
        }
    }
}

struct InspectView;

impl Node<Messages, Messages, crate::Error> for InspectView {
    async fn call(
        &self,
        input: Messages,
        _cancellation: &Cancellation,
    ) -> Result<Signal<Messages>, crate::Error> {
        Ok(Signal::Complete(input))
    }

    fn call_with_view(
        &self,
        _input: Messages,
        view: &View,
        _cancellation: &Cancellation,
    ) -> impl Future<Output = Result<Signal<Messages>, crate::Error>> + Send {
        let observed = serde_json::json!({
            "node_id": view.node_id,
            "exec_id": view.exec_id,
            "attempt_idx": view.attempt_idx,
            "status": view.status,
            "input": view.input,
            "attempts": view.attempts,
            "sources": view.sources,
        });
        async move {
            Ok(Signal::Complete(Messages::new([crate::Message::function(
                observed,
            )])))
        }
    }
}

struct InspectAgent;

impl Agent for InspectAgent {
    type Stream = stream::Iter<std::vec::IntoIter<Result<crate::event::Event, Error>>>;

    async fn run(
        &self,
        _input: AgentInput,
        _stream: bool,
        _cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        Ok(stream::iter(vec![Ok(crate::event::Event::Complete {
            message: crate::Message::new(Role::Assistant),
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }

    fn run_with_view(
        &self,
        _input: AgentInput,
        _stream: bool,
        _cancellation: Cancellation,
        view: &View,
    ) -> impl Future<Output = Result<Self::Stream, Error>> + Send {
        let mut message = crate::Message::new(Role::Assistant);
        message.values = serde_json::Map::from_iter([
            ("node_id".into(), view.node_id.clone().into()),
            ("exec_id".into(), view.exec_id.into()),
            ("attempt_idx".into(), view.attempt_idx.into()),
            ("status".into(), view.status.clone().into()),
        ]);
        async move {
            Ok(stream::iter(vec![Ok(crate::event::Event::Complete {
                message,
                usage: None,
                finish_reason: Some("stop".into()),
            })]))
        }
    }

    async fn reply(&self, _input: AgentInput, _cancellation: Cancellation) -> Result<(), Error> {
        Ok(())
    }
}

struct RetryTool {
    definition: crate::components::tool::ToolDefinition,
    calls: Arc<AtomicI64>,
    fail_first: bool,
}

impl crate::components::tool::Tool for RetryTool {
    fn definition(&self) -> &crate::components::tool::ToolDefinition {
        &self.definition
    }

    async fn call(
        &self,
        call: &crate::components::tool::Call,
        _cancellation: &Cancellation,
    ) -> Result<crate::Message, Error> {
        let count = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_first && count == 0 {
            return Err(Error::new("TIMEOUT", "temporary tool failure"));
        }
        Ok(crate::Message::function(serde_json::json!({
            "tool": call.name,
            "ok": true,
        })))
    }
}

fn workflow_tool_definition(name: &str) -> crate::components::tool::ToolDefinition {
    crate::components::tool::ToolDefinition {
        name: name.into(),
        description: format!("{name} test tool"),
        parameters: serde_json::from_value(serde_json::json!({
            "type": "object",
            "properties": {}
        }))
        .unwrap(),
    }
}

fn workflow_tool_input() -> Messages {
    let mut assistant = crate::Message::new(Role::Assistant);
    assistant.content = vec![
        crate::model::content::part::Part {
            r#type: "tool_call".into(),
            data: crate::components::model::content::part::object(serde_json::json!({
                "call_id": "call_a",
                "name": "first",
                "arguments": {}
            })),
        },
        crate::model::content::part::Part {
            r#type: "tool_call".into(),
            data: crate::components::model::content::part::object(serde_json::json!({
                "call_id": "call_b",
                "name": "second",
                "arguments": {}
            })),
        },
    ];
    Messages::new([assistant])
}

#[derive(Clone, Copy)]
struct FakeAgent {
    mode: &'static str,
}

struct RestoringAgent;

struct WaitingAgent;

impl Agent for WaitingAgent {
    async fn reply(
        &self,
        _: crate::components::agent::Input,
        _: crate::Cancellation,
    ) -> Result<(), crate::Error> {
        Err(crate::Error::new(
            "NOT_FOUND",
            "fixture has no live feedback consumer",
        ))
    }
    type Stream = stream::Iter<std::vec::IntoIter<Result<crate::event::Event, Error>>>;

    async fn run(
        &self,
        _input: AgentInput,
        _stream: bool,
        _cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        if crate::runtime::agent::scope::restore().is_none() {
            crate::runtime::agent::scope::capture(serde_json::json!({
                "type": "agent",
                "status": "waiting",
                "wait_id": "approval"
            }));
            return Err(Error::new("WAITING", "agent is waiting for approval"));
        }
        let mut message = crate::Message::new(Role::Assistant);
        message.content.push(crate::model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                serde_json::json!({"value": "approved"}),
            ),
        });
        Ok(stream::iter(vec![Ok(crate::event::Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

impl Agent for RestoringAgent {
    async fn reply(
        &self,
        _: crate::components::agent::Input,
        _: crate::Cancellation,
    ) -> Result<(), crate::Error> {
        Err(crate::Error::new(
            "NOT_FOUND",
            "fixture has no live feedback consumer",
        ))
    }
    type Stream = stream::Iter<std::vec::IntoIter<Result<crate::event::Event, Error>>>;

    async fn run(
        &self,
        _input: AgentInput,
        _stream: bool,
        _cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        let value = if crate::runtime::agent::scope::restore().is_some() {
            "restored_scene"
        } else {
            "fresh"
        };
        let mut message = crate::Message::new(Role::Assistant);
        message.content.push(crate::model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                serde_json::json!({"value": value}),
            ),
        });
        Ok(stream::iter(vec![Ok(crate::event::Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

struct MeteredAgent {
    calls: Arc<AtomicI64>,
    fail_once: bool,
}

impl Agent for MeteredAgent {
    async fn reply(
        &self,
        _: crate::components::agent::Input,
        _: crate::Cancellation,
    ) -> Result<(), crate::Error> {
        Err(crate::Error::new(
            "NOT_FOUND",
            "fixture has no live feedback consumer",
        ))
    }
    type Stream = stream::Iter<std::vec::IntoIter<Result<crate::event::Event, Error>>>;

    async fn run(
        &self,
        _input: AgentInput,
        _stream: bool,
        _cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let usage = crate::model::token::Usage {
            input: Some(10),
            output: Some(4),
            total: Some(14),
            input_details: Some(crate::model::token::InputDetails { cached: Some(3) }),
            output_details: Some(crate::model::token::OutputDetails { think: Some(2) }),
            details: serde_json::Map::from_iter([("raw".into(), serde_json::json!(call))]),
        };
        let mut message = crate::Message::new(Role::Assistant);
        message.content.push(crate::model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                serde_json::json!({"value":"answer"}),
            ),
        });
        message.usage = Some(usage.clone());
        let mut events = vec![Ok(crate::event::Event::Complete {
            message,
            usage: Some(usage),
            finish_reason: Some("stop".into()),
        })];
        if self.fail_once && call == 0 {
            events.push(Err(Error::new("TIMEOUT", "model stream interrupted")));
        }
        Ok(stream::iter(events))
    }
}

fn user_input() -> Messages {
    let mut message = crate::Message::new(Role::User);
    message.content.push(crate::model::content::part::Part {
        r#type: "text".into(),
        data: crate::components::model::content::part::object(serde_json::json!({"value":"hello"})),
    });
    Messages::new([message])
}

fn workflow_input() -> Messages {
    let mut messages = user_input();
    messages.0[0].metadata = serde_json::Map::from_iter([
        ("tenant_id".into(), serde_json::json!(1)),
        ("user_id".into(), serde_json::json!(2)),
        ("agent_id".into(), serde_json::json!(3)),
        ("session_id".into(), serde_json::json!(4)),
        ("chat_id".into(), serde_json::json!(5)),
    ]);
    messages
}

type WorkflowMemoryConfig = crate::memory::Config<
    crate::memory::short::Pair<
        crate::core::memory::short::session::InMemory,
        crate::core::memory::short::chat::InMemory,
    >,
    crate::core::memory::InMemory,
>;

fn workflow_memory() -> WorkflowMemoryConfig {
    use crate::memory::short::session::Store as _;
    let config = WorkflowMemoryConfig::new(
        crate::memory::short::Pair::new(
            crate::core::memory::short::session::InMemory::new(),
            crate::core::memory::short::chat::InMemory::new(),
        ),
        crate::core::memory::InMemory::new(),
    );
    block_on(config.session().create(
        crate::memory::short::session::Session {
            id: 4,
            tenant_id: 1,
            user_id: 2,
            r#type: crate::memory::short::Type::Workflow,
            title: None,
            summary: None,
            usage: None,
            status: crate::memory::short::session::Status::Active,
            metadata: serde_json::Map::new(),
            created_time: 0,
            updated_time: 0,
        },
        &Cancellation::new(),
    ))
    .unwrap();
    config
}

#[test]
fn workflow_usage_counts_failed_and_successful_agent_attempts_once() {
    let mut builder = Builder::new("metered");
    builder
        .add_agent_node(
            "agent",
            MeteredAgent {
                calls: Arc::new(AtomicI64::new(0)),
                fail_once: true,
            },
        )
        .unwrap();
    builder
        .set_retry(
            "agent",
            RetryPolicy {
                max_retries: 1,
                min_delay_ms: 0,
                max_delay_ms: 0,
            },
        )
        .unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(91, workflow.clone()).unwrap();
    block_on(runtime.execute(user_input(), Arc::new(|| 12), Cancellation::new())).unwrap();
    let state = runtime.state();
    let execution = &state.nodes["agent"].executions[&12];
    assert_eq!(execution.attempts.len(), 2);
    assert_eq!(execution.usage.as_ref().unwrap().total, Some(28));
    let total = state.usage.as_ref().unwrap();
    assert_eq!(total.total, Some(28));
    assert_eq!(total.input_details.as_ref().unwrap().cached, Some(6));
    assert_eq!(total.output_details.as_ref().unwrap().think, Some(4));
    assert!(total.details.is_empty());
    assert_eq!(
        execution.attempts[0].steps[0].attempts[0]
            .usage
            .as_ref()
            .unwrap()
            .details["raw"],
        0
    );
    let mut snapshot = runtime.snapshot().unwrap();
    snapshot["usage"]["total"] = serde_json::json!(999);
    snapshot["nodes"]["agent"]["executions"]["12"]["usage"]["total"] = serde_json::json!(999);
    let mut restored = Runtime::from_snapshot(91, workflow, snapshot).unwrap();
    assert_eq!(restored.state().usage.as_ref().unwrap().total, Some(28));
    assert_eq!(
        restored.state().nodes["agent"].executions[&12]
            .usage
            .as_ref()
            .unwrap()
            .total,
        Some(28)
    );
}

#[test]
fn node_receives_only_its_read_only_local_view() {
    let mut builder = Builder::new("node-view");
    builder.add_node("inspect", InspectView).unwrap();
    builder.add_edge(START, "inspect").unwrap();
    builder.add_edge("inspect", END).unwrap();
    let mut runtime = Runtime::new(93, builder.compile().unwrap()).unwrap();
    let result =
        block_on(runtime.execute(user_input(), Arc::new(|| 9300), Cancellation::new())).unwrap();
    let observed = &result.as_slice()[0].values;
    assert_eq!(observed["node_id"], "inspect");
    assert_eq!(observed["exec_id"], 9300);
    assert_eq!(observed["attempt_idx"], 0);
    assert_eq!(observed["status"], "running");
    assert!(observed["input"].is_array());
    assert_eq!(observed["attempts"].as_array().unwrap().len(), 1);
    assert!(observed["sources"].as_object().unwrap().is_empty());
}

#[test]
fn agent_node_receives_the_same_local_view_contract() {
    let mut builder = Builder::new("agent-view");
    builder.add_agent_node("inspect", InspectAgent).unwrap();
    builder.add_edge(START, "inspect").unwrap();
    builder.add_edge("inspect", END).unwrap();
    let mut runtime = Runtime::new(94, builder.compile().unwrap()).unwrap();
    let result =
        block_on(runtime.execute(user_input(), Arc::new(|| 9400), Cancellation::new())).unwrap();
    let values = &result.as_slice()[0].values;
    assert_eq!(values["node_id"], "inspect");
    assert_eq!(values["exec_id"], 9400);
    assert_eq!(values["attempt_idx"], 0);
    assert_eq!(values["status"], "running");
}

#[test]
fn nested_workflow_usage_is_included_without_billing_the_parent_twice() {
    let mut child = Builder::new("metered-child");
    child
        .add_agent_node(
            "agent",
            MeteredAgent {
                calls: Arc::new(AtomicI64::new(0)),
                fail_once: false,
            },
        )
        .unwrap();
    child.add_edge(START, "agent").unwrap();
    child.add_edge("agent", END).unwrap();
    let child = child.compile().unwrap();
    let mut parent = Builder::new("metered-parent");
    parent.add_graph_node("child", &child).unwrap();
    parent.add_edge(START, "child").unwrap();
    parent.add_edge("child", END).unwrap();
    let mut runtime = Runtime::new(92, parent.compile().unwrap()).unwrap();
    let ids = Arc::new(AtomicI64::new(20));
    block_on(runtime.execute(
        user_input(),
        Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
        Cancellation::new(),
    ))
    .unwrap();
    let execution = &runtime.state().nodes["child"].executions[&20];
    assert_eq!(execution.usage.as_ref().unwrap().total, Some(14));
    assert_eq!(
        execution.attempts[0]
            .graph
            .as_ref()
            .unwrap()
            .usage
            .as_ref()
            .unwrap()
            .total,
        Some(14)
    );
    assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(14));
}

impl Agent for FakeAgent {
    async fn reply(
        &self,
        _: crate::components::agent::Input,
        _: crate::Cancellation,
    ) -> Result<(), crate::Error> {
        Err(crate::Error::new(
            "NOT_FOUND",
            "fixture has no live feedback consumer",
        ))
    }
    type Stream = stream::Iter<std::vec::IntoIter<Result<crate::event::Event, Error>>>;

    async fn run(
        &self,
        input: AgentInput,
        _stream: bool,
        _cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        assert_eq!(input.payload.len(), 1);
        assert_eq!(input.payload[0].r#type, "text");
        if self.mode == "error" {
            return Err(Error::new("BUSINESS", "agent failed"));
        }
        let make_message = |role| {
            let mut message = crate::Message::new(role);
            message.content.push(crate::model::content::part::Part {
                r#type: "text".into(),
                data: crate::components::model::content::part::object(
                    serde_json::json!({"value": "agent result"}),
                ),
            });
            message
        };
        let first = (self.mode == "many").then(|| crate::event::Event::Complete {
            message: make_message(Role::Tool),
            usage: None,
            finish_reason: Some("tool".into()),
        });
        let finish_reason = if self.mode == "cancel" {
            "cancelled"
        } else {
            "stop"
        };
        let final_event = crate::event::Event::Complete {
            message: make_message(Role::Assistant),
            usage: None,
            finish_reason: Some(finish_reason.into()),
        };
        let events: Vec<Result<crate::event::Event, Error>> =
            first.into_iter().chain([final_event]).map(Ok).collect();
        Ok(stream::iter(events))
    }
}

#[test]
fn runtime_owns_one_state_and_exposes_checkpoint_json() {
    let mut builder = Builder::new("runtime");
    builder.add_node("answer", crate::node::Echo).unwrap();
    builder.add_edge(START, "answer").unwrap();
    builder.add_edge("answer", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(1, workflow.clone()).unwrap();
    assert_eq!(runtime.ready_nodes(), vec!["answer"]);
    runtime
        .start_batch(vec![("answer".into(), 1, Vec::new())])
        .unwrap();
    runtime
        .complete("answer", 1, Messages::default(), vec![END.into()])
        .unwrap();
    assert!(runtime.advance_step());
    let snapshot = runtime.snapshot().unwrap();
    assert_eq!(snapshot["step"], 1);
    assert_eq!(
        snapshot["nodes"]["answer"]["executions"]["1"]["status"],
        "succeeded"
    );
    assert_eq!(snapshot["definition_id"], "runtime");
    assert_eq!(snapshot["version"], runtime.definition().version);
    let mut restored = Runtime::from_snapshot(1, workflow, snapshot).unwrap();
    assert_eq!(restored.state().step, 1);
}

#[test]
fn run_executes_agent_node_and_preserves_agent_steps() {
    let mut builder = Builder::new("agent-node");
    builder
        .add_agent_node("agent", FakeAgent { mode: "single" })
        .unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(31, workflow).unwrap();
    let mut input = crate::Message::new(Role::User);
    input.content.push(crate::model::content::part::Part {
        r#type: "text".into(),
        data: crate::components::model::content::part::object(
            serde_json::json!({"value": "hello"}),
        ),
    });
    let ids = Arc::new(|| 3100);
    let result =
        block_on(runtime.execute(Messages::new([input]), ids, Cancellation::new())).unwrap();
    assert_eq!(result.as_slice()[0].role, Role::Assistant);
    let execution = &runtime.state().nodes["agent"].executions[&3100];
    assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
    assert_eq!(execution.attempts[0].steps.len(), 1);
    assert_eq!(execution.attempts[0].messages.as_slice().len(), 1);
}

#[test]
fn agent_node_keeps_each_complete_as_a_step_and_commits_last_assistant() {
    let mut builder = Builder::new("agent-many");
    builder
        .add_agent_node("agent", FakeAgent { mode: "many" })
        .unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(32, workflow).unwrap();
    let mut input = crate::Message::new(Role::User);
    input.content.push(crate::model::content::part::Part {
        r#type: "text".into(),
        data: crate::components::model::content::part::object(
            serde_json::json!({"value": "hello"}),
        ),
    });
    let result = block_on(runtime.execute(
        Messages::new([input]),
        Arc::new(|| 3200),
        Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(result.as_slice()[0].role, Role::Assistant);
    let attempt = &runtime.state().nodes["agent"].executions[&3200].attempts[0];
    assert_eq!(attempt.steps.len(), 2);
    assert_eq!(attempt.messages.as_slice().len(), 2);
}

#[test]
fn cancelled_agent_does_not_activate_end() {
    let mut builder = Builder::new("agent-cancel");
    builder
        .add_agent_node("agent", FakeAgent { mode: "cancel" })
        .unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(33, workflow).unwrap();
    let mut input = crate::Message::new(Role::User);
    input.content.push(crate::model::content::part::Part {
        r#type: "text".into(),
        data: crate::components::model::content::part::object(
            serde_json::json!({"value": "hello"}),
        ),
    });
    let _ = block_on(runtime.execute(
        Messages::new([input]),
        Arc::new(|| 3300),
        Cancellation::new(),
    ));
    assert_eq!(
        runtime.state().nodes["agent"].executions[&3300].status,
        node::ExecutionStatus::Cancelled
    );
    assert_ne!(runtime.state().status, Status::Succeeded);
}

#[test]
fn failed_agent_is_recorded_without_successful_outgoing_relations() {
    let mut builder = Builder::new("agent-failure");
    builder
        .add_agent_node("agent", FakeAgent { mode: "error" })
        .unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(34, workflow).unwrap();
    let mut input = crate::Message::new(Role::User);
    input.content.push(crate::model::content::part::Part {
        r#type: "text".into(),
        data: crate::components::model::content::part::object(
            serde_json::json!({"value": "hello"}),
        ),
    });
    let _ = block_on(runtime.execute(
        Messages::new([input]),
        Arc::new(|| 3400),
        Cancellation::new(),
    ));
    let execution = &runtime.state().nodes["agent"].executions[&3400];
    assert_eq!(execution.status, node::ExecutionStatus::Failed);
    assert!(execution.to.is_empty());
    assert_eq!(
        execution.error.as_ref().map(|error| error.code.as_str()),
        Some("BUSINESS")
    );
}

#[test]
fn run_executes_a_dag_and_commits_each_barrier() {
    let mut builder = Builder::new("run");
    builder.add_node("a", crate::node::Echo).unwrap();
    builder.add_node("b", crate::node::Echo).unwrap();
    builder.add_edge(START, "a").unwrap();
    builder.add_edge("a", "b").unwrap();
    builder.add_edge("b", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(11, workflow).unwrap();
    let initial = Messages::new([crate::Message::function(serde_json::json!({
        "query": "hello"
    }))]);
    let next = Arc::new(std::sync::atomic::AtomicI64::new(100));
    let ids = {
        let next = Arc::clone(&next);
        Arc::new(move || next.fetch_add(1, std::sync::atomic::Ordering::AcqRel))
    };
    let result = block_on(runtime.execute(initial.clone(), ids, Cancellation::new())).unwrap();
    assert_eq!(result, initial);
    assert_eq!(runtime.state().step, 2);
    assert_eq!(runtime.state().status, Status::Succeeded);
    assert_eq!(
        runtime.state().nodes["a"].executions[&100].attempts.len(),
        1
    );
    assert_eq!(
        runtime.state().nodes["b"].executions[&101].attempts.len(),
        1
    );
}

#[test]
fn run_persists_a_conditional_route_in_execution_to() {
    let mut builder = Builder::new("conditional-run");
    for id in ["router", "left", "right"] {
        builder.add_node(id, crate::node::Echo).unwrap();
    }
    builder.add_edge(START, "router").unwrap();
    builder
        .add_conditional_edges(
            "router",
            |messages| {
                let route = messages
                    .as_slice()
                    .last()
                    .and_then(|message| message.values.get("route"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("left");
                Ok(vec![route.to_owned()])
            },
            ["left", "right"],
        )
        .unwrap();
    builder.add_edge("left", END).unwrap();
    builder.add_edge("right", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(14, workflow).unwrap();
    let initial = Messages::new([crate::Message::function(serde_json::json!({
        "route": "right"
    }))]);
    let next = Arc::new(std::sync::atomic::AtomicI64::new(400));
    let ids = {
        let next = Arc::clone(&next);
        Arc::new(move || next.fetch_add(1, std::sync::atomic::Ordering::AcqRel))
    };
    block_on(runtime.execute(initial, ids, Cancellation::new())).unwrap();
    assert_eq!(
        runtime.state().nodes["router"].executions[&400].to,
        vec!["right"]
    );
    assert!(runtime.state().nodes["left"].executions.is_empty());
    assert_eq!(runtime.state().status, Status::Succeeded);
}

#[test]
fn runtime_can_save_and_restore_its_single_checkpoint() {
    let mut builder = Builder::new("checkpoint-run");
    builder.add_node("answer", crate::node::Echo).unwrap();
    builder.add_edge(START, "answer").unwrap();
    builder.add_edge("answer", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(16, workflow.clone()).unwrap();
    let ids = Arc::new(|| 600);
    block_on(runtime.execute(Messages::default(), ids, Cancellation::new())).unwrap();
    let checkpoint = crate::core::checkpoint::InMemory::new();
    block_on(runtime.save_checkpoint(&checkpoint, &Cancellation::new())).unwrap();
    let snapshot = block_on(checkpoint.load(16, &Cancellation::new())).unwrap();
    let mut restored = Runtime::from_snapshot(16, workflow, snapshot).unwrap();
    assert_eq!(restored.state().status, Status::Succeeded);
}

#[test]
fn configured_checkpoint_saves_one_closeout_snapshot_after_success() {
    let mut builder = Builder::new("checkpoint-closeout");
    builder.add_node("answer", crate::node::Echo).unwrap();
    builder.add_edge(START, "answer").unwrap();
    builder.add_edge("answer", END).unwrap();
    let workflow = builder.compile().unwrap();
    let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
    let mut runtime = Runtime::new(160, workflow.clone())
        .unwrap()
        .with_checkpoint(checkpoint.clone());

    block_on(runtime.execute(Messages::default(), Arc::new(|| 6_000), Cancellation::new()))
        .unwrap();

    let snapshot = block_on(checkpoint.load(160, &Cancellation::new())).unwrap();
    let mut restored = Runtime::from_snapshot(160, workflow, snapshot).unwrap();
    assert_eq!(restored.state().status, Status::Succeeded);
}

#[test]
fn configured_checkpoint_preserves_cancelled_state_even_when_run_is_cancelled() {
    let mut builder = Builder::new("checkpoint-cancel");
    builder.add_node("answer", crate::node::Echo).unwrap();
    builder.add_edge(START, "answer").unwrap();
    builder.add_edge("answer", END).unwrap();
    let workflow = builder.compile().unwrap();
    let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
    let mut runtime = Runtime::new(161, workflow)
        .unwrap()
        .with_checkpoint(checkpoint.clone());
    let cancellation = Cancellation::new();
    cancellation.cancel();

    let error = block_on(runtime.execute(Messages::default(), Arc::new(|| 6_001), cancellation))
        .unwrap_err();
    assert_eq!(error.code, "CANCELLED");
    let snapshot = block_on(checkpoint.load(161, &Cancellation::new())).unwrap();
    assert_eq!(snapshot["status"], "cancelled");
}

#[test]
fn workflow_memory_writes_chat_graph_and_state_only_at_closeout() {
    use crate::memory::short::chat::Store as _;

    let mut builder = Builder::new("workflow-memory-closeout");
    builder
        .add_agent_node("agent", FakeAgent { mode: "single" })
        .unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let workflow = builder.compile().unwrap();
    let config = workflow_memory();
    let chats = config.clone();
    let mut runtime = Runtime::new(5, workflow.clone())
        .unwrap()
        .memory(config)
        .unwrap();

    block_on(runtime.execute(workflow_input(), Arc::new(|| 5_001), Cancellation::new())).unwrap();

    let chat = block_on(chats.chat().read(5, &Cancellation::new())).unwrap();
    assert_eq!(chat.r#type, crate::memory::short::Type::Workflow);
    assert_eq!(chat.status, crate::memory::short::chat::Status::Completed);
    assert_eq!(
        chat.graph,
        Some(serde_json::to_value(runtime.definition()).unwrap())
    );
    assert_eq!(
        chat.message.as_ref().map(|message| message.role.clone()),
        Some(Role::Assistant)
    );
    assert_eq!(chat.state, Some(runtime.snapshot().unwrap()));
    assert_eq!(chat.state.as_ref().unwrap()["chat_id"], 5);
    assert_eq!(chat.state.as_ref().unwrap()["type"], "workflow");
    assert_eq!(chat.state.as_ref().unwrap()["status"], "succeeded");
}

#[test]
fn terminal_workflow_snapshot_is_idempotent_with_memory() {
    use crate::memory::short::chat::Store as _;

    let mut builder = Builder::new("workflow-terminal-restore");
    builder
        .add_agent_node("agent", FakeAgent { mode: "single" })
        .unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let workflow = builder.compile().unwrap();
    let config = workflow_memory();
    let chats = config.clone();
    let mut first = Runtime::new(5, workflow.clone())
        .unwrap()
        .memory(config.clone())
        .unwrap();
    block_on(first.execute(workflow_input(), Arc::new(|| 5_010), Cancellation::new())).unwrap();
    let snapshot = first.snapshot().unwrap();

    let mut restored = Runtime::from_snapshot(5, workflow, snapshot)
        .unwrap()
        .memory(config)
        .unwrap();
    block_on(restored.execute(workflow_input(), Arc::new(|| 5_011), Cancellation::new())).unwrap();

    assert_eq!(
        block_on(chats.chat().read(5, &Cancellation::new()))
            .unwrap()
            .status,
        crate::memory::short::chat::Status::Completed
    );
    assert_eq!(
        block_on(chats.chat().read(5, &Cancellation::new()))
            .unwrap()
            .state
            .unwrap()["chat_id"],
        5
    );
}

#[test]
fn workflow_snapshot_rejects_a_different_chat_id() {
    let mut builder = Builder::new("workflow-checkpoint-identity");
    builder.add_node("answer", crate::node::Echo).unwrap();
    builder.add_edge(START, "answer").unwrap();
    builder.add_edge("answer", END).unwrap();
    let workflow = builder.compile().unwrap();
    let runtime = Runtime::new(5, workflow.clone()).unwrap();
    let error = match Runtime::from_snapshot(6, workflow, runtime.snapshot().unwrap()) {
        Ok(_) => panic!("a checkpoint for another chat_id must be rejected"),
        Err(error) => error,
    };
    assert_eq!(error.code, "CONFLICT");
}

#[test]
fn workflow_memory_keeps_waiting_chat_open_until_resume_closeout() {
    use crate::memory::short::chat::Store as _;

    let mut builder = Builder::new("workflow-memory-wait");
    builder
        .add_node("approval", WaitOnce(AtomicBool::new(false)))
        .unwrap();
    builder.add_edge(START, "approval").unwrap();
    builder.add_edge("approval", END).unwrap();
    let workflow = builder.compile().unwrap();
    let config = workflow_memory();
    let chats = config.clone();
    let mut runtime = Runtime::new(5, workflow).unwrap().memory(config).unwrap();
    let initial = workflow_input();

    assert_eq!(
        block_on(runtime.execute(initial.clone(), Arc::new(|| 5_002), Cancellation::new(),))
            .unwrap_err()
            .code,
        "WAITING"
    );
    assert_eq!(
        block_on(chats.chat().read(5, &Cancellation::new()))
            .unwrap()
            .status,
        crate::memory::short::chat::Status::Running
    );
    assert!(
        block_on(chats.chat().read(5, &Cancellation::new()))
            .unwrap()
            .state
            .is_none()
    );

    runtime
        .resume(
            "approval",
            Messages::new([crate::Message::function(serde_json::json!({
                "approved": true
            }))]),
        )
        .unwrap();
    block_on(runtime.execute(initial, Arc::new(|| 5_003), Cancellation::new())).unwrap();
    assert_eq!(
        block_on(chats.chat().read(5, &Cancellation::new()))
            .unwrap()
            .status,
        crate::memory::short::chat::Status::Completed
    );
    assert!(
        block_on(chats.chat().read(5, &Cancellation::new()))
            .unwrap()
            .state
            .is_some()
    );
}

#[test]
fn workflow_memory_closes_failed_and_cancelled_runs_with_state() {
    use crate::memory::short::chat::Store as _;

    for (mode, expected_chat_status, expected_state) in [
        (
            "error",
            crate::memory::short::chat::Status::Failed,
            "failed",
        ),
        (
            "cancel",
            crate::memory::short::chat::Status::Stopped,
            "cancelled",
        ),
    ] {
        let mut builder = Builder::new(format!("workflow-memory-{mode}"));
        builder.add_agent_node("agent", FakeAgent { mode }).unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = builder.compile().unwrap();
        let config = workflow_memory();
        let chats = config.clone();
        let mut runtime = Runtime::new(5, workflow).unwrap().memory(config).unwrap();
        let error =
            block_on(runtime.execute(workflow_input(), Arc::new(|| 5_004), Cancellation::new()))
                .unwrap_err();
        assert_eq!(
            error.code,
            if mode == "cancel" {
                "CANCELLED"
            } else {
                "BUSINESS"
            }
        );
        assert_eq!(
            block_on(chats.chat().read(5, &Cancellation::new()))
                .unwrap()
                .status,
            expected_chat_status
        );
        assert_eq!(
            block_on(chats.chat().read(5, &Cancellation::new()))
                .unwrap()
                .state
                .unwrap()["status"],
            expected_state
        );
    }
}

#[test]
fn failed_node_can_activate_an_explicit_failure_relation() {
    let mut builder = Builder::new("failure-run");
    let fail = async |_: Messages, _: Cancellation| -> Result<Messages, Error> {
        Err(Error::new("BUSINESS", "failed"))
    };
    builder.add_node("fail", fail).unwrap();
    builder.add_node("handle", crate::node::Echo).unwrap();
    builder.add_edge(START, "fail").unwrap();
    builder
        .add_failure_edges(["fail"], ["handle"], "node_failed")
        .unwrap();
    builder.add_edge("handle", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(17, workflow).unwrap();
    let next = Arc::new(std::sync::atomic::AtomicI64::new(700));
    let ids = {
        let next = Arc::clone(&next);
        Arc::new(move || next.fetch_add(1, std::sync::atomic::Ordering::AcqRel))
    };
    block_on(runtime.execute(Messages::default(), ids, Cancellation::new())).unwrap();
    assert_eq!(runtime.state().nodes["fail"].status, node::Status::Failed);
    assert_eq!(
        runtime.state().nodes["handle"].status,
        node::Status::Succeeded
    );
    assert_eq!(runtime.state().status, Status::Succeeded);
}

#[test]
fn run_executes_a_graph_node_and_keeps_child_state_on_attempt() {
    let mut child_builder = Builder::new("child-run");
    child_builder.add_node("inner", crate::node::Echo).unwrap();
    child_builder.add_edge(START, "inner").unwrap();
    child_builder.add_edge("inner", END).unwrap();
    let child = child_builder.compile().unwrap();

    let mut parent_builder = Builder::new("parent-run");
    parent_builder.add_graph_node("child", &child).unwrap();
    parent_builder.add_edge(START, "child").unwrap();
    parent_builder.add_edge("child", END).unwrap();
    let parent = parent_builder.compile().unwrap();
    let mut runtime = Runtime::new(15, parent).unwrap();
    let initial = Messages::new([crate::Message::function(serde_json::json!({
        "query": "nested"
    }))]);
    let next = Arc::new(std::sync::atomic::AtomicI64::new(500));
    let ids = {
        let next = Arc::clone(&next);
        Arc::new(move || next.fetch_add(1, std::sync::atomic::Ordering::AcqRel))
    };
    block_on(runtime.execute(initial.clone(), ids, Cancellation::new())).unwrap();
    let execution = &runtime.state().nodes["child"].executions[&500];
    assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
    assert_eq!(
        execution.attempts[0].graph.as_ref().unwrap().status,
        Status::Succeeded
    );
    assert_eq!(execution.messages, initial);
}

#[test]
fn nested_graph_wait_resumes_through_the_parent_execution() {
    let mut child_builder = Builder::new("child-wait");
    child_builder
        .add_node("approval", WaitOnce(AtomicBool::new(false)))
        .unwrap();
    child_builder.add_edge(START, "approval").unwrap();
    child_builder.add_edge("approval", END).unwrap();
    let child = child_builder.compile().unwrap();

    let mut parent_builder = Builder::new("parent-wait");
    parent_builder.add_graph_node("child", &child).unwrap();
    parent_builder.add_edge(START, "child").unwrap();
    parent_builder.add_edge("child", END).unwrap();
    let parent = parent_builder.compile().unwrap();
    let mut runtime = Runtime::new(18, parent).unwrap();
    let initial = Messages::default();
    let first = block_on(runtime.execute(initial.clone(), Arc::new(|| 1_200), Cancellation::new()));
    assert_eq!(first.unwrap_err().code, "WAITING");
    assert_eq!(runtime.state().status, Status::Paused);
    assert_eq!(
        runtime.state().nodes["child"].executions[&1_200].status,
        node::ExecutionStatus::Waiting
    );

    runtime
        .resume(
            "child",
            Messages::new([crate::Message::function(
                serde_json::json!({"approved": true}),
            )]),
        )
        .unwrap();
    block_on(runtime.execute(initial, Arc::new(|| 1_201), Cancellation::new())).unwrap();
    let execution = &runtime.state().nodes["child"].executions[&1_200];
    assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
    assert_eq!(execution.attempts.len(), 1);
    assert_eq!(
        execution.attempts[0].graph.as_ref().unwrap().status,
        Status::Succeeded
    );
}

#[test]
fn nested_graph_checkpoint_restores_child_wait_before_parent_resume() {
    let mut child_builder = Builder::new("child-checkpoint-wait");
    child_builder
        .add_node("approval", WaitOnce(AtomicBool::new(false)))
        .unwrap();
    child_builder.add_edge(START, "approval").unwrap();
    child_builder.add_edge("approval", END).unwrap();
    let child = child_builder.compile().unwrap();

    let mut parent_builder = Builder::new("parent-checkpoint-wait");
    parent_builder.add_graph_node("child", &child).unwrap();
    parent_builder.add_edge(START, "child").unwrap();
    parent_builder.add_edge("child", END).unwrap();
    let parent = parent_builder.compile().unwrap();
    let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
    let mut runtime = Runtime::new(19, parent.clone())
        .unwrap()
        .with_checkpoint(checkpoint.clone());

    let initial = Messages::default();
    let first = block_on(runtime.execute(initial.clone(), Arc::new(|| 1_300), Cancellation::new()));
    assert_eq!(first.unwrap_err().code, "WAITING");

    let snapshot = block_on(checkpoint.load(19, &Cancellation::new())).unwrap();
    let child_execution = &snapshot["nodes"]["child"]["executions"]["1300"];
    assert_eq!(child_execution["status"], "waiting");
    assert_eq!(child_execution["attempts"][0]["graph"]["status"], "paused");
    assert_eq!(
        child_execution["attempts"][0]["graph"]["waits"],
        serde_json::json!([])
    );
    assert_eq!(
        child_execution["attempts"][0]["graph"]["nodes"]["approval"]["executions"]["1300"]["waits"]
            ["approval"]["status"],
        "waiting"
    );

    let mut restored = Runtime::from_snapshot(19, parent, snapshot).unwrap();
    restored
        .resume(
            "child",
            Messages::new([crate::Message::function(
                serde_json::json!({"approved": true}),
            )]),
        )
        .unwrap();
    block_on(restored.execute(initial, Arc::new(|| 1_301), Cancellation::new())).unwrap();

    let execution = &restored.state().nodes["child"].executions[&1_300];
    assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
    assert_eq!(execution.attempts.len(), 1);
    assert_eq!(
        execution.attempts[0].graph.as_ref().unwrap().status,
        Status::Succeeded
    );
    assert!(
        execution.attempts[0]
            .graph
            .as_ref()
            .unwrap()
            .waits
            .is_empty()
    );
}

#[test]
fn checkpoint_recovery_retries_waiting_node_without_allocating_a_new_execution() {
    let mut builder = Builder::new("checkpoint-waiting-node");
    builder.add_node("answer", crate::node::Echo).unwrap();
    builder
        .set_retry(
            "answer",
            RetryPolicy {
                max_retries: 1,
                min_delay_ms: 0,
                max_delay_ms: 0,
            },
        )
        .unwrap();
    builder.add_edge(START, "answer").unwrap();
    builder.add_edge("answer", END).unwrap();
    let workflow = builder.compile().unwrap();

    let mut runtime = Runtime::new(1_700, workflow.clone()).unwrap();
    runtime.state_mut().initial = Some(Messages::default());
    runtime
        .start_batch(vec![("answer".into(), 17_001, Vec::new())])
        .unwrap();
    let snapshot = runtime.snapshot().unwrap();
    assert_eq!(
        snapshot["nodes"]["answer"]["executions"]["17001"]["attempts"][0]["status"],
        "running"
    );

    let mut restored = Runtime::from_snapshot(1_700, workflow, snapshot).unwrap();
    let allocated = Arc::new(AtomicI64::new(0));
    let allocated_for_run = Arc::clone(&allocated);
    block_on(restored.execute(
        Messages::default(),
        Arc::new(move || allocated_for_run.fetch_add(1, Ordering::SeqCst)),
        Cancellation::new(),
    ))
    .unwrap();

    let execution = &restored.state().nodes["answer"].executions[&17_001];
    assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
    assert_eq!(execution.attempts.len(), 1);
    assert_eq!(execution.attempts[0].status, node::AttemptStatus::Succeeded);
    assert_eq!(allocated.load(Ordering::SeqCst), 0);
}

#[test]
fn checkpoint_recovery_resumes_agent_attempt_without_allocating_new_identity() {
    let mut builder = Builder::new("checkpoint-waiting-agent");
    builder
        .add_agent_node("agent", FakeAgent { mode: "single" })
        .unwrap();
    builder
        .set_retry(
            "agent",
            RetryPolicy {
                max_retries: 1,
                min_delay_ms: 0,
                max_delay_ms: 0,
            },
        )
        .unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let workflow = builder.compile().unwrap();

    let mut runtime = Runtime::new(1_701, workflow.clone()).unwrap();
    runtime.state_mut().initial = Some(user_input());
    runtime
        .start_batch(vec![("agent".into(), 17_011, Vec::new())])
        .unwrap();
    let snapshot = runtime.snapshot().unwrap();

    let mut restored = Runtime::from_snapshot(1_701, workflow, snapshot).unwrap();
    let allocated = Arc::new(AtomicI64::new(0));
    let allocated_for_run = Arc::clone(&allocated);
    block_on(restored.execute(
        user_input(),
        Arc::new(move || allocated_for_run.fetch_add(1, Ordering::SeqCst)),
        Cancellation::new(),
    ))
    .unwrap();

    let execution = &restored.state().nodes["agent"].executions[&17_011];
    assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
    assert_eq!(execution.attempts.len(), 1);
    assert_eq!(execution.attempts[0].status, node::AttemptStatus::Succeeded);
    assert_eq!(execution.attempts[0].idx, 0);
    assert_eq!(allocated.load(Ordering::SeqCst), 0);
}

#[test]
fn embedded_agent_wait_uses_workflow_wait_and_resumes_same_execution() {
    let mut builder = Builder::new("embedded-agent-wait");
    builder.add_agent_node("agent", WaitingAgent).unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let workflow = builder.compile().unwrap();

    let mut runtime = Runtime::new(1_704, workflow).unwrap();
    let first = block_on(runtime.execute(user_input(), Arc::new(|| 17_041), Cancellation::new()));
    assert_eq!(first.unwrap_err().code, "WAITING");
    assert_eq!(runtime.state().status, Status::Paused);
    let execution = &runtime.state().nodes["agent"].executions[&17_041];
    assert_eq!(execution.status, node::ExecutionStatus::Waiting);
    assert_eq!(execution.attempts.len(), 1);
    assert_eq!(execution.attempts[0].status, node::AttemptStatus::Waiting);
    assert!(runtime.state().waits.is_empty());
    assert_eq!(
        runtime.state().pause("agent").unwrap().unwrap().wait_id,
        "approval"
    );

    runtime
        .resume(
            "agent",
            Messages::new([crate::Message::function(serde_json::json!({
                "approved": true
            }))]),
        )
        .unwrap();
    block_on(runtime.execute(user_input(), Arc::new(|| 17_042), Cancellation::new())).unwrap();

    let execution = &runtime.state().nodes["agent"].executions[&17_041];
    assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
    assert_eq!(execution.attempts.len(), 1);
    assert_eq!(execution.attempts[0].idx, 0);
    assert!(runtime.state().waits.is_empty());
}

#[test]
fn checkpoint_recovery_passes_embedded_agent_scene_to_the_same_attempt() {
    let mut builder = Builder::new("checkpoint-embedded-agent-scene");
    builder.add_agent_node("agent", RestoringAgent).unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    let workflow = builder.compile().unwrap();

    let mut runtime = Runtime::new(1_703, workflow.clone()).unwrap();
    runtime.state_mut().initial = Some(user_input());
    runtime
        .start_batch(vec![("agent".into(), 17_031, Vec::new())])
        .unwrap();
    runtime
        .state_mut()
        .nodes
        .get_mut("agent")
        .unwrap()
        .executions
        .get_mut(&17_031)
        .unwrap()
        .attempts[0]
        .agent = Some(serde_json::json!({
        "type": "agent",
        "status": "running",
        "messages": [{"role": "assistant", "content": []}]
    }));
    let snapshot = runtime.snapshot().unwrap();

    let mut restored = Runtime::from_snapshot(1_703, workflow, snapshot).unwrap();
    let allocated = Arc::new(AtomicI64::new(0));
    let allocated_for_run = Arc::clone(&allocated);
    block_on(restored.execute(
        user_input(),
        Arc::new(move || allocated_for_run.fetch_add(1, Ordering::SeqCst)),
        Cancellation::new(),
    ))
    .unwrap();

    let execution = &restored.state().nodes["agent"].executions[&17_031];
    assert_eq!(execution.attempts.len(), 1);
    assert_eq!(execution.attempts[0].idx, 0);
    assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
    assert_eq!(
        execution.messages.0[0].content[0].data["value"],
        serde_json::json!("restored_scene")
    );
    assert_eq!(allocated.load(Ordering::SeqCst), 0);
}

#[test]
fn checkpoint_recovery_resumes_running_node_without_retry_budget() {
    let mut builder = Builder::new("checkpoint-waiting-no-retry");
    builder.add_node("answer", crate::node::Echo).unwrap();
    builder.add_edge(START, "answer").unwrap();
    builder.add_edge("answer", END).unwrap();
    let workflow = builder.compile().unwrap();

    let mut runtime = Runtime::new(1_702, workflow.clone()).unwrap();
    runtime.state_mut().initial = Some(Messages::default());
    runtime
        .start_batch(vec![("answer".into(), 17_021, Vec::new())])
        .unwrap();
    let snapshot = runtime.snapshot().unwrap();

    let mut restored = Runtime::from_snapshot(1_702, workflow, snapshot).unwrap();
    let allocated = Arc::new(AtomicI64::new(0));
    let allocated_for_run = Arc::clone(&allocated);
    let result = block_on(restored.execute(
        Messages::default(),
        Arc::new(move || allocated_for_run.fetch_add(1, Ordering::SeqCst)),
        Cancellation::new(),
    ))
    .unwrap();

    assert!(result.is_empty());
    let execution = &restored.state().nodes["answer"].executions[&17_021];
    assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
    assert_eq!(execution.attempts.len(), 1);
    assert_eq!(execution.attempts[0].status, node::AttemptStatus::Succeeded);
    assert_eq!(allocated.load(Ordering::SeqCst), 0);
}

#[test]
fn run_waits_at_a_barrier_instead_of_activating_downstream() {
    let mut builder = Builder::new("wait");
    builder.add_node("a", crate::node::Echo).unwrap();
    builder.add_node("b", crate::node::Echo).unwrap();
    builder.add_edge(START, "a").unwrap();
    builder.add_edge("a", "b").unwrap();
    builder.add_edge("b", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(12, workflow).unwrap();
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let ids = Arc::new(|| 201);
    assert_eq!(
        block_on(runtime.execute(Messages::default(), ids, cancellation))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(runtime.state().status, Status::Cancelled);
}

#[test]
fn resolved_wait_reuses_the_same_execution_id_and_attempt() {
    let mut builder = Builder::new("resume-wait");
    builder
        .add_node("approval", WaitOnce(AtomicBool::new(false)))
        .unwrap();
    builder.add_edge(START, "approval").unwrap();
    builder.add_edge("approval", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(13, workflow).unwrap();
    let mut request = crate::Message::new(crate::components::message::Role::User);
    request
        .content
        .push(crate::components::model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                serde_json::json!({"value":"request"}),
            ),
        });
    let initial = Messages::new([request]);
    let ids = Arc::new(AtomicI64::new(700));
    let first = block_on(runtime.execute(
        initial.clone(),
        Arc::new({
            let ids = Arc::clone(&ids);
            move || ids.fetch_add(1, Ordering::AcqRel)
        }),
        Cancellation::new(),
    ));
    assert_eq!(first.unwrap_err().code, "WAITING");
    let mut approved = crate::Message::new(crate::components::message::Role::User);
    approved
        .content
        .push(crate::components::model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                serde_json::json!({"value":"approved"}),
            ),
        });
    let approved = Messages::new([approved]);
    runtime.resume("approval", approved.clone()).unwrap();
    runtime.resume("approval", approved).unwrap();
    let mut changed = crate::Message::new(crate::components::message::Role::User);
    changed
        .content
        .push(crate::components::model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                serde_json::json!({"value":"changed"}),
            ),
        });
    assert_eq!(
        runtime
            .resume("approval", Messages::new([changed]))
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    let output = block_on(runtime.execute(
        initial,
        Arc::new({
            let ids = Arc::clone(&ids);
            move || ids.fetch_add(1, Ordering::AcqRel)
        }),
        Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(
        output.as_slice()[0].content[0].data,
        crate::components::model::content::part::object(serde_json::json!({"value":"approved"}))
    );
    let execution = &runtime.state().nodes["approval"].executions[&700];
    assert_eq!(execution.attempts.len(), 1);
    assert_eq!(execution.attempts[0].idx, 0);
    assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
    assert_eq!(ids.load(Ordering::Acquire), 701);
}

#[test]
fn fan_out_relation_activates_one_target_execution() {
    let mut builder = Builder::new("fan-out");
    builder.add_node("source", crate::node::Echo).unwrap();
    builder.add_node("target", crate::node::Echo).unwrap();
    builder.add_edge(START, "source").unwrap();
    builder.add_fan_out_edges("source", "target").unwrap();
    builder.add_edge("target", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(14, workflow).unwrap();
    let ids = Arc::new(AtomicI64::new(800));
    block_on(runtime.execute(
        Messages::default(),
        Arc::new({
            let ids = Arc::clone(&ids);
            move || ids.fetch_add(1, Ordering::AcqRel)
        }),
        Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(runtime.state().nodes["target"].executions.len(), 1);
    assert_eq!(
        runtime.state().nodes["target"].executions[&801].from[0].exec_id,
        800
    );
}

#[test]
fn loop_relation_uses_the_declared_condition_and_new_execution_ids() {
    let mut builder = Builder::new("loop");
    builder.add_node("first", crate::node::Echo).unwrap();
    builder.add_node("second", crate::node::Echo).unwrap();
    builder.add_edge(START, "first").unwrap();
    builder.add_edge("first", "second").unwrap();
    builder.add_edge("second", END).unwrap();
    let enabled = Arc::new(AtomicBool::new(true));
    let condition = Arc::clone(&enabled);
    builder
        .add_loop_edge("second", "first", move |_| {
            condition.swap(false, Ordering::AcqRel)
        })
        .unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(15, workflow).unwrap();
    let ids = Arc::new(AtomicI64::new(900));
    block_on(runtime.execute(
        Messages::default(),
        Arc::new({
            let ids = Arc::clone(&ids);
            move || ids.fetch_add(1, Ordering::AcqRel)
        }),
        Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(runtime.state().nodes["first"].executions.len(), 2);
    assert_eq!(runtime.state().nodes["second"].executions.len(), 2);
    assert_eq!(
        runtime.state().nodes["first"].executions[&902].from[0].exec_id,
        901
    );
}

#[test]
fn join_failure_waits_for_all_sources_and_activates_handler() {
    let mut builder = Builder::new("join-failure");
    builder
        .add_node("failed", |_input: Messages, _| async {
            Err::<Messages, crate::Error>(crate::Error::new("FAILED", "boom"))
        })
        .unwrap();
    builder.add_node("ok", crate::node::Echo).unwrap();
    builder.add_node("handler", crate::node::Echo).unwrap();
    builder.add_edge(START, "failed").unwrap();
    builder.add_edge(START, "ok").unwrap();
    builder.add_join_edge(["failed", "ok"], "handler").unwrap();
    builder
        .add_failure_edges(["failed", "ok"], ["handler"], "join_failed")
        .unwrap();
    builder.add_edge("handler", END).unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(16, workflow).unwrap();
    let ids = Arc::new(AtomicI64::new(1_000));
    block_on(runtime.execute(
        Messages::default(),
        Arc::new({
            let ids = Arc::clone(&ids);
            move || ids.fetch_add(1, Ordering::AcqRel)
        }),
        Cancellation::new(),
    ))
    .unwrap();
    let handler = runtime.state().nodes["handler"]
        .executions
        .values()
        .next()
        .unwrap();
    assert_eq!(handler.from.len(), 2);
    assert_eq!(handler.status, node::ExecutionStatus::Succeeded);
}

#[test]
fn runtime_retries_only_transient_node_errors() {
    let mut builder = Builder::new("retry");
    builder
        .add_node("retry", Flaky(AtomicBool::new(false)))
        .unwrap();
    builder.add_edge(START, "retry").unwrap();
    builder.add_edge("retry", END).unwrap();
    builder
        .set_retry(
            "retry",
            RetryPolicy {
                max_retries: 1,
                min_delay_ms: 0,
                max_delay_ms: 0,
            },
        )
        .unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(17, workflow).unwrap();
    block_on(runtime.execute(Messages::default(), Arc::new(|| 1_100), Cancellation::new()))
        .unwrap();
    let execution = &runtime.state().nodes["retry"].executions[&1_100];
    assert_eq!(execution.attempts.len(), 2);
    assert_eq!(execution.attempts[0].status, node::AttemptStatus::Failed);
    assert_eq!(execution.attempts[1].status, node::AttemptStatus::Succeeded);
}

#[test]
fn runtime_tool_retry_skips_successful_call_ids() {
    let first_calls = Arc::new(AtomicI64::new(0));
    let second_calls = Arc::new(AtomicI64::new(0));
    let mut builder = Builder::new("tool-retry-dedup");
    builder
        .add_node(
            "tools",
            crate::runtime::workflow::tool::ToolNode::new(RetryTool {
                definition: workflow_tool_definition("first"),
                calls: first_calls.clone(),
                fail_first: false,
            })
            .tool(RetryTool {
                definition: workflow_tool_definition("second"),
                calls: second_calls.clone(),
                fail_first: true,
            }),
        )
        .unwrap();
    builder.add_edge(START, "tools").unwrap();
    builder.add_edge("tools", END).unwrap();
    builder
        .set_retry(
            "tools",
            RetryPolicy {
                max_retries: 1,
                min_delay_ms: 0,
                max_delay_ms: 0,
            },
        )
        .unwrap();
    let workflow = builder.compile().unwrap();
    let mut runtime = Runtime::new(18, workflow).unwrap();
    block_on(runtime.execute(
        workflow_tool_input(),
        Arc::new(|| 1_200),
        Cancellation::new(),
    ))
    .unwrap();

    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert_eq!(second_calls.load(Ordering::SeqCst), 2);
    let execution = &runtime.state().nodes["tools"].executions[&1_200];
    assert_eq!(execution.attempts.len(), 2);
    assert_eq!(execution.attempts[0].messages.as_slice().len(), 1);
    assert_eq!(execution.attempts[1].messages.as_slice().len(), 2);
}

fn retry_workflow(calls: Arc<AtomicI64>, policy: RetryPolicy) -> CompiledWorkflow {
    let mut builder = Builder::new("async-retry");
    builder
        .add_node("retry", move |input: Messages, _: Cancellation| {
            let calls = calls.clone();
            async move {
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    Err(Error::new("TIMEOUT", "first call failed"))
                } else {
                    Ok(input)
                }
            }
        })
        .unwrap();
    builder.add_edge(START, "retry").unwrap();
    builder.add_edge("retry", END).unwrap();
    builder.set_retry("retry", policy).unwrap();
    builder.compile().unwrap()
}

#[tokio::test(start_paused = true)]
async fn runtime_waits_for_backoff_before_starting_the_next_attempt() {
    let calls = Arc::new(AtomicI64::new(0));
    let workflow = retry_workflow(
        calls.clone(),
        RetryPolicy {
            max_retries: 1,
            min_delay_ms: 100,
            max_delay_ms: 100,
        },
    );
    let mut runtime = Runtime::new(90, workflow).unwrap();
    let mut run =
        Box::pin(runtime.execute(Messages::default(), Arc::new(|| 900), Cancellation::new()));
    assert!(futures_util::poll!(run.as_mut()).is_pending());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    tokio::time::advance(std::time::Duration::from_millis(99)).await;
    assert!(futures_util::poll!(run.as_mut()).is_pending());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    run.await.unwrap();
    let execution = &runtime.state().nodes["retry"].executions[&900];
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(execution.attempts.len(), 2);
    assert_eq!(
        execution.attempts[0].error.as_ref().unwrap().code,
        "TIMEOUT"
    );
    assert_eq!(execution.attempts[1].status, node::AttemptStatus::Succeeded);
}

#[tokio::test(start_paused = true)]
async fn cancellation_during_backoff_preserves_the_failed_attempt() {
    let calls = Arc::new(AtomicI64::new(0));
    let workflow = retry_workflow(
        calls.clone(),
        RetryPolicy {
            max_retries: 1,
            min_delay_ms: 100,
            max_delay_ms: 100,
        },
    );
    let mut runtime = Runtime::new(90, workflow).unwrap();
    let cancellation = Cancellation::new();
    let mut run =
        Box::pin(runtime.execute(Messages::default(), Arc::new(|| 900), cancellation.clone()));
    assert!(futures_util::poll!(run.as_mut()).is_pending());
    cancellation.cancel();
    assert_eq!(run.await.unwrap_err().code, "CANCELLED");
    let execution = &runtime.state().nodes["retry"].executions[&900];
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(execution.status, node::ExecutionStatus::Cancelled);
    assert!(execution.error.is_none());
    assert_eq!(execution.attempts.len(), 1);
    assert_eq!(execution.attempts[0].status, node::AttemptStatus::Failed);
    assert_eq!(
        execution.attempts[0].error.as_ref().unwrap().code,
        "TIMEOUT"
    );
}

#[test]
fn retry_without_a_timer_returns_an_error_without_replaying_the_node() {
    let calls = Arc::new(AtomicI64::new(0));
    let workflow = retry_workflow(
        calls.clone(),
        RetryPolicy {
            max_retries: 1,
            min_delay_ms: 100,
            max_delay_ms: 100,
        },
    );
    let mut runtime = Runtime::new(90, workflow).unwrap();
    let error =
        block_on(runtime.execute(Messages::default(), Arc::new(|| 900), Cancellation::new()))
            .unwrap_err();
    assert_eq!(error.code, "UNAVAILABLE");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let execution = &runtime.state().nodes["retry"].executions[&900];
    assert_eq!(execution.attempts.len(), 1);
    assert_eq!(
        execution.attempts[0].error.as_ref().unwrap().code,
        "TIMEOUT"
    );
}

#[test]
fn resumed_attempt_does_not_reset_the_execution_retry_budget() {
    let calls = Arc::new(AtomicI64::new(0));
    let policy = RetryPolicy {
        max_retries: 1,
        min_delay_ms: 0,
        max_delay_ms: 0,
    };
    let workflow = retry_workflow(calls.clone(), policy.clone());
    let invocation = block_on(invoke_with_retry(
        Target::Node(workflow.executor("retry").unwrap()),
        serde_json::to_value(Messages::default()).unwrap(),
        View::default(),
        Cancellation::new(),
        policy,
        Arc::new(|| 900),
        90,
        1,
        node::Attempt::default(),
        None,
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(invocation.attempts.len(), 1);
    assert_eq!(invocation.attempts[0].idx, 1);
    assert!(matches!(invocation.outcome, Outcome::Failed(_)));
}

#[test]
fn checkpoint_recovery_toolnode_reuses_committed_call_ids() {
    let first_calls = Arc::new(AtomicI64::new(0));
    let second_calls = Arc::new(AtomicI64::new(0));
    let mut builder = Builder::new("tool-checkpoint-recovery");
    builder
        .add_node(
            "tools",
            crate::runtime::workflow::tool::ToolNode::new(RetryTool {
                definition: workflow_tool_definition("first"),
                calls: first_calls.clone(),
                fail_first: false,
            })
            .tool(RetryTool {
                definition: workflow_tool_definition("second"),
                calls: second_calls.clone(),
                fail_first: false,
            }),
        )
        .unwrap();
    builder.add_edge(START, "tools").unwrap();
    builder.add_edge("tools", END).unwrap();
    let workflow = builder.compile().unwrap();

    let mut committed = crate::Message::function(serde_json::json!({
        "tool": "first",
        "ok": true,
    }));
    committed
        .metadata
        .insert("call_id".into(), serde_json::json!("call_a"));
    let invocation = block_on(invoke_with_retry(
        Target::Node(workflow.executor("tools").unwrap()),
        serde_json::to_value(workflow_tool_input()).unwrap(),
        View::default(),
        Cancellation::new(),
        RetryPolicy::default(),
        Arc::new(|| 1_300),
        19,
        0,
        node::Attempt {
            messages: Messages::new([committed]),
            ..Default::default()
        },
        None,
    ));

    assert!(matches!(invocation.outcome, Outcome::Complete(_)));
    assert_eq!(first_calls.load(Ordering::SeqCst), 0);
    assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    assert_eq!(invocation.attempts[0].messages.as_slice().len(), 2);
}

fn recovery_tools(calls: &[Arc<AtomicI64>; 3], parallel: bool, fail_first: bool) -> tool::ToolNode {
    let node = tool::ToolNode::new(RetryTool {
        definition: workflow_tool_definition("first"),
        calls: calls[0].clone(),
        fail_first: false,
    })
    .tool(RetryTool {
        definition: workflow_tool_definition("second"),
        calls: calls[1].clone(),
        fail_first,
    })
    .tool(RetryTool {
        definition: workflow_tool_definition("third"),
        calls: calls[2].clone(),
        fail_first: false,
    });
    if parallel { node.parallel() } else { node }
}

fn recovery_workflow(node: tool::ToolNode) -> CompiledWorkflow {
    let mut builder = Builder::new("tool-process-recovery");
    builder.add_node("tools", node).unwrap();
    builder.add_edge(START, "tools").unwrap();
    builder.add_edge("tools", END).unwrap();
    builder
        .set_retry(
            "tools",
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
fn tool_checkpoint_recovers_in_a_fresh_process_without_replaying_successes() {
    const FIXTURE: &str = "HEART_TOOL_RECOVERY_FIXTURE";
    const MODE: &str = "HEART_TOOL_RECOVERY_PARALLEL";
    const RETRY: &str = "HEART_TOOL_RECOVERY_RETRY";
    let calls = std::array::from_fn(|_| Arc::new(AtomicI64::new(0)));
    if let Some(path) = std::env::var_os(FIXTURE) {
        let parallel = std::env::var_os(MODE).is_some();
        let prior_attempt = std::env::var_os(RETRY).is_some();
        let snapshot: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let workflow = recovery_workflow(recovery_tools(&calls, parallel, !prior_attempt));
        let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
        let mut runtime = Runtime::from_snapshot(19, workflow, snapshot)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        let input = runtime.state().initial.clone().unwrap();
        let output = block_on(runtime.execute(
            input.clone(),
            Arc::new(|| panic!("recovery must reuse the existing exec_id")),
            Cancellation::new(),
        ))
        .unwrap();

        assert_eq!(calls[0].load(Ordering::SeqCst), 0);
        let retries = if prior_attempt { 1 } else { 2 };
        assert_eq!(calls[1].load(Ordering::SeqCst), retries);
        assert_eq!(calls[2].load(Ordering::SeqCst), i64::from(!parallel));
        let execution = &runtime.state().nodes["tools"].executions[&1_300];
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(execution.attempts[0].idx, 0);
        assert_eq!(execution.attempts[0].status, node::AttemptStatus::Failed);
        assert_eq!(
            execution.attempts[0].error.as_ref().unwrap().code,
            "TIMEOUT"
        );
        assert_eq!(execution.attempts[1].idx, 1);
        assert_eq!(execution.attempts[1].status, node::AttemptStatus::Succeeded);
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert!(execution.error.is_none());
        let ids = output
            .as_slice()
            .iter()
            .map(|message| message.metadata["call_id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["call_a", "call_b", "call_c"]);
        let terminal = block_on(checkpoint.load(19, &Cancellation::new())).unwrap();
        assert_eq!(terminal, runtime.snapshot().unwrap());
        // A second process-local restore of the saved terminal result is
        // a read, not another tool invocation or attempt.
        let mut terminal_runtime = Runtime::from_snapshot(
            19,
            recovery_workflow(recovery_tools(&calls, parallel, !prior_attempt)),
            terminal,
        )
        .unwrap();
        assert_eq!(
            block_on(terminal_runtime.execute(
                input,
                Arc::new(|| panic!("terminal restore must not allocate an id")),
                Cancellation::new(),
            ))
            .unwrap(),
            output
        );
        assert_eq!(calls[1].load(Ordering::SeqCst), retries);
        return;
    }

    for (parallel, prior_attempt) in [(false, false), (true, false), (false, true), (true, true)] {
        let calls = std::array::from_fn(|_| Arc::new(AtomicI64::new(0)));
        let tools = recovery_tools(&calls, parallel, true);
        let mut input = workflow_tool_input();
        input.0[0].content.push(crate::model::content::part::Part {
            r#type: "tool_call".into(),
            data: crate::model::content::part::object(serde_json::json!({
                "call_id": "call_c", "name": "third", "arguments": {}
            })),
        });
        let partial = block_on(tools.call_with_recovery(
            input.clone(),
            Cancellation::new(),
            node::Attempt::default(),
        ))
        .unwrap_err();
        assert_eq!(partial.error.code, "TIMEOUT");
        assert_eq!(partial.completed.0.len(), if parallel { 2 } else { 1 });
        let mut runtime = Runtime::new(19, recovery_workflow(tools)).unwrap();
        runtime.state_mut().initial = Some(input);
        runtime
            .start_batch(vec![("tools".into(), 1_300, vec![])])
            .unwrap();
        // Model the saved in-flight scene, not an already terminal failed
        // run: these sub-call facts are present, but normal output and
        // downstream activation have not been committed.
        runtime
            .state_mut()
            .nodes
            .get_mut("tools")
            .unwrap()
            .executions
            .get_mut(&1_300)
            .unwrap()
            .attempts[0]
            .messages = partial.completed;
        if prior_attempt {
            let execution = runtime
                .state_mut()
                .nodes
                .get_mut("tools")
                .unwrap()
                .executions
                .get_mut(&1_300)
                .unwrap();
            execution.attempts[0].status = node::AttemptStatus::Failed;
            execution.attempts[0].error = Some(partial.error);
            // A crash after starting the next Attempt must not lose
            // successful calls retained only in the previous Attempt.
            execution.attempts.push(node::Attempt {
                idx: 1,
                ..Default::default()
            });
        }
        assert!(
            runtime.state().nodes["tools"].executions[&1_300]
                .messages
                .0
                .is_empty()
        );
        let checkpoint = crate::core::checkpoint::InMemory::new();
        block_on(runtime.save_checkpoint(&checkpoint, &Cancellation::new())).unwrap();
        let saved = block_on(checkpoint.load(19, &Cancellation::new())).unwrap();
        let path = std::env::temp_dir().join(format!(
            "heart-tool-recovery-{}-{}-{parallel}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, serde_json::to_vec(&saved).unwrap()).unwrap();
        drop(runtime);
        let mut process = std::process::Command::new(std::env::current_exe().unwrap());
        process.args([
                "--exact",
                "runtime::workflow::tests::tool_checkpoint_recovers_in_a_fresh_process_without_replaying_successes",
                "--nocapture",
            ]).env(FIXTURE, &path).env_remove(MODE).env_remove(RETRY);
        if parallel {
            process.env(MODE, "1");
        }
        if prior_attempt {
            process.env(RETRY, "1");
        }
        let result = process.output();
        std::fs::remove_file(path).unwrap();
        let result = result.unwrap();
        assert!(
            result.status.success(),
            "child recovery failed: {}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
