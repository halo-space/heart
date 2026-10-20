//! One lifecycle implementation, actual per-execution state and native APIs.
use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message, Messages,
    agent::{self, toolkit::Toolkit},
    event::Event,
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
    node::{Node, Signal},
    runtime::workflow::{
        CompiledWorkflow, Runtime, WorkflowBuilder,
        graph::{END, START},
        tool::ToolNode,
    },
    tool::{Call, Evaluation, Tool, ToolDefinition},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
    },
};

type Counts = Arc<Mutex<BTreeMap<String, [usize; 5]>>>;
fn count(counts: &Counts, owner: &str, phase: usize) {
    counts.lock().unwrap().entry(owner.into()).or_default()[phase] += 1;
}
struct Operation {
    definition: ToolDefinition,
    counts: Counts,
    pause: AtomicBool,
}
#[derive(Clone)]
struct SharedTool(Arc<Operation>);
impl Tool for SharedTool {
    fn definition(&self) -> &ToolDefinition {
        &self.0.definition
    }
    async fn before(&self, call: &mut Call, _: &Cancellation) -> Result<(), Error> {
        count(&self.0.counts, call.arguments["owner"].as_str().unwrap(), 0);
        assert_eq!(call.arguments["n"], 1);
        call.arguments.insert("n".into(), json!(2));
        Ok(())
    }
    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        count(&self.0.counts, call.arguments["owner"].as_str().unwrap(), 1);
        assert_eq!(call.arguments["n"], 2);
        Ok(Message::function(
            json!({"owner":call.arguments["owner"],"n":2}),
        ))
    }
    async fn after(
        &self,
        call: &Call,
        message: &mut Message,
        _: &Cancellation,
    ) -> Result<(), Error> {
        let owner = call.arguments["owner"].as_str().unwrap();
        count(&self.0.counts, owner, 2);
        assert_eq!(message.values["owner"], owner);
        assert_eq!(message.values["n"], 2);
        if owner == "A" && self.0.pause.load(Ordering::SeqCst) {
            return Err(Error::new("WAITING", "A after waits"));
        }
        message.values.insert("n".into(), json!(3));
        Ok(())
    }
    async fn eval(
        &self,
        call: &Call,
        message: &Message,
        _: &Cancellation,
    ) -> Result<Evaluation, Error> {
        count(&self.0.counts, call.arguments["owner"].as_str().unwrap(), 3);
        assert_eq!(message.values["n"], 3);
        Ok(Evaluation::passed())
    }
}
#[derive(Clone)]
struct SharedModel(Counts);
impl Chat for SharedModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let owner = request
            .messages
            .as_slice()
            .iter()
            .find(|message| message.role == Role::System)
            .unwrap()
            .content[0]
            .data["value"]
            .as_str()
            .unwrap();
        count(&self.0, owner, 4);
        let done = request.messages.as_slice().iter().any(|message| {
            message.values.get("owner") == Some(&json!(owner))
                && message.values.get("n") == Some(&json!(3))
        });
        let mut message = Message::new(Role::Assistant);
        if done {
            message.values.insert("answer".into(), json!(owner));
        } else {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data:
                    json!({"call_id":"same_call","name":"shared","arguments":{"owner":owner,"n":1}})
                        .as_object()
                        .unwrap()
                        .clone(),
            });
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                input: Some(1),
                output: Some(1),
                total: Some(2),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}
fn shared() -> SharedTool {
    SharedTool(Arc::new(Operation {definition:ToolDefinition {name:"shared".into(),description:"shared actual instance".into(),parameters:json!({"type":"object","properties":{"owner":{"type":"string"},"n":{"type":"integer"}},"required":["owner","n"],"additionalProperties":false}).as_object().unwrap().clone()},counts:Default::default(),pause:AtomicBool::new(true)}))
}
fn graph(react: bool, tool: SharedTool) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("shared-lifecycle-owners");
    let model = SharedModel(tool.0.counts.clone());
    for (id, owner) in [("a", "A"), ("b", "B")] {
        let toolkit = Toolkit::new().tools(tool.clone());
        if react {
            builder
                .add_agent_node(
                    id,
                    agent::react()
                        .model(model.clone())
                        .system_prompt(owner)
                        .toolkit(toolkit)
                        .build()
                        .unwrap(),
                )
                .unwrap();
        } else {
            builder
                .add_agent_node(
                    id,
                    agent::chat()
                        .model(model.clone())
                        .system_prompt(owner)
                        .toolkit(toolkit)
                        .build()
                        .unwrap(),
                )
                .unwrap();
        }
        builder.add_edge(START, id).unwrap();
    }
    builder.add_node("merge", halo_agents::node::Echo).unwrap();
    builder.add_join(["a", "b"], "merge").unwrap();
    builder.add_edge("merge", END).unwrap();
    builder.compile().unwrap()
}
static IDS: AtomicI64 = AtomicI64::new(500_000);
fn next_id() -> i64 {
    IDS.fetch_add(1, Ordering::SeqCst)
}
fn input() -> Messages {
    let mut message = Message::new(Role::User);
    message.metadata = json!({"chat_id":3001,"session_id":4001,"agent_id":5001})
        .as_object()
        .unwrap()
        .clone();
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"original shared task"})
            .as_object()
            .unwrap()
            .clone(),
    });
    Messages::new([message])
}
fn scene<'a>(snapshot: &'a Value, node: &str) -> &'a Value {
    &snapshot["nodes"][node]["executions"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()["attempts"][0]["agent"]
}

#[tokio::test]
async fn shared_model_tool_and_call_id_keep_distinct_workflow_node_scenes() {
    for react in [false, true] {
        let tool = shared();
        let runtime = Runtime::new(3001, graph(react, tool.clone()))
            .unwrap()
            .with_max_concurrency(2)
            .unwrap();
        let events = runtime
            .run(input(), Arc::new(next_id), Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(
            matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
            "{events:?}"
        );
        let saved = runtime.snapshot().unwrap();
        assert_eq!(
            scene(&saved, "a")["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["stage"],
            "after"
        );
        assert_eq!(
            scene(&saved, "a")["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["message"]["values"]["owner"],
            "A"
        );
        assert_eq!(
            scene(&saved, "b")["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["stage"],
            "committed"
        );
        assert_eq!(
            scene(&saved, "b")["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["message"]["values"]["owner"],
            "B"
        );
        assert_eq!(tool.0.counts.lock().unwrap()["A"], [1, 1, 1, 0, 1]);
        assert_eq!(tool.0.counts.lock().unwrap()["B"], [1, 1, 1, 1, 2]);
        let b = saved["nodes"]["b"].clone();
        tool.0.pause.store(false, Ordering::SeqCst);
        let mut restored = Runtime::from_snapshot(
            3001,
            graph(react, tool.clone()),
            serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap(),
        )
        .unwrap();
        restored
            .resume(
                "a",
                Messages::new([Message::function(json!({"continue":true}))]),
            )
            .unwrap();
        let events = restored
            .run(input(), Arc::new(next_id), Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        let done = restored.snapshot().unwrap();
        assert_eq!(done["status"], "succeeded");
        assert_eq!(
            done["nodes"]["b"], b,
            "A restoration must not rewrite successful B"
        );
        assert_eq!(tool.0.counts.lock().unwrap()["A"], [1, 1, 2, 1, 2]);
        assert_eq!(tool.0.counts.lock().unwrap()["B"], [1, 1, 1, 1, 2]);
        assert_eq!(done["usage"]["total"], 8);
    }
}

#[derive(Clone)]
struct Plain {
    definition: ToolDefinition,
    calls: Arc<AtomicUsize>,
}
impl Tool for Plain {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Message::function(json!({"n":call.arguments["n"]})))
    }
}
#[tokio::test]
async fn absent_hooks_pass_through_direct_tool_toolkit_and_workflow() {
    let calls = Arc::new(AtomicUsize::new(0));
    let tool = Plain {
        definition: ToolDefinition {
            name: "plain".into(),
            description: "no custom lifecycle hooks".into(),
            parameters:
                json!({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]})
                    .as_object()
                    .unwrap()
                    .clone(),
        },
        calls: calls.clone(),
    };
    let call = Call {
        call_id: "plain_1".into(),
        name: "plain".into(),
        arguments: json!({"n":7}).as_object().unwrap().clone(),
    };
    let c = Cancellation::new();
    let (direct, eval) = halo_agents::tool::execute(&tool, call.clone(), &c)
        .await
        .unwrap();
    assert!(eval.passed);
    let (toolkit, eval) = Toolkit::new()
        .tools(tool.clone())
        .execute(call.clone(), &c)
        .await
        .unwrap();
    assert!(eval.passed);
    assert_eq!(direct, toolkit);
    let mut selection = Message::new(Role::Assistant);
    selection.content.push(Part {
        r#type: "tool_call".into(),
        data: serde_json::to_value(&call)
            .unwrap()
            .as_object()
            .unwrap()
            .clone(),
    });
    let Signal::Complete(messages) = ToolNode::new(tool)
        .call(Messages::new([selection]), c)
        .await
        .unwrap()
    else {
        panic!("default Hooks cannot wait")
    };
    assert_eq!(messages.as_slice()[0].values, direct.values);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}
