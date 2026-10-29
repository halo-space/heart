//! Live-provider lifecycle ownership: actual per-execution state and native APIs.
use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::message::Role,
    agent::model::chat,
    agent::model::content::part::Part,
    agent::node::{Node, Signal},
    agent::provider::vllm::{Config, Vllm, chat::Chat as VllmChat},
    agent::tool::{Call, Evaluation, Tool, ToolDefinition},
    agent::workflow::{
        CompiledWorkflow, Runtime, WorkflowBuilder,
        graph::{END, START},
        tool::ToolNode,
    },
    agent::{self},
};
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
};

type Counts = Arc<Mutex<[usize; 4]>>;
fn count(counts: &Counts, phase: usize) {
    counts.lock().unwrap()[phase] += 1;
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
        count(&self.0.counts, 0);
        assert_eq!(call.arguments["n"], 1);
        call.arguments.insert("n".into(), json!(2));
        Ok(())
    }
    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        count(&self.0.counts, 1);
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
        count(&self.0.counts, 2);
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
        _call: &Call,
        message: &Message,
        _: &Cancellation,
    ) -> Result<Evaluation, Error> {
        count(&self.0.counts, 3);
        assert_eq!(message.values["n"], 3);
        Ok(Evaluation::passed())
    }
}
fn shared() -> SharedTool {
    SharedTool(Arc::new(Operation {definition:ToolDefinition {name:"shared".into(),description:"shared actual instance".into(),parameters:json!({"type":"object","properties":{"owner":{"type":"string"},"n":{"type":"integer"}},"required":["owner","n"],"additionalProperties":false}).as_object().unwrap().clone()},counts:Default::default(),pause:AtomicBool::new(true)}))
}
fn graph(react: bool, tool: SharedTool, model: VllmChat) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("shared-lifecycle-owners");
    for (id, owner) in [("a", "A"), ("b", "B")] {
        let capabilities = tool.clone();
        if react {
            builder
                .add_agent_node(
                    id,
                    agent::react()
                        .model(model.clone())
                    .system_prompt(format!(
                        "You are workflow node {owner}. You must call the shared tool exactly once with owner set to the exact string {owner:?} and n set to 1. Do not answer before the tool call. After receiving the tool result, provide a brief answer that includes the owner."
                    ))
                        .tools(capabilities)
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
                    .system_prompt(format!(
                        "You are workflow node {owner}. You must call the shared tool exactly once with owner set to the exact string {owner:?} and n set to 1. Do not answer before the tool call. After receiving the tool result, provide a brief answer that includes the owner."
                    ))
                        .tools(capabilities)
                        .build()
                        .unwrap(),
                )
                .unwrap();
        }
        builder.add_edge(START, id).unwrap();
    }
    builder.add_node("merge", halo::agent::node::Echo).unwrap();
    builder.add_join(["a", "b"], "merge").unwrap();
    builder.add_edge("merge", END).unwrap();
    builder.compile().unwrap()
}
fn live_model() -> Result<VllmChat, Error> {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL")
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_VLLM_BASE_URL is required"))?,
        api_key: std::env::var("HEART_VLLM_API_KEY")
            .map_err(|_| Error::new("AUTHENTICATION", "HEART_VLLM_API_KEY is required"))?,
        ..Config::default()
    })?;
    let model_name = std::env::var("HEART_VLLM_MODEL")
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_VLLM_MODEL is required"))?;
    provider.chat(
        chat::Config::new(model_name).options(serde_json::Map::from_iter([(
            "chat_template_kwargs".into(),
            json!({ "enable_thinking": false }),
        )])),
    )
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
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn shared_model_tool_and_call_id_keep_distinct_workflow_node_scenes() {
    for react in [false, true] {
        let tool = shared();
        let model = live_model().expect("construct live VLLM Chat");
        let runtime = Runtime::new(3001, graph(react, tool.clone(), model.clone()))
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
        assert_eq!(*tool.0.counts.lock().unwrap(), [2, 2, 2, 1]);
        let b = saved["nodes"]["b"].clone();
        tool.0.pause.store(false, Ordering::SeqCst);
        let mut restored = Runtime::from_snapshot(
            3001,
            graph(react, tool.clone(), model),
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
        assert_eq!(*tool.0.counts.lock().unwrap(), [2, 2, 3, 2]);
        assert!(done["usage"]["total"].as_u64().unwrap_or_default() > 0);
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
async fn absent_hooks_pass_through_direct_tool_mounted_capabilities_and_workflow() {
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
    let (direct, eval) = halo::agent::tool::execute(&tool, call.clone(), &c)
        .await
        .unwrap();
    assert!(eval.passed);
    let (capabilities, eval) = halo::agent::tool::execute(&tool, call.clone(), &c)
        .await
        .unwrap();
    assert!(eval.passed);
    assert_eq!(direct, capabilities);
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
        .call(Messages::new([selection]), &c)
        .await
        .unwrap()
    else {
        panic!("default Hooks cannot wait")
    };
    assert_eq!(messages.as_slice()[0].values, direct.values);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}
