use std::{
    future::pending,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures_executor::block_on;
use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message, Messages,
    agent::{self, Agent, Input, toolkit::Toolkit},
    core,
    event::Event,
    mcp::{Definition as McpDefinition, Server},
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
    skill::{Definition, Skill},
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::json;

#[derive(Clone, Copy)]
enum Mode {
    Echo,
    Error,
    Pending,
    Cancel,
}

struct Echo {
    definition: Definition,
    calls: Arc<AtomicUsize>,
    mode: Mode,
}

fn skill(name: &str, mode: Mode) -> (Echo, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    (
        Echo {
            definition: Definition {
                name: name.into(),
                ..Default::default()
            },
            calls: calls.clone(),
            mode,
        },
        calls,
    )
}

impl Skill for Echo {
    fn definition(&self) -> &Definition {
        &self.definition
    }
    async fn run(&self, input: Messages, cancellation: &Cancellation) -> Result<Messages, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Echo => Ok(input),
            Mode::Error => Err(Error::new("BUSINESS_ERROR", "original").with_details("unchanged")),
            Mode::Pending => pending().await,
            Mode::Cancel => {
                cancellation.cancel();
                Ok(input)
            }
        }
    }
}

struct Remote {
    definition: McpDefinition,
    lists: Arc<AtomicUsize>,
}

fn remote() -> (Remote, Arc<AtomicUsize>) {
    let lists = Arc::new(AtomicUsize::new(0));
    (
        Remote {
            definition: McpDefinition {
                name: "server".into(),
                ..Default::default()
            },
            lists: lists.clone(),
        },
        lists,
    )
}

fn tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "research".into(),
        description: "research".into(),
        parameters: serde_json::from_value(json!({"type":"object","properties":{}})).unwrap(),
    }
}

impl Server for Remote {
    fn definition(&self) -> &McpDefinition {
        &self.definition
    }
    async fn list_tools(&self, _: &Cancellation) -> Result<Vec<ToolDefinition>, Error> {
        self.lists.fetch_add(1, Ordering::SeqCst);
        Ok(vec![tool_definition()])
    }
    async fn call(&self, _: Call, _: &Cancellation) -> Result<Message, Error> {
        panic!("remote must not execute")
    }
}

struct Local(ToolDefinition);
impl Tool for Local {
    fn definition(&self) -> &ToolDefinition {
        &self.0
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        panic!("tool must not execute")
    }
}

#[test]
fn registration_is_deferred_and_all_message_fields_are_preserved() {
    let (echo, calls) = skill("research", Mode::Echo);
    let toolkit = Toolkit::new().skills(vec![echo]).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(toolkit.is_empty());
    assert_eq!(toolkit.len(), 0);
    assert!(toolkit.definitions().is_empty());
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "think".into(),
        data: json!({"value":"consider"}).as_object().unwrap().clone(),
    });
    message.values.insert("answer".into(), json!(42));
    message.metadata.insert("chat_id".into(), json!(77));
    message.usage = Some(Usage {
        input: Some(10),
        output: Some(2),
        total: Some(12),
        ..Default::default()
    });
    let input = Messages::new([message, Message::function(json!({"x":1}))]);
    let output = block_on(toolkit.run("research", input.clone(), &Cancellation::new())).unwrap();
    assert_eq!(output, input);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        block_on(toolkit.run("research", Messages::default(), &Cancellation::new()))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn heterogeneous_skills_keep_their_native_contract_and_instructions() {
    let (echo, calls) = skill("echo", Mode::Echo);
    let markdown = "---\nname: research\ndescription: Research\n---\nUse sources.";
    let research =
        core::skill::Skill::from_markdown(markdown, |input: Messages, _: Cancellation| async {
            Ok::<_, Error>(input)
        })
        .unwrap();
    let toolkit = Toolkit::new()
        .skills(vec![echo])
        .unwrap()
        .skills(vec![research])
        .unwrap();
    let input = Message::function(json!({"query":"weather"}));
    let output = block_on(toolkit.run(
        "research",
        Messages::new([input.clone()]),
        &Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(output.0[0].role, Role::System);
    assert_eq!(output.0[0].content[0].data["value"], "Use sources.");
    assert_eq!(output.0[1], input);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(
        block_on(toolkit.run("echo", Messages::default(), &Cancellation::new()))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn missing_skill_does_not_fall_back_to_a_tool_or_discover_mcp() {
    let (remote, lists) = remote();
    let toolkit = Toolkit::new()
        .tool(Local(tool_definition()))
        .mcps(vec![remote]);
    assert_eq!(
        block_on(toolkit.run("research", Messages::default(), &Cancellation::new()))
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert_eq!(lists.load(Ordering::SeqCst), 0);
}

#[test]
fn explicit_skill_dispatch_does_not_discover_unrelated_mcp() {
    let (echo, calls) = skill("local", Mode::Echo);
    let (remote, lists) = remote();
    let toolkit = Toolkit::new()
        .mcps(vec![remote])
        .skills(vec![echo])
        .unwrap();
    block_on(toolkit.run("local", Messages::default(), &Cancellation::new())).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(lists.load(Ordering::SeqCst), 0);
}

#[test]
fn blank_and_duplicate_names_are_rejected_without_calls() {
    let (echo, calls) = skill("  ", Mode::Echo);
    assert_eq!(
        Toolkit::new().skills(vec![echo]).err().unwrap().code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let (first, calls) = skill("research", Mode::Echo);
    let (second, _) = skill("research", Mode::Echo);
    assert_eq!(
        Toolkit::new()
            .skills(vec![first, second])
            .err()
            .unwrap()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let (echo, _) = skill("research", Mode::Echo);
    assert_eq!(
        Toolkit::new()
            .tool(Local(tool_definition()))
            .skills(vec![echo])
            .err()
            .unwrap()
            .code,
        "INVALID_ARGUMENTS"
    );
    let (echo, calls) = skill("research", Mode::Echo);
    let toolkit = Toolkit::new()
        .skills(vec![echo])
        .unwrap()
        .tool(Local(tool_definition()));
    assert_eq!(
        block_on(toolkit.run("research", Messages::default(), &Cancellation::new()))
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

struct Model(Arc<AtomicUsize>);
impl Chat for Model {
    type Stream = stream::Empty<Result<Event, Error>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(stream::empty())
    }
}

#[test]
fn mcp_skill_name_collision_is_rejected_before_model_work() {
    let (echo, calls) = skill("research", Mode::Echo);
    let (remote, lists) = remote();
    let model_calls = Arc::new(AtomicUsize::new(0));
    let toolkit = Toolkit::new()
        .skills(vec![echo])
        .unwrap()
        .mcps(vec![remote]);
    let agent = agent::chat()
        .model(Model(model_calls.clone()))
        .toolkit(toolkit)
        .build()
        .unwrap();
    let error = block_on(agent.run(Input::default(), Cancellation::new()))
        .err()
        .unwrap();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert_eq!(lists.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(model_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn errors_are_direct_and_never_retried() {
    let (echo, calls) = skill("research", Mode::Error);
    let toolkit = Toolkit::new().skills(vec![echo]).unwrap();
    assert_eq!(
        block_on(toolkit.run("research", Messages::default(), &Cancellation::new())).unwrap_err(),
        Error::new("BUSINESS_ERROR", "original").with_details("unchanged")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn precancelled_call_never_enters_skill_and_ready_cancel_never_returns_success() {
    let (echo, calls) = skill("research", Mode::Echo);
    let toolkit = Toolkit::new().skills(vec![echo]).unwrap();
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        block_on(toolkit.run("research", Messages::default(), &cancellation))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let (echo, calls) = skill("research", Mode::Cancel);
    let toolkit = Toolkit::new().skills(vec![echo]).unwrap();
    assert_eq!(
        block_on(toolkit.run("research", Messages::default(), &Cancellation::new()))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn pending_skill_cancellation_wakes_dispatch() {
    let (echo, calls) = skill("research", Mode::Pending);
    let toolkit = Toolkit::new().skills(vec![echo]).unwrap();
    let cancellation = Cancellation::new();
    let token = cancellation.clone();
    let task = tokio::spawn(async move {
        tokio::task::yield_now().await;
        token.cancel();
    });
    assert_eq!(
        tokio::time::timeout(
            Duration::from_secs(1),
            toolkit.run("research", Messages::default(), &cancellation)
        )
        .await
        .unwrap()
        .unwrap_err()
        .code,
        "CANCELLED"
    );
    task.await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn skill_registration_does_not_advertise_model_tools() {
    struct AssertNoTools;
    impl Chat for AssertNoTools {
        type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
        async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
            // ReAct advertises its built-in planning control, not Skills.
            assert_eq!(
                request
                    .tools
                    .iter()
                    .map(|tool| tool.name.as_str())
                    .collect::<Vec<_>>(),
                vec!["plan"]
            );
            assert_eq!(
                request.tool_choice,
                Some(halo_agents::model::chat::ToolChoice::Auto)
            );
            Ok(stream::iter(vec![Ok(Event::Complete {
                message: Message::function(json!({"answer":"done"})),
                usage: None,
                finish_reason: None,
            })]))
        }
    }
    let (echo, calls) = skill("research", Mode::Echo);
    let agent = agent::react()
        .model(AssertNoTools)
        .toolkit(Toolkit::new().skills(vec![echo]).unwrap())
        .build()
        .unwrap();
    block_on(async {
        let output = agent
            .run(Input::default(), Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert_eq!(output.len(), 1);
        assert!(output[0].is_ok());
    });
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
