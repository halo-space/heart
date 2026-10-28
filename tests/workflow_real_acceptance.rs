//! End-to-end Workflow acceptance with a real DeepSeek-backed Agent node.

use std::sync::{
    Arc,
    atomic::{AtomicI64, AtomicUsize, Ordering},
};

use futures_util::StreamExt;
use halo::{
    agent::{
        self, Cancellation, Error, Message, Messages,
        checkpoint::Checkpoint as _,
        event::Event,
        message::Role,
        model::chat,
        model::content::part::Part,
        permission::{Behavior, Decision, Permission, Request as PermissionRequest},
        provider::{
            deepseek::DeepSeek,
            vllm::{Config, Vllm},
        },
        tool::{Call, Tool, ToolDefinition},
    },
    runtime::workflow::{
        Runtime, WorkflowBuilder,
        graph::{END, START},
    },
};
use serde_json::json;

struct ApprovalClock(Arc<AtomicUsize>);

impl Tool for ApprovalClock {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: std::sync::LazyLock<ToolDefinition> =
            std::sync::LazyLock::new(|| ToolDefinition {
                name: "current_time".into(),
                description: "返回当前 Unix 时间戳（秒）".into(),
                parameters: serde_json::from_value(json!({
                    "type":"object",
                    "properties":{},
                    "additionalProperties":false
                }))
                .expect("valid clock schema"),
            });
        &DEFINITION
    }

    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| Error::new("INTERNAL", error.to_string()))?
            .as_secs();
        Ok(Message::function(json!({"unix_seconds":timestamp})))
    }
}

struct AskForApproval;

fn next_workflow_approval_id() -> i64 {
    static NEXT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(8_273_100);
    NEXT.fetch_add(1, Ordering::SeqCst)
}

impl Permission for AskForApproval {
    async fn check(&self, _: &PermissionRequest, _: &Cancellation) -> Result<Decision, Error> {
        Ok(Decision {
            behavior: Behavior::Ask,
            reason: Some("Approve reading the live clock".into()),
        })
    }
}

#[tokio::test]
#[ignore = "requires a real DeepSeek API key"]
async fn workflow_runs_embedded_agent_with_real_model() {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct the real DeepSeek provider");
    let model = provider
        .chat(
            chat::Config::new(
                std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
            )
            .options(serde_json::Map::from_iter([(
                "thinking".into(),
                json!({"type":"enabled"}),
            )])),
        )
        .expect("construct the real Chat model");
    let worker = agent::chat()
        .model(model)
        .system_prompt("Follow the user's exact output instruction.")
        .build()
        .expect("build the real Agent");

    let mut builder = WorkflowBuilder::new("live-deepseek-agent-workflow");
    builder
        .add_agent_node("agent", worker)
        .expect("add the Agent as a Workflow node");
    builder.add_edge(START, "agent").expect("connect START");
    builder
        .add_edge("agent", END)
        .expect("connect Agent to END");

    let workflow = builder.compile().expect("compile the Workflow");
    let mut runtime = Runtime::new(8_271_001, workflow).expect("create Workflow Runtime");
    let next_id = Arc::new(AtomicI64::new(8_271_002));
    let ids = Arc::clone(&next_id);
    let mut user = Message::new(Role::User);
    user.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"Reply with exactly this token and nothing else: HEART_WORKFLOW_AGENT_OK"})
            .as_object()
            .unwrap()
            .clone(),
    });
    let mut events = runtime
        .run(
            Messages::new([user]),
            Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        )
        .await
        .expect("run the real Workflow");

    let mut final_text = String::new();
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event.expect("Workflow event")
            && message.role == halo::agent::message::Role::Assistant
        {
            final_text = message
                .content
                .iter()
                .filter(|part| part.r#type == "text")
                .filter_map(|part| part.data.get("value").and_then(|value| value.as_str()))
                .collect::<String>();
        }
    }
    drop(events);

    assert!(
        final_text.contains("HEART_WORKFLOW_AGENT_OK"),
        "real model did not follow the requested output: {final_text:?}"
    );
    let execution = runtime.state().nodes["agent"]
        .executions
        .values()
        .next()
        .expect("Workflow should persist the Agent execution");
    assert_eq!(
        execution.status,
        halo::runtime::workflow::node::ExecutionStatus::Succeeded
    );
    assert!(execution.attempts[0].usage.is_some());
    println!("LIVE_WORKFLOW_AGENT_OUTPUT={final_text}");
}

#[tokio::test]
#[ignore = "requires a real DeepSeek API key"]
async fn workflow_forwards_a_real_partial_complete_after_cancellation() {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct the real DeepSeek provider");
    let model = provider
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct the real Chat model");
    let worker = agent::chat()
        .model(model)
        .system_prompt(
            "Write a detailed, multi-paragraph explanation of how the water cycle works.",
        )
        .build()
        .expect("build the real Agent");

    let mut builder = WorkflowBuilder::new("live-workflow-agent-cancellation");
    builder
        .add_agent_node("agent", worker)
        .expect("add the Agent as a Workflow node");
    builder.add_edge(START, "agent").expect("connect START");
    builder
        .add_edge("agent", END)
        .expect("connect Agent to END");
    let workflow = builder.compile().expect("compile the Workflow");
    let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
    let runtime = Runtime::new(8_272_001, workflow)
        .expect("create Workflow Runtime")
        .with_checkpoint(checkpoint.clone());
    let next_id = Arc::new(AtomicI64::new(8_272_002));
    let ids = Arc::clone(&next_id);
    let mut user = Message::new(Role::User);
    user.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"Explain the water cycle in detail."})
            .as_object()
            .unwrap()
            .clone(),
    });
    let cancellation = Cancellation::new();
    let mut events = runtime
        .run(
            Messages::new([user]),
            Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
            cancellation.clone(),
        )
        .await
        .expect("run the real Workflow");

    let first = tokio::time::timeout(std::time::Duration::from_secs(60), events.next())
        .await
        .expect("real Provider should stream its first Delta")
        .expect("Workflow should emit a Delta")
        .expect("real Provider event should succeed");
    assert!(matches!(first, Event::Delta { .. }));
    cancellation.cancel();

    let rest = tokio::time::timeout(std::time::Duration::from_secs(60), async {
        events.collect::<Vec<_>>().await
    })
    .await
    .expect("Workflow should close after cancellation");
    let partial = rest
        .iter()
        .filter_map(|event| match event.as_ref().ok()? {
            Event::Complete {
                message,
                usage,
                finish_reason,
            } if finish_reason.as_deref() == Some("cancelled") => Some((message, usage.as_ref())),
            _ => None,
        })
        .next_back()
        .expect("Workflow should forward the Agent's partial cancelled Complete");
    assert!(!partial.0.content.is_empty() || !partial.0.values.is_empty());
    assert!(partial.1.and_then(|usage| usage.total).unwrap_or_default() > 0);
    assert!(
        rest.iter()
            .any(|event| matches!(event, Err(error) if error.code == "CANCELLED"))
    );
    assert_eq!(
        checkpoint
            .load(8_272_001, &Cancellation::new())
            .await
            .unwrap()["status"],
        "cancelled"
    );
    println!("LIVE_WORKFLOW_CANCELLED_PARTIAL={:?}", partial.0);
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_workflow_agent_waits_for_approval_then_resumes_the_live_tool_step() {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL").expect("set HEART_VLLM_BASE_URL"),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("set HEART_VLLM_API_KEY"),
        ..Config::default()
    })
    .expect("construct real VLLM provider");
    let model = provider
        .chat(
            chat::Config::new(std::env::var("HEART_VLLM_MODEL").expect("set HEART_VLLM_MODEL"))
                .options(serde_json::Map::from_iter([(
                    "chat_template_kwargs".into(),
                    json!({"enable_thinking":false}),
                )])),
        )
        .expect("construct real Chat");
    let calls = Arc::new(AtomicUsize::new(0));
    let worker = agent::react()
        .model(model)
        .system_prompt(
            "必须调用 current_time 获取真实 Unix 时间戳。等待用户批准后才能执行工具；最终回答只能使用工具返回的时间戳。",
        )
        .tools(ApprovalClock(Arc::clone(&calls)))
        .permission(AskForApproval)
        .with_id_fn(next_workflow_approval_id)
        .build()
        .expect("build embedded ReAct Agent");

    let mut builder = WorkflowBuilder::new("live-workflow-agent-approval");
    builder
        .add_agent_node("agent", worker)
        .expect("add Agent node");
    builder.add_edge(START, "agent").expect("connect START");
    builder.add_edge("agent", END).expect("connect END");
    let workflow = builder.compile().expect("compile Workflow");
    let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
    let runtime = Runtime::new(8_273_001, workflow)
        .expect("create Workflow Runtime")
        .with_checkpoint(checkpoint.clone());

    let mut user = Message::new(Role::User);
    user.metadata = serde_json::from_value(json!({"chat_id":8_273_001,"session_id":8_273_002}))
        .expect("valid run identity");
    user.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"查询当前 Unix 时间戳，先申请工具权限。"})
            .as_object()
            .unwrap()
            .clone(),
    });
    let mut events = runtime
        .run(
            Messages::new([user]),
            Arc::new(|| 8_273_003),
            Cancellation::new(),
        )
        .await
        .expect("start Workflow");
    let reply_id = tokio::time::timeout(std::time::Duration::from_secs(60), async {
        loop {
            let event = events.next().await.expect("Workflow remains active")?;
            if let Event::Complete { message, .. } = event
                && let Some(id) = message
                    .metadata
                    .get("reply_id")
                    .and_then(serde_json::Value::as_i64)
            {
                return Ok::<_, Error>(id);
            }
        }
    })
    .await
    .expect("real model should ask for approval")
    .expect("approval question event");
    assert_eq!(calls.load(Ordering::SeqCst), 0, "approval gates the Tool");

    let mut approval = Message::new(Role::User);
    approval.metadata.insert("chat_id".into(), json!(8_273_001));
    approval
        .metadata
        .insert("session_id".into(), json!(8_273_002));
    approval.metadata.insert("reply_id".into(), json!(reply_id));
    approval.content.push(Part {
        r#type: "json".into(),
        data: json!({"value":{"approved":true}})
            .as_object()
            .unwrap()
            .clone(),
    });
    runtime
        .reply(Messages::new([approval]), Cancellation::new())
        .await
        .expect("submit approval to the embedded Agent");
    let output = tokio::time::timeout(std::time::Duration::from_secs(60), async {
        events.collect::<Vec<_>>().await
    })
    .await
    .expect("Workflow should finish after approval");
    assert!(output.iter().all(Result::is_ok), "{output:?}");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "Tool executes once after approval"
    );

    let timestamp = output
        .iter()
        .filter_map(|event| match event.as_ref().ok()? {
            Event::Complete { message, .. } if message.role == Role::Assistant => Some(message),
            _ => None,
        })
        .flat_map(|message| message.content.iter())
        .filter(|part| part.r#type == "text")
        .filter_map(|part| part.data.get("value").and_then(|value| value.as_str()))
        .flat_map(|text| text.split(|character: char| !character.is_ascii_digit()))
        .find(|candidate| candidate.len() == 10)
        .and_then(|candidate| candidate.parse::<u64>().ok())
        .expect("final response must use the live Tool timestamp");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_secs();
    assert!(timestamp.abs_diff(now) <= 10);
    let saved = checkpoint
        .load(8_273_001, &Cancellation::new())
        .await
        .expect("Workflow Checkpoint was persisted");
    assert_eq!(saved["status"], "succeeded");
    println!("LIVE_WORKFLOW_APPROVAL_TIMESTAMP={timestamp}");
}
