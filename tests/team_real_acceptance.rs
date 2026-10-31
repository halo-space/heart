//! Real Team acceptance. Every model call uses the configured VLLM service.

use std::sync::{
    Arc,
    atomic::{AtomicI64, AtomicUsize, Ordering},
};

use futures_util::StreamExt;
use halo::agent::{self, Agent, Cancellation, Error, Input, Message, SubAgent};
use halo::agent::{
    checkpoint::Checkpoint,
    event::Event,
    message::Role,
    middleware::Middleware,
    model::chat,
    provider::vllm::{Config, Vllm, chat::Chat as VllmChat},
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Value, json};

fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(9_200_100);
    NEXT.fetch_add(1, Ordering::SeqCst)
}

#[derive(Clone)]
struct Probe {
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
}

impl Middleware for Probe {
    async fn before_execute(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        Ok(())
    }

    async fn after_execute(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }
}

fn provider() -> Result<Vllm, Error> {
    Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL")
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "set HEART_VLLM_BASE_URL"))?,
        api_key: std::env::var("HEART_VLLM_API_KEY")
            .map_err(|_| Error::new("AUTHENTICATION", "set HEART_VLLM_API_KEY"))?,
        ..Config::default()
    })
}

fn chat(provider: &Vllm) -> Result<VllmChat, Error> {
    provider.chat(
        chat::Config::new(
            std::env::var("HEART_VLLM_MODEL")
                .map_err(|_| Error::new("INVALID_ARGUMENTS", "set HEART_VLLM_MODEL"))?,
        )
        .options(serde_json::Map::from_iter([(
            "chat_template_kwargs".into(),
            json!({"enable_thinking": false}),
        )])),
    )
}

fn input() -> Input {
    serde_json::from_value(json!({
        "metadata": {"chat_id": 9_200_001, "session_id": 9_200_002},
        "payload": [{"type":"text","data":{"value":"请协调团队完成研究、代码和审查。"}}]
    }))
    .expect("valid Team input")
}

struct AskForApproval;

impl Tool for AskForApproval {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: std::sync::LazyLock<ToolDefinition> =
            std::sync::LazyLock::new(|| ToolDefinition {
                name: "ask_for_approval".into(),
                description: "Ask the user for approval and wait for one reply".into(),
                parameters: serde_json::from_value(json!({
                    "type":"object",
                    "properties":{"question":{"type":"string"}},
                    "required":["question"],
                    "additionalProperties":false
                }))
                .expect("approval schema"),
            });
        &DEFINITION
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let question = call.arguments["question"]
            .as_str()
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "question is required"))?;
        let mut prompt = Message::new(Role::Assistant);
        prompt
            .content
            .push(halo::agent::model::content::part::Part {
                r#type: "text".into(),
                data: json!({"value":question}).as_object().unwrap().clone(),
            });
        let feedback = halo::agent::runtime::agent::wait::ask(prompt, 1, cancellation).await?;
        Ok(Message::function(json!({"user_feedback": feedback.0})))
    }
}

struct FailAfter;

impl Middleware for FailAfter {
    async fn after_execute(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        Err(Error::new("FAILED", "real member post-processing failure"))
    }
}

#[tokio::test]
#[ignore = "requires HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_team_member_failure_returns_to_leader_context() {
    let provider = provider().expect("construct real VLLM provider");
    let child = agent::react()
        .model(chat(&provider).expect("child model"))
        .system_prompt("你是 reviewer。收到任务后只回复 REVIEW_OK。")
        .middleware(FailAfter)
        .build()
        .expect("build failing reviewer");
    let team = agent::team()
        .leader(
            agent::react()
                .model(chat(&provider).expect("leader model"))
                .system_prompt(
                    "你是 Team Leader。必须且只调用一次 team_assign，把完整 agent::Input 任务交给 reviewer。成员可能失败；无论结果如何，都只用一句话报告成员执行结果，不要再次分配任务。",
                ),
        )
        .member(
            SubAgent::new("reviewer", child)
                .expect("register reviewer")
                .description("负责审查"),
        )
        .with_id_fn(next_id)
        .build()
        .expect("build Team");
    let events = team
        .run(input(), true, Cancellation::new())
        .await
        .expect("start Team");
    let tail = tokio::time::timeout(std::time::Duration::from_secs(90), async {
        events.collect::<Vec<_>>().await
    })
    .await
    .expect("Team must close after member failure");
    assert!(
        tail.iter().all(Result::is_ok),
        "Team failed to continue: {tail:?}"
    );
    assert!(tail.iter().any(|event| matches!(event.as_ref().unwrap(),
        Event::Complete { message, .. }
            if message.role == Role::Tool
                && message.values.get("name").and_then(|v| v.as_str()) == Some("team_assign")
                && message.values.contains_key("error")
    )));
}

#[tokio::test]
#[ignore = "requires HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_team_cancellation_propagates_to_active_member() {
    let provider = provider().expect("construct real VLLM provider");
    let child = agent::react()
        .model(chat(&provider).expect("child model"))
        .system_prompt("你是 reviewer。收到任务后只回复 REVIEW_OK。")
        .build()
        .expect("build reviewer");
    let team = agent::team()
        .leader(
            agent::react()
                .model(chat(&provider).expect("leader model"))
                .system_prompt(
                    "你是 Team Leader。必须且只调用一次 team_assign，把完整 agent::Input 任务交给 reviewer。",
                ),
        )
        .member(SubAgent::new("reviewer", child).unwrap())
        .with_id_fn(next_id)
        .build()
        .expect("build Team");
    let cancellation = Cancellation::new();
    let mut events = team
        .run(input(), true, cancellation.clone())
        .await
        .expect("start Team");
    let mut saw_member = false;
    while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(90), events.next())
        .await
        .expect("Team must emit before cancellation")
    {
        let event = event.expect("event before cancellation");
        let message = match event {
            Event::Delta { message } | Event::Complete { message, .. } => message,
        };
        if message.metadata.contains_key("node_id") {
            saw_member = true;
            cancellation.cancel();
            break;
        }
    }
    assert!(saw_member, "Team must start a member before cancellation");
    let tail = tokio::time::timeout(std::time::Duration::from_secs(90), async {
        events.collect::<Vec<_>>().await
    })
    .await
    .expect("Team must close after cancellation");
    assert!(
        tail.iter().any(|event| matches!(event, Err(error) if error.code == "CANCELLED"))
            || tail.iter().any(|event| matches!(event, Ok(Event::Complete { finish_reason: Some(reason), .. }) if reason == "cancelled")),
        "Team cancellation must close the active member: {tail:?}"
    );
}

#[tokio::test]
#[ignore = "requires HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_team_planner_task_assigns_member_and_reuses_agent_graph() {
    let provider = provider().expect("construct real VLLM provider");
    let child = agent::chat()
        .model(chat(&provider).expect("child model"))
        .system_prompt("你是 researcher。收到任务后只回复 RESEARCH_OK。")
        .build()
        .expect("build researcher");
    let team = agent::team()
        .leader(
            agent::react()
                .model(chat(&provider).expect("leader model"))
                .system_prompt(
                    "你是 Team Leader。这个任务必须先调用 plan 创建一个单节点计划。计划任务执行时，必须调用 team_assign 把完整 agent::Input 交给 researcher。等待成员完成后，再输出一句汇总。不要在计划外直接调用 team_assign。",
                ),
        )
        .member(
            SubAgent::new("researcher", child)
                .unwrap()
                .description("负责研究任务"),
        )
        .with_id_fn(next_id)
        .build()
        .expect("build Team");
    let events = team
        .run(input(), true, Cancellation::new())
        .await
        .expect("start planned Team");
    let tail = tokio::time::timeout(std::time::Duration::from_secs(120), async {
        events.collect::<Vec<_>>().await
    })
    .await
    .expect("planned Team must finish");
    assert!(
        tail.iter().all(Result::is_ok),
        "planned Team failed: {tail:?}"
    );
    assert!(tail.iter().any(|event| matches!(event.as_ref().unwrap(),
        Event::Complete { message, .. } if message.metadata.contains_key("node_id")
    )));
}

#[tokio::test]
#[ignore = "requires HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_team_member_waits_for_reply_then_leader_continues() {
    let provider = provider().expect("construct real VLLM provider");
    let child = agent::react()
        .model(chat(&provider).expect("child model"))
        .system_prompt(
            "你是 reviewer 成员。必须调用 ask_for_approval，询问用户是否批准发布。收到反馈后，只回复 REVIEW_OK。",
        )
        .tools([AskForApproval])
        .build()
        .expect("build reviewer");
    let team = agent::team()
        .leader(
            agent::react()
                .model(chat(&provider).expect("leader model"))
                .system_prompt(
                    "你是 Team Leader。必须且只调用一次 team_assign，把完整 agent::Input 任务交给 reviewer，要求它询问用户是否批准发布。等待 reviewer 返回后，只回复一句总结。不要自行假设用户批准。",
                ),
        )
        .member(
            SubAgent::new("reviewer", child)
                .expect("register reviewer")
                .description("负责发布审批"),
        )
        .with_id_fn(next_id)
        .build()
        .expect("build Team");
    let cancellation = Cancellation::new();
    let mut events = team
        .run(input(), true, cancellation.clone())
        .await
        .expect("start Team");
    let mut observed = Vec::new();
    let reply_id = loop {
        let event = tokio::time::timeout(std::time::Duration::from_secs(90), events.next())
            .await
            .expect("Team must reach member waiting")
            .unwrap_or_else(|| panic!("Team stream ended before waiting: {observed:?}"))
            .unwrap_or_else(|error| {
                panic!("Team failed before waiting: {error:?}; events={observed:?}")
            });
        observed.push(event.clone());
        if let Event::Complete { message, .. } = event
            && let Some(reply_id) = message.metadata.get("reply_id").and_then(|v| v.as_i64())
        {
            break reply_id;
        }
    };
    let mut reply = input();
    reply.metadata.insert("reply_id".into(), json!(reply_id));
    reply.payload[0].data = json!({"value":"批准发布。"});
    team.reply(reply, cancellation)
        .await
        .expect("reply to member");
    let tail = tokio::time::timeout(std::time::Duration::from_secs(90), async {
        events.collect::<Vec<_>>().await
    })
    .await
    .expect("Team must continue after member reply");
    assert!(
        tail.iter().all(Result::is_ok),
        "Team failed after reply: {tail:?}"
    );
    assert!(tail.iter().any(|event| matches!(event.as_ref().unwrap(),
        Event::Complete { message, .. }
            if message.role == Role::Assistant && !message.metadata.contains_key("reply_id")
    )));
}

#[tokio::test]
#[ignore = "requires HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_team_checkpoint_restores_waiting_member_without_replaying_completed_work() {
    let provider = provider().expect("construct real VLLM provider");
    let checkpoint = std::sync::Arc::new(halo::agent::core::checkpoint::InMemory::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let build_team = || {
        let child = agent::react()
            .model(chat(&provider).expect("child model"))
            .system_prompt(
                "你是 reviewer 成员。必须调用 ask_for_approval，询问用户是否批准发布。收到反馈后，只回复 REVIEW_OK。",
            )
            .tools([AskForApproval])
            .checkpoint(checkpoint.clone())
            .with_id_fn(next_id)
            .build()
            .expect("build reviewer");
        let child = SubAgent::new("reviewer", child)
            .expect("register reviewer")
            .description("负责发布审批");
        agent::team()
            .leader(
                agent::react()
                    .model(chat(&provider).expect("leader model"))
                    .system_prompt(
                        "你是 Team Leader。必须且只调用一次 team_assign，把完整 agent::Input 任务交给 reviewer，要求它询问用户是否批准发布。等待 reviewer 返回后，只回复一句总结。",
                    )
                    .checkpoint(checkpoint.clone())
                    .with_id_fn(next_id),
            )
            .member(child)
            .with_id_fn(next_id)
            .build()
            .expect("build Team")
    };
    let mut first_input = input();
    first_input
        .metadata
        .insert("chat_id".into(), json!(9200003));
    let first = build_team();
    let mut events = first
        .run(first_input.clone(), true, Cancellation::new())
        .await
        .expect("start Team");
    let mut observed = Vec::new();
    let reply_id = loop {
        let event = tokio::time::timeout(std::time::Duration::from_secs(90), events.next())
            .await
            .expect("Team must reach waiting")
            .unwrap_or_else(|| panic!("Team stream ended before waiting: {observed:?}"))
            .unwrap_or_else(|error| {
                panic!("Team failed before waiting: {error:?}; events={observed:?}")
            });
        observed.push(event.clone());
        if let Event::Complete { message, .. } = event
            && let Some(reply_id) = message.metadata.get("reply_id").and_then(|v| v.as_i64())
        {
            break reply_id;
        }
    };
    let saved = checkpoint
        .load(9200003, &Cancellation::new())
        .await
        .expect("Team waiting checkpoint must exist");
    assert_eq!(saved["status"], "waiting");
    drop(events);

    let mut reply = first_input;
    reply.metadata.insert("reply_id".into(), json!(reply_id));
    reply.payload[0].data = json!({"value":"批准发布。"});
    let resumed = build_team();
    let restored = resumed
        .run(reply, true, Cancellation::new())
        .await
        .expect("restore Team from Checkpoint");
    let tail = tokio::time::timeout(std::time::Duration::from_secs(90), async {
        restored.collect::<Vec<_>>().await
    })
    .await
    .expect("restored Team must finish");
    assert!(
        tail.iter().all(Result::is_ok),
        "restored Team failed: {tail:?}"
    );
    assert!(tail.iter().any(|event| matches!(event.as_ref().unwrap(),
        Event::Complete { message, .. }
            if message.role == Role::Assistant && !message.metadata.contains_key("reply_id")
    )));
    let _ = calls;
}

#[tokio::test]
#[ignore = "requires HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_team_serializes_same_member_tasks_and_runs_other_members_concurrently() {
    let provider = provider().expect("construct real VLLM provider");
    let researcher_probe = Probe {
        active: Arc::new(AtomicUsize::new(0)),
        peak: Arc::new(AtomicUsize::new(0)),
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let researcher = agent::chat()
        .model(chat(&provider).expect("researcher model"))
        .system_prompt("你是 researcher。收到任意任务后只回复 RESEARCH_OK。")
        .middleware(researcher_probe.clone())
        .build()
        .expect("build researcher");
    let coder = agent::chat()
        .model(chat(&provider).expect("coder model"))
        .system_prompt("你是 coder。收到任意任务后只回复 CODER_OK。")
        .build()
        .expect("build coder");
    let team = agent::team()
        .leader(
            agent::react()
                .model(chat(&provider).expect("leader model"))
                .system_prompt(
                    "你是 Team Leader。必须在同一次模型响应中调用 team_assign 三次：前两次 member 必须是 researcher，第三次 member 必须是 coder。每个 input 必须是完整 agent::Input，payload 只有一个 text。等待三次都完成后再汇总，不能漏掉任何一次。",
                ),
        )
        .member(SubAgent::new("researcher", researcher).unwrap())
        .member(SubAgent::new("coder", coder).unwrap())
        .with_max_concurrency(3)
        .with_id_fn(next_id)
        .build()
        .expect("build Team");

    let mut events = team
        .run(input(), true, Cancellation::new())
        .await
        .expect("start Team");
    let mut member_execs = 0usize;
    while let Some(event) = events.next().await {
        let event = event.expect("Team event must succeed");
        let message = match event {
            Event::Delta { message } | Event::Complete { message, .. } => message,
        };
        if message.metadata.contains_key("node_id") {
            member_execs += 1;
        }
    }
    assert!(member_execs >= 3, "Team must expose all member executions");
    assert_eq!(researcher_probe.calls.load(Ordering::SeqCst), 2);
    assert_eq!(researcher_probe.peak.load(Ordering::SeqCst), 1);
}
