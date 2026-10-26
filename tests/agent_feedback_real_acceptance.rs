//! Human feedback round-trip through a real DeepSeek-backed Agent.

use futures_util::StreamExt;
use halo::agent::{
    Agent, Input, SubAgent,
    checkpoint::Checkpoint,
    event::Event,
    message::Role,
    model::chat,
    provider::deepseek::DeepSeek,
    tool::{Call, Tool, ToolDefinition},
};
use halo::agent::{Cancellation, Error, Message};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicI64, AtomicUsize, Ordering},
};

fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(813_100);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

struct AskForApproval {
    definition: ToolDefinition,
    count: usize,
    calls: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
}

impl AskForApproval {
    fn new(count: usize) -> Self {
        Self {
            definition: ToolDefinition {
                name: "ask_for_approval".into(),
                description: "Ask the user for approval and return the user's reply".into(),
                parameters: json!({
                    "type":"object",
                    "properties":{
                        "question":{"type":"string"},
                        "count":{"type":"integer","minimum":1}
                    },
                    "required":["question","count"],
                    "additionalProperties":false
                })
                .as_object()
                .unwrap()
                .clone(),
            },
            count,
            calls: None,
        }
    }

    fn with_calls(mut self, calls: std::sync::Arc<std::sync::atomic::AtomicUsize>) -> Self {
        self.calls = Some(calls);
        self
    }
}

impl Tool for AskForApproval {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        if let Some(calls) = &self.calls {
            calls.fetch_add(1, Ordering::SeqCst);
        }
        let question = call.arguments["question"]
            .as_str()
            .filter(|question| !question.trim().is_empty())
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "question is required"))?;
        let mut prompt = Message::new(Role::Assistant);
        prompt
            .content
            .push(halo::agent::model::content::part::Part {
                r#type: "text".into(),
                data: json!({"value":question}).as_object().unwrap().clone(),
            });
        let count = call.arguments["count"]
            .as_u64()
            .and_then(|count| usize::try_from(count).ok())
            .filter(|count| *count > 0)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "count must be positive"))?;
        if count != self.count {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "model requested a different feedback count than the test scenario",
            ));
        }
        let feedback = halo::runtime::agent::wait::ask(prompt, count, cancellation).await?;
        Ok(Message::function(json!({"user_feedback":feedback.0})))
    }
}

fn provider_chat() -> impl chat::Chat {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct real DeepSeek provider");
    provider
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct real Chat")
}

async fn real_agent_waits_for_feedback_then_continues(count: usize, chat_id: i64) {
    let agent = halo::agent::chat()
        .model(provider_chat())
        .system_prompt(
            format!("先调用 ask_for_approval 工具询问用户是否批准，工具参数 count 必须设为 {count}。等待用户提交全部 {count} 份反馈后，只根据反馈给出一句简短确认。"),
        )
        .tools([AskForApproval::new(count)])
        .with_id_fn(next_id)
        .build()
        .expect("build Agent");
    let input: Input = serde_json::from_value(json!({
        "metadata":{"chat_id":chat_id,"session_id":813002},
        "payload":[{"type":"text","data":{"value":"请询问我是否批准部署。"}}]
    }))
    .unwrap();
    let cancellation = Cancellation::new();
    let mut events = agent
        .run(input, true, cancellation.clone())
        .await
        .expect("start real Agent run");

    let mut observed = Vec::new();
    let reply_id = loop {
        let event = tokio::time::timeout(std::time::Duration::from_secs(60), events.next())
            .await
            .expect("real model must reach the approval request in time")
            .unwrap_or_else(|| panic!("Agent stream ended before approval; events: {observed:#?}"))
            .expect("Agent event succeeds");
        observed.push(event.clone());
        if let Event::Complete { message, .. } = event
            && let Some(reply_id) = message
                .metadata
                .get("reply_id")
                .and_then(|value| value.as_i64())
        {
            break reply_id;
        }
    };

    for index in 0..count {
        let reply: Input = serde_json::from_value(json!({
            "metadata":{"chat_id":chat_id,"session_id":813002,"reply_id":reply_id},
            "payload":[{"type":"text","data":{"value":format!("批准部署，补充反馈 {}", index + 1)}}]
        }))
        .unwrap();
        agent
            .reply(reply, cancellation.clone())
            .await
            .expect("deliver user feedback to the waiting run");
    }

    let tail = tokio::time::timeout(std::time::Duration::from_secs(60), async {
        events.collect::<Vec<_>>().await
    })
    .await
    .expect("Agent must continue after feedback");
    assert!(
        tail.iter().all(Result::is_ok),
        "Agent failed after reply: {tail:?}"
    );
    let final_message = tail
        .iter()
        .filter_map(|event| match event.as_ref().unwrap() {
            Event::Complete { message, .. }
                if message.role == Role::Assistant
                    && !message.metadata.contains_key("reply_id") =>
            {
                Some(message)
            }
            _ => None,
        })
        .next_back()
        .expect("Agent must emit a final assistant message after reply");
    assert!(!final_message.content.is_empty() || !final_message.values.is_empty());
    println!("LIVE_AGENT_REPLY_FINAL={final_message:?}");
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends live model requests"]
async fn real_agent_waits_for_one_user_reply_then_continues() {
    real_agent_waits_for_feedback_then_continues(1, 813001).await;
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends live model requests"]
async fn real_agent_waits_for_all_three_user_replies_then_continues() {
    real_agent_waits_for_feedback_then_continues(3, 813003).await;
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends live model requests"]
async fn real_parent_delegates_feedback_to_child_and_continues_after_reply() {
    let calls = Arc::new(AtomicUsize::new(0));
    let checkpoint = Arc::new(halo::core::checkpoint::InMemory::new());
    let child = halo::agent::chat()
        .model(provider_chat())
        .system_prompt(
            "你是审批子智能体。必须调用 ask_for_approval 工具询问用户是否批准发布。工具 count 必须为 1。收到用户反馈后，只依据反馈给父智能体一句简短结论。",
        )
        .tools([AskForApproval::new(1).with_calls(Arc::clone(&calls))])
        .with_id_fn(next_id)
        .build()
        .expect("build child Agent");
    let child = SubAgent::new("reviewer", child)
        .expect("register child Agent")
        .description("询问用户发布审批，并把真实反馈返回给父智能体");
    let parent = halo::agent::chat()
        .model(provider_chat())
        .system_prompt(
            "你是主智能体。必须且只调用一次 agent_spawn 委派给 reviewer 子智能体，要求它询问用户是否批准发布。等待子智能体返回后，用一句话报告结论，不要自行假设用户已批准。",
        )
        .subagents([child])
        .checkpoint(checkpoint.clone())
        .with_id_fn(next_id)
        .build()
        .expect("build parent Agent");

    let input: Input = serde_json::from_value(json!({
        "metadata":{"chat_id":813004,"session_id":813002},
        "payload":[{"type":"text","data":{"value":"请确认是否批准发布版本 2.1。"}}]
    }))
    .expect("valid parent input");
    let cancellation = halo::agent::Cancellation::new();
    let mut events = parent
        .run(input, true, cancellation.clone())
        .await
        .expect("start live parent Agent");
    let mut observed = Vec::new();
    let reply_id = loop {
        let event = tokio::time::timeout(std::time::Duration::from_secs(90), events.next())
            .await
            .expect("live parent/child must reach the approval request")
            .unwrap_or_else(|| panic!("stream ended before child asked; events: {observed:#?}"))
            .expect("parent/child execution succeeds while asking");
        observed.push(event.clone());
        if let Event::Complete { message, .. } = event
            && let Some(id) = message
                .metadata
                .get("reply_id")
                .and_then(|value| value.as_i64())
        {
            break id;
        }
    };

    let reply: Input = serde_json::from_value(json!({
        "metadata":{"chat_id":813004,"session_id":813002,"reply_id":reply_id},
        "payload":[{"type":"text","data":{"value":"批准发布版本 2.1。"}}]
    }))
    .expect("valid child feedback");
    parent
        .reply(reply, cancellation)
        .await
        .expect("deliver feedback to the waiting child Step");

    let tail = tokio::time::timeout(std::time::Duration::from_secs(90), async {
        events.collect::<Vec<_>>().await
    })
    .await
    .expect("parent must continue after child feedback");
    assert!(tail.iter().all(Result::is_ok), "child run failed: {tail:?}");
    assert!(
        tail.iter().any(
            |event| matches!(event.as_ref().unwrap(), Event::Complete { message, .. }
            if message.role == Role::Assistant && !message.metadata.contains_key("reply_id")
                && (!message.content.is_empty() || !message.values.is_empty()))
        ),
        "parent must produce a final message from the child's real feedback: {tail:?}"
    );
    assert!(
        observed
            .iter()
            .any(|event| matches!(event, Event::Complete { message, .. }
            if message.metadata.contains_key("reply_id"))),
        "the user-facing wait must be emitted on the original parent stream"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1, "child Tool runs once");
    let saved = checkpoint
        .load(813004, &halo::agent::Cancellation::new())
        .await
        .expect("final parent/child State remains checkpointed");
    assert_eq!(saved["status"], "succeeded");
    assert_eq!(saved["waits"], json!([]));
    println!("LIVE_PARENT_CHILD_FEEDBACK_TAIL={tail:?}");
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends live model requests"]
async fn real_child_feedback_restores_partial_collection_from_outer_checkpoint() {
    let calls = Arc::new(AtomicUsize::new(0));
    let checkpoint = Arc::new(halo::core::checkpoint::InMemory::new());
    let make_parent = || {
        let child = halo::agent::chat()
            .model(provider_chat())
            .system_prompt(
                "你是审批子智能体。必须调用 ask_for_approval 工具询问用户是否批准发布，工具 count 必须为 2。工具结果里已经包含两份反馈后，不要再次调用该工具；只依据这两份反馈给父智能体一句简短结论。",
            )
            .tools([AskForApproval::new(2).with_calls(Arc::clone(&calls))])
            .with_id_fn(next_id)
            .build()
            .expect("build child Agent");
        let child = SubAgent::new("reviewer", child)
            .expect("register child Agent")
            .description("收齐两份用户审批反馈后返回结论");
        halo::agent::chat()
            .model(provider_chat())
            .system_prompt(
                "你是主智能体。必须且只调用一次 agent_spawn 委派给 reviewer 子智能体，要求它询问用户是否批准发布。若对话里已有子智能体结果，不要再次委派；等待子智能体完成后，用一句话报告结论。",
            )
            .subagents([child])
            .checkpoint(checkpoint.clone())
            .with_id_fn(next_id)
            .build()
            .expect("build parent Agent")
    };

    let input: Input = serde_json::from_value(json!({
        "metadata":{"chat_id":813005,"session_id":813002},
        "payload":[{"type":"text","data":{"value":"请确认是否批准发布版本 2.2。"}}]
    }))
    .expect("valid parent input");
    let first_parent = make_parent();
    let mut events = first_parent
        .run(input.clone(), true, halo::agent::Cancellation::new())
        .await
        .expect("start live parent Agent");
    let reply_id = loop {
        let event = tokio::time::timeout(std::time::Duration::from_secs(90), events.next())
            .await
            .expect("real child must ask for feedback")
            .unwrap_or_else(|| panic!("stream ended before the child question"))
            .expect("parent/child execution succeeds while asking");
        if let Event::Complete { message, .. } = event
            && let Some(id) = message
                .metadata
                .get("reply_id")
                .and_then(|value| value.as_i64())
        {
            break id;
        }
    };

    let first_reply: Input = serde_json::from_value(json!({
        "metadata":{"chat_id":813005,"session_id":813002,"reply_id":reply_id},
        "payload":[{"type":"text","data":{"value":"第一份反馈：批准。"}}]
    }))
    .expect("valid first feedback");
    first_parent
        .reply(first_reply, halo::agent::Cancellation::new())
        .await
        .expect("save first feedback");
    assert!(futures_util::poll!(events.next()).is_pending());
    let partial = checkpoint
        .load(813005, &halo::agent::Cancellation::new())
        .await
        .expect("outer checkpoint contains the child scene");
    assert_eq!(partial["waits"].as_array().unwrap().len(), 1);
    drop(events);

    // A new Agent instance loads the outer checkpoint. The incoming reply is
    // the second contribution; no model is allowed to recreate the first wait.
    let resumed = make_parent();
    let mut second_reply = input;
    second_reply
        .metadata
        .insert("reply_id".into(), json!(reply_id));
    second_reply.payload[0].data = json!({"value":"第二份反馈：仍然批准。"});
    let restored_events = resumed
        .run(second_reply, true, halo::agent::Cancellation::new())
        .await
        .expect("restore the parent from its outer Checkpoint");
    let tail = tokio::time::timeout(std::time::Duration::from_secs(120), async {
        restored_events.collect::<Vec<_>>().await
    })
    .await
    .expect("restored child wait should continue without a new process-local consumer");
    assert!(
        tail.iter().all(Result::is_ok),
        "restored run failed: {tail:?}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "restore re-enters the waiting Tool once to consume its saved wait"
    );
    assert!(
        tail.iter().all(
            |event| !matches!(event.as_ref().unwrap(), Event::Complete { message, .. }
            if message.metadata.contains_key("reply_id"))
        ),
        "restoring the partial wait must not present a second question"
    );
    assert!(
        tail.iter().any(
            |event| matches!(event.as_ref().unwrap(), Event::Complete { message, .. }
            if message.role == Role::Assistant && !message.metadata.contains_key("reply_id")
                && (!message.content.is_empty() || !message.values.is_empty()))
        ),
        "restored child and parent must both finish from the saved feedback facts"
    );
}
