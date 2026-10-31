//! Live VLLM-backed static Team with a ReAct Leader.
//!
//! Required: HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicI64, Ordering};

use futures_util::StreamExt;
use halo::agent::{self, Agent, Cancellation, Error, Input, Payload, SubAgent};
use halo::agent::{
    event::Event,
    model::chat,
    provider::vllm::{Config, Vllm, chat::Chat as VllmChat},
};
use serde_json::json;

static NEXT_ID: AtomicI64 = AtomicI64::new(9_101_000);

fn next_id() -> i64 {
    NEXT_ID.fetch_add(1, Ordering::SeqCst)
}

fn model(provider: &Vllm) -> Result<VllmChat, Error> {
    let model_name = std::env::var("HEART_VLLM_MODEL")
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_VLLM_MODEL is required"))?;
    provider.chat(
        chat::Config::new(model_name).options(serde_json::Map::from_iter([(
            "chat_template_kwargs".into(),
            json!({"enable_thinking": false}),
        )])),
    )
}

fn member(provider: &Vllm, name: &str) -> Result<SubAgent, Error> {
    let agent = agent::chat()
        .model(model(provider)?)
        .system_prompt(format!(
            "你是 {name} 团队成员。收到任务后只回复 {name}_OK，不调用工具。"
        ))
        .build()?;
    SubAgent::new(name, agent).map(|member| member.description(format!("负责{name}相关工作")))
}

async fn run() -> Result<(), Error> {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL")
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_VLLM_BASE_URL is required"))?,
        api_key: std::env::var("HEART_VLLM_API_KEY")
            .map_err(|_| Error::new("AUTHENTICATION", "HEART_VLLM_API_KEY is required"))?,
        ..Config::default()
    })?;
    let leader = agent::react()
        .model(model(&provider)?)
        .system_prompt(
            "你是团队 Leader。必须在同一次模型响应中调用 team_assign 三次，成员依次为 researcher、coder、reviewer。每个 input 必须是完整 agent::Input JSON，包含 metadata 和 payload，payload 中有一个 type=text 的文本项。等待三个成员都完成后，再用一句话汇总结果，不要自行回答。",
        );
    let team = agent::team()
        .leader(leader)
        .member(member(&provider, "researcher")?)
        .member(member(&provider, "coder")?)
        .member(member(&provider, "reviewer")?)
        .with_max_concurrency(3)
        .with_id_fn(next_id)
        .build()?;
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"请协调三个成员分别完成研究、代码和审查，然后汇总结果。"}),
            mime_type: None,
        }],
        ..Default::default()
    };
    let mut events = team.run(input, true, Cancellation::new()).await?;
    let mut members = BTreeSet::new();
    while let Some(event) = events.next().await {
        let event = event?;
        let message = match event {
            Event::Delta { message } | Event::Complete { message, .. } => message,
        };
        if let Some(node) = message.metadata.get("node_id").and_then(|v| v.as_str()) {
            members.insert(node.to_owned());
        }
        println!("{message:?}");
    }
    if members.len() < 3 {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "the live Team did not complete three member executions",
        ));
    }
    println!("team_member_execution_count={}", members.len());
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    run().await
}
