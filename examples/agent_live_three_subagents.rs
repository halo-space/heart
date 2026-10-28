//! Live VLLM-backed parent Agent with three concurrently delegated SubAgents.
//!
//! Required environment variables: HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and
//! HEART_VLLM_MODEL. The parent is instructed to emit three agent_spawn calls
//! in one model response; the example verifies that three sourced child
//! executions complete on the parent Event stream.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicI64, Ordering};

use futures_util::StreamExt;
use halo::agent::{self, Agent, Cancellation, Error, Input, Payload, SubAgent};
use halo::agent::{
    event::Event,
    model::chat,
    provider::vllm::{Config, Vllm, chat::Chat as VllmChat},
    trace::fastrace::{Config as TraceConfig, ConsoleReporter, flush, set_reporter},
};
use serde_json::json;

static NEXT_ID: AtomicI64 = AtomicI64::new(9_001_000);

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

fn child(provider: &Vllm, name: &str) -> Result<SubAgent, Error> {
    let agent = agent::chat()
        .model(model(provider)?)
        .system_prompt(format!(
            "你是 {name} 子 Agent。收到任务后只回复 {name}_OK，不调用工具。"
        ))
        .build()?;
    SubAgent::new(name, agent)
}

async fn run() -> Result<(), Error> {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL")
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_VLLM_BASE_URL is required"))?,
        api_key: std::env::var("HEART_VLLM_API_KEY")
            .map_err(|_| Error::new("AUTHENTICATION", "HEART_VLLM_API_KEY is required"))?,
        ..Config::default()
    })?;
    let parent = agent::react()
        .model(model(&provider)?)
        .system_prompt(
            "必须在同一次模型响应中分别调用 agent_spawn 三次，name 依次为 researcher、coder、reviewer。三个子任务都完成后，再汇总三个结果。不要直接回答，不要漏掉任何一个子 Agent。",
        )
        .subagents([
            child(&provider, "researcher")?,
            child(&provider, "coder")?,
            child(&provider, "reviewer")?,
        ])
        .with_id_fn(next_id)
        .with_max_concurrency(3)
        .build()?;
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"请并发委派 researcher、coder、reviewer，然后报告三个子 Agent 的结果。"}),
            mime_type: None,
        }],
        ..Default::default()
    };

    let mut events = parent.run(input, true, Cancellation::new()).await?;
    let mut child_executions = BTreeSet::new();
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } | Event::Complete { message, .. } => {
                if let (Some(node_id), Some(exec_id)) = (
                    message
                        .metadata
                        .get("node_id")
                        .and_then(|value| value.as_str()),
                    message
                        .metadata
                        .get("exec_id")
                        .and_then(|value| value.as_i64()),
                ) {
                    child_executions.insert((node_id.to_owned(), exec_id));
                }
                println!("{message:?}");
            }
        }
    }
    println!("child_execution_count={}", child_executions.len());
    if child_executions.len() < 3 {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "the live parent did not complete three sourced SubAgent executions",
        ));
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    set_reporter(ConsoleReporter, TraceConfig::default());
    let result = run().await;
    flush();
    result
}
