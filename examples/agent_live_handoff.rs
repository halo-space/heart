//! Live VLLM-backed Team Handoff.
//!
//! Required: HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL.
//! The Leader transfers control to the reviewer; it does not resume after the
//! reviewer returns its final Assistant message.

use std::sync::atomic::{AtomicI64, Ordering};

use futures_util::StreamExt;
use halo::agent::{self, Agent, Cancellation, Error, Input, Payload, SubAgent};
use halo::agent::{
    event::Event,
    model::chat,
    provider::vllm::{Config, Vllm, chat::Chat as VllmChat},
};
use serde_json::json;

static NEXT_ID: AtomicI64 = AtomicI64::new(9_301_000);

fn next_id() -> i64 {
    NEXT_ID.fetch_add(1, Ordering::SeqCst)
}

fn model(provider: &Vllm) -> Result<VllmChat, Error> {
    let name = std::env::var("HEART_VLLM_MODEL")
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_VLLM_MODEL is required"))?;
    provider.chat(
        chat::Config::new(name).options(serde_json::Map::from_iter([(
            "chat_template_kwargs".into(),
            json!({"enable_thinking": false}),
        )])),
    )
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL")
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_VLLM_BASE_URL is required"))?,
        api_key: std::env::var("HEART_VLLM_API_KEY")
            .map_err(|_| Error::new("AUTHENTICATION", "HEART_VLLM_API_KEY is required"))?,
        ..Config::default()
    })?;

    let reviewer = agent::chat()
        .model(model(&provider)?)
        .system_prompt("你是 reviewer。收到任务后只回复 HANDOFF_OK，不调用工具。")
        .build()?;
    let team = agent::team()
        .leader(
            agent::react()
                .model(model(&provider)?)
                .system_prompt(
                    "你是 Team Leader。必须直接调用 agent_handoff，把完整 agent::Input 交给 reviewer。禁止调用 plan、team_assign 或其他工具；调用后不要继续输出。",
                ),
        )
        .member(SubAgent::new("reviewer", reviewer)?.description("负责最终回复"))
        .handoff("leader", "reviewer")
        .with_id_fn(next_id)
        .build()?;

    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value": "请把这项任务转交 reviewer 完成。"}),
            mime_type: None,
        }],
        ..Default::default()
    };
    let mut events = team.run(input, true, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => println!("DELTA {message:?}"),
            Event::Complete {
                message,
                usage,
                finish_reason,
            } => {
                println!(
                    "COMPLETE role={:?} finish_reason={finish_reason:?} usage={usage:?} message={message:?}",
                    message.role
                );
                if finish_reason.as_deref() == Some("handoff") {
                    break;
                }
            }
        }
    }
    Ok(())
}
