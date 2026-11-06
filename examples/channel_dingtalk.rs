//! Real DingTalk Stream channel connected to a real DeepSeek Agent.
//!
//! Required environment variables: DINGTALK_CLIENT_ID, DINGTALK_CLIENT_SECRET,
//! and DEEPSEEK_API_KEY. The process keeps the Stream connection open.

use std::sync::Arc;

use halo::agent::model::chat;
use halo::agent::{self, Agent, Cancellation, Error};
use halo::agent::{channels::dingtalk, provider::deepseek::DeepSeek};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(
            std::env::var("DEEPSEEK_API_KEY")
                .map_err(|_| Error::new("AUTHENTICATION", "DEEPSEEK_API_KEY is required"))?,
        )
        .build()?;
    let model = provider.chat(
        chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        )
        .options(serde_json::Map::from_iter([(
            "thinking".into(),
            json!({"type": "disabled"}),
        )])),
    )?;
    let agent = Arc::new(
        agent::chat()
            .model(model)
            .system_prompt("你是钉钉里的简洁中文助手。")
            .build()?,
    );

    let channel = dingtalk::Channel::new(
        dingtalk::Config::new(
            std::env::var("DINGTALK_CLIENT_ID")
                .map_err(|_| Error::new("AUTHENTICATION", "DINGTALK_CLIENT_ID is required"))?,
            std::env::var("DINGTALK_CLIENT_SECRET")
                .map_err(|_| Error::new("AUTHENTICATION", "DINGTALK_CLIENT_SECRET is required"))?,
        ),
        move |inbound: dingtalk::Inbound, reply: dingtalk::Reply| {
            let agent = Arc::clone(&agent);
            async move {
                let events = agent.run(inbound.input, true, Cancellation::new()).await?;
                reply.events(events).await
            }
        },
    )?;
    channel.start().await
}
