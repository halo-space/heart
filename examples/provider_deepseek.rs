//! Real DeepSeek Chat Completions example.
//!
//! Required environment variables: DEEPSEEK_API_KEY and optionally
//! DEEPSEEK_BASE_URL/DEEPSEEK_MODEL. The provider maps reasoning_content to
//! the shared `think` content part and ordinary output to `text`.
use futures_util::StreamExt;
use halo::{
    agent::event::Event,
    agent::message::{Message, Messages, Role},
    agent::model::{chat, chat::Chat as _, content::part::Part},
    agent::provider::deepseek::DeepSeek,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), halo::agent::model::Error> {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
            halo::agent::model::Error::new("AUTHENTICATION", "DEEPSEEK_API_KEY is required")
        })?)
        .build()?;
    let chat = provider.chat(
        chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        )
        .options(serde_json::Map::from_iter([(
            "thinking".into(),
            json!({"type": "enabled"}),
        )])),
    )?;

    let mut user = Message::new(Role::User);
    user.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "请只回复：DeepSeek Rust 适配测试成功"})
            .as_object()
            .unwrap()
            .clone(),
    });
    let mut events = chat
        .stream(chat::Request {
            messages: Messages::new([user]),
            stream: true,
            max_tokens: Some(128),
            ..Default::default()
        })
        .await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => {
                for part in message.content {
                    let value = part.data["value"].as_str().unwrap_or_default();
                    match part.r#type.as_str() {
                        "think" => eprint!("[think] {value}"),
                        _ => print!("{value}"),
                    }
                }
            }
            Event::Complete {
                usage,
                finish_reason,
                ..
            } => {
                println!("\nusage={usage:?}, finish_reason={finish_reason:?}");
            }
        }
    }
    Ok(())
}
