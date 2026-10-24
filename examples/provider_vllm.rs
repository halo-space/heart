//! VLLM OpenAI-compatible Chat example.
//!
//! Required environment variables: HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and
//! HEART_VLLM_MODEL. The provider returns VLLM `delta.reasoning` as the shared
//! `think` content part and ordinary deltas as `text` parts.
use futures_util::StreamExt;
use halo::{
    event::Event,
    message::{Message, Messages, Role},
    model::{chat, chat::Chat as _, content::part::Part},
    provider::vllm::{Config, Vllm},
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), halo::agent::model::Error> {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8000/v1".into()),
        api_key: std::env::var("HEART_VLLM_API_KEY").map_err(|_| {
            halo::agent::model::Error::new("AUTHENTICATION", "HEART_VLLM_API_KEY is required")
        })?,
        ..Config::default()
    })?;
    let chat = provider.chat(chat::Config {
        model_name: std::env::var("HEART_VLLM_MODEL").map_err(|_| {
            halo::agent::model::Error::new("INVALID_ARGUMENTS", "HEART_VLLM_MODEL is required")
        })?,
        options: serde_json::Map::from_iter([
            ("top_k".into(), json!(20)),
            (
                "chat_template_kwargs".into(),
                json!({"enable_thinking": true}),
            ),
        ]),
    })?;
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "介绍平路运河"})
            .as_object()
            .unwrap()
            .clone(),
    });
    let mut events = chat
        .stream(chat::Request {
            messages: Messages::new([message]),
            stream: true,
            max_tokens: Some(512),
            ..Default::default()
        })
        .await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => {
                for part in message.content {
                    print!("{}", part.data["value"].as_str().unwrap_or_default());
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
