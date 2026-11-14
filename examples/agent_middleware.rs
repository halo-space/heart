//! Real DeepSeek Agent + Middleware example.
//!
//! The provider configuration is loaded from the ignored `.env` file. The
//! middleware changes the model request and annotates the final message; the
//! model response itself is produced by the configured remote provider.
use futures_util::StreamExt;
use halo::agent::middleware::Middleware;
use halo::agent::model::chat;
use halo::agent::provider::deepseek::DeepSeek;
use halo::agent::{self, Agent, Cancellation, Error, Input, Payload};
use serde_json::{Value, json};

struct MarkAnswer;

impl Middleware for MarkAnswer {
    async fn before_model(&self, request: &mut Value, _: &Cancellation) -> Result<(), Error> {
        request["temperature"] = json!(0.2);
        Ok(())
    }

    async fn after_execute(&self, message: &mut Value, _: &Cancellation) -> Result<(), Error> {
        message["values"]["checked"] = json!(true);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenvy::dotenv().ok();
    let provider =
        DeepSeek::builder()
            .base_url(
                std::env::var("DEEPSEEK_BASE_URL")
                    .unwrap_or_else(|_| "https://api.deepseek.com".into()),
            )
            .api_key(std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
                Error::new("AUTHENTICATION", "DEEPSEEK_API_KEY is required in .env")
            })?)
            .build()?;
    let chat = provider.chat(chat::Config::new(
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
    ))?;
    let agent = agent::chat()
        .model(chat)
        .system_prompt("你是一个简洁的助手。回答后不要调用工具。")
        .middleware(MarkAnswer)
        .build()?;
    let input = Input {
        metadata: Default::default(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"请用一句话介绍 Rust 的所有权模型。"}),
            mime_type: None,
        }],
    };
    let mut events = agent.run(input, true, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        println!("{:?}", event?);
    }
    Ok(())
}
