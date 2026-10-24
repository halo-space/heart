//! Explicit Provider -> model binding -> shared Event, with no Runtime required.
use futures_util::StreamExt;
use halo::{
    agent::event::Event,
    agent::message::{Message, Messages, Role},
    agent::model::{
        chat::{self, Chat as _},
        content::part::Part,
    },
    agent::provider::openai::{Config, OpenAI},
};
use serde_json::{Map, json};

#[tokio::main]
async fn main() -> Result<(), halo::agent::model::Error> {
    // This example makes a paid API request ONLY when explicitly run by a user.
    // Compilation/tests never execute main or read actual credentials.
    let provider = OpenAI::new(Config {
        api_key: std::env::var("OPENAI_API_KEY").map_err(|_| {
            halo::agent::model::Error::new("AUTHENTICATION", "OPENAI_API_KEY is required")
        })?,
        base_url: std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
        ..Config::default()
    })?;
    let chat = provider.chat(chat::Config {
        model_name: std::env::var("OPENAI_MODEL").map_err(|_| {
            halo::agent::model::Error::new("INVALID_ARGUMENTS", "OPENAI_MODEL is required")
        })?,
        options: Map::new(),
    })?;
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "你好"}).as_object().unwrap().clone(),
    });
    let mut events = chat
        .stream(chat::Request {
            messages: Messages::new([message]),
            stream: true,
            ..Default::default()
        })
        .await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => {
                for part in message.content {
                    if part.r#type == "text" {
                        print!("{}", part.data["value"].as_str().unwrap_or_default());
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
