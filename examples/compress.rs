//! Explicit standalone compression through an application-selected Chat.
//! Running this example calls a real model and may incur cost:
//! HALO_API_KEY=... HALO_MODEL=... cargo run --example compress
//! Optional HALO_BASE_URL selects an OpenAI-compatible endpoint.
//! Automated cargo tests only compile this example, never run that model call.

use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::compression::Compressor as _,
    agent::core,
    agent::message::Role,
    agent::model::{chat, content::part::Part},
    agent::prompt,
    agent::provider::openai::{Config, OpenAI},
};
use serde_json::{Map, json};

fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":value}).as_object().unwrap().clone(),
    });
    message
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let api_key = std::env::var("HALO_API_KEY").map_err(|_| {
        Error::new(
            "INVALID_ARGUMENTS",
            "set HALO_API_KEY to explicitly run this example",
        )
    })?;
    let model_name = std::env::var("HALO_MODEL").map_err(|_| {
        Error::new(
            "INVALID_ARGUMENTS",
            "set HALO_MODEL to explicitly run this example",
        )
    })?;
    let provider = OpenAI::new(Config {
        base_url: std::env::var("HALO_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
        api_key,
        ..Config::default()
    })?;
    let chat = provider.chat(chat::Config {
        model_name,
        options: Map::new(),
    })?;
    let template = prompt::Template::new().message(text(
        Role::System,
        "将后面的对话压缩成简短的中文上下文，保留用户意图和已有结论，只返回摘要正文。",
    ));
    let component = core::compression::Compressor::new(
        chat,
        chat::Request {
            messages: template.format(&prompt::Variables::new())?,
            stream: true,
            max_tokens: Some(256),
            ..Default::default()
        },
    )?;
    let history = Messages::new([
        text(Role::User, "北京天气好的话，我周末想去露营。"),
        text(
            Role::Assistant,
            "出发前确认天气、风力和营地开放情况，并准备保暖衣物。",
        ),
    ]);
    let compressed = component.compress(history, &Cancellation::new()).await?;
    println!(
        "{}",
        serde_json::to_string_pretty(&compressed).map_err(|_| Error::new(
            "INVALID_RESPONSE",
            "could not serialize complete compression result"
        ))?
    );
    Ok(())
}
