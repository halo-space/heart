//! OpenAI-compatible multimodal example with real image input.
//!
//! Required environment variables:
//! - HEART_MULTIMODAL_BASE_URL
//! - HEART_MULTIMODAL_API_KEY
//! - HEART_MULTIMODAL_MODEL
//! - HEART_MULTIMODAL_IMAGE_URL or HEART_MULTIMODAL_IMAGE_DATA_URL

use futures_util::StreamExt;
use halo::agent::event::Event;
use halo::agent::message::{Message, Role};
use halo::agent::model::content::part::Part;
use halo::agent::model::multimodal::{self, Multimodal as _};
use halo::agent::provider::openai::OpenAI;
use halo::agent::{Error, Messages};
use serde_json::{Map, json};

fn env(name: &str) -> Result<String, Error> {
    std::env::var(name).map_err(|_| Error::new("INVALID_ARGUMENTS", format!("{name} is required")))
}

fn text(value: &str) -> Part {
    Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!(value))]),
        mime_type: None,
    }
}

fn image(kind: &str, value: String, mime_type: &str) -> Part {
    let value = if kind == "base64" {
        value
            .split_once(";base64,")
            .map(|(_, payload)| payload.to_owned())
            .unwrap_or(value)
    } else {
        value
    };
    Part {
        r#type: "image".into(),
        data: Map::from_iter([(
            String::from("source"),
            json!({"kind": kind, "value": value}),
        )]),
        mime_type: Some(mime_type.into()),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let provider = OpenAI::builder()
        .base_url(env("HEART_MULTIMODAL_BASE_URL")?)
        .api_key(env("HEART_MULTIMODAL_API_KEY")?)
        .build()?;
    let model = provider.multimodal(multimodal::Config::new(env("HEART_MULTIMODAL_MODEL")?))?;

    let (kind, value) = match std::env::var("HEART_MULTIMODAL_IMAGE_DATA_URL") {
        Ok(value) if !value.trim().is_empty() => ("base64", value),
        _ => ("url", env("HEART_MULTIMODAL_IMAGE_URL")?),
    };
    let mime_type =
        std::env::var("HEART_MULTIMODAL_IMAGE_MIME").unwrap_or_else(|_| "image/png".into());

    let mut system = Message::new(Role::System);
    system.content.push(text("你是一个图片描述助手。"));
    let mut user = Message::new(Role::User);
    user.content.push(text("请用一句话描述这张图片。"));
    user.content.push(image(kind, value, &mime_type));

    let mut events = model
        .stream(multimodal::Request {
            messages: Messages::new([system, user]),
            modalities: vec!["text".into()],
            stream: true,
            ..multimodal::Request::default()
        })
        .await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => println!("DELTA {message:?}"),
            Event::Complete { message, usage, .. } => {
                println!("COMPLETE {message:?} usage={usage:?}");
            }
        }
    }
    Ok(())
}
