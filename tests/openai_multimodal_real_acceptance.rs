//! Opt-in real OpenAI-compatible multimodal acceptance.
//!
//! The endpoint must expose a vision-capable model. No canned response or
//! protocol fixture is used.

use futures_util::StreamExt;
use halo::agent::event::Event;
use halo::agent::message::{Message, Role};
use halo::agent::model::{
    content::part::Part,
    multimodal::{self, Multimodal as _},
};
use halo::agent::provider::openai::OpenAI;
use halo::agent::{Error, Messages};
use serde_json::{Map, json};

fn env(name: &str) -> Result<String, Error> {
    std::env::var(name).map_err(|_| Error::new("INVALID_ARGUMENTS", format!("set {name}")))
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

#[tokio::test]
#[ignore = "requires a real vision-capable OpenAI-compatible endpoint and image"]
async fn real_openai_compatible_multimodal_returns_text() {
    let provider = OpenAI::builder()
        .base_url(env("HEART_MULTIMODAL_BASE_URL").unwrap())
        .api_key(env("HEART_MULTIMODAL_API_KEY").unwrap())
        .build()
        .unwrap();
    let model = provider
        .multimodal(multimodal::Config::new(
            env("HEART_MULTIMODAL_MODEL").unwrap(),
        ))
        .unwrap();
    let (kind, value) = match std::env::var("HEART_MULTIMODAL_IMAGE_DATA_URL") {
        Ok(value) if !value.trim().is_empty() => ("base64", value),
        _ => ("url", env("HEART_MULTIMODAL_IMAGE_URL").unwrap()),
    };
    let mime_type =
        std::env::var("HEART_MULTIMODAL_IMAGE_MIME").unwrap_or_else(|_| "image/png".into());

    let mut system = Message::new(Role::System);
    system.content.push(text("You describe images briefly."));
    let mut user = Message::new(Role::User);
    user.content
        .push(text("Describe this image in one sentence."));
    user.content.push(image(kind, value, &mime_type));

    let mut events = model
        .stream(multimodal::Request {
            messages: Messages::new([system, user]),
            modalities: vec!["text".into()],
            stream: true,
            ..multimodal::Request::default()
        })
        .await
        .unwrap();
    let mut complete = None;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event.unwrap() {
            complete = Some(message);
        }
    }
    let message = complete.expect("multimodal provider must emit Complete");
    assert!(message.content.iter().any(|part| part.r#type == "text"));
}
