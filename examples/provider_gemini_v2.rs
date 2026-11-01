//! Real Gemini V2 TTS and multimodal Provider example.
//!
//! Required environment variables:
//! - `GEMINI_API_KEY`
//! - `GEMINI_TEST_IMAGE_URL` or `GEMINI_TEST_IMAGE_DATA_URL` for the multimodal request

use futures_util::StreamExt;
use halo::agent::model::{
    content::part::Part,
    multimodal::{self, Multimodal},
    tts::{self, Tts},
};
use halo::agent::provider::gemini::Gemini;
use halo::agent::{Error, Message, Messages, event::Event, message::Role};
use serde_json::{Map, json};

fn env(name: &str) -> Result<String, Error> {
    std::env::var(name).map_err(|_| Error::new("INVALID_ARGUMENTS", format!("{name} is required")))
}

fn text_message(text: &str) -> Message {
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!(text))]),
    });
    message
}

fn image_message(prompt: &str, source_kind: &str, source_value: &str, mime_type: &str) -> Message {
    let mut message = text_message(prompt);
    message.content.push(Part {
        r#type: "image".into(),
        data: Map::from_iter([(
            String::from("source"),
            json!({
                "kind": source_kind,
                "value": source_value,
                "mime_type": mime_type
            }),
        )]),
    });
    message
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let provider = Gemini::builder().api_key(env("GEMINI_API_KEY")?).build()?;

    let tts = provider.tts(tts::Config::new("gemini-3.8-flash-tts"))?;
    let mut audio_events = tts
        .stream(tts::Request {
            messages: Messages::new([text_message("这是 Heart 的 Gemini TTS 真实测试。")]),
            voice: Some("Kore".into()),
            format: Some("audio/wav".into()),
            stream: false,
            ..tts::Request::default()
        })
        .await?;
    while let Some(event) = audio_events.next().await {
        match event? {
            Event::Complete { message, .. } => {
                println!("TTS complete: {} content part(s)", message.content.len());
            }
            Event::Delta { .. } => {}
        }
    }

    let image_mime =
        std::env::var("GEMINI_TEST_IMAGE_MIME").unwrap_or_else(|_| "image/jpeg".into());
    let (image_kind, image_value) = match std::env::var("GEMINI_TEST_IMAGE_DATA_URL") {
        Ok(value) if !value.trim().is_empty() => ("base64", value),
        _ => ("url", env("GEMINI_TEST_IMAGE_URL")?),
    };
    let multimodal_model =
        std::env::var("GEMINI_MULTIMODAL_MODEL").unwrap_or_else(|_| "gemini-3.8-flash".into());
    let multimodal = provider.multimodal(multimodal::Config::new(multimodal_model))?;
    let mut vision_events = multimodal
        .stream(multimodal::Request {
            messages: Messages::new([image_message(
                "请描述这张图片。",
                image_kind,
                &image_value,
                &image_mime,
            )]),
            stream: false,
            ..multimodal::Request::default()
        })
        .await?;
    while let Some(event) = vision_events.next().await {
        match event? {
            Event::Complete { message, .. } => {
                println!(
                    "multimodal complete: {} content part(s)",
                    message.content.len()
                );
            }
            Event::Delta { .. } => {}
        }
    }
    Ok(())
}
