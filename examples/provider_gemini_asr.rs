//! Real Gemini audio transcription example.
//!
//! Required environment variables:
//! - `GEMINI_API_KEY`
//! - `GEMINI_TEST_AUDIO_DATA_URL`, a `data:<mime>;base64,<payload>` value

use futures_util::StreamExt;
use halo::agent::event::Event;
use halo::agent::message::{Message, Role};
use halo::agent::model::{
    asr::{self, Asr},
    content::part::Part,
};
use halo::agent::provider::gemini::Gemini;
use halo::agent::{Error, Messages};
use serde_json::{Map, json};

fn env(name: &str) -> Result<String, Error> {
    std::env::var(name).map_err(|_| Error::new("INVALID_ARGUMENTS", format!("{name} is required")))
}

fn audio_message(data_url: &str) -> Result<Message, Error> {
    let (header, value) = data_url.split_once(",").ok_or_else(|| {
        Error::new(
            "INVALID_ARGUMENTS",
            "GEMINI_TEST_AUDIO_DATA_URL must be a Data URL",
        )
    })?;
    let mime_type = header
        .strip_prefix("data:")
        .and_then(|value| value.split_once(';'))
        .map(|(mime, _)| mime)
        .filter(|mime| !mime.trim().is_empty())
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "audio Data URL has no MIME type"))?;
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "text".into(),
        data: Map::from_iter([(
            String::from("value"),
            json!("请完整转写这段音频中的语音，只输出转写文本，不要描述音频。"),
        )]),
        mime_type: None,
    });
    message.content.push(Part {
        r#type: "audio".into(),
        data: Map::from_iter([(
            String::from("source"),
            json!({"kind": "base64", "value": value}),
        )]),
        mime_type: Some(mime_type.into()),
    });
    Ok(message)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let provider = Gemini::builder().api_key(env("GEMINI_API_KEY")?).build()?;
    let model = std::env::var("GEMINI_ASR_MODEL")
        .or_else(|_| std::env::var("GEMINI_MULTIMODAL_MODEL"))
        .unwrap_or_else(|_| "gemini-3.5-flash-lite".into());
    let asr = provider.asr(asr::Config::new(model))?;
    let mut events = asr
        .stream(asr::Request {
            messages: Messages::new([audio_message(&env("GEMINI_TEST_AUDIO_DATA_URL")?)?]),
            stream: false,
            ..asr::Request::default()
        })
        .await?;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event? {
            for part in message.content {
                if part.r#type == "text"
                    && let Some(text) = part.data.get("value").and_then(|value| value.as_str())
                {
                    println!("{text}");
                }
            }
        }
    }
    Ok(())
}
