//! Live Gemini V2 TTS and multimodal acceptance tests.
//!
//! These tests are intentionally ignored by default. They require a real
//! Gemini API key and a real image Data URL supplied by the caller.

use futures_util::StreamExt;
use halo::agent::event::Event;
use halo::agent::message::{Message, Role};
use halo::agent::model::{
    content::part::Part,
    multimodal::{self, Multimodal as _},
    tts::{self, Tts as _},
};
use halo::agent::provider::gemini::Gemini;
use halo::agent::{Error, Messages};
use serde_json::{Map, Value, json};

fn required(name: &str) -> Result<String, Error> {
    std::env::var(name).map_err(|_| {
        Error::new(
            "AUTHENTICATION",
            format!("set {name} for Gemini acceptance"),
        )
    })
}

fn text_message(text: &str) -> Message {
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!(text))]),
    });
    message
}

fn image_message(data_url: String) -> Result<Message, Error> {
    let mime_type = data_url
        .strip_prefix("data:")
        .and_then(|value| value.split_once(';'))
        .map(|(mime, _)| mime.to_owned())
        .filter(|mime| !mime.trim().is_empty())
        .ok_or_else(|| {
            Error::new(
                "INVALID_ARGUMENTS",
                "GEMINI_TEST_IMAGE_DATA_URL must be a Data URL",
            )
        })?;
    let mut message = text_message("请描述这张真实图片。");
    message.content.push(Part {
        r#type: "image".into(),
        data: Map::from_iter([(
            String::from("source"),
            json!({
                "kind": "base64",
                "value": data_url,
                "mime_type": mime_type
            }),
        )]),
    });
    Ok(message)
}

fn has_audio(message: &Message) -> bool {
    message.content.iter().any(|part| {
        part.r#type == "audio"
            && part
                .data
                .get("source")
                .and_then(Value::as_object)
                .and_then(|source| source.get("value"))
                .and_then(Value::as_str)
                .is_some_and(|value| value.starts_with("data:audio/"))
    })
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY and sends a live Gemini TTS request"]
async fn gemini_tts_returns_real_audio_complete() {
    let provider = Gemini::builder()
        .api_key(required("GEMINI_API_KEY").unwrap())
        .build()
        .unwrap();
    let tts = provider
        .tts(tts::Config::new(
            std::env::var("GEMINI_TTS_MODEL").unwrap_or_else(|_| "gemini-3.8-flash-tts".into()),
        ))
        .unwrap();
    let mut events = tts
        .stream(tts::Request {
            messages: Messages::new([text_message("这是 Heart 的真实 Gemini TTS 验收。")]),
            voice: Some("Kore".into()),
            format: Some("audio/wav".into()),
            stream: false,
            ..tts::Request::default()
        })
        .await
        .unwrap();
    let mut complete = None;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, usage, .. } = event.unwrap() {
            complete = Some((message, usage));
        }
    }
    let (message, _usage) = complete.expect("Gemini TTS must emit Complete");
    assert!(has_audio(&message), "Gemini TTS must return an audio Part");
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY and GEMINI_TEST_IMAGE_DATA_URL"]
async fn gemini_multimodal_returns_real_complete() {
    let provider = Gemini::builder()
        .api_key(required("GEMINI_API_KEY").unwrap())
        .build()
        .unwrap();
    let model =
        std::env::var("GEMINI_MULTIMODAL_MODEL").unwrap_or_else(|_| "gemini-3.5-flash-lite".into());
    let multimodal = provider.multimodal(multimodal::Config::new(model)).unwrap();
    let mut events = multimodal
        .stream(multimodal::Request {
            messages: Messages::new([image_message(
                required("GEMINI_TEST_IMAGE_DATA_URL").unwrap(),
            )
            .unwrap()]),
            stream: false,
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
    let message = complete.expect("Gemini multimodal must emit Complete");
    assert!(
        message.content.iter().any(|part| part.r#type == "text"),
        "Gemini multimodal response must contain text"
    );
}
