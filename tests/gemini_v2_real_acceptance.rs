//! Live Gemini V2 TTS and multimodal acceptance tests.
//!
//! These tests are intentionally ignored by default. They require a real
//! Gemini API key and real image/audio Data URLs supplied by the caller.

use futures_util::StreamExt;
use halo::agent::event::Event;
use halo::agent::message::{Message, Role};
use halo::agent::model::{
    chat::{self, Chat},
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
        mime_type: None,
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
    let value = data_url
        .split_once(";base64,")
        .map(|(_, payload)| payload.to_owned())
        .unwrap_or(data_url);
    let mut message = text_message("请描述这张真实图片。");
    message.content.push(Part {
        r#type: "image".into(),
        data: Map::from_iter([(
            String::from("source"),
            json!({
                "kind": "base64",
                "value": value
            }),
        )]),
        mime_type: Some(mime_type),
    });
    Ok(message)
}

fn audio_message(data_url: String) -> Result<Message, Error> {
    let mime_type = data_url
        .strip_prefix("data:")
        .and_then(|value| value.split_once(';'))
        .map(|(mime, _)| mime.to_owned())
        .filter(|mime| !mime.trim().is_empty())
        .ok_or_else(|| {
            Error::new(
                "INVALID_ARGUMENTS",
                "GEMINI_TEST_AUDIO_DATA_URL must be a Data URL",
            )
        })?;
    let value = data_url
        .split_once(";base64,")
        .map(|(_, payload)| payload.to_owned())
        .unwrap_or(data_url);
    let mut message = text_message("请描述这段真实音频。");
    message.content.push(Part {
        r#type: "audio".into(),
        data: Map::from_iter([(
            String::from("source"),
            json!({"kind": "base64", "value": value}),
        )]),
        mime_type: Some(mime_type),
    });
    Ok(message)
}

fn video_message(data_url: String) -> Result<Message, Error> {
    let mime_type = data_url
        .strip_prefix("data:")
        .and_then(|value| value.split_once(';'))
        .map(|(mime, _)| mime.to_owned())
        .filter(|mime| !mime.trim().is_empty())
        .ok_or_else(|| {
            Error::new(
                "INVALID_ARGUMENTS",
                "GEMINI_TEST_VIDEO_DATA_URL must be a Data URL",
            )
        })?;
    let value = data_url
        .split_once(";base64,")
        .map(|(_, payload)| payload.to_owned())
        .unwrap_or(data_url);
    let mut message = text_message("请描述这段真实视频。");
    message.content.push(Part {
        r#type: "video".into(),
        data: Map::from_iter([(
            String::from("source"),
            json!({"kind": "base64", "value": value}),
        )]),
        mime_type: Some(mime_type),
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
                .is_some_and(|value| !value.trim().is_empty())
    })
}

async fn complete_chat(model: &impl chat::Chat, messages: Messages) -> Result<Message, Error> {
    let events = model
        .stream(chat::Request {
            messages,
            stream: false,
            ..chat::Request::default()
        })
        .await?;
    futures_util::pin_mut!(events);
    let mut complete = None;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event? {
            complete = Some(message);
        }
    }
    complete.ok_or_else(|| Error::new("INVALID_RESPONSE", "Gemini Chat emitted no Complete"))
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY and sends a live Gemini Chat request"]
async fn gemini_chat_returns_real_text_complete() {
    let provider = Gemini::builder()
        .api_key(required("GEMINI_API_KEY").unwrap())
        .build()
        .unwrap();
    let model = provider
        .chat(chat::Config::new(
            std::env::var("GEMINI_CHAT_MODEL")
                .or_else(|_| std::env::var("GEMINI_MULTIMODAL_MODEL"))
                .unwrap_or_else(|_| "gemini-3.8-flash".into()),
        ))
        .unwrap();
    let mut system = Message::new(Role::System);
    system.content.push(Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!("你是一个简短的助手。"))]),
        mime_type: None,
    });
    let mut events = model
        .stream(chat::Request {
            messages: Messages::new([system, text_message("请只回复：Gemini Chat 已接通。")]),
            stream: false,
            ..chat::Request::default()
        })
        .await
        .unwrap();
    let mut complete = None;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event.unwrap() {
            complete = Some(message);
        }
    }
    let message = complete.expect("Gemini Chat must emit Complete");
    assert!(
        message.content.iter().any(|part| part.r#type == "text"),
        "Gemini Chat response must contain text"
    );
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY and sends two live Gemini Chat requests"]
async fn gemini_chat_continues_with_turn_id() {
    let provider = Gemini::builder()
        .api_key(required("GEMINI_API_KEY").unwrap())
        .build()
        .unwrap();
    let model = provider
        .chat(chat::Config::new(
            std::env::var("GEMINI_CHAT_MODEL")
                .or_else(|_| std::env::var("GEMINI_MULTIMODAL_MODEL"))
                .unwrap_or_else(|_| "gemini-3.8-flash".into()),
        ))
        .unwrap();

    let mut system = Message::new(Role::System);
    system.content.push(Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!("你是一个简短的助手。"))]),
        mime_type: None,
    });
    let first_user = text_message("请记住这个暗号：蓝色河流。只回复已记住。");
    let first = complete_chat(&model, Messages::new([system.clone(), first_user.clone()]))
        .await
        .unwrap();
    assert!(
        first
            .metadata
            .get("turn_id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .is_some(),
        "Gemini Chat Complete must include turn_id"
    );

    let second = complete_chat(
        &model,
        Messages::new([
            system,
            first_user,
            first,
            text_message("我刚才让你记住的暗号是什么？只回复暗号。"),
        ]),
    )
    .await
    .unwrap();
    assert!(
        second
            .metadata
            .get("turn_id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .is_some(),
        "continued Gemini Chat Complete must include turn_id"
    );
    assert!(
        second.content.iter().any(|part| {
            part.r#type == "text"
                && part
                    .data
                    .get("value")
                    .and_then(Value::as_str)
                    .is_some_and(|value| value.contains("蓝色河流"))
        }),
        "continued Gemini Chat response must use the previous turn"
    );
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

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY and GEMINI_TEST_AUDIO_DATA_URL"]
async fn gemini_multimodal_accepts_real_audio_input() {
    let provider = Gemini::builder()
        .api_key(required("GEMINI_API_KEY").unwrap())
        .build()
        .unwrap();
    let model =
        std::env::var("GEMINI_MULTIMODAL_MODEL").unwrap_or_else(|_| "gemini-3.5-flash-lite".into());
    let multimodal = provider.multimodal(multimodal::Config::new(model)).unwrap();
    let mut events = multimodal
        .stream(multimodal::Request {
            messages: Messages::new([audio_message(
                required("GEMINI_TEST_AUDIO_DATA_URL").unwrap(),
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
    let message = complete.expect("Gemini audio multimodal must emit Complete");
    assert!(
        message.content.iter().any(|part| part.r#type == "text"),
        "Gemini audio multimodal response must contain text"
    );
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY and GEMINI_TEST_VIDEO_DATA_URL"]
async fn gemini_multimodal_accepts_real_video_input() {
    let provider = Gemini::builder()
        .api_key(required("GEMINI_API_KEY").unwrap())
        .build()
        .unwrap();
    let model =
        std::env::var("GEMINI_MULTIMODAL_MODEL").unwrap_or_else(|_| "gemini-3.5-flash-lite".into());
    let multimodal = provider.multimodal(multimodal::Config::new(model)).unwrap();
    let mut events = multimodal
        .stream(multimodal::Request {
            messages: Messages::new([video_message(
                required("GEMINI_TEST_VIDEO_DATA_URL").unwrap(),
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
    let message = complete.expect("Gemini video multimodal must emit Complete");
    assert!(
        message.content.iter().any(|part| part.r#type == "text"),
        "Gemini video multimodal response must contain text"
    );
}
