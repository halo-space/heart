//! Live DeepSeek-V4.1-Flash image and file input acceptance tests.
//!
//! These tests are ignored by default. They require a real DeepSeek API key
//! and a real image Data URL supplied by the caller.

use futures_util::StreamExt;
use halo::agent::event::Event;
use halo::agent::message::{Message, Role};
use halo::agent::model::chat;
use halo::agent::model::content::part::Part;
use halo::agent::provider::deepseek::DeepSeek;
use halo::agent::{Error, Messages};
use serde_json::{Map, Value, json};

fn required(name: &str) -> Result<String, Error> {
    std::env::var(name).map_err(|_| Error::new("AUTHENTICATION", format!("set {name}")))
}

fn text(value: &str) -> Part {
    Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!(value))]),
        mime_type: None,
    }
}

fn media_message(kind: &str, source_kind: &str, source_value: &str, mime: Option<&str>) -> Message {
    let mut user = Message::new(Role::User);
    user.content.push(text(match kind {
        "file" => "请读取这个文件并用一句话说明内容。",
        _ => "请用一句话描述这张图片。",
    }));
    let mut source = Map::from_iter([
        ("kind".into(), json!(source_kind)),
        ("value".into(), json!(source_value)),
    ]);
    if kind == "file" {
        source.insert("filename".into(), json!("sample.png"));
    }
    user.content.push(Part {
        r#type: kind.into(),
        data: Map::from_iter([("source".into(), Value::Object(source))]),
        mime_type: mime.map(str::to_owned),
    });
    user
}

async fn complete_text(model: &impl chat::Chat, messages: Messages) -> Result<String, Error> {
    let events = model
        .stream(chat::Request {
            messages,
            stream: false,
            ..chat::Request::default()
        })
        .await?;
    futures_util::pin_mut!(events);
    let mut text = String::new();
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event? {
            for part in message.content {
                if part.r#type == "text"
                    && let Some(value) = part.data.get("value").and_then(Value::as_str)
                {
                    text.push_str(value);
                }
            }
        }
    }
    if text.trim().is_empty() {
        return Err(Error::new("INVALID_RESPONSE", "DeepSeek returned no text"));
    }
    Ok(text)
}

fn messages_with_media(
    kind: &str,
    source_kind: &str,
    source_value: &str,
    mime: Option<&str>,
) -> Messages {
    let mut system = Message::new(Role::System);
    system.content.push(text("你是一个简短的视觉理解助手。"));
    let mut user = media_message(kind, source_kind, source_value, mime);
    let mut content = vec![text("请用一句话描述这个输入。")];
    content.extend(user.content);
    user.content = content;
    Messages::new([system, user])
}

fn deepseek() -> Result<DeepSeek, Error> {
    DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(required("DEEPSEEK_API_KEY")?)
        .build()
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and DEEPSEEK_TEST_IMAGE_DATA_URL"]
async fn deepseek_flash_accepts_real_image_input() {
    let provider = deepseek().unwrap();
    let model = provider
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .unwrap();
    let data_url = required("DEEPSEEK_TEST_IMAGE_DATA_URL").unwrap();
    let value = data_url
        .split_once(";base64,")
        .map(|(_, value)| value)
        .unwrap_or(&data_url);
    let result = complete_text(
        &model,
        messages_with_media("image", "base64", value, Some("image/png")),
    )
    .await;
    assert!(result.is_ok(), "DeepSeek image request failed: {result:?}");
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and DEEPSEEK_TEST_IMAGE_DATA_URL"]
async fn deepseek_flash_accepts_real_file_input() {
    let provider = deepseek().unwrap();
    let model = provider
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .unwrap();
    let data_url = required("DEEPSEEK_TEST_IMAGE_DATA_URL").unwrap();
    let value = data_url
        .split_once(";base64,")
        .map(|(_, value)| value)
        .unwrap_or(&data_url);
    let result = complete_text(
        &model,
        messages_with_media("file", "base64", value, Some("image/png")),
    )
    .await;
    assert!(result.is_ok(), "DeepSeek file request failed: {result:?}");
}
