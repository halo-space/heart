//! Live DeepSeek tool-image context acceptance.
//!
//! This test uses a real tool-shaped Message returned by a screenshot tool.
//! It is ignored by default and requires a live DeepSeek API request.

use futures_util::StreamExt;
use halo::agent::event::Event;
use halo::agent::message::{Message, Role};
use halo::agent::model::chat::{self, Chat as _};
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
        data: Map::from_iter([("value".into(), json!(value))]),
        mime_type: None,
    }
}

fn image_tool_message(data_url: String) -> Result<Message, Error> {
    let mime_type = data_url
        .strip_prefix("data:")
        .and_then(|value| value.split_once(';'))
        .map(|(mime, _)| mime.to_owned())
        .filter(|mime| !mime.trim().is_empty())
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "image input must be a Data URL"))?;
    let value = data_url
        .split_once(";base64,")
        .map(|(_, payload)| payload.to_owned())
        .unwrap_or(data_url);
    let mut message = Message::new(Role::Tool);
    message.metadata.insert("call_id".into(), json!("call_1"));
    message
        .metadata
        .insert("name".into(), json!("get_screenshot"));
    message.content.push(Part {
        r#type: "image".into(),
        data: Map::from_iter([("source".into(), json!({"kind": "base64", "value": value}))]),
        mime_type: Some(mime_type),
    });
    Ok(message)
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and DEEPSEEK_TEST_IMAGE_DATA_URL"]
async fn deepseek_reads_real_image_from_tool_message() {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(required("DEEPSEEK_API_KEY").unwrap())
        .build()
        .unwrap();
    let model = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into());
    let options = Map::from_iter([("thinking".into(), json!({"type": "disabled"}))]);
    let chat = provider
        .chat(chat::Config::new(model).options(options))
        .unwrap();

    let mut system = Message::new(Role::System);
    system.content.push(text("你是一个简短的视觉理解助手。"));
    let mut user = Message::new(Role::User);
    user.content.push(text(
        "截图工具已经返回结果，请根据工具图片回答截图中有什么。",
    ));
    let mut assistant = Message::new(Role::Assistant);
    assistant.content.push(Part {
        r#type: "tool_call".into(),
        data: Map::from_iter([
            ("call_id".into(), json!("call_1")),
            ("name".into(), json!("get_screenshot")),
            ("arguments".into(), json!({})),
        ]),
        mime_type: None,
    });

    let mut events = chat
        .stream(chat::Request {
            messages: Messages::new([
                system,
                user,
                assistant,
                image_tool_message(required("DEEPSEEK_TEST_IMAGE_DATA_URL").unwrap()).unwrap(),
            ]),
            stream: false,
            ..chat::Request::default()
        })
        .await
        .unwrap();
    let mut answer = String::new();
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event.unwrap() {
            for part in message.content {
                if part.r#type == "text"
                    && let Some(value) = part.data.get("value").and_then(Value::as_str)
                {
                    answer.push_str(value);
                }
            }
        }
    }
    assert!(!answer.trim().is_empty(), "DeepSeek returned no answer");
}
