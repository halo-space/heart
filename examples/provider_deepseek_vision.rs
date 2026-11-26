//! Real DeepSeek-V4.1-Flash image input through the shared Chat contract.
//!
//! Required environment variables: `DEEPSEEK_API_KEY` and
//! `DEEPSEEK_TEST_IMAGE_DATA_URL`. Optional variables are
//! `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL`.

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
        data: Map::from_iter([(String::from("value"), json!(value))]),
        mime_type: None,
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let data_url = required("DEEPSEEK_TEST_IMAGE_DATA_URL")?;
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

    let mut system = Message::new(Role::System);
    system.content.push(text("你是一个简短的视觉理解助手。"));
    let mut user = Message::new(Role::User);
    user.content.push(text("请用一句话描述这张图片。"));
    user.content.push(Part {
        r#type: "image".into(),
        data: Map::from_iter([(
            String::from("source"),
            json!({"kind": "base64", "value": value}),
        )]),
        mime_type: Some(mime_type),
    });

    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(required("DEEPSEEK_API_KEY")?)
        .build()?;
    let model = provider.chat(chat::Config::new(
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
    ))?;
    let events = model
        .stream(chat::Request {
            messages: Messages::new([system, user]),
            stream: false,
            ..chat::Request::default()
        })
        .await?;
    futures_util::pin_mut!(events);
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } | Event::Complete { message, .. } => {
                for part in message.content {
                    if part.r#type == "text"
                        && let Some(value) = part.data.get("value").and_then(Value::as_str)
                    {
                        println!("{value}");
                    }
                }
            }
        }
    }
    Ok(())
}
