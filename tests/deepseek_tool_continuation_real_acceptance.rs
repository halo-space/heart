//! Live DeepSeek multi-turn tool continuation acceptance.
//!
//! The first request asks the real model to call `current_time`. The test then
//! sends the returned assistant tool call together with a real local timestamp
//! and verifies that the next model request produces the final text response.
//! It is ignored by default because it sends live requests and consumes quota.

use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use halo::agent::event::Event;
use halo::agent::message::{Message, Messages, Role};
use halo::agent::model::{chat, chat::ToolChoice, content::part::Part};
use halo::agent::provider::deepseek::DeepSeek;
use halo::agent::{Error, tool::ToolDefinition};
use serde_json::{Map, Value, json};

fn required(name: &str) -> Result<String, Error> {
    std::env::var(name).map_err(|_| Error::new("AUTHENTICATION", format!("set {name}")))
}

fn text_part(value: impl Into<String>) -> Part {
    Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!(value.into()))]),
        mime_type: None,
    }
}

fn text_message(role: Role, value: impl Into<String>) -> Message {
    let mut message = Message::new(role);
    message.content.push(text_part(value));
    message
}

fn tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "current_time".into(),
        description: "读取当前 Unix 时间戳（秒）".into(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }))
        .expect("valid tool schema"),
    }
}

async fn complete(model: &impl chat::Chat, request: chat::Request) -> Result<Message, Error> {
    let events = model.stream(request).await?;
    futures_util::pin_mut!(events);
    let mut complete = None;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event? {
            complete = Some(message);
        }
    }
    complete.ok_or_else(|| Error::new("INVALID_RESPONSE", "DeepSeek emitted no Complete"))
}

fn tool_result(call_id: &str, name: &str, timestamp: u64) -> Message {
    let mut message = Message::new(Role::Tool);
    message.metadata.insert("call_id".into(), json!(call_id));
    message.metadata.insert("name".into(), json!(name));
    message.content.push(text_part(
        serde_json::to_string(&json!({"unix_seconds": timestamp})).unwrap(),
    ));
    message
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends two live Chat requests"]
async fn deepseek_chat_continues_after_real_tool_call() {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(required("DEEPSEEK_API_KEY").unwrap())
        .build()
        .unwrap();
    let tools = vec![tool_definition()];
    let model = provider
        .chat(
            chat::Config::new(
                std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
            )
            .options(Map::from_iter([(
                "thinking".into(),
                json!({"type": "disabled"}),
            )])),
        )
        .unwrap();

    let system = text_message(
        Role::System,
        "你是一个工具调用助手。需要时间时必须调用 current_time，不能猜测时间。",
    );
    let user = text_message(Role::User, "现在的 Unix 时间是多少？必须调用工具后再回答。");
    let first = complete(
        &model,
        chat::Request {
            messages: Messages::new([system.clone(), user.clone()]),
            tools: tools.clone(),
            tool_choice: Some(ToolChoice::Required),
            stream: false,
            ..chat::Request::default()
        },
    )
    .await
    .unwrap();

    let call = first
        .content
        .iter()
        .find(|part| part.r#type == "tool_call")
        .expect("DeepSeek must return a tool call");
    let call_id = call
        .data
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .expect("tool call must include call_id")
        .to_owned();
    let name = call
        .data
        .get("name")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .expect("tool call must include name")
        .to_owned();
    assert_eq!(name, "current_time");

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after Unix epoch")
        .as_secs();
    let second = complete(
        &model,
        chat::Request {
            messages: Messages::new([system, user, first, tool_result(&call_id, &name, timestamp)]),
            tools,
            tool_choice: Some(ToolChoice::None),
            stream: false,
            ..chat::Request::default()
        },
    )
    .await
    .unwrap();
    assert!(
        second.content.iter().any(|part| {
            part.r#type == "text"
                && part
                    .data
                    .get("value")
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.trim().is_empty())
        }),
        "DeepSeek must produce final text after the tool result"
    );
}
