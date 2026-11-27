//! Live structured-output acceptance for the OpenAI-compatible providers.
//!
//! These tests are ignored by default because they spend provider quota. They
//! use the configured real providers and validate the shared `Message` output,
//! rather than inspecting provider-specific response JSON.

use futures_util::StreamExt;
use halo::agent::Error;
use halo::agent::event::Event;
use halo::agent::message::{Message, Messages, Role};
use halo::agent::model::{chat, chat::ResponseFormat, content::part::Part};
use halo::agent::provider::{
    deepseek::DeepSeek,
    vllm::{Config as VllmConfig, Vllm},
};
use serde_json::{Map, Value, json};

fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: Map::from_iter([(String::from("value"), json!(value))]),
        mime_type: None,
    });
    message
}

async fn complete(model: &impl chat::Chat, request: chat::Request) -> Result<Message, Error> {
    let events = model.stream(request).await?;
    futures_util::pin_mut!(events);
    let mut result = None;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event? {
            result = Some(message);
        }
    }
    result.ok_or_else(|| Error::new("INVALID_RESPONSE", "provider emitted no Complete"))
}

fn json_object(message: &Message) -> Value {
    let value = message
        .content
        .iter()
        .find(|part| part.r#type == "text")
        .and_then(|part| part.data.get("value"))
        .and_then(Value::as_str)
        .expect("structured output must contain a text part");
    let parsed: Value = serde_json::from_str(value).expect("structured output must be JSON");
    assert!(
        parsed.is_object(),
        "structured output must be a JSON object"
    );
    parsed
}

fn answer_schema() -> ResponseFormat {
    ResponseFormat::JsonSchema {
        name: "answer".into(),
        schema: json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        }),
        strict: true,
    }
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends a live structured-output request"]
async fn deepseek_returns_real_json_object_output() {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct DeepSeek provider");
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
        .expect("construct DeepSeek Chat");
    let message = complete(
        &model,
        chat::Request {
            messages: Messages::new([
                text(Role::System, "只输出合法 JSON 对象，不要 Markdown 或解释。"),
                text(Role::User, "返回一个对象，字段 answer 的值为北京。"),
            ]),
            response_format: Some(ResponseFormat::JsonObject),
            stream: false,
            ..chat::Request::default()
        },
    )
    .await
    .expect("DeepSeek structured-output request");
    let value = json_object(&message);
    assert_eq!(value.get("answer").and_then(Value::as_str), Some("北京"));
}

#[tokio::test]
#[ignore = "requires HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn vllm_returns_real_json_object_output() {
    let provider = Vllm::new(VllmConfig {
        base_url: std::env::var("HEART_VLLM_BASE_URL").expect("set HEART_VLLM_BASE_URL"),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("set HEART_VLLM_API_KEY"),
        ..VllmConfig::default()
    })
    .expect("construct VLLM provider");
    let model = provider
        .chat(
            chat::Config::new(std::env::var("HEART_VLLM_MODEL").expect("set HEART_VLLM_MODEL"))
                .options(Map::from_iter([(
                    "chat_template_kwargs".into(),
                    json!({"enable_thinking": false}),
                )])),
        )
        .expect("construct VLLM Chat");
    let message = complete(
        &model,
        chat::Request {
            messages: Messages::new([
                text(Role::System, "只输出合法 JSON 对象，不要 Markdown 或解释。"),
                text(Role::User, "返回一个对象，字段 answer 的值为北京。"),
            ]),
            response_format: Some(ResponseFormat::JsonObject),
            stream: false,
            ..chat::Request::default()
        },
    )
    .await
    .expect("VLLM structured-output request");
    let value = json_object(&message);
    assert_eq!(value.get("answer").and_then(Value::as_str), Some("北京"));
}

#[tokio::test]
#[ignore = "requires HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn vllm_returns_real_json_schema_output() {
    let provider = Vllm::new(VllmConfig {
        base_url: std::env::var("HEART_VLLM_BASE_URL").expect("set HEART_VLLM_BASE_URL"),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("set HEART_VLLM_API_KEY"),
        ..VllmConfig::default()
    })
    .expect("construct VLLM provider");
    let model = provider
        .chat(
            chat::Config::new(std::env::var("HEART_VLLM_MODEL").expect("set HEART_VLLM_MODEL"))
                .options(Map::from_iter([(
                    "chat_template_kwargs".into(),
                    json!({"enable_thinking": false}),
                )])),
        )
        .expect("construct VLLM Chat");
    let message = complete(
        &model,
        chat::Request {
            messages: Messages::new([
                text(
                    Role::System,
                    "只输出符合 Schema 的合法 JSON 对象，不要 Markdown 或解释。",
                ),
                text(Role::User, "返回一个对象，字段 answer 的值为北京。"),
            ]),
            response_format: Some(answer_schema()),
            stream: false,
            ..chat::Request::default()
        },
    )
    .await
    .expect("VLLM JSON Schema request");
    let value = json_object(&message);
    assert_eq!(value, Value::Object(message.values.clone()));
    assert_eq!(value.get("answer").and_then(Value::as_str), Some("北京"));
}
