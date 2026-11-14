//! Real DeepSeek Agent + Tool + Permission example.
//!
//! The model, tool selection and permission path are live. Provider settings
//! are loaded from the ignored `.env` file.
use futures_util::StreamExt;
use halo::agent::model::chat;
use halo::agent::permission::{Behavior, Decision, Request as PermissionRequest};
use halo::agent::provider::deepseek::DeepSeek;
use halo::agent::tool::{Call, Tool, ToolDefinition};
use halo::agent::{self, Agent, Cancellation, Error, Input, Message, Payload};
use serde_json::json;
use std::sync::LazyLock;

struct Lookup;

impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: LazyLock<ToolDefinition> = LazyLock::new(|| ToolDefinition {
            name: "lookup_weather".into(),
            description: "查询指定城市的天气。".into(),
            parameters: serde_json::from_value(json!({
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"],
                "additionalProperties":false
            }))
            .expect("static tool schema is valid"),
        });
        &DEFINITION
    }

    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        let city = call
            .arguments
            .get("city")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("北京");
        Ok(Message::function(json!({
            "city": city,
            "weather": "晴朗",
            "temperature_c": 22
        })))
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenvy::dotenv().ok();
    let provider =
        DeepSeek::builder()
            .base_url(
                std::env::var("DEEPSEEK_BASE_URL")
                    .unwrap_or_else(|_| "https://api.deepseek.com".into()),
            )
            .api_key(std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
                Error::new("AUTHENTICATION", "DEEPSEEK_API_KEY is required in .env")
            })?)
            .build()?;
    let chat = provider.chat(chat::Config::new(
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
    ))?;
    let policy = halo::agent::core::permission::Permission::new(
        |request: PermissionRequest, _: Cancellation| async move {
            if request.action == "call" && request.resource == "lookup_weather" {
                Ok(Decision {
                    behavior: Behavior::Allow,
                    reason: None,
                })
            } else {
                Ok(Decision {
                    behavior: Behavior::Deny,
                    reason: Some("tool is not allowed".into()),
                })
            }
        },
    );
    let agent = agent::react()
        .model(chat)
        .system_prompt("需要天气信息时必须调用 lookup_weather 工具，然后根据工具结果回答。")
        .tools(Lookup)
        .permission(policy)
        .build()?;
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"北京今天天气怎么样？"}),
            mime_type: None,
        }],
        ..Default::default()
    };
    let mut events = agent.run(input, true, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        println!("{:?}", event?);
    }
    Ok(())
}
