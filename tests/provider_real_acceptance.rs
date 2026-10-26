//! Opt-in real-provider acceptance. It sends a live request to the configured
//! VLLM endpoint; no HTTP fixture or scripted provider response is used.

use futures_util::StreamExt;
use halo::agent::{
    event::Event,
    message::{Message, Messages, Role},
    model::{
        chat::{self, Chat as _},
        content::part::Part,
    },
    provider::vllm::{Config, Vllm},
};
use serde_json::json;

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn configured_vllm_returns_a_real_assistant_message_and_usage() {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL").expect("set HEART_VLLM_BASE_URL"),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("set HEART_VLLM_API_KEY"),
        ..Config::default()
    })
    .expect("construct VLLM provider");
    let model_name = std::env::var("HEART_VLLM_MODEL").expect("set HEART_VLLM_MODEL");
    let model = provider
        .chat(
            chat::Config::new(model_name.clone()).options(serde_json::Map::from_iter([(
                "chat_template_kwargs".into(),
                json!({"enable_thinking":false}),
            )])),
        )
        .expect("construct Chat model");

    let mut user = Message::new(Role::User);
    user.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"用一句话说明什么是京杭大运河。"})
            .as_object()
            .unwrap()
            .clone(),
    });
    let events = model
        .stream(chat::Request {
            messages: Messages::new([user]),
            stream: true,
            max_tokens: Some(128),
            ..Default::default()
        })
        .await
        .expect("send request to the real VLLM provider")
        .collect::<Vec<_>>()
        .await;

    let complete = events
        .iter()
        .filter_map(|event| match event.as_ref().expect("provider event") {
            Event::Complete {
                message,
                usage,
                finish_reason,
            } => Some((message, usage, finish_reason)),
            Event::Delta { .. } => None,
        })
        .next_back()
        .expect("real provider must return a Complete event");
    let (message, usage, finish_reason) = complete;

    assert_eq!(message.role, Role::Assistant);
    assert!(
        !message.content.is_empty() || !message.values.is_empty(),
        "real provider returned an empty assistant message: {message:?}"
    );
    assert_eq!(message.metadata["provider_name"], "vllm");
    assert_eq!(message.metadata["model_name"], model_name);
    assert!(
        usage
            .as_ref()
            .and_then(|usage| usage.total)
            .unwrap_or_default()
            > 0
    );
    assert!(
        finish_reason
            .as_deref()
            .is_some_and(|reason| !reason.is_empty())
    );

    println!("LIVE_PROVIDER_MESSAGE={message:?}");
    println!("LIVE_PROVIDER_USAGE={usage:?} FINISH_REASON={finish_reason:?}");
}
