//! A real Provider response is the source of the Chat closeout persisted by
//! the Agent; this acceptance test never fabricates a model event.
use chat::Store as _;
use futures_util::StreamExt;
use halo::agent::{Cancellation, Message};
use halo::{
    agent::{
        self, Agent, Input, Payload,
        event::Event,
        memory::{
            self,
            short::{chat, session},
        },
        message::Role,
        model::chat::Config as ChatConfig,
        provider::deepseek::DeepSeek,
    },
    core,
};
use serde_json::{Map, json};
use session::Store as _;

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends a live model request"]
async fn real_provider_complete_and_usage_are_saved_at_chat_closeout() {
    let memory = memory::Config::new(
        memory::short::Pair::new(
            core::memory::short::session::InMemory::new(),
            core::memory::short::chat::InMemory::new(),
        ),
        core::memory::InMemory::new(),
    );
    memory
        .session()
        .create(
            session::Session {
                id: 10,
                tenant_id: 1,
                user_id: 2,
                r#type: memory::short::Type::Agent,
                title: None,
                summary: None,
                usage: None,
                status: session::Status::Active,
                metadata: Map::new(),
                created_time: 0,
                updated_time: 0,
            },
            &Cancellation::new(),
        )
        .await
        .expect("create session");

    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct real DeepSeek provider");
    let model = provider
        .chat(ChatConfig::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct real Chat");
    let agent = agent::chat()
        .model(model)
        .memory(memory.clone())
        .build()
        .expect("build memory-backed Chat Agent");

    let chat_id = 813_610;
    let input = Input {
        metadata: json!({
            "tenant_id": 1,
            "user_id": 2,
            "agent_id": 3,
            "session_id": 10,
            "chat_id": chat_id,
        })
        .as_object()
        .expect("metadata object")
        .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"2 加 2 等于几？请用一句简短中文回答。"}),
            mime_type: None,
        }],
    };
    let mut events = agent
        .run(input, true, Cancellation::new())
        .await
        .expect("start real Chat run");
    let mut final_message: Option<Message> = None;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event.expect("real Agent event")
            && message.role == Role::Assistant
        {
            final_message = Some(message);
        }
    }
    let final_message = final_message.expect("real provider should return an assistant message");
    assert!(final_message.content.iter().any(|part| {
        part.r#type == "text"
            && part
                .data
                .get("value")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|text| !text.is_empty())
    }));

    let saved = memory
        .chat()
        .read(chat_id, &Cancellation::new())
        .await
        .expect("read the saved Chat closeout");
    assert_eq!(saved.status, chat::Status::Completed);
    let saved_message = saved.message.as_ref().expect("saved assistant message");
    assert_eq!(saved_message.role, final_message.role);
    assert_eq!(saved_message.content, final_message.content);
    assert_eq!(saved_message.values, final_message.values);
    assert_eq!(saved_message.usage, final_message.usage);
    assert!(!saved_message.metadata.contains_key("chat_id"));
    let saved_usage = saved.usage.as_ref().expect("aggregated Chat Usage");
    let provider_usage = final_message.usage.as_ref().expect("real Provider Usage");
    assert_eq!(saved_usage.input, provider_usage.input);
    assert_eq!(saved_usage.output, provider_usage.output);
    assert_eq!(saved_usage.total, provider_usage.total);
    assert!(saved.state.as_ref().is_some_and(|state| state.is_object()));
    println!(
        "LIVE_CHAT_CLOSEOUT chat_id={} status={:?} usage={:?}",
        saved.id, saved.status, saved.usage
    );
}
