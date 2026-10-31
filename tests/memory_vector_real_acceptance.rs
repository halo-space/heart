//! Real long-memory vector preparation acceptance.
//!
//! The test uses a real DeepSeek Agent, a real Zhipu Embedding-3 component and
//! the real in-memory vector-capable backend. Compression/extraction are
//! application components that preserve the actual historical input so this
//! acceptance isolates vector preparation from model output formatting drift.

use std::time::Duration;

use futures_util::StreamExt;
use halo::agent::{self, Agent, Cancellation, Error, Input, Message, Payload};
use halo::agent::{
    core,
    memory::short::{self, chat, session},
    memory::{self, Memory, extraction::Extractor as ExtractorContract},
    message::Role,
    model::{self, embed::Embed as _},
    provider::{deepseek::DeepSeek, zhipu::Zhipu},
};
use serde_json::{Map, Value, json};

use chat::Store as _;
use session::Store as _;

fn next_id() -> i64 {
    static NEXT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(8_300_000);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(model::content::part::Part {
        r#type: "text".into(),
        data: json!({"value": value}).as_object().unwrap().clone(),
    });
    message
}

fn deepseek() -> Result<DeepSeek, Error> {
    DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(
            std::env::var("DEEPSEEK_API_KEY")
                .map_err(|_| Error::new("AUTHENTICATION", "set DEEPSEEK_API_KEY"))?,
        )
        .timeout(90)
        .build()
}

struct InputCompressor;

impl halo::agent::compression::Compressor for InputCompressor {
    fn size(&self) -> Option<u64> {
        Some(1)
    }

    async fn compress(
        &self,
        input: halo::agent::Messages,
        _: &Cancellation,
    ) -> Result<halo::agent::Messages, Error> {
        let content = input
            .as_slice()
            .iter()
            .flat_map(|message| &message.content)
            .filter_map(|part| part.data.get("value").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(halo::agent::Messages::new([text(
            Role::Assistant,
            &format!("历史内容：{content}"),
        )]))
    }
}

struct InputExtractor;

impl ExtractorContract for InputExtractor {
    async fn extract(
        &self,
        input: &halo::agent::Messages,
        _: &Cancellation,
    ) -> Result<halo::agent::Messages, Error> {
        let content = input
            .as_slice()
            .iter()
            .flat_map(|message| &message.content)
            .filter_map(|part| part.data.get("value").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(" ");
        let mut message = Message::new(Role::Assistant);
        message.values = json!({"user":{"fact":format!("历史内容：{content}")}})
            .as_object()
            .unwrap()
            .clone();
        Ok(halo::agent::Messages::new([message]))
    }

    async fn merge(
        &self,
        current: &memory::Item,
        content: &str,
        _: &Cancellation,
    ) -> Result<String, Error> {
        Ok(format!("{}；{}", current.content, content))
    }
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and HEART_EMBED_API_KEY"]
async fn real_long_memory_writeback_prepares_and_stores_vector() -> Result<(), Error> {
    let deepseek = deepseek()?;
    let model_name = std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into());
    let model = deepseek.chat(model::chat::Config::new(model_name))?;

    let zhipu = Zhipu::new(halo::agent::provider::zhipu::Config {
        base_url: std::env::var("HEART_EMBED_BASE_URL")
            .unwrap_or_else(|_| "https://open.bigmodel.cn/api/paas/v4".into()),
        api_key: std::env::var("HEART_EMBED_API_KEY")
            .map_err(|_| Error::new("AUTHENTICATION", "set HEART_EMBED_API_KEY"))?,
        ..Default::default()
    })?;
    let embed = zhipu.embed(model::embed::Config {
        model_name: std::env::var("HEART_EMBED_MODEL").unwrap_or_else(|_| "embedding-3".into()),
        options: Map::new(),
    })?;
    let probe = embed
        .embed(model::embed::Request {
            input: "周末露营前查看天气和风力".into(),
            dimensions: None,
            options: Map::new(),
        })
        .await?;
    let probe_vector = probe.values["vector"]
        .as_array()
        .ok_or_else(|| Error::new("INVALID_RESPONSE", "probe embedding has no vector"))?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or_else(|| Error::new("INVALID_RESPONSE", "probe vector is not numeric"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let long = core::memory::InMemory::with_vector_dimension(probe_vector.len())?;
    let short = core::memory::short::InMemory::new();
    let memory = memory::Config::new(short, long)
        .with_compression(InputCompressor)
        .with_extractor(InputExtractor)
        .with_id_fn(next_id)
        .with_embed(embed);
    let cancellation = Cancellation::new();
    memory
        .session()
        .create(
            session::Session {
                id: 8_300_001,
                tenant_id: 1,
                user_id: 8_300_002,
                r#type: short::Type::Agent,
                title: None,
                summary: None,
                usage: None,
                status: session::Status::Active,
                metadata: Map::new(),
                created_time: 1,
                updated_time: 1,
            },
            &cancellation,
        )
        .await?;
    memory
        .chat()
        .create(
            chat::Chat {
                id: 8_300_003,
                tenant_id: 1,
                user_id: 8_300_002,
                agent_id: 8_300_004,
                session_id: 8_300_001,
                r#type: short::Type::Agent,
                status: chat::Status::Completed,
                input: Some(chat::Request {
                    metadata: Map::from_iter([
                        ("tenant_id".into(), json!(1)),
                        ("user_id".into(), json!(8_300_002)),
                        ("agent_id".into(), json!(8_300_004)),
                        ("session_id".into(), json!(8_300_001)),
                    ]),
                    payload: vec![chat::Payload {
                        r#type: "text".into(),
                        data: json!({"value":"用户明确偏好周末去露营，并希望以后使用简体中文简洁回答"}),
                        mime_type: None,
                    }],
                }),
                message: Some(text(Role::Assistant, "出发前应查看天气和风力。")),
                usage: Some(model::token::Usage {
                    input: Some(100),
                    output: Some(100),
                    total: Some(200),
                    ..Default::default()
                }),
                graph: None,
                state: None,
                metadata: Map::new(),
                started_time: Some(1),
                completed_time: Some(2),
                created_time: 1,
                updated_time: 2,
            },
            &cancellation,
        )
        .await?;

    let agent = agent::chat()
        .model(model)
        .system_prompt("只回复 VECTOR_MEMORY_OK")
        .memory(memory.clone())
        .build()?;
    let input = Input {
        metadata: Map::from_iter([
            ("tenant_id".into(), json!(1)),
            ("user_id".into(), json!(8_300_002)),
            ("agent_id".into(), json!(8_300_004)),
            ("session_id".into(), json!(8_300_001)),
            ("chat_id".into(), json!(8_300_005)),
        ]),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"请只回复 VECTOR_MEMORY_OK"}),
            mime_type: None,
        }],
    };
    let mut events = agent.run(input, false, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        event?;
    }

    for _ in 0..120 {
        for kind in ["summary", "fact", "preference"] {
            let options = Map::from_iter([(
                "filter".into(),
                json!({"user_id":8300002,"agent_id":null,"session_id":null,"type":kind}),
            )]);
            let result = memory
                .long()
                .vector(
                    memory::vector::Request {
                        vector: probe_vector.clone(),
                        options,
                    },
                    &Cancellation::new(),
                )
                .await?;
            if !result.is_empty() {
                println!("LIVE_MEMORY_VECTOR_CONTENT={}", result[0].content);
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Err(Error::new(
        "TIMEOUT",
        "automatic long-memory vector writeback did not complete",
    ))
}
