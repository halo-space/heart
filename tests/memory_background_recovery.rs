//! I09 acceptance: a persisted Session Memory scene is resumed by a fresh
//! process. The worker uses a real DeepSeek Chat and real Elasticsearch
//! short/long-memory backends; no model or persistence mock is involved.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use elasticsearch::{
    Elasticsearch,
    http::transport::{SingleNodeConnectionPool, TransportBuilder},
    indices::IndicesDeleteParts,
};
use futures_util::StreamExt;
use halo::{
    agent::{self, Agent, Cancellation, Error, Input, Payload},
    agent::{core, memory, model::chat, provider::deepseek::DeepSeek},
};
use serde_json::{Map, Value, json};

use halo::agent::memory::short::{
    self, Store as _, chat::Store as _, session, session::Store as _,
};

fn environment() -> (Elasticsearch, String, String, String) {
    let url = std::env::var("HEART_ES_URL").expect("set HEART_ES_URL");
    let suffix = std::env::var("HEART_I09_SUFFIX").unwrap_or_else(|_| "default".into());
    let session_index = format!("heart-i09-session-{suffix}");
    let chat_index = format!("heart-i09-chat-{suffix}");
    let long_index = format!("heart-i09-long-{suffix}");
    let transport = TransportBuilder::new(SingleNodeConnectionPool::new(url.parse().unwrap()))
        .disable_proxy()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("build Elasticsearch transport");
    (
        Elasticsearch::new(transport),
        session_index,
        chat_index,
        long_index,
    )
}

fn provider() -> Result<DeepSeek, Error> {
    DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
            Error::new(
                "INVALID_ARGUMENTS",
                "set DEEPSEEK_API_KEY for I09 acceptance",
            )
        })?)
        .timeout(60)
        .build()
}

fn text(value: impl Into<String>) -> halo::agent::Message {
    let mut message = halo::agent::Message::new(halo::agent::message::Role::System);
    message
        .content
        .push(halo::agent::model::content::part::Part {
            r#type: "text".into(),
            data: json!({"value": value.into()}).as_object().unwrap().clone(),
        });
    message
}

fn pending_scene() -> Value {
    json!({
        "stage": "summary",
        "base": {
            "summary": null,
            "usage": null,
            "from": null,
            "to": null
        },
        "summary": "这是已经持久化、等待提交的摘要。",
        "usage": null,
        "from": 1,
        "to": 1,
        "extraction": null
    })
}

fn id() -> i64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_millis();
    i64::try_from(millis).expect("timestamp fits i64")
}

async fn worker() -> Result<(), Error> {
    let chat_id = std::env::var("HEART_I09_CHAT_ID")
        .expect("worker chat id")
        .parse::<i64>()
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid worker chat id"))?;
    let session_id = std::env::var("HEART_I09_SESSION_ID")
        .expect("worker session id")
        .parse::<i64>()
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid worker session id"))?;
    let (client, session_index, chat_index, long_index) = environment();
    let short = core::memory::short::es::new(
        client.clone(),
        core::memory::short::es::Config::new(session_index, chat_index),
    )?;
    let long = core::memory::long::es::new(client, long_index)?;
    let compression_chat = provider()?.chat(chat::Config::new(
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
    ))?;
    let model = provider()?.chat(chat::Config::new(
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
    ))?;
    let compressor = core::compression::Compressor::new(
        compression_chat,
        chat::Request {
            messages: halo::agent::Messages::new([text(
                "只返回一段简短摘要正文，不要解释，不要 Markdown。",
            )]),
            stream: true,
            max_tokens: Some(128),
            ..Default::default()
        },
    )?
    .with_size(1);
    let memory = memory::Config::new(short, long).with_compression(compressor);
    let agent = agent::chat().model(model).memory(memory.clone()).build()?;
    let input = Input {
        metadata: Map::from_iter([
            ("tenant_id".into(), json!(1)),
            ("user_id".into(), json!(2)),
            ("agent_id".into(), json!(3)),
            ("session_id".into(), json!(session_id)),
            ("chat_id".into(), json!(chat_id)),
        ]),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"请只回复 I09_WORKER_OK"}),
            mime_type: None,
        }],
    };
    let mut events = agent.run(input, false, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        event?;
    }
    let mut session = memory
        .session()
        .read(session_id, &Cancellation::new())
        .await?;
    for _ in 0..60 {
        if !session.metadata.contains_key("memory") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        session = memory
            .session()
            .read(session_id, &Cancellation::new())
            .await?;
    }
    if session.metadata.contains_key("memory") {
        return Err(Error::new(
            "TIMEOUT",
            "worker finished before the persisted Memory scene was closed",
        ));
    }
    if session.summary.as_deref() != Some("这是已经持久化、等待提交的摘要。") {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "worker did not commit the persisted summary",
        ));
    }
    Ok(())
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and an isolated Elasticsearch service"]
async fn persisted_memory_scene_recovers_in_a_fresh_process() {
    let suffix = format!("{}-{}", std::process::id(), id());
    let (client, session_index, chat_index, long_index) = {
        let url = std::env::var("HEART_ES_URL").expect("set HEART_ES_URL");
        let transport = TransportBuilder::new(SingleNodeConnectionPool::new(url.parse().unwrap()))
            .disable_proxy()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();
        (
            Elasticsearch::new(transport),
            format!("heart-i09-session-{suffix}"),
            format!("heart-i09-chat-{suffix}"),
            format!("heart-i09-long-{suffix}"),
        )
    };
    for index in [&session_index, &chat_index, &long_index] {
        client
            .indices()
            .create(elasticsearch::indices::IndicesCreateParts::Index(index))
            .send()
            .await
            .expect("create isolated I09 index");
    }
    let session_id = id();
    let chat_id = session_id + 1;
    let short = core::memory::short::es::new(
        client.clone(),
        core::memory::short::es::Config::new(session_index.clone(), chat_index.clone()),
    )
    .unwrap();
    let session = session::Session {
        id: session_id,
        tenant_id: 1,
        user_id: 2,
        r#type: short::Type::Agent,
        title: Some("I09 recovery".into()),
        summary: None,
        usage: None,
        status: session::Status::Active,
        metadata: Map::from_iter([("memory".into(), pending_scene())]),
        created_time: 0,
        updated_time: 0,
    };
    short
        .session()
        .create(session, &Cancellation::new())
        .await
        .unwrap();
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "persisted_memory_scene_worker",
            "--nocapture",
        ])
        .env("HEART_I09_SUFFIX", &suffix)
        .env("HEART_I09_SESSION_ID", session_id.to_string())
        .env("HEART_I09_CHAT_ID", chat_id.to_string())
        .output()
        .expect("start fresh I09 worker");
    assert!(
        result.status.success(),
        "worker failed\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let recovered = short
        .session()
        .read(session_id, &Cancellation::new())
        .await
        .unwrap();
    assert!(!recovered.metadata.contains_key("memory"));
    assert_eq!(
        recovered.summary.as_deref(),
        Some("这是已经持久化、等待提交的摘要。")
    );
    short
        .session()
        .delete(session_id, &Cancellation::new())
        .await
        .unwrap();
    short
        .chat()
        .delete(chat_id, &Cancellation::new())
        .await
        .ok();
    client
        .indices()
        .delete(IndicesDeleteParts::Index(&[
            &session_index,
            &chat_index,
            &format!("heart-i09-long-{suffix}"),
        ]))
        .send()
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "explicit subprocess worker; invoked by persisted_memory_scene_recovers_in_a_fresh_process"]
async fn persisted_memory_scene_worker() {
    worker().await.expect("I09 worker recovery");
}
