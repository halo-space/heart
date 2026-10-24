//! Offline Agent + short history + explicitly selected long memory.
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::event::Event,
    agent::memory::{
        self, Memory,
        short::{self, chat, session},
    },
    agent::message::Role,
    agent::model::{
        chat::{Chat, Request},
        content::part::Part,
    },
    agent::{self, Agent, Input, Payload},
    core,
};
use serde_json::{Map, json};

use chat::Store as _;
use session::Store as _;

struct Model;
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        for message in request.messages.as_slice() {
            for part in &message.content {
                if let Some(value) = part.data.get("value").and_then(serde_json::Value::as_str) {
                    println!("{:?}: {value}", message.role);
                }
            }
        }
        let mut message = Message::new(Role::Assistant);
        message.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"天气晴朗，适合规划露营。"})
                .as_object()
                .unwrap()
                .clone(),
        });
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Each backend initializes its own internal storage. The composition step
    // only combines the already-created short and long capabilities.
    let short = core::memory::short::InMemory::new();
    let long = core::memory::InMemory::new();
    let memory = memory::Config::new(short, long).with_scopes(["user", "agent"]);
    let cancellation = Cancellation::new();
    memory
        .session()
        .create(
            session::Session {
                id: 101,
                tenant_id: 1,
                user_id: 2,
                r#type: short::Type::Agent,
                title: Some("天气与露营".into()),
                summary: None,
                usage: None,
                status: session::Status::Active,
                metadata: Map::new(),
                created_time: 0,
                updated_time: 0,
            },
            &cancellation,
        )
        .await?;
    let mut answer = Message::new(Role::Assistant);
    answer.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"北京今天晴。"}).as_object().unwrap().clone(),
    });
    memory
        .chat()
        .create(
            chat::Chat {
                id: 201,
                tenant_id: 1,
                user_id: 2,
                agent_id: 3,
                session_id: 101,
                r#type: short::Type::Agent,
                status: chat::Status::Completed,
                input: Some(chat::Request {
                    metadata: Map::new(),
                    payload: vec![chat::Payload {
                        r#type: "text".into(),
                        data: json!({"value":"北京天气怎么样？"}),
                        mime_type: None,
                    }],
                }),
                message: Some(answer),
                usage: None,
                graph: None,
                state: None,
                metadata: Map::new(),
                started_time: None,
                completed_time: None,
                created_time: 0,
                updated_time: 0,
            },
            &cancellation,
        )
        .await?;
    memory
        .long()
        .write(
            memory::Request {
                id: 401,
                r#type: memory::Type::Preference,
                user_id: 2,
                agent_id: None,
                session_id: None,
                from: 1,
                to: 2,
                content: "用户喜欢亲近自然的周末活动。".into(),
                metadata: None,
                vector: None,
            },
            &cancellation,
        )
        .await?;

    let agent = agent::chat().model(Model).memory(memory.clone()).build()?;
    let input = Input {
        metadata: json!({"tenant_id":1,"user_id":2,"agent_id":3,"session_id":101,"chat_id":202})
            .as_object()
            .unwrap()
            .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"天气好的话，我想出去露营。"}),
            mime_type: None,
        }],
    };
    let mut events = agent.run(input, true, cancellation).await?;
    while let Some(event) = events.next().await {
        println!("{:?}", event?);
    }
    let saved = memory.chat().read(202, &Cancellation::new()).await?;
    assert_eq!(saved.status, chat::Status::Completed);
    println!("Saved Chat {}: {:?}", saved.id, saved.status);
    // Graph/state require the actual integrated Runtime State; no fake logs.
    Ok(())
}
