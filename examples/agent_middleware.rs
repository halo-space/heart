//! Run with a local deterministic model, no network, credentials or Runtime.
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::event::Event,
    agent::message::Role,
    agent::middleware::Middleware,
    agent::model::chat::{Chat, Request},
    agent::{self, Agent, Input, Payload},
};
use serde_json::{Value, json};

struct Answer;
impl Chat for Answer {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        message.values.insert("answer".into(), json!("hello"));
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

struct MarkAnswer;
impl Middleware for MarkAnswer {
    async fn before_model(&self, request: &mut Value, _: &Cancellation) -> Result<(), Error> {
        request["temperature"] = json!(0.2);
        Ok(())
    }
    async fn after_execute(&self, message: &mut Value, _: &Cancellation) -> Result<(), Error> {
        message["values"]["checked"] = json!(true);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let agent = agent::chat().model(Answer).middleware(MarkAnswer).build()?;
    let input = Input {
        metadata: Default::default(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"hello"}),
            mime_type: None,
        }],
    };
    let mut events = agent.run(input, true, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        println!("{:?}", event?);
    }
    Ok(())
}
