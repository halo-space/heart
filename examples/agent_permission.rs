//! Deterministic local model/tool example; no network, credentials or Runtime.
use std::sync::{
    LazyLock,
    atomic::{AtomicUsize, Ordering},
};

use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::event::Event,
    agent::message::Role,
    agent::model::{
        chat::{Chat, Request},
        content::part::Part,
    },
    agent::permission::{Behavior, Decision, Request as PermissionRequest},
    agent::tool::{Call, Tool, ToolDefinition},
    agent::{self, Agent, Input},
    core,
};
use serde_json::json;

struct Lookup;
impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: LazyLock<ToolDefinition> = LazyLock::new(|| ToolDefinition {
            name: "lookup".into(),
            description: "example lookup".into(),
            parameters: serde_json::from_value(json!({"type":"object","properties":{}})).unwrap(),
        });
        &DEFINITION
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        Ok(Message::function(json!({"temperature":22})))
    }
}

struct Model(AtomicUsize);
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"c1","name":"lookup","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        } else {
            message
                .values
                .insert("answer".into(), json!("lookup completed"));
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let policy = core::permission::Permission::new(
        |request: PermissionRequest, _: Cancellation| async move {
            if request.action == "call" && request.resource == "lookup" {
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
        .model(Model(AtomicUsize::new(0)))
        .tools(Lookup)
        .permission(policy)
        .build()?;
    let mut events = agent
        .run(Input::default(), true, Cancellation::new())
        .await?;
    while let Some(event) = events.next().await {
        println!("{:?}", event?);
    }
    Ok(())
}
