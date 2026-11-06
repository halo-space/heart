//! DingTalk Stream channel adapter.
//!
//! The adapter uses the Rust DingTalk Stream SDK for the WebSocket transport,
//! then exposes Heart's `Input` and `Message` contracts to the application
//! handler. Runtime state, routing IDs, and checkpoints remain application and
//! runtime responsibilities.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::pkg::dingtalk::{
    AsyncChatbotHandler, ChatbotMessage, ChatbotReplier, Credential, DingTalkStreamClient,
    MessageBody,
};
use futures_util::{Stream, StreamExt};
use serde_json::{Map, Value};

use crate::Error;
use crate::agent::{Input, Payload};
use crate::components::event::Event;
use crate::components::message::Role;

/// A parsed DingTalk chatbot message with the original platform message kept
/// for replying through the session webhook or OpenAPI.
#[derive(Clone, Debug)]
pub struct Inbound {
    pub input: Input,
    pub message_id: String,
    pub conversation_id: Option<String>,
    pub sender_id: Option<String>,
    message: ChatbotMessage,
}

impl Inbound {
    /// Parse one DingTalk Stream callback body into the shared Agent input.
    pub fn parse(body: &MessageBody) -> Result<Self, Error> {
        let value: Value = serde_json::from_str(&body.data)
            .map_err(|error| Error::new("INVALID_ARGUMENTS", error.to_string()))?;
        let message = ChatbotMessage::from_value(&value)
            .map_err(|error| Error::new("INVALID_ARGUMENTS", error.to_string()))?;
        let message_id = message
            .message_id
            .clone()
            .or_else(|| body.headers.message_id.clone())
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "DingTalk messageId is missing"))?;
        let text = message
            .get_text_list()
            .map(|parts| parts.join("\n"))
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "DingTalk message has no text"))?;

        let mut metadata = Map::new();
        metadata.insert("channel".into(), Value::String("dingtalk".into()));
        metadata.insert("message_id".into(), Value::String(message_id.clone()));
        insert_optional(
            &mut metadata,
            "conversation_id",
            message.conversation_id.clone(),
        );
        insert_optional(&mut metadata, "sender_id", message.sender_id.clone());
        insert_optional(
            &mut metadata,
            "conversation_type",
            message.conversation_type.clone(),
        );

        Ok(Self {
            input: Input {
                metadata,
                payload: vec![Payload {
                    r#type: "text".into(),
                    data: serde_json::json!({"value": text}),
                    mime_type: None,
                }],
            },
            message_id,
            conversation_id: message.conversation_id.clone(),
            sender_id: message.sender_id.clone(),
            message,
        })
    }

    pub fn message(&self) -> &ChatbotMessage {
        &self.message
    }
}

fn insert_optional(metadata: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if let Some(value) = value.filter(|value| !value.trim().is_empty()) {
        metadata.insert(key.into(), Value::String(value));
    }
}

/// Sends text back to the conversation that produced an inbound message.
#[derive(Clone)]
pub struct Reply {
    replier: ChatbotReplier,
    message: ChatbotMessage,
}

impl Reply {
    pub fn new(replier: ChatbotReplier, message: ChatbotMessage) -> Self {
        Self { replier, message }
    }

    pub async fn text(&self, text: &str) -> Result<(), Error> {
        self.replier
            .reply_text(text, &self.message)
            .await
            .map(|_| ())
            .map_err(|error| Error::new("TRANSPORT", error.to_string()))
    }

    /// Consume a Heart Event stream and send the final visible Assistant text.
    /// Delta events remain available to callers that need platform-specific
    /// draft updates, while ordinary DingTalk text replies close on Complete.
    pub async fn events<S>(&self, events: S) -> Result<(), Error>
    where
        S: Stream<Item = Result<Event, Error>> + Send + 'static,
    {
        futures_util::pin_mut!(events);
        while let Some(event) = events.next().await {
            match event {
                Err(error) => {
                    self.text(&error.message).await?;
                    return Err(error);
                }
                Ok(Event::Delta { .. }) => {}
                Ok(Event::Complete { message, .. }) => {
                    if message.role != Role::Assistant {
                        continue;
                    }
                    let text = message
                        .content
                        .iter()
                        .filter(|part| part.r#type == "text")
                        .filter_map(|part| part.data.get("value").and_then(Value::as_str))
                        .collect::<String>();
                    if !text.is_empty() {
                        self.text(&text).await?;
                    }
                }
            }
        }
        Ok(())
    }
}

type HandlerFuture = Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'static>>;

/// Application-owned binding from a DingTalk message to an Agent, Workflow,
/// or Harness entry point.
pub trait Handler: Send + Sync + 'static {
    fn handle(&self, inbound: Inbound, reply: Reply) -> HandlerFuture;
}

impl<F, Fut> Handler for F
where
    F: Fn(Inbound, Reply) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), Error>> + Send + 'static,
{
    fn handle(&self, inbound: Inbound, reply: Reply) -> HandlerFuture {
        Box::pin(self(inbound, reply))
    }
}

struct Callback<H> {
    handler: Arc<H>,
    replier: Arc<std::sync::OnceLock<ChatbotReplier>>,
    seen: Arc<Mutex<std::collections::VecDeque<String>>>,
}

impl<H> AsyncChatbotHandler for Callback<H>
where
    H: Handler,
{
    fn process(&self, body: &MessageBody) {
        let inbound = match Inbound::parse(body) {
            Ok(inbound) => inbound,
            Err(_) => return,
        };
        let accepted = self.seen.lock().map(|mut seen| {
            if seen
                .iter()
                .any(|message_id| message_id == &inbound.message_id)
            {
                false
            } else {
                seen.push_back(inbound.message_id.clone());
                if seen.len() > 4096 {
                    seen.pop_front();
                }
                true
            }
        });
        if !accepted.unwrap_or(false) {
            return;
        }
        let Some(replier) = self.replier.get().cloned() else {
            return;
        };
        let reply = Reply::new(replier, inbound.message.clone());
        let handler = Arc::clone(&self.handler);
        tokio::spawn(async move {
            let _ = handler.handle(inbound, reply).await;
        });
    }
}

/// Configuration for one DingTalk Stream channel instance.
#[derive(Clone, Debug)]
pub struct Config {
    pub client_id: String,
    pub client_secret: String,
}

impl Config {
    pub fn new(client_id: impl Into<String>, client_secret: impl Into<String>) -> Self {
        Self {
            client_id: client_id.into(),
            client_secret: client_secret.into(),
        }
    }

    fn validate(&self) -> Result<(), Error> {
        if self.client_id.trim().is_empty() || self.client_secret.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "DingTalk client_id and client_secret are required",
            ));
        }
        Ok(())
    }
}

/// DingTalk Stream channel. The caller owns the target binding and controls
/// how `Inbound::input` is routed to Agent, Workflow, or Harness.
pub struct Channel<H> {
    client: DingTalkStreamClient,
    _handler: std::marker::PhantomData<fn() -> H>,
}

impl<H> Channel<H>
where
    H: Handler,
{
    pub fn new(config: Config, handler: H) -> Result<Self, Error> {
        config.validate()?;
        let credential = Credential::new(config.client_id, config.client_secret);
        let replier = Arc::new(std::sync::OnceLock::new());
        let seen = Arc::new(Mutex::new(std::collections::VecDeque::new()));
        let client = DingTalkStreamClient::builder(credential)
            .register_async_chatbot_handler(
                ChatbotMessage::TOPIC,
                Callback {
                    handler: Arc::new(handler),
                    replier: Arc::clone(&replier),
                    seen,
                },
            )
            .build();
        let _ = replier.set(client.chatbot_replier());
        Ok(Self {
            client,
            _handler: std::marker::PhantomData,
        })
    }

    pub async fn start(mut self) -> Result<(), Error> {
        self.client
            .start()
            .await
            .map_err(|error| Error::new("TRANSPORT", error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, Inbound};
    use crate::pkg::dingtalk::{Headers, MessageBody};

    fn body(data: &str) -> MessageBody {
        MessageBody {
            spec_version: "1.0".into(),
            headers: Headers {
                message_id: Some("header-message".into()),
                ..Headers::default()
            },
            data: data.into(),
            extensions: Default::default(),
        }
    }

    #[test]
    fn parses_text_into_the_shared_input_contract() {
        let inbound = Inbound::parse(&body(
            r#"{
                "msgId":"message-1",
                "conversationId":"conversation-1",
                "senderId":"sender-1",
                "conversationType":"2",
                "msgtype":"text",
                "text":{"content":"北京天气怎么样"}
            }"#,
        ))
        .unwrap();

        assert_eq!(inbound.message_id, "message-1");
        assert_eq!(inbound.conversation_id.as_deref(), Some("conversation-1"));
        assert_eq!(inbound.sender_id.as_deref(), Some("sender-1"));
        assert_eq!(inbound.input.metadata["channel"], "dingtalk");
        assert_eq!(inbound.input.payload[0].r#type, "text");
        assert_eq!(inbound.input.payload[0].data["value"], "北京天气怎么样");
    }

    #[test]
    fn rejects_messages_without_text_or_identity() {
        assert!(
            Inbound::parse(&body(
                r#"{"msgId":"message-1","msgtype":"picture","content":{}}"#,
            ))
            .is_err()
        );
        let mut missing_identity = body(r#"{"msgtype":"text","text":{"content":"hello"}}"#);
        missing_identity.headers.message_id = None;
        assert!(Inbound::parse(&missing_identity).is_err());
    }

    #[test]
    fn config_requires_both_credentials() {
        assert!(Config::new("", "secret").validate().is_err());
        assert!(Config::new("client", "").validate().is_err());
        assert!(Config::new("client", "secret").validate().is_ok());
    }
}
