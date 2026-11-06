//! Feishu/Lark long-connection channel adapter.
//!
//! The adapter uses Feishu's WebSocket event channel, converts ordinary text
//! messages into Heart's shared `Input`, deduplicates platform deliveries and
//! replies through the Feishu message API. Runtime state, routing IDs and
//! checkpoints remain application and runtime responsibilities.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::Error;
use crate::agent::{Input, Payload};
use crate::components::event::Event;
use crate::components::message::Role;
use crate::pkg::feishu::{
    EventDispatcher, LarkClient, LarkError, MessageSender, P2MessageReceiveV1, RequestOption,
    WsClient,
};
use futures_util::{Stream, StreamExt};
use serde_json::{Map, Value};

/// A parsed Feishu message with the platform identifiers retained as routing
/// metadata for the application.
#[derive(Clone, Debug)]
pub struct Inbound {
    pub input: Input,
    pub message_id: String,
    pub chat_id: Option<String>,
    pub sender_id: Option<String>,
    pub chat_type: Option<String>,
}

impl Inbound {
    pub(crate) fn parse(event: P2MessageReceiveV1) -> Result<Self, Error> {
        let message = event.message;
        if message.message_type != "text" {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "Feishu channel currently accepts text messages only",
            ));
        }
        let message_id = nonempty(message.message_id)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "Feishu message_id is missing"))?;
        let text = parse_text(&message.content)
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "Feishu message has no text"))?;
        let sender_id = sender_id(&event.sender);
        let chat_id = nonempty(message.chat_id);
        let chat_type = nonempty(message.chat_type);

        let mut metadata = Map::new();
        metadata.insert("channel".into(), Value::String("feishu".into()));
        metadata.insert("message_id".into(), Value::String(message_id.clone()));
        insert_optional(&mut metadata, "chat_id", chat_id.clone());
        insert_optional(&mut metadata, "chat_type", chat_type.clone());
        insert_optional(&mut metadata, "sender_id", sender_id.clone());
        insert_optional(
            &mut metadata,
            "sender_type",
            nonempty(event.sender.sender_type),
        );
        insert_optional(
            &mut metadata,
            "tenant_key",
            nonempty(event.sender.tenant_key),
        );
        insert_optional(&mut metadata, "root_id", nonempty(message.root_id));
        insert_optional(&mut metadata, "parent_id", nonempty(message.parent_id));
        insert_optional(&mut metadata, "thread_id", nonempty(message.thread_id));

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
            chat_id,
            sender_id,
            chat_type,
        })
    }
}

fn parse_text(content: &str) -> Option<String> {
    serde_json::from_str::<Value>(content)
        .ok()
        .and_then(|value| value.get("text").and_then(Value::as_str).map(str::to_owned))
}

fn sender_id(sender: &MessageSender) -> Option<String> {
    sender
        .sender_id
        .as_ref()
        .and_then(|id| {
            id.open_id()
                .or_else(|| id.user_id())
                .or_else(|| id.union_id())
        })
        .map(str::to_owned)
}

fn nonempty(value: String) -> Option<String> {
    (!value.trim().is_empty()).then_some(value)
}

fn insert_optional(metadata: &mut Map<String, Value>, key: &str, value: Option<String>) {
    if let Some(value) = value {
        metadata.insert(key.into(), Value::String(value));
    }
}

/// Sends text as a reply to the Feishu message that produced an inbound event.
#[derive(Clone)]
pub struct Reply {
    client: LarkClient,
    message_id: String,
}

impl Reply {
    pub async fn text(&self, text: &str) -> Result<(), Error> {
        self.client
            .channel_messaging()
            .reply_text(&self.message_id, text, false, &RequestOption::default())
            .await
            .map(|_| ())
            .map_err(|error| Error::new("TRANSPORT", error.to_string()))
    }

    /// Consume a Heart Event stream and send each visible Assistant text from
    /// a Complete event. Delta events are ignored by this adapter.
    pub async fn events<S>(&self, events: S) -> Result<(), Error>
    where
        S: Stream<Item = Result<Event, Error>> + Send,
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

pub type HandlerFuture = Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'static>>;

/// Application-owned binding from a Feishu message to an Agent, Workflow or
/// Harness entry point.
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
    client: LarkClient,
    seen: Arc<Mutex<std::collections::VecDeque<String>>>,
}

impl<H> Callback<H>
where
    H: Handler,
{
    async fn process(&self, event: P2MessageReceiveV1) -> Result<(), LarkError> {
        let Ok(inbound) = Inbound::parse(event) else {
            return Ok(());
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
            return Ok(());
        }
        let reply = Reply {
            client: self.client.clone(),
            message_id: inbound.message_id.clone(),
        };
        let handler = Arc::clone(&self.handler);
        tokio::spawn(async move {
            let _ = handler.handle(inbound, reply).await;
        });
        Ok(())
    }
}

/// Configuration for one Feishu/Lark long-connection channel instance.
#[derive(Clone, Debug)]
pub struct Config {
    pub app_id: String,
    pub app_secret: String,
}

impl Config {
    pub fn new(app_id: impl Into<String>, app_secret: impl Into<String>) -> Self {
        Self {
            app_id: app_id.into(),
            app_secret: app_secret.into(),
        }
    }

    fn validate(&self) -> Result<(), Error> {
        if self.app_id.trim().is_empty() || self.app_secret.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "Feishu app_id and app_secret are required",
            ));
        }
        Ok(())
    }
}

/// Feishu WebSocket channel. The caller owns the target binding and controls
/// how `Inbound::input` is routed to Agent, Workflow or Harness.
pub struct Channel<H> {
    client: WsClient,
    _handler: std::marker::PhantomData<fn() -> H>,
}

impl<H> Channel<H>
where
    H: Handler,
{
    pub fn new(config: Config, handler: H) -> Result<Self, Error> {
        config.validate()?;
        let client = LarkClient::builder(config.app_id, config.app_secret)
            .build()
            .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
        let seen = Arc::new(Mutex::new(std::collections::VecDeque::new()));
        let callback = Arc::new(Callback {
            handler: Arc::new(handler),
            client: client.clone(),
            seen,
        });
        let dispatcher = EventDispatcher::new("", "").on_p2_im_message_receive_v1(move |event| {
            let callback = Arc::clone(&callback);
            async move { callback.process(event).await }
        });
        Ok(Self {
            client: client.ws_client(dispatcher).auto_reconnect(true),
            _handler: std::marker::PhantomData,
        })
    }

    pub async fn start(self) -> Result<(), Error> {
        self.client
            .start()
            .await
            .map_err(|error| Error::new("TRANSPORT", error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, Inbound};
    use crate::pkg::feishu::{Message, MessageSender, P2MessageReceiveV1};

    fn event(message_type: &str, content: &str) -> P2MessageReceiveV1 {
        let mut message = Message::default();
        message.message_id = "om_001".into();
        message.chat_id = "oc_001".into();
        message.chat_type = "p2p".into();
        message.message_type = message_type.into();
        message.content = content.into();
        let mut sender = MessageSender::default();
        sender.sender_type = "user".into();
        sender.tenant_key = "tenant_001".into();
        let mut event = P2MessageReceiveV1::default();
        event.message = message;
        event.sender = sender;
        event
    }

    #[test]
    fn parses_text_into_the_shared_input_contract() {
        let inbound = Inbound::parse(event("text", r#"{"text":"北京天气怎么样"}"#)).unwrap();
        assert_eq!(inbound.message_id, "om_001");
        assert_eq!(inbound.chat_id.as_deref(), Some("oc_001"));
        assert_eq!(inbound.input.metadata["channel"], "feishu");
        assert_eq!(inbound.input.payload[0].r#type, "text");
        assert_eq!(inbound.input.payload[0].data["value"], "北京天气怎么样");
    }

    #[test]
    fn rejects_non_text_or_empty_messages() {
        assert!(Inbound::parse(event("post", r#"{"title":"title"}"#)).is_err());
        assert!(Inbound::parse(event("text", r#"{"text":""}"#)).is_err());
        assert!(Inbound::parse(event("text", "not-json")).is_err());
    }

    #[test]
    fn config_requires_both_credentials() {
        assert!(Config::new("", "secret").validate().is_err());
        assert!(Config::new("app", "").validate().is_err());
        assert!(Config::new("app", "secret").validate().is_ok());
    }
}
