use super::{Config, Inbound};
use larksuite_oapi_sdk_rs::events::im::{Message, MessageSender, P2MessageReceiveV1};

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
