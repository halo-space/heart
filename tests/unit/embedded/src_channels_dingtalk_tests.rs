use super::{Config, Inbound};
use dingtalk_stream::messages::frames::{Headers, MessageBody};

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
    assert!(
        Inbound::parse(&body(
            r#"{
                    "msgId":"message-2",
                    "msgtype":"richText",
                    "content":{"richText":[{"text":"这不是纯文本输入"}]}
                }"#,
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
