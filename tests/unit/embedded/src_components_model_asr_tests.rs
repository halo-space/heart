use super::{Config, Request};
use crate::components::message::{Message, Messages, Role};
use crate::components::model::content::part::{Part, object};
use serde_json::json;

fn audio_message() -> Message {
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "audio".into(),
        data: object(json!({
            "source": {"kind": "base64", "value": "AAAA"}
        })),
        mime_type: Some("audio/wav".into()),
    });
    message
}

#[test]
fn request_requires_shared_audio_part() {
    let mut request = Request {
        messages: Messages::new([audio_message()]),
        ..Request::default()
    };
    request.validate().unwrap();
    request.messages.0[0].content[0].r#type = "text".into();
    assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");
}

#[test]
fn request_rejects_missing_audio_source() {
    let mut message = audio_message();
    message.content[0].data = object(json!({"value": "AAAA"}));
    let request = Request {
        messages: Messages::new([message]),
        ..Request::default()
    };
    assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");
}

#[test]
fn config_keeps_provider_defaults_separate() {
    let config = Config::new("speech-model");
    assert_eq!(config.model_name, "speech-model");
    assert!(config.options.is_empty());
}
