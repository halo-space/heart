use super::{Config, Request};
use crate::components::message::{Message, Messages, Role};
use crate::components::model::content::part::{Part, object};
use serde_json::json;

#[test]
fn request_requires_text_and_valid_speed() {
    let mut request = Request {
        messages: Messages::new([{
            let mut message = Message::new(Role::User);
            message.content.push(Part {
                r#type: "text".into(),
                data: object(json!({"value":"hello"})),
            });
            message
        }]),
        ..Request::default()
    };
    request.validate().unwrap();
    request.speed = Some(0.0);
    assert!(request.validate().is_err());
}

#[test]
fn config_keeps_voice_defaults_out_of_request() {
    let config = Config::new("voice-model");
    assert_eq!(config.model_name, "voice-model");
    assert!(config.options.is_empty());
}
