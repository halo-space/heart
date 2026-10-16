//! Shared projection from the Agent input envelope into a user Message.

use serde_json::{Map, Value};

use crate::{
    agent::Input,
    components::message::{Message, Role},
    components::model::content::part::Part,
};

pub(crate) fn user_message(input: Input) -> Message {
    let content = input
        .payload
        .into_iter()
        .map(|payload| {
            let mut data = match payload.data {
                Value::Object(object) => object,
                value => Map::from_iter([(String::from("value"), value)]),
            };
            if let Some(mime_type) = payload.mime_type {
                data.insert("mime_type".into(), Value::String(mime_type));
            }
            Part {
                r#type: payload.r#type,
                data,
            }
        })
        .collect();
    let mut message = Message {
        role: Role::User,
        content,
        values: Map::new(),
        metadata: input.metadata,
        usage: None,
    };
    crate::runtime::observation::clear_source(&mut message);
    message
}
