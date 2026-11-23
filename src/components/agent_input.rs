//! Shared projection from the Agent input envelope into a user Message.

use serde_json::{Map, Value};

use crate::{
    agent::Input,
    components::message::{Message, Role},
    components::model::content::part::Part,
};

#[doc(hidden)]
pub fn user_message(input: Input) -> Message {
    let content = input
        .payload
        .into_iter()
        .map(|payload| {
            let data = match payload.data {
                Value::Object(object) => object,
                value => Map::from_iter([(String::from("value"), value)]),
            };
            Part {
                r#type: payload.r#type,
                data,
                mime_type: payload.mime_type,
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
    for key in ["chat_id", "node_id", "exec_id", "reply_id"] {
        message.metadata.remove(key);
    }
    message
}
