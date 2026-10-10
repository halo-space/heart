use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::model::content::Part;
use crate::components::model::token::Usage;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
    Function,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Part>,
    pub values: Map<String, Value>,
    pub metadata: Map<String, Value>,
    pub usage: Option<Usage>,
}

impl Message {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            content: Vec::new(),
            values: Map::new(),
            metadata: Map::new(),
            usage: None,
        }
    }
    /// Creates a function-role message from structured function output values.
    pub fn function(values: Value) -> Self {
        let values = match values {
            Value::Object(values) => values,
            value => Map::from_iter([(String::from("value"), value)]),
        };

        Self {
            role: Role::Function,
            values,
            content: Vec::new(),
            metadata: Map::new(),
            usage: None,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize, JsonSchema)]
pub struct Messages(pub Vec<Message>);

impl Messages {
    pub fn new(messages: impl IntoIterator<Item = Message>) -> Self {
        Self(messages.into_iter().collect())
    }

    pub fn push(&mut self, message: Message) {
        self.0.push(message);
    }

    pub fn as_slice(&self) -> &[Message] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
