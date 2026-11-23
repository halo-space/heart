use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Canonical ordered content part used by `Message`.
pub mod part {
    use super::*;

    /// Convert JSON into the canonical Part data map.
    ///
    /// Message content keeps its stable object-shaped `data` contract. Scalar
    /// input is represented by the same `value` member used by `Message` and
    /// Agent input instead of panicking at a public construction boundary.
    #[doc(hidden)]
    pub fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(object) => object,
            value => Map::from_iter([(String::from("value"), value)]),
        }
    }

    #[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
    pub struct Part {
        #[serde(rename = "type")]
        pub r#type: String,
        pub data: Map<String, Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub mime_type: Option<String>,
    }
}
