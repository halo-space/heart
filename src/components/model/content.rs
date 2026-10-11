use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Canonical ordered content part used by `Message`.
pub mod part {
    use super::*;

    /// Convert a JSON object literal into the canonical Part data map.
    /// Non-object values are rejected at the construction boundary instead of
    /// leaking a second scalar representation into Message content.
    pub(crate) fn object(value: Value) -> Map<String, Value> {
        value
            .as_object()
            .cloned()
            .expect("Part data must be a JSON object")
    }

    #[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
    pub struct Part {
        #[serde(rename = "type")]
        pub r#type: String,
        pub data: Map<String, Value>,
    }
}
