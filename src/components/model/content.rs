use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Ordered content part in a Message.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
pub struct Part {
    #[serde(rename = "type")]
    pub r#type: String,
    pub data: Value,
}

/// Compatibility path for callers that already use `content::part::Part`.
/// The public type itself is defined in this file.
pub mod part {
    pub use super::Part;
}
