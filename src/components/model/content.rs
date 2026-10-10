use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Canonical ordered content part used by `Message`.
pub mod part {
    use super::*;

    #[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
    pub struct Part {
        #[serde(rename = "type")]
        pub r#type: String,
        pub data: Value,
    }
}

/// Compatibility re-export for callers using the earlier `content::Part` path.
pub use part::Part;
