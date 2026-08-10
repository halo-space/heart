use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Options {
    pub dimensions: Option<u32>,
    pub encoding_format: Option<EncodingFormat>,
    pub user: Option<String>,
    pub extra_headers: Option<BTreeMap<String, String>>,
    pub extra_query: Option<Value>,
    pub extra_body: Option<Value>,
    pub timeout: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            dimensions: None,
            encoding_format: None,
            user: None,
            extra_headers: None,
            extra_query: None,
            extra_body: None,
            timeout: None,
        }
    }
}

impl Options {
    pub fn validate(&self) -> Result<(), Error> {
        if self.dimensions == Some(0) {
            return Err(Error::invalid_input(
                "embedding dimensions must be positive",
            ));
        }

        Ok(())
    }
}
