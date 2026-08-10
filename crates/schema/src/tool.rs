use serde::{Deserialize, Serialize};

use crate::JsonSchema;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Definition {
    pub name: String,
    pub description: String,
    pub parameters: JsonSchema,
    pub strict: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Execute {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub code: Option<String>,
    pub is_retry: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorKind {
    InvalidInput,
    Execution,
    Transport,
    Timeout,
    Cancelled,
    Decode,
}
