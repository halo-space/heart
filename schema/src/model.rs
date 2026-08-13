use serde::{Deserialize, Serialize};

pub mod audio;
pub mod chat;
pub mod embedding;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input: Option<u64>,
    pub cached: Option<u64>,
    pub output: Option<u64>,
    pub reasoning: Option<u64>,
    pub total: Option<u64>,
    pub duration: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    InvalidInput,
    Unsupported,
    Api,
    Transport,
    Timeout,
    Cancelled,
    Decode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
    pub code: Option<String>,
    pub is_retry: bool,
}

impl Error {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidInput,
            message: message.into(),
            code: None,
            is_retry: false,
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Unsupported,
            message: message.into(),
            code: None,
            is_retry: false,
        }
    }
}
