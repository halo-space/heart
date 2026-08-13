use serde::{Deserialize, Serialize};

use crate::{message::Message, model::Usage};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Delta {
        message: Message,
    },
    Complete {
        message: Message,
        usage: Option<Usage>,
        finish_reason: Option<String>,
    },
}
