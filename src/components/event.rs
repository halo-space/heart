use serde::{Deserialize, Serialize};

use crate::components::message::Message;
use crate::components::model::token::Usage;

/// Shared message event used by every stream-capable model or runtime call.
///
/// This component does not own execution state. Moving the event here does not
/// make complete-returning components such as Embed or Function emit a stream.
/// Complete represents an invocation result, not its containing Node/Workflow
/// status. Runtime states are not independently broadcast through this enum.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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
