use serde::{Deserialize, Serialize};

use crate::components::message::Message;
use crate::components::model::token::Usage;

/// Shared observation event used by Chat calls and Agent runs.
///
/// This component does not own execution state. Moving the event here does not
/// make complete-returning components such as Embed or Function emit a stream.
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
