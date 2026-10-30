use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Identifies one externally visible top-level run.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Context {
    pub thread_id: String,
    pub run_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<String>,
}

impl Context {
    pub fn new(thread_id: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            run_id: run_id.into(),
            parent_run_id: None,
        }
    }
}

/// AG-UI wire events emitted by the protocol adapter.
///
/// The enum models the protocol boundary only. It is not used by Agent or
/// Workflow Runtime and does not replace Heart's Event contract.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Event {
    #[serde(rename = "RUN_STARTED")]
    RunStarted {
        #[serde(rename = "threadId")]
        thread_id: String,
        #[serde(rename = "runId")]
        run_id: String,
        #[serde(rename = "parentRunId", skip_serializing_if = "Option::is_none")]
        parent_run_id: Option<String>,
    },
    #[serde(rename = "RUN_FINISHED")]
    RunFinished {
        #[serde(rename = "threadId")]
        thread_id: String,
        #[serde(rename = "runId")]
        run_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        outcome: Option<Value>,
    },
    #[serde(rename = "RUN_ERROR")]
    RunError {
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        code: Option<String>,
    },
    #[serde(rename = "TEXT_MESSAGE_START")]
    TextMessageStart {
        #[serde(rename = "messageId")]
        message_id: String,
        role: String,
    },
    #[serde(rename = "TEXT_MESSAGE_CONTENT")]
    TextMessageContent {
        #[serde(rename = "messageId")]
        message_id: String,
        delta: String,
    },
    #[serde(rename = "TEXT_MESSAGE_END")]
    TextMessageEnd {
        #[serde(rename = "messageId")]
        message_id: String,
    },
    #[serde(rename = "REASONING_MESSAGE_START")]
    ReasoningMessageStart {
        #[serde(rename = "messageId")]
        message_id: String,
    },
    #[serde(rename = "REASONING_MESSAGE_CONTENT")]
    ReasoningMessageContent {
        #[serde(rename = "messageId")]
        message_id: String,
        delta: String,
    },
    #[serde(rename = "REASONING_MESSAGE_END")]
    ReasoningMessageEnd {
        #[serde(rename = "messageId")]
        message_id: String,
    },
    #[serde(rename = "TOOL_CALL_START")]
    ToolCallStart {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolCallName")]
        tool_call_name: String,
    },
    #[serde(rename = "TOOL_CALL_ARGS")]
    ToolCallArgs {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        delta: String,
    },
    #[serde(rename = "TOOL_CALL_END")]
    ToolCallEnd {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
    },
    #[serde(rename = "TOOL_CALL_RESULT")]
    ToolCallResult {
        #[serde(rename = "messageId")]
        message_id: String,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        content: Value,
    },
    #[serde(rename = "STATE_SNAPSHOT")]
    StateSnapshot { snapshot: Value },
    #[serde(rename = "STATE_DELTA")]
    StateDelta { delta: Value },
    #[serde(rename = "CUSTOM")]
    Custom { name: String, value: Value },
}
