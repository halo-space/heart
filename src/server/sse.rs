//! SSE framing for protocol responses.
//!
//! This module only serializes one already-converted protocol response. It
//! does not own an HTTP connection or a Runtime event stream.

use crate::Error;
use crate::components::protocol::a2a::StreamResponse;

/// Encode one A2A response as a standard SSE `data` frame.
pub fn encode(response: &StreamResponse) -> Result<String, Error> {
    let data = serde_json::to_string(response)
        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
    Ok(format!("data: {data}\n\n"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::encode;
    use crate::components::protocol::a2a::{Context, StreamResponse, Task, TaskState, TaskStatus};

    #[test]
    fn encodes_a2a_response_as_one_sse_frame() {
        let context = Context::new(2001, 3001, "task-1");
        let response = StreamResponse::Task(Task {
            id: context.task_id(),
            context_id: context.context_id(),
            status: TaskStatus {
                state: TaskState::Working,
                message: None,
            },
            artifacts: Vec::new(),
            history: Vec::new(),
        });

        let frame = encode(&response).unwrap();
        assert!(frame.starts_with("data: "));
        assert!(frame.ends_with("\n\n"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                frame.strip_prefix("data: ").unwrap().trim(),
            )
            .unwrap()["task"]["id"],
            json!("task-1")
        );
    }
}
