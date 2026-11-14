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
        serde_json::from_str::<serde_json::Value>(frame.strip_prefix("data: ").unwrap().trim(),)
            .unwrap()["task"]["id"],
        json!("task-1")
    );
}
