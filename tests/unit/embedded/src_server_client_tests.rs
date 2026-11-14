use serde_json::json;

use super::Client;
use crate::components::protocol::a2a::{StreamResponse, Task, TaskState, TaskStatus};

#[test]
fn validates_endpoint_and_decodes_events() {
    assert!(Client::new(" ").is_err());
    let response = StreamResponse::Task(Task {
        id: "task-1".into(),
        context_id: "session-1".into(),
        status: TaskStatus {
            state: TaskState::Working,
            message: None,
        },
        artifacts: Vec::new(),
        history: Vec::new(),
    });
    let bytes = serde_json::to_vec(&response).unwrap();
    assert_eq!(
        serde_json::to_value(Client::decode_event(&bytes).unwrap()).unwrap()["task"]["id"],
        json!("task-1")
    );
}
