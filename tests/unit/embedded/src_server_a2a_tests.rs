use serde_json::json;

use super::{Action, dispatch};
use crate::components::protocol::a2a::{Context, Message, Part, Role, SendMessageRequest};

fn request(task_id: Option<&str>) -> SendMessageRequest {
    SendMessageRequest {
        message: Message {
            role: Role::User,
            parts: vec![Part::text("继续")],
            message_id: "message-1".into(),
            context_id: Some("2001".into()),
            task_id: task_id.map(str::to_owned),
            reference_task_ids: Vec::new(),
            metadata: Default::default(),
        },
        metadata: Default::default(),
    }
}

#[test]
fn routes_new_message_to_run() {
    let result = dispatch(Context::new(2001, 3001, "task-1"), request(None)).unwrap();
    assert_eq!(result.action, Action::Run);
    assert_eq!(result.input.metadata["chat_id"], json!(3001));
    assert_eq!(result.input.payload[0].r#type, "text");
}

#[test]
fn routes_existing_task_to_reply() {
    let result = dispatch(Context::new(2001, 3001, "task-1"), request(Some("task-1"))).unwrap();
    assert_eq!(result.action, Action::Reply);
}

#[test]
fn rejects_a_different_task() {
    let error = dispatch(Context::new(2001, 3001, "task-1"), request(Some("task-2"))).unwrap_err();
    assert_eq!(error.code, "CONFLICT");
}
