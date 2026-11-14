use serde_json::json;

use super::{ErrorObject, Method, Request, Response, event};
use crate::Error;
use crate::components::protocol::a2a::{Context, StreamResponse, Task, TaskState, TaskStatus};

#[test]
fn validates_json_rpc_version_and_method() {
    let request = Request {
        jsonrpc: "2.0".into(),
        id: json!(1),
        method: "message/send".into(),
        params: json!({}),
    };
    request.validate().unwrap();
    assert_eq!(request.method().unwrap(), Method::Send);
}

#[test]
fn rejects_invalid_json_rpc_envelope() {
    let request = Request {
        jsonrpc: "1.0".into(),
        id: json!(1),
        method: "message/send".into(),
        params: json!({}),
    };
    assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");
}

#[test]
fn rejects_unknown_method() {
    let request = Request {
        jsonrpc: "2.0".into(),
        id: json!(1),
        method: "tasks/unknown".into(),
        params: json!({}),
    };
    assert_eq!(request.method().unwrap_err().code, "METHOD_NOT_FOUND");
}

#[test]
fn builds_result_and_error_responses() {
    let result = Response::result(json!(1), json!({"ok": true}));
    assert_eq!(result.jsonrpc, "2.0");
    assert!(result.error.is_none());

    let error = Response::error(
        json!(1),
        ErrorObject {
            code: -32600,
            message: "Invalid Request".into(),
            data: None,
        },
    );
    assert!(error.result.is_none());
    assert_eq!(error.error.unwrap().code, -32600);
}

#[test]
fn maps_heart_error_to_json_rpc_error() {
    let response = Response::from_error(json!(1), &Error::new("INVALID_ARGUMENTS", "bad input"));
    let error = response.error.unwrap();
    assert_eq!(error.code, -32602);
    assert_eq!(error.data.unwrap()["code"], json!("INVALID_ARGUMENTS"));
}

#[test]
fn converts_a2a_response_to_axum_sse_event() {
    let response = StreamResponse::Task(Task {
        id: "task-1".into(),
        context_id: Context::new(2001, 3001, "task-1").context_id(),
        status: TaskStatus {
            state: TaskState::Working,
            message: None,
        },
        artifacts: Vec::new(),
        history: Vec::new(),
    });
    let _ = event(&response).unwrap();
}
