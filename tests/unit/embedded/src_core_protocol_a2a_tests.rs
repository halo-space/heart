use futures_util::StreamExt;
use serde_json::json;

use super::adapt;
use crate::Error;
use crate::components::event::Event as HeartEvent;
use crate::components::message::{Message, Role};
use crate::components::model::content::part::Part;
use crate::components::protocol::a2a::{Context, StreamResponse, TaskState, TaskStatus};

#[tokio::test]
async fn starts_task_and_finishes_after_the_source_ends() {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "北京晴天"}).as_object().unwrap().clone(),
    });
    let source = futures_util::stream::iter(vec![Ok(HeartEvent::Complete {
        message,
        usage: None,
        finish_reason: Some("stop".into()),
    })]);

    let output = adapt(source, Context::new(2001, 3001, "task-3001"))
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        output.first(),
        Some(Ok(StreamResponse::Task(task)))
            if task.id == "task-3001"
                && task.context_id == "2001"
                && task.status.state == TaskState::Working
    ));
    assert!(output.iter().any(|item| matches!(
        item,
        Ok(StreamResponse::TaskArtifactUpdateEvent(update))
            if update.artifact.parts[0].text.as_deref() == Some("北京晴天")
    )));
    assert!(matches!(
        output.last(),
        Some(Ok(StreamResponse::TaskStatusUpdateEvent(update)))
            if update.status.state == TaskState::Completed
    ));
}

#[tokio::test]
async fn maps_waiting_without_creating_a_tool_result_wrapper() {
    let source = futures_util::stream::once(async { Err(Error::new("WAITING", "需要审批")) })
        .chain(futures_util::stream::pending());
    let mut output = Box::pin(adapt(source, Context::new(2001, 3001, "task-3001")));
    assert!(matches!(
        output.next().await,
        Some(Ok(StreamResponse::Task(_)))
    ));
    assert!(matches!(
        output.next().await,
        Some(Ok(StreamResponse::TaskStatusUpdateEvent(update)))
            if update.status.state == TaskState::InputRequired
                && update.status.message.is_none()
    ));
}

#[tokio::test]
async fn continues_after_waiting_when_the_live_run_resumes() {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "已恢复"}).as_object().unwrap().clone(),
    });
    let source = futures_util::stream::iter(vec![
        Err(Error::new("WAITING", "需要审批")),
        Ok(HeartEvent::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        }),
    ]);
    let output = adapt(source, Context::new(2001, 3001, "task-3001"))
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().any(|item| matches!(
        item,
        Ok(StreamResponse::TaskStatusUpdateEvent(update))
            if update.status.state == TaskState::InputRequired
    )));
    assert!(output.iter().any(|item| matches!(
        item,
        Ok(StreamResponse::TaskArtifactUpdateEvent(update))
            if update.artifact.parts[0].text.as_deref() == Some("已恢复")
    )));
    assert!(matches!(
        output.last(),
        Some(Ok(StreamResponse::TaskStatusUpdateEvent(update)))
            if update.status.state == TaskState::Completed
    ));
}

#[tokio::test]
async fn maps_cancellation_to_canceled() {
    let source = futures_util::stream::iter(vec![Err(Error::new("CANCELLED", "用户取消运行"))]);
    let output = adapt(source, Context::new(2001, 3001, "task-3001"))
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        output.last(),
        Some(Ok(StreamResponse::TaskStatusUpdateEvent(update)))
            if update.status.state == TaskState::Canceled
                && update.status.message.as_ref().is_some_and(|message|
                    message.metadata.get("code") == Some(&json!("CANCELLED")))
    ));
}

#[tokio::test]
async fn maps_cancelled_complete_to_canceled_instead_of_completed() {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "已收到的部分输出"})
            .as_object()
            .unwrap()
            .clone(),
    });
    let source = futures_util::stream::iter(vec![Ok(HeartEvent::Complete {
        message,
        usage: None,
        finish_reason: Some("cancelled".into()),
    })]);
    let output = adapt(source, Context::new(2001, 3001, "task-3001"))
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().any(|item| matches!(
        item,
        Ok(StreamResponse::TaskArtifactUpdateEvent(update))
            if update.artifact.parts[0].text.as_deref() == Some("已收到的部分输出")
    )));
    assert!(matches!(
        output.last(),
        Some(Ok(StreamResponse::TaskStatusUpdateEvent(update)))
            if update.status.state == TaskState::Canceled
    ));
    assert!(!output.iter().any(|item| matches!(
        item,
        Ok(StreamResponse::TaskStatusUpdateEvent(update))
            if update.status.state == TaskState::Completed
    )));
}

#[tokio::test]
async fn preserves_tool_call_as_structured_a2a_data() {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "tool_call".into(),
        data: json!({
            "call_id": "call-1",
            "name": "ask_for_approval",
            "arguments": {"count": 1}
        })
        .as_object()
        .unwrap()
        .clone(),
    });
    let source = futures_util::stream::iter(vec![Ok(HeartEvent::Complete {
        message,
        usage: None,
        finish_reason: Some("stop".into()),
    })]);
    let output = adapt(source, Context::new(2001, 3001, "task-3001"))
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().any(|item| matches!(
        item,
        Ok(StreamResponse::TaskArtifactUpdateEvent(update))
            if update.artifact.parts[0].data.as_ref().is_some_and(|data|
                data["type"] == json!("tool_call")
                    && data["data"]["name"] == json!("ask_for_approval")
            )
    )));
}

#[tokio::test]
async fn maps_reply_id_complete_to_input_required() {
    let mut message = Message::new(Role::Assistant);
    message.metadata.insert("reply_id".into(), json!(42));
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "请批准"}).as_object().unwrap().clone(),
    });
    let source = futures_util::stream::once(async move {
        Ok(HeartEvent::Complete {
            message,
            usage: None,
            finish_reason: None,
        })
    })
    .chain(futures_util::stream::pending());
    let mut output = Box::pin(adapt(source, Context::new(2001, 3001, "task-3001")));
    assert!(matches!(
        output.next().await,
        Some(Ok(StreamResponse::Task(_)))
    ));
    assert!(matches!(
        output.next().await,
        Some(Ok(StreamResponse::TaskArtifactUpdateEvent(_)))
    ));
    assert!(matches!(
        output.next().await,
        Some(Ok(StreamResponse::TaskStatusUpdateEvent(update)))
            if update.status.state == TaskState::InputRequired
                && update.status.message.is_none()
    ));
}

#[test]
fn keeps_inbound_part_metadata_in_input_metadata() {
    let mut part = crate::components::protocol::a2a::Part::text("report");
    part.filename = Some("report.txt".into());
    part.metadata.insert("source".into(), json!("upload"));
    let message = crate::components::protocol::a2a::Message {
        role: crate::components::protocol::a2a::Role::User,
        parts: vec![part],
        message_id: "m-1".into(),
        context_id: Some("2001".into()),
        task_id: None,
        reference_task_ids: Vec::new(),
        metadata: Default::default(),
    };
    let input = super::input(&message, &Context::new(2001, 3001, "task-1")).unwrap();
    assert_eq!(input.metadata["a2a_parts"][0]["index"], json!(0));
    assert_eq!(
        input.metadata["a2a_parts"][0]["filename"],
        json!("report.txt")
    );
    assert_eq!(
        input.metadata["a2a_parts"][0]["metadata"]["source"],
        json!("upload")
    );
}

#[tokio::test]
async fn keeps_function_data_as_an_artifact_part() {
    let mut message = Message::function(json!({"data": "ok"}));
    message.metadata.insert("message_id".into(), json!("m-1"));
    let source = futures_util::stream::iter(vec![Ok(HeartEvent::Delta { message })]);
    let output = adapt(source, Context::new(2001, 3001, "task-3001"))
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().any(|item| matches!(
        item,
        Ok(StreamResponse::TaskArtifactUpdateEvent(update))
            if update.artifact.artifact_id == "m-1"
                && update.artifact.parts[0].data == Some(json!({"data": "ok"}))
    )));
}

#[tokio::test]
async fn keeps_delta_and_complete_in_one_artifact_without_message_id() {
    let mut delta = Message::new(Role::Assistant);
    delta.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "北京"}).as_object().unwrap().clone(),
    });
    let mut complete = Message::new(Role::Assistant);
    complete.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "北京今天晴朗"})
            .as_object()
            .unwrap()
            .clone(),
    });
    let output = adapt(
        futures_util::stream::iter(vec![
            Ok(HeartEvent::Delta { message: delta }),
            Ok(HeartEvent::Complete {
                message: complete,
                usage: None,
                finish_reason: Some("stop".into()),
            }),
        ]),
        Context::new(2001, 3001, "task-3001"),
    )
    .collect::<Vec<_>>()
    .await;
    let ids = output
        .iter()
        .filter_map(|response| match response {
            Ok(StreamResponse::TaskArtifactUpdateEvent(update)) => {
                Some(update.artifact.artifact_id.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["task-3001:artifact:1", "task-3001:artifact:1"]);
}

#[tokio::test]
async fn does_not_expose_think_parts_to_a2a() {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "think".into(),
        data: json!({"value": "内部思考"}).as_object().unwrap().clone(),
    });
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value": "公开答案"}).as_object().unwrap().clone(),
    });
    let source = futures_util::stream::iter(vec![Ok(HeartEvent::Complete {
        message,
        usage: None,
        finish_reason: Some("stop".into()),
    })]);

    let output = adapt(source, Context::new(2001, 3001, "task-3001"))
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().any(|item| matches!(
        item,
        Ok(StreamResponse::TaskArtifactUpdateEvent(update))
            if update.artifact.parts.len() == 1
                && update.artifact.parts[0].text.as_deref() == Some("公开答案")
    )));
}

#[tokio::test]
async fn rejects_unknown_heart_content_types() {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "audio".into(),
        data: json!({"value": "unsupported"}).as_object().unwrap().clone(),
    });
    let source = futures_util::stream::iter(vec![Ok(HeartEvent::Delta { message })]);

    let output = adapt(source, Context::new(2001, 3001, "task-3001"))
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        output.last(),
        Some(Ok(StreamResponse::TaskStatusUpdateEvent(update)))
            if update.status.state == TaskState::Failed
                && update.status.message.as_ref().is_some_and(|message|
                    message.metadata.get("code") == Some(&json!("A2A_INVALID_PART")))
    ));
}

#[test]
fn serializes_a2a_wire_enum_names() {
    let status = serde_json::to_value(TaskState::InputRequired).unwrap();
    assert_eq!(status, json!("TASK_STATE_INPUT_REQUIRED"));

    let message = crate::components::protocol::a2a::Message {
        role: crate::components::protocol::a2a::Role::Agent,
        parts: vec![crate::components::protocol::a2a::Part::text("ok")],
        message_id: "m-1".into(),
        context_id: None,
        task_id: None,
        reference_task_ids: Vec::new(),
        metadata: serde_json::Map::new(),
    };
    assert_eq!(
        serde_json::to_value(&message).unwrap()["role"],
        json!("ROLE_AGENT")
    );

    let status_event = crate::components::protocol::a2a::TaskStatusUpdateEvent {
        task_id: "task-1".into(),
        context_id: "context-1".into(),
        status: TaskStatus {
            state: TaskState::Completed,
            message: None,
        },
        metadata: serde_json::Map::new(),
    };
    let status_json = serde_json::to_value(status_event).unwrap();
    assert_eq!(status_json["taskId"], json!("task-1"));
    assert!(status_json.get("final").is_none());
}
