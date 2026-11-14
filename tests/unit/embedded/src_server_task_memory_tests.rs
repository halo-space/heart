use super::{Memory, MessageState};
use crate::components::protocol::a2a::{
    Message, Part, Role, StreamResponse, TaskState, TaskStatus, TaskStatusUpdateEvent,
};

fn message(id: &str, text: &str) -> Message {
    Message {
        role: Role::User,
        parts: vec![Part::text(text)],
        message_id: id.into(),
        context_id: None,
        task_id: None,
        reference_task_ids: Vec::new(),
        metadata: Default::default(),
    }
}

#[test]
fn creates_deduplicates_and_records_projection_events() {
    let tasks = Memory::new();
    let context = tasks.create(1, 2).unwrap();
    assert_eq!(tasks.pending().unwrap().len(), 0);
    assert_eq!(
        tasks
            .message(&context.task_id, &message("m-1", "a"))
            .unwrap(),
        MessageState::Accepted
    );
    assert_eq!(
        tasks
            .message(&context.task_id, &message("m-1", "a"))
            .unwrap(),
        MessageState::Duplicate
    );
    assert_eq!(
        tasks
            .message(&context.task_id, &message("m-2", "b"))
            .unwrap(),
        MessageState::Accepted
    );

    tasks
        .record(
            &context.task_id,
            StreamResponse::TaskStatusUpdateEvent(TaskStatusUpdateEvent {
                task_id: context.task_id.clone(),
                context_id: context.context_id(),
                status: TaskStatus {
                    state: TaskState::Working,
                    message: None,
                },
                metadata: Default::default(),
            }),
        )
        .unwrap();
    assert_eq!(tasks.pending().unwrap().len(), 1);
    assert_eq!(
        tasks.task(&context.task_id).unwrap().status.state,
        TaskState::Working
    );
}
