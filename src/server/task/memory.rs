//! In-memory implementation of the A2A task projection.
//!
//! This store owns only the externally visible task snapshot and idempotency
//! facts; Runtime State and Checkpoint remain outside this module.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use super::{MessageState, Store};
use crate::Error;
use crate::components::protocol::a2a::{Context, Message, StreamResponse, Task, TaskState};

#[derive(Clone, Default)]
pub struct Memory {
    next: Arc<AtomicU64>,
    entries: Arc<Mutex<BTreeMap<String, TaskRecord>>>,
}

#[derive(Clone)]
struct TaskRecord {
    context: Context,
    task: Task,
    messages: BTreeMap<String, Vec<u8>>,
}

impl Memory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create one server task. The generated ID is opaque to Heart and is
    /// only used by the A2A projection.
    pub fn create(&self, session_id: i64, chat_id: i64) -> Result<Context, Error> {
        if session_id <= 0 || chat_id <= 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "session_id and chat_id must be positive",
            ));
        }
        let task_id = format!("task-{}", self.next.fetch_add(1, Ordering::Relaxed) + 1);
        let context = Context::new(session_id, chat_id, task_id.clone());
        let task = Task {
            id: task_id.clone(),
            context_id: context.context_id(),
            status: crate::components::protocol::a2a::TaskStatus {
                state: TaskState::Submitted,
                message: None,
            },
            artifacts: Vec::new(),
            history: Vec::new(),
        };
        self.entries.lock().map_err(lock_error)?.insert(
            task_id,
            TaskRecord {
                context: context.clone(),
                task,
                messages: BTreeMap::new(),
            },
        );
        Ok(context)
    }

    pub fn task(&self, task_id: &str) -> Result<Task, Error> {
        self.entries
            .lock()
            .map_err(lock_error)?
            .get(task_id)
            .map(|entry| entry.task.clone())
            .ok_or_else(|| Error::new("NOT_FOUND", "A2A task was not found"))
    }

    pub fn pending(&self) -> Result<Vec<Task>, Error> {
        Ok(self
            .entries
            .lock()
            .map_err(lock_error)?
            .values()
            .map(|entry| entry.task.clone())
            .filter(|task| {
                matches!(
                    task.status.state,
                    TaskState::Working | TaskState::InputRequired | TaskState::AuthRequired
                )
            })
            .collect())
    }

    pub fn context(&self, task_id: &str) -> Result<Context, Error> {
        self.entries
            .lock()
            .map_err(lock_error)?
            .get(task_id)
            .map(|entry| entry.context.clone())
            .ok_or_else(|| Error::new("NOT_FOUND", "A2A task was not found"))
    }

    /// Accept one A2A message ID. Repeating the same ID and exact payload is
    /// idempotent; reusing it with another payload is a conflict.
    pub fn message(&self, task_id: &str, message: &Message) -> Result<MessageState, Error> {
        let encoded = serde_json::to_vec(message)
            .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
        let mut entries = self.entries.lock().map_err(lock_error)?;
        let entry = entries
            .get_mut(task_id)
            .ok_or_else(|| Error::new("NOT_FOUND", "A2A task was not found"))?;
        match entry.messages.get(&message.message_id) {
            Some(previous) if previous == &encoded => Ok(MessageState::Duplicate),
            Some(_) => Err(Error::new(
                "CONFLICT",
                "A2A messageId was reused with different content",
            )),
            None => {
                entry.messages.insert(message.message_id.clone(), encoded);
                Ok(MessageState::Accepted)
            }
        }
    }

    pub fn record(&self, task_id: &str, response: StreamResponse) -> Result<(), Error> {
        let mut entries = self.entries.lock().map_err(lock_error)?;
        let entry = entries
            .get_mut(task_id)
            .ok_or_else(|| Error::new("NOT_FOUND", "A2A task was not found"))?;
        if terminal(&entry.task) {
            return Ok(());
        }
        apply(&mut entry.task, &response);
        Ok(())
    }
}

#[allow(clippy::manual_async_fn)]
impl Store for Memory {
    fn create(
        &self,
        session_id: i64,
        chat_id: i64,
    ) -> impl Future<Output = Result<Context, Error>> + Send {
        async move { Self::create(self, session_id, chat_id) }
    }

    fn task(&self, task_id: &str) -> impl Future<Output = Result<Task, Error>> + Send {
        async move { Self::task(self, task_id) }
    }

    fn pending(&self) -> impl Future<Output = Result<Vec<Task>, Error>> + Send {
        async move { Self::pending(self) }
    }

    fn context(&self, task_id: &str) -> impl Future<Output = Result<Context, Error>> + Send {
        async move { Self::context(self, task_id) }
    }

    fn message(
        &self,
        task_id: &str,
        message: &Message,
    ) -> impl Future<Output = Result<MessageState, Error>> + Send {
        async move { Self::message(self, task_id, message) }
    }

    fn record(
        &self,
        task_id: &str,
        response: StreamResponse,
    ) -> impl Future<Output = Result<(), Error>> + Send {
        async move { Self::record(self, task_id, response) }
    }
}

fn apply(task: &mut Task, response: &StreamResponse) {
    match response {
        StreamResponse::Task(next) => *task = next.clone(),
        StreamResponse::Message(message) => task.history.push(message.clone()),
        StreamResponse::TaskStatusUpdateEvent(update) => task.status = update.status.clone(),
        StreamResponse::TaskArtifactUpdateEvent(update) => {
            if update.append
                && let Some(previous) = task
                    .artifacts
                    .iter_mut()
                    .find(|artifact| artifact.artifact_id == update.artifact.artifact_id)
            {
                previous.parts.extend(update.artifact.parts.clone());
                previous.metadata = update.artifact.metadata.clone();
            } else {
                task.artifacts.push(update.artifact.clone());
            }
        }
    }
}

fn terminal(task: &Task) -> bool {
    matches!(
        task.status.state,
        TaskState::Completed | TaskState::Failed | TaskState::Canceled | TaskState::Rejected
    )
}

fn lock_error<T>(_: std::sync::PoisonError<T>) -> Error {
    Error::new("CONFLICT", "A2A task projection lock is poisoned")
}

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_server_task_memory_tests.rs"]
mod tests;
