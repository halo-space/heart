//! Conversion between Heart events and the A2A task stream.
//!
//! This module deliberately does not own an HTTP client, a task store, or a
//! Runtime. The transport layer supplies a Heart Event stream and decides how
//! to serialize the returned A2A values.

use std::collections::VecDeque;
use std::pin::Pin;

use futures_core::Stream;
use futures_util::{StreamExt, stream};
use serde_json::{Map, Value};

use crate::Error;
use crate::components::agent::{Input, Payload};
use crate::components::event::Event as HeartEvent;
use crate::components::message::Message as HeartMessage;
use crate::components::model::content::part::Part as HeartPart;
pub use crate::components::protocol::a2a::{
    AgentCapabilities, AgentCard, Artifact, Context, Message, Part, Role, SendMessageRequest,
    Skill, StreamResponse, Task, TaskArtifactUpdateEvent, TaskState, TaskStatus,
    TaskStatusUpdateEvent,
};

/// Convert an inbound A2A message into Heart's public Agent input envelope.
///
/// This is a data conversion only. It does not select an Agent, invoke a
/// Runtime, or resolve a task. The Server layer supplies the local Context and
/// decides whether the resulting input is passed to `run` or `reply`.
pub fn input(message: &Message, context: &Context) -> Result<Input, Error> {
    if message.message_id.trim().is_empty() {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "A2A messageId must not be empty",
        ));
    }
    if !matches!(message.role, Role::User) {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "A2A inbound message must use the user role",
        ));
    }
    if let Some(context_id) = &message.context_id
        && context_id != &context.context_id()
    {
        return Err(Error::new(
            "CONFLICT",
            "A2A contextId does not match the selected Heart session",
        ));
    }

    let payload = message
        .parts
        .iter()
        .map(input_payload)
        .collect::<Result<Vec<_>, _>>()?;
    let mut metadata = message.metadata.clone();
    metadata.insert("chat_id".into(), context.chat_id.into());
    metadata.insert("session_id".into(), context.session_id.into());
    metadata.insert("task_id".into(), context.task_id().into());
    metadata.insert("message_id".into(), message.message_id.clone().into());
    let part_metadata = message
        .parts
        .iter()
        .enumerate()
        .filter_map(|(index, part)| {
            if part.filename.is_none() && part.metadata.is_empty() {
                return None;
            }
            Some(serde_json::json!({
                "index": index,
                "filename": part.filename,
                "metadata": part.metadata,
            }))
        })
        .collect::<Vec<_>>();
    if !part_metadata.is_empty() {
        metadata.insert("a2a_parts".into(), Value::Array(part_metadata));
    }

    Ok(Input { metadata, payload })
}

fn input_payload(part: &Part) -> Result<Payload, Error> {
    let mut count = 0;
    if part.text.is_some() {
        count += 1;
    }
    if part.raw.is_some() {
        count += 1;
    }
    if part.url.is_some() {
        count += 1;
    }
    if part.data.is_some() {
        count += 1;
    }
    if count != 1 {
        return Err(Error::new(
            "A2A_INVALID_PART",
            "A2A message Part must contain exactly one content field",
        ));
    }

    let (r#type, data) = if let Some(text) = &part.text {
        ("text", Value::String(text.clone()))
    } else if let Some(raw) = &part.raw {
        ("raw", Value::String(raw.clone()))
    } else if let Some(url) = &part.url {
        ("url", Value::String(url.clone()))
    } else {
        ("json", part.data.clone().expect("validated A2A data Part"))
    };

    Ok(Payload {
        r#type: r#type.into(),
        data,
        mime_type: part.media_type.clone(),
    })
}

type InputStream<'a> = Pin<Box<dyn Stream<Item = Result<HeartEvent, Error>> + Send + 'a>>;

/// Convert a top-level Heart run into an A2A task lifecycle stream.
///
/// The first item is always a `Task` in `WORKING` state. Heart `Delta` and
/// `Complete` events become artifact updates; a Heart `Complete` never closes
/// the A2A task because Agent and ReAct runs may contain several internal model
/// completions. The task closes only when the source stream ends or returns an
/// error.
pub fn adapt<'a, S>(
    events: S,
    context: Context,
) -> impl Stream<Item = Result<StreamResponse, Error>> + Send + 'a
where
    S: Stream<Item = Result<HeartEvent, Error>> + Send + 'a,
{
    let state = State {
        source: Box::pin(events),
        context,
        queue: VecDeque::new(),
        started: false,
        terminal: false,
        sequence: 0,
        active_artifact: None,
    };

    stream::unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.queue.pop_front() {
                return Some((Ok(event), state));
            }

            if !state.started {
                state.started = true;
                return Some((Ok(StreamResponse::Task(state.initial_task())), state));
            }

            if state.terminal {
                return None;
            }

            match state.source.next().await {
                Some(Ok(event)) => {
                    let cancelled = matches!(
                        &event,
                        HeartEvent::Complete {
                            finish_reason: Some(reason),
                            ..
                        } if reason == "cancelled"
                    );
                    let waiting = matches!(
                        &event,
                        HeartEvent::Complete { message, .. }
                            if message.metadata.contains_key("reply_id")
                    );
                    match state.convert(event) {
                        Ok(Some(response)) => state.queue.push_back(response),
                        Ok(None) => {}
                        Err(_error) if cancelled => {}
                        Err(error) => {
                            state.terminal = true;
                            state.queue.push_back(StreamResponse::TaskStatusUpdateEvent(
                                state.error_status(&error),
                            ));
                        }
                    }
                    if waiting && !cancelled && !state.terminal {
                        state.queue.push_back(StreamResponse::TaskStatusUpdateEvent(
                            state.error_status(&Error::new("WAITING", "Agent requires a reply")),
                        ));
                    }
                    if cancelled {
                        state.terminal = true;
                        state.queue.push_back(StreamResponse::TaskStatusUpdateEvent(
                            state.error_status(&Error::new("CANCELLED", "Agent run cancelled")),
                        ));
                    }
                }
                Some(Err(error)) => {
                    state.queue.push_back(StreamResponse::TaskStatusUpdateEvent(
                        state.error_status(&error),
                    ));
                    if error.code != "WAITING" {
                        state.terminal = true;
                    }
                }
                None => {
                    state.terminal = true;
                    state.queue.push_back(StreamResponse::TaskStatusUpdateEvent(
                        state.finished_status(),
                    ));
                }
            }
        }
    })
}

struct State<'a> {
    source: InputStream<'a>,
    context: Context,
    queue: VecDeque<StreamResponse>,
    started: bool,
    terminal: bool,
    sequence: u64,
    active_artifact: Option<String>,
}

impl State<'_> {
    fn initial_task(&self) -> Task {
        Task {
            id: self.context.task_id(),
            context_id: self.context.context_id(),
            status: TaskStatus {
                state: TaskState::Working,
                message: None,
            },
            artifacts: Vec::new(),
            history: Vec::new(),
        }
    }

    fn convert(&mut self, event: HeartEvent) -> Result<Option<StreamResponse>, Error> {
        match event {
            HeartEvent::Delta { message } => self
                .artifact_update(message, true, false)
                .map(|update| update.map(StreamResponse::TaskArtifactUpdateEvent)),
            HeartEvent::Complete { message, .. } => self
                .artifact_update(message, false, true)
                .map(|update| update.map(StreamResponse::TaskArtifactUpdateEvent)),
        }
    }

    fn artifact_update(
        &mut self,
        message: HeartMessage,
        append: bool,
        last_chunk: bool,
    ) -> Result<Option<TaskArtifactUpdateEvent>, Error> {
        let parts = message_parts(&message)?;
        let artifact_id = self.artifact_id(&message, last_chunk);
        if parts.is_empty() {
            return Ok(None);
        }
        Ok(Some(TaskArtifactUpdateEvent {
            task_id: self.context.task_id(),
            context_id: self.context.context_id(),
            artifact: Artifact {
                artifact_id,
                name: None,
                parts,
                metadata: message.metadata,
            },
            append,
            last_chunk,
            metadata: Map::new(),
        }))
    }

    fn artifact_id(&mut self, message: &HeartMessage, last_chunk: bool) -> String {
        if let Some(id) = message.metadata.get("message_id").and_then(Value::as_str) {
            return id.to_owned();
        }
        if let (Some(node_id), Some(exec_id)) = (
            message.metadata.get("node_id").and_then(Value::as_str),
            message.metadata.get("exec_id").and_then(Value::as_i64),
        ) {
            return format!("{node_id}:{exec_id}");
        }
        let id = self.active_artifact.clone().unwrap_or_else(|| {
            self.sequence += 1;
            format!("{}:artifact:{}", self.context.task_id(), self.sequence)
        });
        if last_chunk {
            self.active_artifact = None;
        } else {
            self.active_artifact = Some(id.clone());
        }
        id
    }

    fn error_status(&self, error: &Error) -> TaskStatusUpdateEvent {
        let (state, message) = match error.code.as_str() {
            "WAITING" => (TaskState::InputRequired, None),
            "CANCELLED" => (
                TaskState::Canceled,
                Some(status_message(&error.message, &error.code)),
            ),
            _ => (
                TaskState::Failed,
                Some(status_message(&error.message, &error.code)),
            ),
        };
        TaskStatusUpdateEvent {
            task_id: self.context.task_id(),
            context_id: self.context.context_id(),
            status: TaskStatus { state, message },
            metadata: Map::new(),
        }
    }

    fn finished_status(&self) -> TaskStatusUpdateEvent {
        TaskStatusUpdateEvent {
            task_id: self.context.task_id(),
            context_id: self.context.context_id(),
            status: TaskStatus {
                state: TaskState::Completed,
                message: None,
            },
            metadata: Map::new(),
        }
    }
}

fn status_message(text: &str, code: &str) -> Message {
    let mut metadata = Map::new();
    metadata.insert("code".into(), Value::String(code.into()));
    Message {
        role: Role::Agent,
        parts: vec![Part::text(text)],
        message_id: format!("status:{code}"),
        context_id: None,
        task_id: None,
        reference_task_ids: Vec::new(),
        metadata,
    }
}

fn message_parts(message: &HeartMessage) -> Result<Vec<Part>, Error> {
    if message.content.is_empty() {
        return if message.values.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![Part::data(Value::Object(message.values.clone()))])
        };
    }

    message
        .content
        .iter()
        .filter_map(|part| match part.r#type.as_str() {
            "think" => None,
            _ => Some(content_part(part)),
        })
        .collect()
}

fn content_part(part: &HeartPart) -> Result<Part, Error> {
    match part.r#type.as_str() {
        "text" => part
            .data
            .get("value")
            .or_else(|| part.data.get("text"))
            .and_then(Value::as_str)
            .map(Part::text)
            .ok_or_else(|| {
                Error::new("A2A_INVALID_PART", "text content is missing a string value")
            }),
        "function" => Ok(Part::data(Value::Object(part.data.clone()))),
        "tool_call" => {
            let mut data = Map::new();
            data.insert("type".into(), Value::String("tool_call".into()));
            data.insert("data".into(), Value::Object(part.data.clone()));
            Ok(Part::data(Value::Object(data)))
        }
        other => Err(Error::new(
            "A2A_INVALID_PART",
            format!("unsupported Heart content type: {other}"),
        )),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_core_protocol_a2a_tests.rs"]
mod tests;
