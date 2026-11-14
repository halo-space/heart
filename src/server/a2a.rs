//! A2A request classification and Heart input conversion.
//!
//! HTTP, JSON-RPC and SSE bindings call this module. It deliberately does not
//! own a transport connection, Task store, Agent State or execution loop.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_core::Stream;
use futures_util::{StreamExt, stream};
use serde_json::{Map, Value};
use tokio::sync::broadcast;

use crate::Error;
use crate::components::agent::{Agent, Input};
use crate::components::protocol::a2a::{
    Context, Message, Part, Role, SendMessageRequest, StreamResponse, Task, TaskState, TaskStatus,
    TaskStatusUpdateEvent,
};
use crate::server::task::{MessageState, Store, memory::Memory};

type Projection = Pin<Box<dyn Stream<Item = Result<StreamResponse, Error>> + Send + 'static>>;

#[derive(Clone, Default)]
struct Live {
    tasks: Arc<Mutex<BTreeMap<String, task::Handle>>>,
}

mod task {
    use super::StreamResponse;
    use tokio::sync::broadcast;

    #[derive(Clone)]
    pub(super) struct Handle {
        pub(super) cancel: crate::Cancellation,
        pub(super) events: broadcast::Sender<StreamResponse>,
    }
}

impl Live {
    fn create(&self, task_id: &str) {
        let (events, _) = broadcast::channel(64);
        self.tasks.lock().expect("A2A live lock poisoned").insert(
            task_id.to_owned(),
            task::Handle {
                cancel: crate::Cancellation::new(),
                events,
            },
        );
    }

    fn ensure(&self, task_id: &str) -> bool {
        if let Ok(mut tasks) = self.tasks.lock()
            && !tasks.contains_key(task_id)
        {
            let (events, _) = broadcast::channel(64);
            tasks.insert(
                task_id.to_owned(),
                task::Handle {
                    cancel: crate::Cancellation::new(),
                    events,
                },
            );
            return true;
        }
        false
    }

    fn cancellation(&self, task_id: &str) -> Result<crate::Cancellation, Error> {
        self.tasks
            .lock()
            .map_err(|_| Error::new("CONFLICT", "A2A live lock is poisoned"))?
            .get(task_id)
            .map(|handle| handle.cancel.clone())
            .ok_or_else(|| Error::new("CONFLICT", "A2A task is not attached to a live run"))
    }

    fn cancel(&self, task_id: &str) -> Result<(), Error> {
        let tasks = self
            .tasks
            .lock()
            .map_err(|_| Error::new("CONFLICT", "A2A live lock is poisoned"))?;
        if let Some(task) = tasks.get(task_id) {
            task.cancel.cancel();
        }
        Ok(())
    }

    fn publish(&self, task_id: &str, response: StreamResponse) {
        if let Ok(tasks) = self.tasks.lock()
            && let Some(task) = tasks.get(task_id)
        {
            let _ = task.events.send(response);
        }
    }

    fn remove(&self, task_id: &str) {
        if let Ok(mut tasks) = self.tasks.lock() {
            tasks.remove(task_id);
        }
    }

    fn subscribe(&self, task_id: &str) -> Result<broadcast::Receiver<StreamResponse>, Error> {
        let tasks = self
            .tasks
            .lock()
            .map_err(|_| Error::new("CONFLICT", "A2A live lock is poisoned"))?;
        let Some(task) = tasks.get(task_id) else {
            let (_, receiver) = broadcast::channel(1);
            return Ok(receiver);
        };
        Ok(task.events.subscribe())
    }
}

/// The Heart entry point selected for one A2A message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    /// Start a new Heart run because the A2A message has no task ID.
    Run,
    /// Submit the message to an existing Heart run through `reply`.
    Reply,
}

/// Result of the transport-neutral A2A dispatch decision.
#[derive(Clone, Debug, PartialEq)]
pub struct Dispatch {
    pub action: Action,
    pub context: Context,
    pub input: Input,
}

/// Classify an A2A `SendMessage` request and convert its message to Heart
/// input. The caller remains responsible for invoking `run` or `reply` and
/// for managing the returned Event stream.
pub fn dispatch(context: Context, request: SendMessageRequest) -> Result<Dispatch, Error> {
    let action = match request.message.task_id.as_deref() {
        None => Action::Run,
        Some(task_id) if task_id == context.task_id => Action::Reply,
        Some(_) => {
            return Err(Error::new(
                "CONFLICT",
                "A2A taskId does not match the selected Heart task",
            ));
        }
    };
    let input = crate::core::protocol::a2a::input(&request.message, &context)?;
    Ok(Dispatch {
        action,
        context,
        input,
    })
}

/// Validate the task ID independently when a transport already has a parsed
/// message and only needs to check whether it can continue the selected task.
pub fn task_matches(context: &Context, message: &Message) -> bool {
    message.task_id.as_deref() == Some(context.task_id.as_str())
}

/// A transport-neutral A2A server for one Agent instance.
///
/// The server owns only the A2A task projection. Agent state, waiting and
/// checkpoint recovery remain inside the Agent implementation.
pub struct Service<A, S = Memory>
where
    A: Agent + 'static,
    S: Store,
{
    agent: Arc<A>,
    store: S,
    live: Live,
    projection: Arc<tokio::sync::Mutex<()>>,
}

impl<A, S> Clone for Service<A, S>
where
    A: Agent + 'static,
    S: Store,
{
    fn clone(&self) -> Self {
        Self {
            agent: self.agent.clone(),
            store: self.store.clone(),
            live: self.live.clone(),
            projection: self.projection.clone(),
        }
    }
}

impl<A> Service<A, Memory>
where
    A: Agent + 'static,
{
    pub fn new(agent: A) -> Self {
        Self {
            agent: Arc::new(agent),
            store: Memory::new(),
            live: Live::default(),
            projection: Arc::new(tokio::sync::Mutex::new(())),
        }
    }
}

impl<A, S> Service<A, S>
where
    A: Agent + 'static,
    S: Store,
{
    pub fn with_store(agent: A, store: S) -> Self {
        Self {
            agent: Arc::new(agent),
            store,
            live: Live::default(),
            projection: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub async fn task(&self, task_id: &str) -> Result<Task, Error> {
        self.store.task(task_id).await
    }

    /// Return unfinished tasks that a Server or Daemon can reattach after a
    /// process restart. Runtime Checkpoint data remains outside this store.
    pub async fn pending(&self) -> Result<Vec<Task>, Error> {
        self.store.pending().await
    }

    /// Reattach every unfinished task exposed by the Task Store.
    pub async fn recover_pending(&self) -> Result<Vec<Task>, Error>
    where
        A: Send + Sync,
    {
        let pending = self.pending().await?;
        let mut recovered = Vec::with_capacity(pending.len());
        for task in pending {
            recovered.push(self.recover(&task.id).await?);
        }
        Ok(recovered)
    }

    /// Reattach one unfinished task to the existing Agent and its persisted
    /// Checkpoint. This restores only the server driver; Runtime owns the
    /// execution State, waiting and recovery facts.
    pub async fn recover(&self, task_id: &str) -> Result<Task, Error>
    where
        A: Send + Sync,
    {
        let context = self.context(task_id).await?;
        let task = self.store.task(task_id).await?;
        if terminal(&task) {
            return Ok(task);
        }
        if !self.live.ensure(task_id) {
            return self.store.task(task_id).await;
        }
        let input = Input {
            metadata: Map::from_iter([
                ("session_id".into(), Value::from(context.session_id)),
                ("chat_id".into(), Value::from(context.chat_id)),
            ]),
            payload: Vec::new(),
        };
        let service = self.clone();
        tokio::spawn(async move {
            service.background_drive(context, input).await;
        });
        self.store.task(task_id).await
    }

    pub async fn cancel(&self, task_id: &str) -> Result<Task, Error> {
        self.live.cancel(task_id)?;
        self.store.task(task_id).await
    }

    /// Create the task projection and convert the first message. The caller
    /// decides whether to keep the resulting stream attached or drive it in
    /// the background.
    pub async fn begin(&self, request: SendMessageRequest) -> Result<(Context, Input), Error> {
        if request.message.task_id.is_some() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "message/send must not include taskId",
            ));
        }
        let (session_id, chat_id) = ids(&request)?;
        let context = self.store.create(session_id, chat_id).await?;
        self.live.create(&context.task_id);
        self.accept_message(&context, &request.message).await?;
        let input = crate::core::protocol::a2a::input(&request.message, &context)?;
        Ok((context, input))
    }

    /// Start a new A2A task, drive it in the background, and return its
    /// current projection. The background driver remains alive through
    /// `WAITING` so a later `reply` can resume the same Agent stream.
    pub async fn send(&self, request: SendMessageRequest) -> Result<Task, Error>
    where
        A: Send + Sync,
    {
        let (context, input) = self.begin(request).await?;
        let task_id = context.task_id.clone();
        let service = self.clone();
        tokio::spawn(async move {
            service.background_drive(context, input).await;
        });
        self.store.task(&task_id).await
    }

    /// Drive a started task to completion. A WAITING response keeps the
    /// underlying Agent stream alive until a later `reply` resumes it.
    pub async fn drive(&self, context: Context, input: Input) -> Result<(), Error> {
        let events = self.stream_from(context, input).await?;
        futures_util::pin_mut!(events);
        while let Some(response) = events.next().await {
            response?;
        }
        Ok(())
    }

    async fn background_drive(&self, context: Context, input: Input) {
        if let Err(error) = self.drive(context.clone(), input).await {
            let terminal = self
                .store
                .task(&context.task_id)
                .await
                .map(|task| terminal(&task))
                .unwrap_or(false);
            if !terminal {
                let _ = self.record_error(&context, &error).await;
            }
        }
        if self
            .store
            .task(&context.task_id)
            .await
            .map(|task| terminal(&task))
            .unwrap_or(false)
        {
            self.live.remove(&context.task_id);
        }
    }

    /// Start a task stream after `begin` has created its projection.
    pub async fn stream_from(
        &self,
        context: Context,
        input: Input,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, Error>> + Send + 'static>>, Error>
    {
        let cancellation = self.live.cancellation(&context.task_id)?;
        let stream = match self.agent.run(input, true, cancellation).await {
            Ok(stream) => stream,
            Err(error) => {
                self.record_error(&context, &error).await?;
                let task = self.store.task(&context.task_id).await?;
                let status = error_status(&context, &error);
                self.live.remove(&context.task_id);
                return Ok(Box::pin(stream::iter([
                    Ok(StreamResponse::Task(task)),
                    Ok(status),
                ])));
            }
        };
        self.record_working(&context).await?;
        let store = self.store.clone();
        let live = self.live.clone();
        let projection = self.projection.clone();
        let task_id = context.task_id.clone();
        let error_service = self.clone();
        let error_context = context.clone();
        let output = crate::core::protocol::a2a::adapt(stream, context);
        Ok(Box::pin(output.then(move |response| {
            let store = store.clone();
            let live = live.clone();
            let projection = projection.clone();
            let task_id = task_id.clone();
            let error_service = error_service.clone();
            let error_context = error_context.clone();
            async move {
                let response = match response {
                    Ok(response) => response,
                    Err(error) => {
                        // A Runtime/Event stream error is still a terminal
                        // externally visible task fact. Persist it before
                        // returning so detached HTTP drivers and later
                        // resubscriptions observe canceled/failed instead of
                        // a task stranded in working.
                        error_service.record_error(&error_context, &error).await?;
                        live.remove(&task_id);
                        return Err(error);
                    }
                };
                let _guard = projection.lock().await;
                store.record(&task_id, response.clone()).await?;
                live.publish(&task_id, response.clone());
                if response_terminal(&response) {
                    live.remove(&task_id);
                }
                Ok(response)
            }
        })))
    }

    /// Start a new A2A task and expose the projected Task/Artifact stream.
    pub async fn stream(
        &self,
        request: SendMessageRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, Error>> + Send + 'static>>, Error>
    {
        let (context, input) = self.begin(request).await?;
        self.stream_from(context, input).await
    }

    /// Submit one message to the existing task's waiting Agent.
    pub async fn reply(&self, request: SendMessageRequest) -> Result<Task, Error> {
        let task_id = request
            .message
            .task_id
            .as_deref()
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "reply requires taskId"))?;
        let context = self.context(task_id).await?;
        let task = self.store.task(task_id).await?;
        if !matches!(
            task.status.state,
            TaskState::InputRequired | TaskState::AuthRequired
        ) {
            return Err(Error::new(
                "CONFLICT",
                "A2A reply requires a task waiting for input",
            ));
        }
        if self.accept_message(&context, &request.message).await? == MessageState::Duplicate {
            return self.store.task(task_id).await;
        }
        let input = crate::core::protocol::a2a::input(&request.message, &context)?;
        self.agent
            .reply(input, self.live.cancellation(task_id)?)
            .await?;
        self.store.task(task_id).await
    }

    /// Return a snapshot followed by events published after subscription.
    pub async fn resubscribe(&self, task_id: &str) -> Result<Projection, Error> {
        let _guard = self.projection.lock().await;
        let task = self.store.task(task_id).await?;
        let receiver = self.live.subscribe(task_id)?;
        let done = terminal(&task);
        let store = self.store.clone();
        let task_id = task_id.to_owned();
        Ok(Box::pin(stream::unfold(
            (Some(task), receiver, done, store, task_id),
            |(snapshot, mut receiver, done, store, task_id)| async move {
                if let Some(task) = snapshot {
                    return Some((
                        Ok(StreamResponse::Task(task)),
                        (None, receiver, done, store, task_id),
                    ));
                }
                if done {
                    return None;
                }
                match receiver.recv().await {
                    Ok(response) => {
                        let done = response_terminal(&response);
                        Some((Ok(response), (None, receiver, done, store, task_id)))
                    }
                    Err(broadcast::error::RecvError::Lagged(_count)) => {
                        match store.task(&task_id).await {
                            Ok(task) => {
                                let done = terminal(&task);
                                Some((
                                    Ok(StreamResponse::Task(task)),
                                    (None, receiver, done, store, task_id),
                                ))
                            }
                            Err(error) => {
                                Some((Err(error), (None, receiver, true, store, task_id)))
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => None,
                }
            },
        )))
    }

    async fn context(&self, task_id: &str) -> Result<Context, Error> {
        self.store.context(task_id).await
    }

    async fn accept_message(
        &self,
        context: &Context,
        message: &Message,
    ) -> Result<MessageState, Error> {
        if !task_matches(context, message) && message.task_id.is_some() {
            return Err(Error::new("CONFLICT", "A2A taskId does not match the task"));
        }
        self.store.message(&context.task_id, message).await
    }

    async fn record_working(&self, context: &Context) -> Result<(), Error> {
        self.record(
            &context.task_id,
            StreamResponse::TaskStatusUpdateEvent(TaskStatusUpdateEvent {
                task_id: context.task_id(),
                context_id: context.context_id(),
                status: TaskStatus {
                    state: TaskState::Working,
                    message: None,
                },
                metadata: Map::new(),
            }),
        )
        .await
    }

    pub(crate) async fn record_error(&self, context: &Context, error: &Error) -> Result<(), Error> {
        self.record(&context.task_id, error_status(context, error))
            .await
    }

    async fn record(&self, task_id: &str, response: StreamResponse) -> Result<(), Error> {
        let _guard = self.projection.lock().await;
        self.store.record(task_id, response.clone()).await?;
        self.live.publish(task_id, response);
        Ok(())
    }
}

fn ids(request: &SendMessageRequest) -> Result<(i64, i64), Error> {
    fn value(request: &SendMessageRequest, name: &str) -> Option<i64> {
        request
            .metadata
            .get(name)
            .or_else(|| request.message.metadata.get(name))
            .and_then(Value::as_i64)
    }
    let session_id = value(request, "session_id")
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "A2A input requires session_id"))?;
    let chat_id = value(request, "chat_id")
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "A2A input requires chat_id"))?;
    Ok((session_id, chat_id))
}

fn status_message(error: &Error) -> Message {
    let mut metadata = Map::new();
    metadata.insert("code".into(), Value::String(error.code.clone()));
    Message {
        role: Role::Agent,
        parts: vec![Part::text(error.message.clone())],
        message_id: format!("status:{}", error.code),
        context_id: None,
        task_id: None,
        reference_task_ids: Vec::new(),
        metadata,
    }
}

pub(crate) fn error_status(context: &Context, error: &Error) -> StreamResponse {
    StreamResponse::TaskStatusUpdateEvent(TaskStatusUpdateEvent {
        task_id: context.task_id(),
        context_id: context.context_id(),
        status: TaskStatus {
            state: if error.code == "CANCELLED" {
                TaskState::Canceled
            } else {
                TaskState::Failed
            },
            message: Some(status_message(error)),
        },
        metadata: Map::new(),
    })
}

fn terminal(task: &Task) -> bool {
    matches!(
        task.status.state,
        TaskState::Completed | TaskState::Failed | TaskState::Canceled | TaskState::Rejected
    )
}

fn response_terminal(response: &StreamResponse) -> bool {
    match response {
        StreamResponse::Task(task) => terminal(task),
        StreamResponse::TaskStatusUpdateEvent(update) => matches!(
            update.status.state,
            TaskState::Completed | TaskState::Failed | TaskState::Canceled | TaskState::Rejected
        ),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
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
        let error =
            dispatch(Context::new(2001, 3001, "task-1"), request(Some("task-2"))).unwrap_err();
        assert_eq!(error.code, "CONFLICT");
    }
}
