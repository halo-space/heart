//! Static Workflow definition and compilation.
//!
//! Static Workflow definitions and their executable Pregel runtime. The
//! builder records an immutable graph definition and the runtime owns the
//! single mutable graph state for one `chat_id`.

mod events;
pub mod graph;
pub mod node;
pub mod tool;

use serde_json::Value;
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;

use crate::Error;
use crate::agent::{Input as AgentInput, Payload as AgentPayload};
use crate::components::message::{Messages, Role};
use crate::components::node::Signal;
use crate::core::memory::history::{History, Run};
use crate::runtime::workflow::node::ExecutionRef;

pub use graph::{
    CompiledWorkflow, Conditional, Definition, Edge, Endpoint, FailureRelation, FanOut,
    FieldSource, GraphDefinition, GraphError, GraphReference, InputBinding, IoSchema, Join, Loop,
    NodeAttrs, NodeDefinition, RetryPolicy, State, Status, Wait, WorkflowBuilder,
};

type CheckpointSave = Arc<
    dyn Fn(
            i64,
            Value,
            crate::Cancellation,
        ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + 'static>>
        + Send
        + Sync,
>;

/// State-owning Workflow runtime façade.
///
/// Scheduling policy and actual async Node invocation remain outside this
/// façade; every mutation is delegated to the single graph State so callers
/// cannot accidentally maintain a second execution model.
pub struct Runtime {
    chat_id: i64,
    workflow: CompiledWorkflow,
    engine: Mutex<Option<Engine>>,
    replies: crate::runtime::agent::reply::Hub,
}

#[derive(Clone)]
struct RecoveryReply {
    response: Messages,
    input: AgentInput,
    path: Vec<graph::ReplyExecution>,
}

/// Moves the one execution into its original stream and returns it on drop.
/// No mutex guard crosses await, no cloned live State or public handle exists.
struct ExecutionOwner<'a> {
    engine: Option<Engine>,
    home: &'a Mutex<Option<Engine>>,
}
impl std::ops::Deref for ExecutionOwner<'_> {
    type Target = Engine;
    fn deref(&self) -> &Engine {
        self.engine.as_ref().expect("owned Workflow execution")
    }
}
impl std::ops::DerefMut for ExecutionOwner<'_> {
    fn deref_mut(&mut self) -> &mut Engine {
        self.engine.as_mut().expect("owned Workflow execution")
    }
}
impl Drop for ExecutionOwner<'_> {
    fn drop(&mut self) {
        if let Some(engine) = self.engine.as_mut()
            && !matches!(
                engine.state.status,
                Status::Succeeded | Status::Failed | Status::Cancelled
            )
            && engine.state.mark_waiting()
        {
            engine.state.status = Status::Paused;
        }
        *self.home.lock().expect("Workflow execution lock poisoned") = self.engine.take();
    }
}

impl Runtime {
    #[cfg(test)]
    async fn execute(
        &mut self,
        initial: Messages,
        ids: Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: crate::Cancellation,
    ) -> Result<Messages, Error> {
        self.engine_mut().execute(initial, ids, cancellation).await
    }
    pub fn new(chat_id: i64, workflow: CompiledWorkflow) -> Result<Self, Error> {
        let engine = Engine::new(chat_id, workflow.clone())?;
        Ok(Self {
            chat_id,
            workflow,
            engine: Mutex::new(Some(engine)),
            replies: Default::default(),
        })
    }
    pub fn from_snapshot(
        chat_id: i64,
        workflow: CompiledWorkflow,
        snapshot: Value,
    ) -> Result<Self, Error> {
        let engine = Engine::from_snapshot(chat_id, workflow.clone(), snapshot)?;
        Ok(Self {
            chat_id,
            workflow,
            engine: Mutex::new(Some(engine)),
            replies: Default::default(),
        })
    }
    fn engine_mut(&mut self) -> &mut Engine {
        self.engine
            .get_mut()
            .expect("workflow execution lock poisoned")
            .as_mut()
            .expect("drop the Workflow run stream before mutating State")
    }
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Result<Self, Error> {
        if max_concurrency == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "max_concurrency must be positive",
            ));
        }
        self.engine_mut().max_concurrency = max_concurrency;
        Ok(self)
    }
    pub fn memory<S, C, T, L>(
        mut self,
        config: crate::memory::Config<S, C, T, L>,
    ) -> Result<Self, Error>
    where
        S: crate::memory::short::session::Store + 'static,
        C: crate::memory::short::chat::Store + 'static,
        T: crate::memory::short::trace::Store + 'static,
        L: crate::memory::Memory + 'static,
    {
        config.validate()?;
        let mut history = History::default();
        history.set(config);
        self.engine_mut().memory = Some(history);
        Ok(self)
    }
    pub fn with_checkpoint<C: crate::checkpoint::Checkpoint + 'static>(
        mut self,
        checkpoint: C,
    ) -> Self {
        let checkpoint = Arc::new(checkpoint);
        self.engine_mut().checkpoint = Some(Arc::new(move |id, state, cancellation| {
            let checkpoint = checkpoint.clone();
            Box::pin(async move { checkpoint.save(id, state, &cancellation).await })
        }));
        self
    }
    pub fn chat_id(&self) -> i64 {
        self.chat_id
    }
    pub fn definition(&self) -> &Definition {
        self.workflow.definition()
    }
    /// Borrow the sole State after dropping/finishing the execution stream.
    /// Live feedback uses reply and never locks or copies this State.
    pub fn state(&mut self) -> &State {
        &self.engine_mut().state
    }
    pub fn state_mut(&mut self) -> &mut State {
        &mut self.engine_mut().state
    }
    pub fn ready_nodes(&mut self) -> Vec<String> {
        self.engine_mut().ready_nodes()
    }
    pub fn start_batch(
        &mut self,
        executions: Vec<(String, i64, Vec<ExecutionRef>)>,
    ) -> Result<(), Error> {
        self.engine_mut().start_batch(executions)
    }
    pub fn complete(
        &mut self,
        node_id: &str,
        exec_id: i64,
        messages: Messages,
        to: Vec<String>,
    ) -> Result<(), Error> {
        self.engine_mut().complete(node_id, exec_id, messages, to)
    }
    pub fn fail(&mut self, node_id: &str, exec_id: i64, error: Error) -> Result<(), Error> {
        self.engine_mut().fail(node_id, exec_id, error)
    }
    pub fn cancel(&mut self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        self.engine_mut().cancel(node_id, exec_id)
    }
    /// Resolved, ended invocations retain their existing node-based recovery.
    pub fn resume(&mut self, node_id: &str, response: Messages) -> Result<(), Error> {
        self.engine_mut().resume(node_id, response)
    }
    pub fn advance_step(&mut self) -> bool {
        self.engine_mut().advance_step()
    }
    pub fn snapshot(&self) -> Result<Value, Error> {
        self.engine
            .try_lock()
            .map_err(|_| Error::new("CONFLICT", "Workflow State is owned by its active stream"))?
            .as_ref()
            .ok_or_else(|| Error::new("CONFLICT", "Workflow State is owned by its active stream"))?
            .snapshot()
    }
    pub async fn save_checkpoint<C: crate::checkpoint::Checkpoint + ?Sized>(
        &self,
        checkpoint: &C,
        cancellation: &crate::Cancellation,
    ) -> Result<(), Error> {
        let state = self.snapshot()?;
        let id = state["chat_id"].as_i64().expect("Workflow identity");
        checkpoint.save(id, state, cancellation).await
    }
    /// Submit to the actual live approval Step. Outputs stay on the original run stream.
    pub async fn reply(
        &self,
        response: Messages,
        cancellation: crate::Cancellation,
    ) -> Result<(), Error> {
        if response.as_slice().len() != 1 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "one live approval submission must identify one question Message",
            ));
        }
        let input = messages_to_agent_input(&response);
        match self.replies.submit(input.clone(), &cancellation) {
            Ok(()) => Ok(()),
            Err(error) if error.code == "NOT_FOUND" => {
                let Ok(mut engine) = self.engine.try_lock() else {
                    return Err(error);
                };
                let Some(engine) = engine.as_mut() else {
                    return Err(error);
                };
                engine.reply_saved(response, input, &cancellation)
            }
            Err(error) => Err(error),
        }
    }
    pub async fn run(
        &self,
        initial_messages: Messages,
        next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: crate::Cancellation,
    ) -> Result<impl futures_core::Stream<Item = Result<crate::event::Event, Error>> + '_, Error>
    {
        let mut engine = {
            let mut home = self
                .engine
                .try_lock()
                .map_err(|_| Error::new("CONFLICT", "Workflow already has an active run stream"))?;
            let engine = home.as_ref().ok_or_else(|| {
                Error::new("CONFLICT", "Workflow already has an active run stream")
            })?;
            engine.validate_run_input(&initial_messages)?;
            ExecutionOwner {
                engine: home.take(),
                home: &self.engine,
            }
        };
        let mut metadata = messages_to_agent_input(&initial_messages).metadata;
        metadata.insert("chat_id".into(), engine.state.chat_id.into());
        let interaction = self.replies.start(metadata)?;
        let owner = interaction.owner();
        Ok(events::stream(events::Output::default(), async move {
            let _owner = owner;
            events::with_interaction(
                interaction,
                engine.execute(initial_messages, next_exec_id, cancellation),
            )
            .await
        }))
    }
}

struct Engine {
    workflow: CompiledWorkflow,
    state: State,
    max_concurrency: usize,
    checkpoint: Option<CheckpointSave>,
    memory: Option<History>,
    history_run: Option<Run>,
    history_closed: bool,
    // Derived from the parent's earlier Attempt.graph, not separate truth.
    // It supplies inherited scenes when a failed nested Graph Node is retried.
    retry_source: Option<Box<graph::State>>,
    // Derived from the saved Agent Step and carried only until the existing
    // execution consumes it. This is transient routing, not saved Runtime state.
    recovery_reply: Option<RecoveryReply>,
    // Saved replies are accepted only by a Runtime explicitly reconstructed
    // from Checkpoint, never by an abandoned live Runtime.
    accepts_saved_replies: bool,
}

impl Engine {
    async fn accept_progress(
        &mut self,
        node_id: &str,
        exec_id: i64,
        mut attempt: node::Attempt,
        event: Option<crate::event::Event>,
        cancellation: &crate::Cancellation,
    ) -> Result<(), Error> {
        let delta = matches!(&event, Some(crate::event::Event::Delta { .. }));
        let question = event.as_ref().and_then(|event| match event {
            crate::event::Event::Complete { message, .. } => {
                message.metadata.get("reply_id").and_then(Value::as_i64)
            }
            _ => None,
        });
        if let Some(id) = question
            && !attempt_question(&attempt, id)?
        {
            return Err(Error::new(
                "INVALID_STATE",
                "Workflow question has no native waiting Step",
            ));
        }
        if !delta {
            let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
            execution.status = if attempt_waiting(&attempt) {
                node::ExecutionStatus::Waiting
            } else {
                node::ExecutionStatus::Running
            };
            let idx = attempt.idx as usize;
            if idx > execution.attempts.len() {
                return Err(Error::new(
                    "INVALID_STATE",
                    "Workflow progress skips an Attempt",
                ));
            }
            if let Some(previous) = execution.attempts.get(idx) {
                attempt.created_time = previous.created_time;
            }
            if idx == execution.attempts.len() {
                execution.attempts.push(attempt);
            } else {
                execution.attempts[idx] = attempt;
            }
            self.state.recompute_usage();
            self.state.status = if self
                .state
                .nodes
                .values()
                .flat_map(|node| node.executions.values())
                .any(|execution| execution.attempts.last().is_some_and(attempt_waiting))
            {
                graph::Status::Paused
            } else {
                graph::Status::Running
            };
        }
        let chat_id = self.state.chat_id;
        self.state.bind_agent_chat_id(chat_id)?;
        if question.is_some() {
            if let Some(checkpoint) = &self.checkpoint {
                checkpoint(
                    self.state.chat_id,
                    self.snapshot()?,
                    crate::Cancellation::new(),
                )
                .await?;
            }
            if cancellation.is_cancelled() {
                return Err(Error::new(
                    "CANCELLED",
                    "Workflow cancelled before presenting approval",
                ));
            }
        }
        self.publish(event, delta).await
    }

    async fn publish(&self, event: Option<crate::event::Event>, delta: bool) -> Result<(), Error> {
        if events::has_parent() {
            let attempt = if delta {
                node::Attempt {
                    idx: events::attempt_index(),
                    ..Default::default()
                }
            } else {
                node::Attempt {
                    idx: events::attempt_index(),
                    status: if self.state.status == graph::Status::Paused {
                        node::AttemptStatus::Waiting
                    } else {
                        node::AttemptStatus::Running
                    },
                    usage: self.state.usage.clone(),
                    graph: Some(Box::new(self.state.clone())),
                    ..Default::default()
                }
            };
            events::progress(attempt, event).await
        } else {
            if let Some(event) = event {
                events::emit(event).await;
            }
            Ok(())
        }
    }
    pub fn new(chat_id: i64, workflow: CompiledWorkflow) -> Result<Self, Error> {
        if chat_id <= 0 {
            return Err(Error::new("INVALID_ARGUMENTS", "chat_id must be positive"));
        }
        let state = State::new(chat_id, workflow.definition());
        Ok(Self {
            workflow,
            state,
            max_concurrency: 100,
            checkpoint: None,
            memory: None,
            history_run: None,
            history_closed: false,
            retry_source: None,
            recovery_reply: None,
            accepts_saved_replies: false,
        })
    }

    pub fn from_snapshot(
        chat_id: i64,
        workflow: CompiledWorkflow,
        snapshot: Value,
    ) -> Result<Self, Error> {
        if chat_id <= 0 {
            return Err(Error::new("INVALID_ARGUMENTS", "chat_id must be positive"));
        }
        let state: State = serde_json::from_value(snapshot)
            .map_err(|error| Error::new("INVALID_CHECKPOINT", error.to_string()))?;
        state.check_identity(chat_id)?;
        if state.definition_id != workflow.definition().definition_id
            || state.version != workflow.definition().version
        {
            return Err(Error::new(
                "CONFLICT",
                "checkpoint definition does not match compiled workflow",
            ));
        }
        let mut runtime = Self {
            workflow,
            state,
            max_concurrency: 100,
            checkpoint: None,
            memory: None,
            history_run: None,
            history_closed: false,
            retry_source: None,
            recovery_reply: None,
            accepts_saved_replies: true,
        };
        if runtime.state.mark_waiting() {
            runtime.state.status = graph::Status::Running;
        }
        runtime.state.recompute_usage();
        Ok(runtime)
    }

    pub fn definition(&self) -> &Definition {
        self.workflow.definition()
    }

    pub fn ready_nodes(&self) -> Vec<String> {
        self.state.ready_nodes(self.definition())
    }

    pub fn start_batch(
        &mut self,
        executions: Vec<(String, i64, Vec<ExecutionRef>)>,
    ) -> Result<(), Error> {
        let definition = self.workflow.definition().clone();
        self.state.start_batch(&definition, executions)
    }

    pub fn complete(
        &mut self,
        node_id: &str,
        exec_id: i64,
        messages: Messages,
        to: Vec<String>,
    ) -> Result<(), Error> {
        self.state
            .complete_execution(node_id, exec_id, messages, to)?;
        self.state.recompute_usage();
        Ok(())
    }

    pub fn fail(&mut self, node_id: &str, exec_id: i64, error: Error) -> Result<(), Error> {
        self.state.fail_execution(node_id, exec_id, error)?;
        self.state.recompute_usage();
        Ok(())
    }

    pub fn cancel(&mut self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        self.state.cancel_execution(node_id, exec_id)?;
        self.state.recompute_usage();
        Ok(())
    }

    /// Submits the external response for an active wait. The next `run` call
    /// resumes the same execution and Attempt; it does not allocate a new
    /// `exec_id`.
    pub fn resume(&mut self, node_id: &str, response: Messages) -> Result<(), Error> {
        self.state.resume(node_id, response)
    }

    fn reply_saved(
        &mut self,
        response: Messages,
        input: AgentInput,
        cancellation: &crate::Cancellation,
    ) -> Result<(), Error> {
        if !self.accepts_saved_replies {
            return Err(Error::new(
                "NOT_FOUND",
                "no restored Workflow Checkpoint is receiving this reply",
            ));
        }
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "reply submission cancelled"));
        }
        if input.metadata.get("chat_id").and_then(Value::as_i64) != Some(self.state.chat_id) {
            return Err(Error::new(
                "CONFLICT",
                "reply belongs to a different Workflow run",
            ));
        }
        if let Some(initial) = &self.state.initial {
            let saved = messages_to_agent_input(initial).metadata;
            for key in ["tenant_id", "user_id", "agent_id", "session_id"] {
                if saved.contains_key(key) && input.metadata.get(key) != saved.get(key) {
                    return Err(Error::new(
                        "CONFLICT",
                        "reply belongs to a different Workflow identity",
                    ));
                }
            }
        }
        if let Some(previous) = &self.recovery_reply {
            return if previous.response == response {
                Ok(())
            } else {
                Err(Error::new(
                    "CONFLICT",
                    "this question already received a different reply",
                ))
            };
        }
        let reply_id = input
            .metadata
            .get("reply_id")
            .and_then(Value::as_i64)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "reply requires reply_id"))?;
        let mut paths = self.state.reply_paths(reply_id)?;
        if paths.len() != 1 {
            return Err(Error::new(
                if paths.is_empty() {
                    "NOT_FOUND"
                } else {
                    "CONFLICT"
                },
                if paths.is_empty() {
                    "reply does not identify a saved waiting question"
                } else {
                    "saved question identity is ambiguous"
                },
            ));
        }
        let path = paths.pop().expect("one validated reply path");
        let scene = agent_scene_for_reply_path(&self.state, &path)?;
        let Some(_) = crate::runtime::agent::reply::prepare_restore(scene, &input)? else {
            return Err(Error::new(
                "NOT_FOUND",
                "reply does not identify a saved waiting question",
            ));
        };

        // Fully recorded Workflow waits already have their response slot in
        // the current State. Earlier waiting snapshots are routed transiently
        // to the existing active Attempt below.
        if self
            .state
            .waits
            .get(&path[0].node_id)
            .is_some_and(|wait| wait.exec_id == path[0].exec_id)
        {
            return self.state.resume(&path[0].node_id, response);
        }
        self.recovery_reply = Some(RecoveryReply {
            response,
            input,
            path,
        });
        Ok(())
    }

    pub fn advance_step(&mut self) -> bool {
        self.state.advance_step()
    }

    /// Produces the exact graph-state JSON accepted by Checkpoint.
    pub fn snapshot(&self) -> Result<Value, Error> {
        serde_json::to_value(&self.state)
            .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))
    }

    fn validate_run_input(&self, initial_messages: &Messages) -> Result<(), Error> {
        self.state.check_identity(self.state.chat_id)?;
        if self
            .state
            .initial
            .as_ref()
            .is_some_and(|initial| initial != initial_messages)
        {
            return Err(Error::new(
                "CONFLICT",
                "workflow initial Messages cannot change after activation",
            ));
        }
        if let Some(chat_id) = messages_to_agent_input(initial_messages)
            .metadata
            .get("chat_id")
            && chat_id.as_i64() != Some(self.state.chat_id)
        {
            return Err(Error::new(
                "CONFLICT",
                "input chat_id does not match this Workflow run",
            ));
        }
        Ok(())
    }

    // Private complete-returning execution engine. The only public run
    // contract is the observation stream; this is not a compatibility API.
    async fn execute(
        &mut self,
        initial_messages: Messages,
        next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: crate::Cancellation,
    ) -> Result<Messages, Error> {
        self.validate_run_input(&initial_messages)?;
        // A terminal snapshot is already closed. Re-running it is a read of
        // the committed result and must not create another Chat/Trace record
        // or invoke any node again.
        if matches!(
            self.state.status,
            graph::Status::Succeeded | graph::Status::Failed | graph::Status::Cancelled
        ) {
            let result = self
                .run_inner(initial_messages, next_exec_id, cancellation)
                .await;
            if result.is_ok() {
                for event in terminal_observations(&self.state) {
                    events::emit(event).await;
                }
            } else if result
                .as_ref()
                .is_err_and(|error| error.code == "CANCELLED")
            {
                for event in cancelled_observations(&self.state) {
                    events::emit(event).await;
                }
            }
            return result;
        }
        self.begin_history(&initial_messages, &cancellation).await?;
        let result = self
            .run_inner(initial_messages, next_exec_id, cancellation)
            .await;
        if let Err(error) = &result
            && error.code != "WAITING"
        {
            if error.code == "CANCELLED" {
                self.cancel_active();
                self.state.status = graph::Status::Cancelled;
            } else if self.state.status != graph::Status::Failed {
                let active = self
                    .state
                    .nodes
                    .iter()
                    .flat_map(|(id, node)| {
                        node.executions
                            .iter()
                            .filter(|(_, execution)| {
                                matches!(
                                    execution.status,
                                    node::ExecutionStatus::Running | node::ExecutionStatus::Waiting
                                )
                            })
                            .map(move |(exec_id, _)| (id.clone(), *exec_id))
                    })
                    .collect::<Vec<_>>();
                for (node_id, exec_id) in active {
                    self.state
                        .fail_execution(&node_id, exec_id, error.clone())?;
                }
                self.state.status = graph::Status::Failed;
            }
            self.state.recompute_usage();
        }
        if let Some(checkpoint) = self.checkpoint.clone() {
            let state = self.snapshot()?;
            checkpoint(self.state.chat_id, state, crate::Cancellation::new()).await?;
        }
        self.close_history(&result).await?;
        result
    }

    async fn begin_history(
        &mut self,
        initial_messages: &Messages,
        cancellation: &crate::Cancellation,
    ) -> Result<(), Error> {
        let Some(history) = self.memory.clone() else {
            return Ok(());
        };
        if self.history_run.is_some() {
            return Ok(());
        }
        let input = messages_to_agent_input(initial_messages);
        let run = if self.state.initial.is_some() {
            history
                .resume_for(input, crate::memory::short::Type::Workflow, cancellation)
                .await?
        } else {
            history
                .begin_for(input, crate::memory::short::Type::Workflow, cancellation)
                .await?
        };
        self.history_run = Some(run);
        Ok(())
    }

    async fn close_history(&mut self, result: &Result<Messages, Error>) -> Result<(), Error> {
        if self.history_closed {
            return Ok(());
        }
        let Some(run) = self.history_run.clone() else {
            return Ok(());
        };
        let status = match result {
            Err(error) if error.code == "WAITING" => return Ok(()),
            Ok(_) => crate::memory::short::chat::Status::Completed,
            Err(error) if error.code == "CANCELLED" => crate::memory::short::chat::Status::Stopped,
            Err(_) => crate::memory::short::chat::Status::Failed,
        };
        run.settle(self.state.usage.clone());
        let graph = serde_json::to_value(self.definition())
            .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))?;
        let state = self.snapshot()?;
        let message = result.as_ref().ok().and_then(|messages| {
            messages
                .as_slice()
                .iter()
                .rev()
                .find(|message| message.role == Role::Assistant)
                .cloned()
        });
        run.close_workflow(status, message, Some(graph), Some(state))
            .await?;
        self.history_closed = true;
        Ok(())
    }

    /// Executes the compiled graph until it reaches END, waiting, failure or
    /// cancellation. The caller's ID source supplies one id for each new
    /// logical activation; retries keep the original id.
    async fn run_inner(
        &mut self,
        initial_messages: Messages,
        next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: crate::Cancellation,
    ) -> Result<Messages, Error> {
        if let Some(existing) = &self.state.initial {
            if existing != &initial_messages {
                return Err(Error::new(
                    "CONFLICT",
                    "workflow initial Messages cannot change after activation",
                ));
            }
        } else {
            self.state.initial = Some(initial_messages);
        }

        loop {
            if self.state.status == graph::Status::Succeeded {
                return Ok(self.end_messages());
            }
            if self.state.status == graph::Status::Failed {
                return Err(self
                    .first_failure()
                    .unwrap_or_else(|| Error::new("FAILED", "workflow execution failed")));
            }
            if self.state.status == graph::Status::Cancelled || cancellation.is_cancelled() {
                self.cancel_active();
                self.state.status = graph::Status::Cancelled;
                return Err(Error::new("CANCELLED", "workflow run cancelled"));
            }
            if let Some((node_id, exec_id, _wait_id, response)) = self.state.resolved_wait() {
                let target = self.target_for(&node_id, exec_id)?;
                let input = match &target {
                    Target::Agent(..) => {
                        let execution = &self.state.nodes[&node_id].executions[&exec_id];
                        let mut input = self.agent_input_for(&node_id, &execution.from)?;
                        let reply = messages_to_agent_input(&response);
                        input.payload = reply.payload;
                        // Reply metadata may supply extra context, but conflicting
                        // ownership still reaches the Agent's canonical check.
                        input.metadata.extend(reply.metadata);
                        serde_json::to_value(input)
                            .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))?
                    }
                    Target::Graph { .. } => self.input_for(
                        &node_id,
                        &self.state.nodes[&node_id].executions[&exec_id].from,
                    )?,
                    Target::Node(_) => serde_json::to_value(response)
                        .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))?,
                };
                let retry = self.definition().nodes[&node_id].retry.clone();
                let start_idx = self.state.nodes[&node_id].executions[&exec_id]
                    .attempts
                    .last()
                    .map(|attempt| attempt.idx)
                    .ok_or_else(|| Error::new("INVALID_STATE", "execution has no attempt"))?;
                let invocation = events::with_source(
                    self.state.chat_id,
                    &node_id,
                    exec_id,
                    invoke_with_retry(
                        target,
                        input,
                        cancellation.clone(),
                        retry,
                        next_exec_id.clone(),
                        self.state.chat_id,
                        start_idx,
                        committed_tool_messages(&self.state.nodes[&node_id].executions[&exec_id]),
                        None,
                    ),
                )
                .await;
                let invocation_graph = invocation.graph;
                self.replace_resumed_attempts(&node_id, exec_id, invocation.attempts)?;
                if let Some(graph) = invocation_graph {
                    self.attach_graph(&node_id, exec_id, graph)?;
                }
                match invocation.outcome {
                    Outcome::Complete(value) => {
                        let messages = self.output_messages(&node_id, value)?;
                        let to = self.outgoing(&node_id, &messages)?;
                        self.state
                            .complete_execution(&node_id, exec_id, messages, to)?;
                    }
                    Outcome::Wait(wait_id) => {
                        self.state.wait_execution(&node_id, exec_id, wait_id)?;
                    }
                    Outcome::Cancelled => {
                        self.state.cancel_execution(&node_id, exec_id)?;
                    }
                    Outcome::Failed(error) => {
                        self.state.fail_execution(&node_id, exec_id, error)?;
                    }
                }
                self.state.recompute_usage();
                self.observe_commit(&node_id, exec_id).await?;
                // Resume has the same Barrier semantics as an ordinary
                // invocation. Other active/waiting siblings still block it.
                self.state.advance_step();
                continue;
            }
            if self.resume_waiting(&next_exec_id, &cancellation).await? {
                self.state.advance_step();
                continue;
            }
            let ready = self.ready_nodes();
            if ready.is_empty() {
                if !self.state.waits.is_empty() {
                    self.state.status = graph::Status::Paused;
                    return Err(Error::new("WAITING", "workflow is waiting for input"));
                }
                if self.reached_end() {
                    self.state.status = graph::Status::Succeeded;
                    return Ok(self.end_messages());
                }
                self.state.status = graph::Status::Failed;
                return Err(self.first_failure().unwrap_or_else(|| {
                    Error::new(
                        "INVALID_STATE",
                        "workflow has no ready node and has not reached END",
                    )
                }));
            }

            let mut specs = Vec::with_capacity(ready.len());
            let mut seen = BTreeSet::new();
            for node_id in ready {
                let exec_id = next_exec_id();
                if !seen.insert(exec_id) {
                    return Err(Error::new("CONFLICT", "exec_id must be globally unique"));
                }
                let from = self.source_executions(&node_id)?;
                let input = self.input_for(&node_id, &from)?;
                let target = if let Some(workflow) = self.workflow.graph_node(&node_id) {
                    Target::Graph {
                        workflow: Arc::new(workflow.clone()),
                        state: None,
                        history: self.inherited_graph(&node_id),
                    }
                } else if let Some(agent) = self.workflow.agent(&node_id) {
                    Target::Agent(agent, None)
                } else {
                    Target::Node(self.workflow.executor(&node_id).ok_or_else(|| {
                        Error::new("UNSUPPORTED", "workflow node has no executor")
                    })?)
                };
                let retry = self.definition().nodes[&node_id].retry.clone();
                specs.push((node_id, exec_id, from, input, target, retry));
            }

            let definition = self.definition().clone();
            self.state.start_batch(
                &definition,
                specs
                    .iter()
                    .map(|(node_id, exec_id, from, _, _, _)| {
                        (node_id.clone(), *exec_id, from.clone())
                    })
                    .collect(),
            )?;

            let limit = self.max_concurrency;
            let task_cancellation = cancellation.clone();
            let task_next_exec_id = Arc::clone(&next_exec_id);
            let chat_id = self.state.chat_id;
            let relay = events::Relay::default();
            let slots = Arc::new(tokio::sync::Semaphore::new(limit));
            let changed = Arc::new(futures_util::task::AtomicWaker::new());
            let interaction = events::interaction();
            let mut active = Vec::new();
            let mut results = specs
                .into_iter()
                .map(|(node_id, exec_id, from, input, target, retry)| {
                    let cancellation = task_cancellation.clone();
                    let next_exec_id = Arc::clone(&task_next_exec_id);
                    let relay = relay.clone();
                    let slot =
                        crate::runtime::agent::reply::Slot::waiting(slots.clone(), changed.clone());
                    let run = interaction.as_ref().map(|run| run.branch(slot.clone()));
                    active.push((node_id.clone(), exec_id, run.clone(), slot.clone()));
                    async move {
                        let owner = run.as_ref().map(|run| run.owner());
                        let entering = match &run {
                            Some(run) => run.continue_execution(&cancellation).await,
                            None => slot.acquire(&cancellation).await,
                        };
                        if let Err(error) = entering {
                            return (
                                node_id,
                                exec_id,
                                from,
                                Invocation {
                                    attempts: vec![node::Attempt {
                                        status: node::AttemptStatus::Cancelled,
                                        ..Default::default()
                                    }],
                                    outcome: Outcome::Failed(error),
                                    graph: None,
                                },
                            );
                        }
                        let result = events::with_optional_interaction(
                            run,
                            events::with_relay(
                                relay,
                                events::with_source(
                                    chat_id,
                                    &node_id,
                                    exec_id,
                                    invoke_with_retry(
                                        target,
                                        input,
                                        cancellation,
                                        retry,
                                        next_exec_id,
                                        chat_id,
                                        0,
                                        Messages::default(),
                                        None,
                                    ),
                                ),
                            ),
                        )
                        .await;
                        drop(owner);
                        slot.release();
                        (node_id, exec_id, from, result)
                    }
                })
                .collect::<futures_util::stream::FuturesUnordered<_>>();

            enum ProgressOrResult<T> {
                Facts(Box<events::Progress>),
                Finished(Box<T>),
            }
            loop {
                let next = std::future::poll_fn(|cx| {
                    changed.register(cx.waker());
                    match results.poll_next_unpin(cx) {
                        std::task::Poll::Ready(Some(result)) => {
                            return std::task::Poll::Ready(Some(ProgressOrResult::Finished(
                                Box::new(result),
                            )));
                        }
                        std::task::Poll::Ready(None) => return std::task::Poll::Ready(None),
                        std::task::Poll::Pending => {}
                    }
                    if let std::task::Poll::Ready(progress) = relay.poll(cx) {
                        return std::task::Poll::Ready(Some(ProgressOrResult::Facts(Box::new(
                            progress,
                        ))));
                    }
                    for (node_id, exec_id, run, _) in &active {
                        if self.state.nodes[node_id].executions[exec_id]
                            .attempts
                            .last()
                            .is_some_and(|attempt| attempt.status == node::AttemptStatus::Waiting)
                            && let Some(run) = run
                            && run.children_blocked()
                        {
                            run.suspend();
                        }
                    }
                    if active.iter().all(|(_, _, _, slot)| !slot.held())
                        && let Some(run) = &interaction
                    {
                        run.suspend();
                    }
                    std::task::Poll::Pending
                })
                .await;
                let Some(next) = next else { break };
                let (node_id, exec_id, _from, result) = match next {
                    ProgressOrResult::Facts(progress) => {
                        let events::Progress {
                            node_id,
                            exec_id,
                            attempt,
                            event,
                            ack,
                        } = *progress;
                        let result = self
                            .accept_progress(&node_id, exec_id, attempt, event, &cancellation)
                            .await;
                        let _ = ack.send(result.clone());
                        result?;
                        continue;
                    }
                    ProgressOrResult::Finished(result) => *result,
                };
                let invocation_graph = result.graph;
                self.replace_attempts(&node_id, exec_id, result.attempts)?;
                if let Some(graph) = invocation_graph {
                    self.attach_graph(&node_id, exec_id, graph)?;
                }
                match result.outcome {
                    Outcome::Complete(value) => {
                        let messages = self.output_messages(&node_id, value)?;
                        let to = self.outgoing(&node_id, &messages)?;
                        self.state
                            .complete_execution(&node_id, exec_id, messages, to)?;
                    }
                    Outcome::Wait(wait_id) => {
                        self.state.wait_execution(&node_id, exec_id, wait_id)?;
                    }
                    Outcome::Cancelled => {
                        self.state.cancel_execution(&node_id, exec_id)?;
                    }
                    Outcome::Failed(error) => {
                        self.state.fail_execution(&node_id, exec_id, error)?;
                    }
                }
                self.state.recompute_usage();
                self.observe_commit(&node_id, exec_id).await?;
            }
            if self.state.waits.is_empty() {
                self.state.status = graph::Status::Running;
            }
            self.state.recompute_usage();
            if cancellation.is_cancelled() {
                self.cancel_active();
                self.state.status = graph::Status::Cancelled;
                return Err(Error::new("CANCELLED", "workflow run cancelled"));
            }
            if self.state.nodes.values().any(|node| {
                node.executions.values().any(|execution| {
                    execution.idx == self.state.step
                        && execution.status == node::ExecutionStatus::Cancelled
                })
            }) {
                self.state.status = graph::Status::Cancelled;
                return Err(Error::new("CANCELLED", "workflow execution cancelled"));
            }
            if !self.state.advance_step() {
                self.state.status = graph::Status::Paused;
                return Err(Error::new(
                    "WAITING",
                    "workflow batch did not reach a barrier",
                ));
            }
        }
    }

    fn reached_end(&self) -> bool {
        self.state.nodes.values().any(|state| {
            state.executions.values().any(|execution| {
                execution.status == node::ExecutionStatus::Succeeded
                    && execution.to.iter().any(|target| target == graph::END)
            })
        })
    }

    fn end_messages(&self) -> Messages {
        let mut messages = Messages::default();
        for node_id in self.definition().nodes.keys() {
            if let Some(state) = self.state.nodes.get(node_id) {
                for execution in state.executions.values() {
                    if execution.status == node::ExecutionStatus::Succeeded
                        && execution.to.iter().any(|target| target == graph::END)
                    {
                        messages.0.extend(execution.messages.0.clone());
                    }
                }
            }
        }
        messages
    }

    fn outgoing(&self, node_id: &str, messages: &Messages) -> Result<Vec<String>, Error> {
        let mut targets = outgoing(self.definition(), node_id);
        let allowed: BTreeSet<String> = self
            .definition()
            .graph
            .conditional_edges
            .iter()
            .filter(|relation| relation.from == node_id)
            .flat_map(|relation| relation.to.iter().cloned())
            .collect();
        if !allowed.is_empty() {
            let selected = self.workflow.route(node_id, messages)?;
            if selected.is_empty() {
                return Err(Error::new(
                    "INVALID_ROUTE",
                    "conditional route selected no target",
                ));
            }
            if selected.iter().any(|target| !allowed.contains(target)) {
                return Err(Error::new(
                    "INVALID_ROUTE",
                    "conditional route selected an undeclared target",
                ));
            }
            targets.retain(|target| !allowed.contains(target));
            targets.extend(selected);
        }
        let loop_targets: Vec<String> = self
            .definition()
            .graph
            .loop_definitions
            .iter()
            .filter(|relation| relation.from == node_id)
            .filter(|_| self.workflow.loop_enabled(node_id, messages))
            .map(|relation| relation.to.clone())
            .collect();
        targets.extend(loop_targets);
        targets.sort();
        targets.dedup();
        Ok(targets)
    }

    fn first_failure(&self) -> Option<Error> {
        self.state
            .nodes
            .values()
            .flat_map(|state| state.executions.values())
            .find_map(|execution| execution.error.clone())
    }

    fn cancel_active(&mut self) {
        let active: Vec<(String, i64)> = self
            .state
            .nodes
            .iter()
            .flat_map(|(node_id, state)| {
                state
                    .executions
                    .values()
                    .filter(|execution| {
                        matches!(
                            execution.status,
                            node::ExecutionStatus::Running | node::ExecutionStatus::Waiting
                        )
                    })
                    .map(|execution| (node_id.clone(), execution.exec_id))
            })
            .collect();
        for (node_id, exec_id) in active {
            let _ = self.state.cancel_execution(&node_id, exec_id);
        }
    }

    fn source_executions(&self, node_id: &str) -> Result<Vec<ExecutionRef>, Error> {
        self.state
            .activation_sources(self.definition(), node_id)
            .ok_or_else(|| Error::new("INVALID_STATE", "node has no committed activation"))
    }

    fn input_for(&self, node_id: &str, from: &[ExecutionRef]) -> Result<Value, Error> {
        if self.is_failure_activation(node_id, from) {
            if self.definition().nodes[node_id].agent {
                return serde_json::to_value(AgentInput::default())
                    .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()));
            }
            if self.workflow.input_is_messages(node_id) {
                return serde_json::to_value(Messages::default())
                    .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()));
            }
            return Ok(Value::Object(serde_json::Map::new()));
        }
        if self.definition().nodes[node_id].agent {
            return serde_json::to_value(self.agent_input_for(node_id, from)?)
                .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()));
        }
        if from.is_empty() {
            if self.workflow.input_is_messages(node_id) {
                return serde_json::to_value(self.state.initial.as_ref().ok_or_else(|| {
                    Error::new("INVALID_STATE", "workflow initial Messages are missing")
                })?)
                .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()));
            }
            let values = self
                .state
                .initial
                .as_ref()
                .ok_or_else(|| {
                    Error::new("INVALID_STATE", "workflow initial Messages are missing")
                })?
                .as_slice()
                .last()
                .map(|message| message.values.clone())
                .unwrap_or_default();
            return Ok(Value::Object(values));
        }
        if self.workflow.input_is_messages(node_id) {
            let mut messages = Messages::default();
            for source in from {
                let execution = self.state.nodes[&source.node_id]
                    .executions
                    .get(&source.exec_id)
                    .ok_or_else(|| Error::new("NOT_FOUND", "source execution missing"))?;
                messages.0.extend(execution.messages.0.clone());
            }
            return serde_json::to_value(messages)
                .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()));
        }
        self.bound_values(node_id, from).map(Value::Object)
    }

    fn agent_input_for(&self, node_id: &str, from: &[ExecutionRef]) -> Result<AgentInput, Error> {
        let attrs = &self.definition().nodes[node_id].attrs.input;
        if !attrs.is_empty() {
            let values = self.bound_values(node_id, from)?;
            return serde_json::from_value(Value::Object(values))
                .map_err(|error| Error::new("INVALID_ARGUMENTS", error.to_string()));
        }
        let messages = if from.is_empty() {
            self.state.initial.clone().unwrap_or_default()
        } else {
            let mut messages = Messages::default();
            for source in from {
                if let Some(execution) = self.state.nodes[&source.node_id]
                    .executions
                    .get(&source.exec_id)
                {
                    messages.0.extend(execution.messages.0.clone());
                }
            }
            messages
        };
        Ok(messages_to_agent_input(&messages))
    }

    // Only field adaptation is shared with Agent execution. Graph activation,
    // execution selection, state types and Pregel behavior remain separate.
    fn bound_values(
        &self,
        node_id: &str,
        from: &[ExecutionRef],
    ) -> Result<serde_json::Map<String, Value>, Error> {
        let bindings =
            self.definition().nodes[node_id]
                .attrs
                .input
                .iter()
                .map(|(field, binding)| {
                    (
                        field.as_str(),
                        binding.from.node_id.as_str(),
                        binding.from.field.as_str(),
                    )
                });
        crate::components::node::input::fields(bindings, |source_id| {
            let source = from
                .iter()
                .find(|source| source.node_id == source_id)
                .ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", format!("missing source {source_id}"))
                })?;
            let execution = self
                .state
                .nodes
                .get(source_id)
                .and_then(|state| state.executions.get(&source.exec_id))
                .ok_or_else(|| Error::new("NOT_FOUND", "source execution missing"))?;
            if execution.status != node::ExecutionStatus::Succeeded {
                return Err(Error::new(
                    "INVALID_STATE",
                    "input source execution has not succeeded",
                ));
            }
            Ok(&execution.messages)
        })
    }

    fn is_failure_activation(&self, node_id: &str, from: &[ExecutionRef]) -> bool {
        self.definition()
            .graph
            .failure_relations
            .iter()
            .any(|relation| {
                relation.to.iter().any(|target| target == node_id)
                    && relation.from.len() == from.len()
                    && relation
                        .from
                        .iter()
                        .all(|source| from.iter().any(|item| &item.node_id == source))
                    && from.iter().any(|source| {
                        self.state
                            .nodes
                            .get(&source.node_id)
                            .and_then(|state| state.executions.get(&source.exec_id))
                            .is_some_and(|execution| {
                                execution.status == node::ExecutionStatus::Failed
                                    && (relation.codes.is_empty()
                                        || execution.error.as_ref().is_some_and(|error| {
                                            relation.codes.iter().any(|code| code == &error.code)
                                        }))
                            })
                    })
                    && (relation.on == "node_failed" || relation.on == "join_failed")
            })
    }

    fn output_messages(&self, node_id: &str, value: Value) -> Result<Messages, Error> {
        if self.workflow.output_is_messages(node_id) {
            serde_json::from_value(value)
                .map_err(|error| Error::new("INVALID_OUTPUT", error.to_string()))
        } else {
            Ok(Messages::new([crate::Message::function(value)]))
        }
    }

    async fn resume_waiting(
        &mut self,
        next_exec_id: &Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: &crate::Cancellation,
    ) -> Result<bool, Error> {
        let recovery_reply = self.recovery_reply.take();
        let target = if let Some(reply) = &recovery_reply {
            let current = reply
                .path
                .first()
                .ok_or_else(|| Error::new("INVALID_STATE", "saved reply path is empty"))?;
            let execution = self
                .state
                .nodes
                .get(&current.node_id)
                .and_then(|node| node.executions.get(&current.exec_id))
                .filter(|execution| {
                    execution.idx == self.state.step
                        && execution.waits.is_empty()
                        && matches!(
                            execution.status,
                            node::ExecutionStatus::Running | node::ExecutionStatus::Waiting
                        )
                })
                .filter(|execution| {
                    execution
                        .attempts
                        .last()
                        .is_some_and(|attempt| attempt.idx == current.idx)
                })
                .ok_or_else(|| Error::new("INVALID_STATE", "saved reply target is not active"))?;
            let _ = execution;
            Some((current.node_id.clone(), current.exec_id))
        } else {
            self.state.waiting_execution()
        };
        let Some((node_id, exec_id)) = target else {
            self.recovery_reply = recovery_reply;
            return Ok(false);
        };
        let execution = self.state.nodes[&node_id].executions[&exec_id].clone();
        self.state
            .node_mut(&node_id)?
            .execution_mut(exec_id)?
            .status = node::ExecutionStatus::Running;
        let previous_idx = execution
            .attempts
            .last()
            .map(|attempt| attempt.idx)
            .ok_or_else(|| Error::new("INVALID_STATE", "waiting execution has no attempt"))?;
        let retry = self.definition().nodes[&node_id].retry.clone();
        let target = self.target_for(&node_id, exec_id)?;
        let mut input = self.input_for(&node_id, &execution.from)?;
        let mut nested_reply = None;
        if let Some(mut reply) = recovery_reply {
            let target = reply.path.remove(0);
            if target.node_id != node_id || target.exec_id != exec_id {
                return Err(Error::new("INVALID_STATE", "saved reply target changed"));
            }
            match self.target_for(&node_id, exec_id)? {
                Target::Agent(_, _) if reply.path.is_empty() => {
                    let mut agent_input: AgentInput = serde_json::from_value(input)
                        .map_err(|error| Error::new("INVALID_STATE", error.to_string()))?;
                    agent_input.payload = reply.input.payload;
                    agent_input.metadata.extend(reply.input.metadata);
                    input = serde_json::to_value(agent_input)
                        .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))?;
                }
                Target::Graph { .. } if !reply.path.is_empty() => {
                    nested_reply = Some(reply);
                }
                _ => {
                    return Err(Error::new(
                        "INVALID_STATE",
                        "saved reply path does not match its Workflow Node",
                    ));
                }
            }
        }
        let invocation = events::with_source(
            self.state.chat_id,
            &node_id,
            exec_id,
            invoke_with_retry(
                target,
                input,
                cancellation.clone(),
                retry,
                Arc::clone(next_exec_id),
                self.state.chat_id,
                // A running Attempt found in a restored snapshot was never
                // committed as a completed/failed attempt. Normalize it to
                // waiting, then resume that same Attempt rather than treating
                // process recovery as a user-visible retry.
                previous_idx,
                committed_tool_messages(&execution),
                nested_reply,
            ),
        )
        .await;
        let invocation_graph = invocation.graph;
        self.replace_resumed_attempts(&node_id, exec_id, invocation.attempts)?;
        if let Some(graph) = invocation_graph {
            self.attach_graph(&node_id, exec_id, graph)?;
        }
        match invocation.outcome {
            Outcome::Complete(value) => {
                let messages = self.output_messages(&node_id, value)?;
                let to = self.outgoing(&node_id, &messages)?;
                self.state
                    .complete_execution(&node_id, exec_id, messages, to)?;
            }
            Outcome::Wait(wait_id) => self.state.wait_execution(&node_id, exec_id, wait_id)?,
            Outcome::Cancelled => self.state.cancel_execution(&node_id, exec_id)?,
            Outcome::Failed(error) => self.state.fail_execution(&node_id, exec_id, error)?,
        }
        self.state.recompute_usage();
        self.observe_commit(&node_id, exec_id).await?;
        Ok(true)
    }

    async fn observe_commit(&self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        let definition = &self.definition().nodes[node_id];
        if definition.agent || definition.workflow.is_some() {
            // Agent calls already forward their real stream, and Graph Node
            // containers must not duplicate the child's observations.
            return Ok(());
        }
        let execution = &self.state.nodes[node_id].executions[&exec_id];
        let messages = if execution.status == node::ExecutionStatus::Succeeded {
            execution
                .messages
                .as_slice()
                .iter()
                .map(events::project_message)
                .collect::<Vec<_>>()
        } else {
            let mut seen = BTreeSet::new();
            execution
                .attempts
                .iter()
                .flat_map(|attempt| attempt.messages.as_slice())
                .filter(|message| {
                    message
                        .metadata
                        .get("call_id")
                        .and_then(Value::as_str)
                        .is_some_and(|call_id| seen.insert(call_id.to_owned()))
                })
                .map(events::project_message)
                .collect::<Vec<_>>()
        };
        for message in messages {
            let mut event = crate::event::Event::Complete {
                message,
                usage: None,
                finish_reason: None,
            };
            crate::runtime::observation::associate(
                &mut event,
                Some(self.state.chat_id),
                Some((node_id, exec_id)),
            );
            // Before presenting a nested node's result, its owning Graph
            // Attempt must receive the already-committed native scene too.
            // This updates memory only, not another Checkpoint write.
            self.publish(Some(event), false).await?;
        }
        Ok(())
    }

    fn target_for(&self, node_id: &str, exec_id: i64) -> Result<Target, Error> {
        if let Some(workflow) = self.workflow.graph_node(node_id) {
            let execution = self.state.nodes[node_id].executions.get(&exec_id);
            let state = execution
                .and_then(|execution| execution.attempts.last())
                .and_then(|attempt| attempt.graph.clone());
            let history = execution
                .and_then(|execution| {
                    execution
                        .attempts
                        .iter()
                        .rev()
                        .skip(1)
                        .find_map(|attempt| attempt.graph.clone())
                })
                .or_else(|| self.inherited_graph(node_id));
            return Ok(Target::Graph {
                workflow: Arc::new(workflow.clone()),
                state,
                history,
            });
        }
        if let Some(agent) = self.workflow.agent(node_id) {
            let restore = self.state.nodes[node_id]
                .executions
                .get(&exec_id)
                .and_then(|execution| execution.attempts.last())
                .and_then(|attempt| attempt.agent.clone());
            return Ok(Target::Agent(agent, restore));
        }
        Ok(Target::Node(self.workflow.executor(node_id).ok_or_else(
            || Error::new("UNSUPPORTED", "workflow node has no executor"),
        )?))
    }

    fn inherited_graph(&self, node_id: &str) -> Option<Box<graph::State>> {
        self.retry_source
            .as_ref()?
            .nodes
            .get(node_id)?
            .executions
            .values()
            .max_by_key(|execution| (execution.idx, execution.exec_id))?
            .attempts
            .last()?
            .graph
            .clone()
    }

    fn replace_attempts(
        &mut self,
        node_id: &str,
        exec_id: i64,
        attempts: Vec<node::Attempt>,
    ) -> Result<(), Error> {
        let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
        if attempts.is_empty() || attempts[0].idx != 0 {
            return Err(Error::new("INVALID_STATE", "invalid attempt history"));
        }
        execution.attempts = attempts;
        execution.usage = execution.attempts.iter().fold(None, |total, attempt| {
            crate::components::model::token::merge(total, attempt.usage.clone())
        });
        Ok(())
    }

    fn replace_resumed_attempts(
        &mut self,
        node_id: &str,
        exec_id: i64,
        mut attempts: Vec<node::Attempt>,
    ) -> Result<(), Error> {
        let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
        let previous = execution
            .attempts
            .last()
            .map(|attempt| attempt.idx)
            .ok_or_else(|| Error::new("INVALID_STATE", "execution has no attempt"))?;
        if attempts.is_empty() || attempts[0].idx != previous {
            return Err(Error::new(
                "INVALID_STATE",
                "invalid resumed attempt history",
            ));
        }
        let original = execution.attempts.last().expect("checked current Attempt");
        attempts[0].created_time = original.created_time;
        if original.agent.is_some() {
            let cumulative = attempts[0]
                .agent
                .as_ref()
                .and_then(|scene| scene.get("usage"))
                .is_some_and(|usage| {
                    serde_json::from_value::<Option<crate::model::token::Usage>>(usage.clone())
                        .is_ok()
                });
            if !cumulative {
                // Early validation/preparation errors may produce no new
                // scene. Custom Agents can also report only segment usage.
                // Neither case erases consumption from the saved Attempt.
                attempts[0].usage = crate::components::model::token::merge(
                    original.usage.clone(),
                    attempts[0].usage.take(),
                );
            }
            if attempts[0].agent.is_none() {
                attempts[0].agent = original.agent.clone();
            }
            // The Agent scene is canonical for recovery. Keep already
            // observed process facts too, rather than replacing the current
            // Workflow Attempt with only the resumed stream segment.
            let mut messages = original.messages.clone();
            messages.0.append(&mut attempts[0].messages.0);
            attempts[0].messages = messages;
            let mut steps = original.steps.clone();
            for (index, step) in attempts[0].steps.iter_mut().enumerate() {
                step.id = format!("complete_{}", original.steps.len() + index + 1);
            }
            steps.append(&mut attempts[0].steps);
            attempts[0].steps = steps;
        }
        execution.attempts.pop();
        execution.attempts.extend(attempts);
        execution.usage = execution.attempts.iter().fold(None, |total, attempt| {
            crate::components::model::token::merge(total, attempt.usage.clone())
        });
        Ok(())
    }

    fn attach_graph(
        &mut self,
        node_id: &str,
        exec_id: i64,
        graph: Box<graph::State>,
    ) -> Result<(), Error> {
        let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
        let attempt = execution
            .attempts
            .last_mut()
            .ok_or_else(|| Error::new("INVALID_STATE", "execution has no attempt"))?;
        attempt.graph = Some(graph);
        Ok(())
    }
}

#[derive(Debug)]
struct Invocation {
    attempts: Vec<node::Attempt>,
    outcome: Outcome,
    graph: Option<Box<graph::State>>,
}

fn attempt_waiting(attempt: &node::Attempt) -> bool {
    attempt
        .agent
        .as_ref()
        .and_then(|scene| scene["waits"].as_array())
        .is_some_and(|waits| !waits.is_empty())
        || attempt.graph.as_ref().is_some_and(|state| {
            !state.waits.is_empty()
                || state
                    .nodes
                    .values()
                    .flat_map(|node| node.executions.values())
                    .any(|execution| execution.attempts.last().is_some_and(attempt_waiting))
        })
}

fn agent_scene_for_reply_path<'a>(
    state: &'a graph::State,
    path: &[graph::ReplyExecution],
) -> Result<&'a Value, Error> {
    let Some((first, rest)) = path.split_first() else {
        return Err(Error::new("INVALID_STATE", "saved reply path is empty"));
    };
    let execution = state
        .nodes
        .get(&first.node_id)
        .and_then(|node| node.executions.get(&first.exec_id))
        .filter(|execution| execution.idx == state.step)
        .ok_or_else(|| Error::new("INVALID_STATE", "saved reply execution is stale"))?;
    let attempt = execution
        .attempts
        .last()
        .filter(|attempt| attempt.idx == first.idx)
        .ok_or_else(|| Error::new("INVALID_STATE", "saved reply Attempt is stale"))?;
    if rest.is_empty() {
        return attempt
            .agent
            .as_ref()
            .ok_or_else(|| Error::new("INVALID_STATE", "saved reply has no Agent scene"));
    }
    agent_scene_for_reply_path(
        attempt
            .graph
            .as_deref()
            .ok_or_else(|| Error::new("INVALID_STATE", "saved reply Graph scene is missing"))?,
        rest,
    )
}

fn attempt_question(attempt: &node::Attempt, id: i64) -> Result<bool, Error> {
    if let Some(scene) = &attempt.agent
        && scene["waits"]
            .as_array()
            .is_some_and(|waits| !waits.is_empty())
        && crate::runtime::agent::wait::question(scene, id)?
    {
        return Ok(true);
    }
    if let Some(state) = &attempt.graph {
        for execution in state
            .nodes
            .values()
            .flat_map(|node| node.executions.values())
        {
            if matches!(
                execution.status,
                node::ExecutionStatus::Running | node::ExecutionStatus::Waiting
            ) && let Some(attempt) = execution.attempts.last()
                && attempt_question(attempt, id)?
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[derive(Debug)]
enum Outcome {
    Complete(Value),
    Wait(String),
    Failed(Error),
    Cancelled,
}

enum Target {
    Node(Arc<dyn graph::ErasedNode + 'static>),
    Agent(Arc<dyn graph::ErasedAgent + 'static>, Option<Value>),
    Graph {
        workflow: Arc<CompiledWorkflow>,
        state: Option<Box<graph::State>>,
        history: Option<Box<graph::State>>,
    },
}

// Reconstruct successful sub-call facts from the entire logical execution,
// not just its latest Attempt: an in-flight retry may have no messages yet.
// This is private ToolNode recovery input, never a partial normal Node output.
fn committed_tool_messages(execution: &node::Execution) -> Messages {
    let mut committed = std::collections::BTreeMap::new();
    for message in execution
        .attempts
        .iter()
        .flat_map(|attempt| attempt.messages.as_slice())
    {
        if let Some(call_id) = message.metadata.get("call_id").and_then(Value::as_str) {
            committed.insert(call_id, message);
        }
    }
    Messages::new(committed.into_values().cloned())
}

// Replay only actual saved cancellation Completes, never synthesize one from
// Delta or turn the cancelled execution into a successful downstream result.
fn terminal_observations(state: &graph::State) -> Vec<crate::event::Event> {
    let mut events = Vec::new();
    for (node_id, node) in &state.nodes {
        for execution in node.executions.values() {
            if execution.status != node::ExecutionStatus::Succeeded
                || !execution.to.iter().any(|to| to == graph::END)
            {
                continue;
            }
            if let Some(child) = execution
                .attempts
                .last()
                .and_then(|attempt| attempt.graph.as_ref())
            {
                events.extend(terminal_observations(child));
            } else {
                for message in execution.messages.as_slice() {
                    let mut event = crate::event::Event::Complete {
                        message: events::project_message(message),
                        usage: message.usage.clone(),
                        finish_reason: None,
                    };
                    crate::runtime::observation::associate(
                        &mut event,
                        Some(state.chat_id),
                        Some((node_id, execution.exec_id)),
                    );
                    events.push(event);
                }
            }
        }
    }
    events
}

fn cancelled_observations(state: &graph::State) -> Vec<crate::event::Event> {
    let mut events = Vec::new();
    for (node_id, node) in &state.nodes {
        for execution in node.executions.values() {
            for attempt in &execution.attempts {
                if let Some(child) = &attempt.graph {
                    events.extend(cancelled_observations(child));
                }
                for (step, message) in attempt.steps.iter().zip(attempt.messages.as_slice()) {
                    if let Some(call) = step.attempts.last()
                        && call.finish_reason.as_deref() == Some("cancelled")
                        && message.role == Role::Assistant
                    {
                        let mut event = crate::event::Event::Complete {
                            message: events::project_message(message),
                            usage: call.usage.clone(),
                            finish_reason: call.finish_reason.clone(),
                        };
                        crate::runtime::observation::associate(
                            &mut event,
                            Some(state.chat_id),
                            Some((node_id, execution.exec_id)),
                        );
                        events.push(event);
                    }
                }
            }
        }
    }
    events
}

// The final argument is the already committed ToolNode context restored from
// Checkpoint; keeping it alongside the retry inputs makes recovery explicit.
#[allow(clippy::too_many_arguments)]
fn invoke_with_retry(
    target: Target,
    input: Value,
    cancellation: crate::Cancellation,
    retry: graph::RetryPolicy,
    next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
    chat_id: i64,
    attempt_start: u32,
    committed: Messages,
    recovery_reply: Option<RecoveryReply>,
) -> Pin<Box<dyn Future<Output = Invocation> + 'static>> {
    Box::pin(async move {
        let mut attempts = Vec::new();
        // ToolNode-only committed call facts. They remain private to this one
        // logical execution and are restored from all saved Attempts on recovery;
        // ordinary Nodes keep the unchanged single-call path.
        let mut committed_tool_messages = committed;
        let mut graph_retry_state = match &target {
            Target::Graph { state, .. } => state.clone(),
            _ => None,
        };
        let mut graph_retry_history = match &target {
            Target::Graph { history, .. } => history.clone(),
            _ => None,
        };
        let retry_policy = retry.clone();
        let mut recovery_reply = recovery_reply;
        // A resumed wait continues the same Attempt. Its existing idx consumes
        // the original execution's retry budget; recovery never resets it.
        for offset in 0..=retry.max_retries.saturating_sub(attempt_start) {
            let idx = attempt_start.saturating_add(offset);
            let mut attempt = node::Attempt {
                idx,
                status: node::AttemptStatus::Running,
                ..Default::default()
            };
            if cancellation.is_cancelled() {
                attempt.status = node::AttemptStatus::Cancelled;
                attempts.push(attempt);
                return Invocation {
                    attempts,
                    outcome: Outcome::Cancelled,
                    graph: None,
                };
            }
            let call_result = match &target {
                Target::Node(executor) => {
                    if let Some(future) = executor.call_with_recovery(
                        input.clone(),
                        cancellation.clone(),
                        committed_tool_messages.clone(),
                    ) {
                        match future.await {
                            Ok(Signal::Complete(value)) => {
                                if let Ok(messages) =
                                    serde_json::from_value::<Messages>(value.clone())
                                {
                                    attempt.messages = messages;
                                }
                                Ok(Signal::Complete(value))
                            }
                            Ok(Signal::Wait(wait)) => Ok(Signal::Wait(wait)),
                            Err(failure) => {
                                committed_tool_messages = failure.completed.clone();
                                attempt.messages = failure.completed;
                                Err(failure.error)
                            }
                        }
                    } else {
                        executor.call(input.clone(), cancellation.clone()).await
                    }
                }
                Target::Agent(executor, restore) => {
                    let mut agent_input: AgentInput = match serde_json::from_value(input.clone()) {
                        Ok(input) => input,
                        Err(error) => {
                            attempt.status = node::AttemptStatus::Failed;
                            let error = Error::new("INVALID_ARGUMENTS", error.to_string());
                            attempt.error = Some(error.clone());
                            attempts.push(attempt);
                            return Invocation {
                                attempts,
                                outcome: Outcome::Failed(error),
                                graph: None,
                            };
                        }
                    };
                    if agent_input
                        .metadata
                        .get("chat_id")
                        .is_some_and(|id| id.as_i64() != Some(chat_id))
                    {
                        let error = Error::new(
                            "CONFLICT",
                            "Agent input chat_id does not match its Workflow",
                        );
                        attempt.status = node::AttemptStatus::Failed;
                        attempt.error = Some(error.clone());
                        attempts.push(attempt);
                        return Invocation {
                            attempts,
                            outcome: Outcome::Failed(error),
                            graph: None,
                        };
                    }
                    agent_input
                        .metadata
                        .insert("chat_id".into(), Value::from(chat_id));
                    if let Some(reply) = recovery_reply.take() {
                        if !reply.path.is_empty() {
                            let error = Error::new(
                                "INVALID_STATE",
                                "saved reply was not routed through its nested Graph",
                            );
                            attempt.status = node::AttemptStatus::Failed;
                            attempt.error = Some(error.clone());
                            attempts.push(attempt);
                            return Invocation {
                                attempts,
                                outcome: Outcome::Failed(error),
                                graph: None,
                            };
                        }
                        agent_input.payload = reply.input.payload;
                        agent_input.metadata.extend(reply.input.metadata);
                    }
                    let run = events::with_attempt(
                        idx,
                        executor.call(agent_input, cancellation.clone(), restore.clone()),
                    )
                    .await;
                    attempt.steps = run.steps;
                    attempt.messages = run.messages;
                    attempt.usage = run.usage;
                    let agent_wait_id = run
                        .scene
                        .as_ref()
                        .and_then(|scene| scene.get("wait_id"))
                        .and_then(Value::as_str)
                        .filter(|wait_id| !wait_id.trim().is_empty())
                        .map(str::to_owned);
                    attempt.agent = run.scene.map(|mut scene| {
                        // An embedded Agent does not own a top-level checkpoint,
                        // so the Workflow supplies the single run identity.
                        scene["chat_id"] = Value::from(chat_id);
                        scene
                    });
                    match run.result {
                        Ok(messages) => serde_json::to_value(messages)
                            .map(Signal::Complete)
                            .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string())),
                        // Agent implementations use the shared Error contract for
                        // an external wait.  Translate it at the Workflow
                        // boundary into the same Signal used by ordinary Nodes;
                        // it must never consume retry budget or become a failure.
                        Err(error) if error.code == "WAITING" => {
                            let wait_id = agent_wait_id
                                .as_deref()
                                .or(error.details.as_deref())
                                .filter(|wait_id| !wait_id.trim().is_empty())
                                .unwrap_or("agent")
                                .to_owned();
                            Ok(Signal::Wait(crate::components::node::Wait { wait_id }))
                        }
                        Err(error) => Err(error),
                    }
                }
                Target::Graph { workflow, .. } => {
                    let nested_reply = recovery_reply.take();
                    match events::with_attempt(
                        idx,
                        run_subgraph(
                            Arc::clone(workflow),
                            input.clone(),
                            cancellation.clone(),
                            Arc::clone(&next_exec_id),
                            chat_id,
                            SubgraphScenes {
                                snapshot: graph_retry_state.clone(),
                                history: graph_retry_history.clone(),
                                reply: nested_reply,
                            },
                        ),
                    )
                    .await
                    {
                        Ok((value, state)) => {
                            attempt.status = node::AttemptStatus::Succeeded;
                            attempt.usage = state.usage.clone();
                            attempts.push(attempt);
                            return Invocation {
                                attempts,
                                outcome: Outcome::Complete(value),
                                graph: Some(Box::new(state)),
                            };
                        }
                        Err(failure) => {
                            if failure.error.code == "WAITING" {
                                let Some(state) = failure.state else {
                                    attempt.status = node::AttemptStatus::Failed;
                                    let error = Error::new(
                                        "INVALID_STATE",
                                        "waiting child workflow did not return its state",
                                    );
                                    attempt.error = Some(error.clone());
                                    attempts.push(attempt);
                                    return Invocation {
                                        attempts,
                                        outcome: Outcome::Failed(error),
                                        graph: None,
                                    };
                                };
                                let Some(wait_id) = (state.waits.len() == 1).then(|| {
                                    state
                                        .waits
                                        .values()
                                        .next()
                                        .expect("one wait")
                                        .wait_id
                                        .clone()
                                }) else {
                                    attempt.status = node::AttemptStatus::Failed;
                                    let error = Error::new(
                                        "CONFLICT",
                                        "waiting child workflow must have exactly one active wait",
                                    );
                                    attempt.error = Some(error.clone());
                                    attempts.push(attempt);
                                    return Invocation {
                                        attempts,
                                        outcome: Outcome::Failed(error),
                                        graph: None,
                                    };
                                };
                                attempt.status = node::AttemptStatus::Waiting;
                                attempt.usage = state.usage.clone();
                                attempt.graph = Some(Box::new(state));
                                attempts.push(attempt);
                                return Invocation {
                                    attempts,
                                    outcome: Outcome::Wait(wait_id),
                                    graph: None,
                                };
                            }
                            if let Some(state) = failure.state {
                                if retry_policy.should_retry(idx, &failure.error) {
                                    // Only a real outer retry inherits successes.
                                    // Checkpoint continuation still uses the exact
                                    // original current-Attempt scene above.
                                    graph_retry_state =
                                        Some(Box::new(retry_graph_state(workflow, &state)));
                                    graph_retry_history = Some(Box::new(state.clone()));
                                }
                                attempt.usage = state.usage.clone();
                                attempt.graph = Some(Box::new(state));
                            }
                            Err(failure.error)
                        }
                    }
                }
            };
            match call_result {
                Ok(crate::node::Signal::Complete(value)) => {
                    attempt.status = node::AttemptStatus::Succeeded;
                    attempts.push(attempt);
                    return Invocation {
                        attempts,
                        outcome: Outcome::Complete(value),
                        graph: None,
                    };
                }
                Ok(crate::node::Signal::Wait(wait)) => {
                    attempt.status = node::AttemptStatus::Waiting;
                    attempts.push(attempt);
                    return Invocation {
                        attempts,
                        outcome: Outcome::Wait(wait.wait_id),
                        graph: None,
                    };
                }
                Err(error) => {
                    if error.code == "CANCELLED" {
                        attempt.status = node::AttemptStatus::Cancelled;
                        attempts.push(attempt);
                        return Invocation {
                            attempts,
                            outcome: Outcome::Cancelled,
                            graph: None,
                        };
                    }
                    attempt.status = node::AttemptStatus::Failed;
                    attempt.error = Some(error.clone());
                    if events::has_parent()
                        && let Err(owner_error) = events::progress(attempt.clone(), None).await
                    {
                        attempts.push(attempt);
                        return Invocation {
                            attempts,
                            outcome: Outcome::Failed(owner_error),
                            graph: None,
                        };
                    }
                    attempts.push(attempt);
                    if !retry_policy.should_retry(idx, &error) {
                        return Invocation {
                            attempts,
                            outcome: Outcome::Failed(error),
                            graph: None,
                        };
                    }
                    if let Err(error) =
                        crate::core::middleware::wait(&retry_policy, idx, &cancellation).await
                    {
                        // Waiting does not constitute a new actual Attempt. Keep
                        // the original failed Attempt and report run cancellation
                        // or a timer setup error separately.
                        return Invocation {
                            attempts,
                            outcome: if error.code == "CANCELLED" {
                                Outcome::Cancelled
                            } else {
                                Outcome::Failed(error)
                            },
                            graph: None,
                        };
                    }
                }
            }
        }
        unreachable!()
    })
}

fn messages_to_agent_input(messages: &Messages) -> AgentInput {
    let message = messages
        .as_slice()
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .or_else(|| messages.as_slice().last());
    let Some(message) = message else {
        return AgentInput::default();
    };
    let payload = message
        .content
        .iter()
        .map(|part| AgentPayload {
            r#type: part.r#type.clone(),
            data: serde_json::Value::Object(part.data.clone()),
            mime_type: part
                .data
                .get("mime_type")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
        .collect();
    AgentInput {
        metadata: message.metadata.clone(),
        payload,
    }
}

struct SubgraphFailure {
    error: Error,
    state: Option<graph::State>,
}

// Derive the next outer Attempt's child scene from existing graph facts. The
// failed Attempt keeps its full unmodified scene; only complete, unaffected
// successful child executions are inherited into this new scene.
fn retry_graph_state(workflow: &CompiledWorkflow, previous: &graph::State) -> graph::State {
    let definition = workflow.definition();
    let mut affected = previous
        .nodes
        .iter()
        .filter(|(_, state)| {
            state
                .executions
                .values()
                .any(|execution| execution.status == node::ExecutionStatus::Failed)
        })
        .map(|(node_id, _)| node_id.clone())
        .collect::<BTreeSet<_>>();
    let mut pending = affected.iter().cloned().collect::<Vec<_>>();
    while let Some(node_id) = pending.pop() {
        let mut downstream = outgoing(definition, &node_id);
        downstream.extend(
            definition
                .graph
                .conditional_edges
                .iter()
                .filter(|relation| relation.from == node_id)
                .flat_map(|relation| relation.to.iter().cloned()),
        );
        downstream.extend(
            definition
                .graph
                .loop_definitions
                .iter()
                .filter(|relation| relation.from == node_id)
                .map(|relation| relation.to.clone()),
        );
        downstream.extend(
            definition
                .graph
                .failure_relations
                .iter()
                .filter(|relation| relation.from.contains(&node_id))
                .flat_map(|relation| relation.to.iter().cloned()),
        );
        for to in downstream {
            if definition.nodes.contains_key(&to) && affected.insert(to.clone()) {
                pending.push(to);
            }
        }
    }
    let mut state = graph::State::new(previous.chat_id, definition);
    state.initial = previous.initial.clone();
    state.step = previous
        .nodes
        .iter()
        .filter(|(node_id, _)| affected.contains(*node_id))
        .flat_map(|(_, node)| node.executions.values().map(|execution| execution.idx))
        .min()
        .unwrap_or(previous.step);
    for (node_id, node) in &previous.nodes {
        if affected.contains(node_id) {
            continue;
        }
        let executions = node
            .executions
            .iter()
            .filter(|(_, execution)| execution.status == node::ExecutionStatus::Succeeded)
            .map(|(exec_id, execution)| (*exec_id, execution.clone()))
            .collect::<std::collections::BTreeMap<_, _>>();
        if !executions.is_empty() {
            state.nodes.insert(
                node_id.clone(),
                node::State {
                    status: node::Status::Succeeded,
                    executions,
                },
            );
        }
    }
    state.recompute_usage();
    state
}

type SubgraphFuture = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<(Value, graph::State), SubgraphFailure>> + 'static>,
>;

struct SubgraphScenes {
    snapshot: Option<Box<graph::State>>,
    history: Option<Box<graph::State>>,
    reply: Option<RecoveryReply>,
}

fn run_subgraph(
    workflow: Arc<CompiledWorkflow>,
    input: Value,
    cancellation: crate::Cancellation,
    next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
    chat_id: i64,
    scenes: SubgraphScenes,
) -> SubgraphFuture {
    Box::pin(async move {
        let SubgraphScenes {
            snapshot,
            history,
            reply,
        } = scenes;
        let messages: Messages =
            serde_json::from_value(input).map_err(|error| SubgraphFailure {
                error: Error::new("INVALID_ARGUMENTS", error.to_string()),
                state: None,
            })?;
        // A nested Graph Node can have a new execution while still reusing
        // its unchanged-input internal successes from the parent's history.
        let stable_input = snapshot
            .as_ref()
            .and_then(|scene| scene.initial.as_ref())
            .unwrap_or(&messages);
        let history = history.filter(|previous| {
            previous.chat_id == chat_id
                && previous.definition_id == workflow.definition().definition_id
                && previous.version == workflow.definition().version
                && previous.initial.as_ref() == Some(stable_input)
        });
        let snapshot = snapshot.or_else(|| {
            history
                .as_ref()
                .map(|previous| Box::new(retry_graph_state(&workflow, previous)))
        });
        let mut runtime = match snapshot {
            Some(snapshot) => Engine::from_snapshot(
                chat_id,
                (*workflow).clone(),
                serde_json::to_value(*snapshot).map_err(|error| SubgraphFailure {
                    error: Error::new("SERIALIZATION_ERROR", error.to_string()),
                    state: None,
                })?,
            )
            .map_err(|error| SubgraphFailure { error, state: None })?,
            None => Engine::new(chat_id, (*workflow).clone())
                .map_err(|error| SubgraphFailure { error, state: None })?,
        };
        runtime.retry_source = history;
        runtime.recovery_reply = reply;
        let run_input = runtime.state.initial.clone().unwrap_or(messages);
        let output = runtime
            .execute(run_input, next_exec_id, cancellation)
            .await
            .map_err(|error| SubgraphFailure {
                error,
                state: Some(runtime.state.clone()),
            })?;
        let state = runtime.state.clone();
        let value = serde_json::to_value(output).map_err(|error| SubgraphFailure {
            error: Error::new("SERIALIZATION_ERROR", error.to_string()),
            state: Some(state.clone()),
        })?;
        Ok((value, state))
    })
}

fn outgoing(definition: &Definition, node_id: &str) -> Vec<String> {
    let mut result: Vec<String> = definition
        .graph
        .ordinary_edges
        .iter()
        .filter(|edge| edge.from.node() == Some(node_id))
        .map(|edge| match &edge.to {
            Endpoint::Node(id) => id.clone(),
            Endpoint::End => graph::END.into(),
            Endpoint::Start => graph::START.into(),
        })
        .collect();
    result.extend(
        definition
            .graph
            .fan_out_definitions
            .iter()
            .filter(|relation| relation.from == node_id)
            .map(|relation| relation.to.clone()),
    );
    result.extend(
        definition
            .graph
            .join_definitions
            .iter()
            .filter(|relation| relation.from.iter().any(|source| source == node_id))
            .map(|relation| relation.to.clone()),
    );
    result.sort();
    result.dedup();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cancellation;
    use crate::agent::{Agent, Input as AgentInput};
    use crate::checkpoint::Checkpoint;
    use crate::components::node::{Node, Signal, Wait as NodeWait};
    use crate::runtime::workflow::graph::{END, START};
    use futures_executor::block_on;
    use futures_util::stream;
    use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

    struct WaitOnce(AtomicBool);

    impl Node<Messages, Messages, crate::Error> for WaitOnce {
        async fn call(
            &self,
            input: Messages,
            _cancellation: Cancellation,
        ) -> Result<Signal<Messages>, crate::Error> {
            if !self.0.swap(true, Ordering::AcqRel) {
                Ok(Signal::Wait(NodeWait {
                    wait_id: "approval".into(),
                }))
            } else {
                Ok(Signal::Complete(input))
            }
        }
    }

    struct Flaky(AtomicBool);

    impl Node<Messages, Messages, crate::Error> for Flaky {
        async fn call(
            &self,
            input: Messages,
            _cancellation: Cancellation,
        ) -> Result<Signal<Messages>, crate::Error> {
            if !self.0.swap(true, Ordering::AcqRel) {
                Err(crate::Error::new("TIMEOUT", "temporary"))
            } else {
                Ok(Signal::Complete(input))
            }
        }
    }

    struct RetryTool {
        definition: crate::components::tool::ToolDefinition,
        calls: Arc<AtomicI64>,
        fail_first: bool,
    }

    impl crate::components::tool::Tool for RetryTool {
        fn definition(&self) -> &crate::components::tool::ToolDefinition {
            &self.definition
        }

        async fn call(
            &self,
            call: &crate::components::tool::Call,
            _cancellation: &Cancellation,
        ) -> Result<crate::Message, Error> {
            let count = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail_first && count == 0 {
                return Err(Error::new("TIMEOUT", "temporary tool failure"));
            }
            Ok(crate::Message::function(serde_json::json!({
                "tool": call.name,
                "ok": true,
            })))
        }
    }

    fn workflow_tool_definition(name: &str) -> crate::components::tool::ToolDefinition {
        crate::components::tool::ToolDefinition {
            name: name.into(),
            description: format!("{name} test tool"),
            parameters: serde_json::from_value(serde_json::json!({
                "type": "object",
                "properties": {}
            }))
            .unwrap(),
        }
    }

    fn workflow_tool_input() -> Messages {
        let mut assistant = crate::Message::new(Role::Assistant);
        assistant.content = vec![
            crate::model::content::part::Part {
                r#type: "tool_call".into(),
                data: crate::components::model::content::part::object(serde_json::json!({
                    "call_id": "call_a",
                    "name": "first",
                    "arguments": {}
                })),
            },
            crate::model::content::part::Part {
                r#type: "tool_call".into(),
                data: crate::components::model::content::part::object(serde_json::json!({
                    "call_id": "call_b",
                    "name": "second",
                    "arguments": {}
                })),
            },
        ];
        Messages::new([assistant])
    }

    #[derive(Clone, Copy)]
    struct FakeAgent {
        mode: &'static str,
    }

    struct RestoringAgent;

    struct WaitingAgent;

    impl Agent for WaitingAgent {
        async fn reply(
            &self,
            _: crate::agent::Input,
            _: crate::Cancellation,
        ) -> Result<(), crate::Error> {
            Err(crate::Error::new(
                "NOT_FOUND",
                "fixture has no live feedback consumer",
            ))
        }
        type Stream = stream::Iter<std::vec::IntoIter<Result<crate::event::Event, Error>>>;

        async fn run(
            &self,
            _input: AgentInput,
            _stream: bool,
            _cancellation: Cancellation,
        ) -> Result<Self::Stream, Error> {
            if crate::runtime::agent::scope::restore().is_none() {
                crate::runtime::agent::scope::capture(serde_json::json!({
                    "type": "agent",
                    "status": "waiting",
                    "wait_id": "approval"
                }));
                return Err(Error::new("WAITING", "agent is waiting for approval"));
            }
            let mut message = crate::Message::new(Role::Assistant);
            message.content.push(crate::model::content::part::Part {
                r#type: "text".into(),
                data: crate::components::model::content::part::object(
                    serde_json::json!({"value": "approved"}),
                ),
            });
            Ok(stream::iter(vec![Ok(crate::event::Event::Complete {
                message,
                usage: None,
                finish_reason: Some("stop".into()),
            })]))
        }
    }

    impl Agent for RestoringAgent {
        async fn reply(
            &self,
            _: crate::agent::Input,
            _: crate::Cancellation,
        ) -> Result<(), crate::Error> {
            Err(crate::Error::new(
                "NOT_FOUND",
                "fixture has no live feedback consumer",
            ))
        }
        type Stream = stream::Iter<std::vec::IntoIter<Result<crate::event::Event, Error>>>;

        async fn run(
            &self,
            _input: AgentInput,
            _stream: bool,
            _cancellation: Cancellation,
        ) -> Result<Self::Stream, Error> {
            let value = if crate::runtime::agent::scope::restore().is_some() {
                "restored_scene"
            } else {
                "fresh"
            };
            let mut message = crate::Message::new(Role::Assistant);
            message.content.push(crate::model::content::part::Part {
                r#type: "text".into(),
                data: crate::components::model::content::part::object(
                    serde_json::json!({"value": value}),
                ),
            });
            Ok(stream::iter(vec![Ok(crate::event::Event::Complete {
                message,
                usage: None,
                finish_reason: Some("stop".into()),
            })]))
        }
    }

    struct MeteredAgent {
        calls: Arc<AtomicI64>,
        fail_once: bool,
    }

    impl Agent for MeteredAgent {
        async fn reply(
            &self,
            _: crate::agent::Input,
            _: crate::Cancellation,
        ) -> Result<(), crate::Error> {
            Err(crate::Error::new(
                "NOT_FOUND",
                "fixture has no live feedback consumer",
            ))
        }
        type Stream = stream::Iter<std::vec::IntoIter<Result<crate::event::Event, Error>>>;

        async fn run(
            &self,
            _input: AgentInput,
            _stream: bool,
            _cancellation: Cancellation,
        ) -> Result<Self::Stream, Error> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let usage = crate::model::token::Usage {
                input: Some(10),
                output: Some(4),
                total: Some(14),
                input_details: Some(crate::model::token::InputDetails { cached: Some(3) }),
                output_details: Some(crate::model::token::OutputDetails { think: Some(2) }),
                details: serde_json::Map::from_iter([("raw".into(), serde_json::json!(call))]),
            };
            let mut message = crate::Message::new(Role::Assistant);
            message.content.push(crate::model::content::part::Part {
                r#type: "text".into(),
                data: crate::components::model::content::part::object(
                    serde_json::json!({"value":"answer"}),
                ),
            });
            message.usage = Some(usage.clone());
            let mut events = vec![Ok(crate::event::Event::Complete {
                message,
                usage: Some(usage),
                finish_reason: Some("stop".into()),
            })];
            if self.fail_once && call == 0 {
                events.push(Err(Error::new("TIMEOUT", "model stream interrupted")));
            }
            Ok(stream::iter(events))
        }
    }

    fn user_input() -> Messages {
        let mut message = crate::Message::new(Role::User);
        message.content.push(crate::model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                serde_json::json!({"value":"hello"}),
            ),
        });
        Messages::new([message])
    }

    fn workflow_input() -> Messages {
        let mut messages = user_input();
        messages.0[0].metadata = serde_json::Map::from_iter([
            ("tenant_id".into(), serde_json::json!(1)),
            ("user_id".into(), serde_json::json!(2)),
            ("agent_id".into(), serde_json::json!(3)),
            ("session_id".into(), serde_json::json!(4)),
            ("chat_id".into(), serde_json::json!(5)),
        ]);
        messages
    }

    type WorkflowMemoryConfig = crate::memory::Config<
        crate::core::memory::short::session::InMemory,
        crate::core::memory::short::chat::InMemory,
        crate::core::memory::short::trace::InMemory,
        crate::core::memory::InMemory,
    >;

    fn workflow_memory() -> (
        WorkflowMemoryConfig,
        Arc<crate::core::memory::short::chat::InMemory>,
        Arc<crate::core::memory::short::trace::InMemory>,
    ) {
        use crate::memory::short::session::Store as _;
        let config = WorkflowMemoryConfig::new(
            crate::core::memory::short::session::InMemory::new(),
            crate::core::memory::short::chat::InMemory::new(),
            crate::core::memory::short::trace::InMemory::new(),
            crate::core::memory::InMemory::new(),
        );
        block_on(config.session.create(
            crate::memory::short::session::Session {
                id: 4,
                tenant_id: 1,
                user_id: 2,
                r#type: crate::memory::short::Type::Workflow,
                title: None,
                summary: None,
                usage: None,
                status: crate::memory::short::session::Status::Active,
                metadata: serde_json::Map::new(),
                created_time: 0,
                updated_time: 0,
            },
            &Cancellation::new(),
        ))
        .unwrap();
        (config.clone(), config.chat.clone(), config.trace.clone())
    }

    #[test]
    fn workflow_usage_counts_failed_and_successful_agent_attempts_once() {
        let mut builder = WorkflowBuilder::new("metered");
        builder
            .add_agent_node(
                "agent",
                MeteredAgent {
                    calls: Arc::new(AtomicI64::new(0)),
                    fail_once: true,
                },
            )
            .unwrap();
        builder
            .set_retry(
                "agent",
                RetryPolicy {
                    max_retries: 1,
                    min_delay_ms: 0,
                    max_delay_ms: 0,
                },
            )
            .unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(91, workflow.clone()).unwrap();
        block_on(runtime.execute(user_input(), Arc::new(|| 12), Cancellation::new())).unwrap();
        let state = runtime.state();
        let execution = &state.nodes["agent"].executions[&12];
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(28));
        let total = state.usage.as_ref().unwrap();
        assert_eq!(total.total, Some(28));
        assert_eq!(total.input_details.as_ref().unwrap().cached, Some(6));
        assert_eq!(total.output_details.as_ref().unwrap().think, Some(4));
        assert!(total.details.is_empty());
        assert_eq!(
            execution.attempts[0].steps[0].attempts[0]
                .usage
                .as_ref()
                .unwrap()
                .details["raw"],
            0
        );
        let mut snapshot = runtime.snapshot().unwrap();
        snapshot["usage"]["total"] = serde_json::json!(999);
        snapshot["nodes"]["agent"]["executions"]["12"]["usage"]["total"] = serde_json::json!(999);
        let mut restored = Runtime::from_snapshot(91, workflow, snapshot).unwrap();
        assert_eq!(restored.state().usage.as_ref().unwrap().total, Some(28));
        assert_eq!(
            restored.state().nodes["agent"].executions[&12]
                .usage
                .as_ref()
                .unwrap()
                .total,
            Some(28)
        );
    }

    #[test]
    fn nested_workflow_usage_is_included_without_billing_the_parent_twice() {
        let mut child = WorkflowBuilder::new("metered-child");
        child
            .add_agent_node(
                "agent",
                MeteredAgent {
                    calls: Arc::new(AtomicI64::new(0)),
                    fail_once: false,
                },
            )
            .unwrap();
        child.add_edge(START, "agent").unwrap();
        child.add_edge("agent", END).unwrap();
        let child = child.compile().unwrap();
        let mut parent = WorkflowBuilder::new("metered-parent");
        parent.add_graph_node("child", &child).unwrap();
        parent.add_edge(START, "child").unwrap();
        parent.add_edge("child", END).unwrap();
        let mut runtime = Runtime::new(92, parent.compile().unwrap()).unwrap();
        let ids = Arc::new(AtomicI64::new(20));
        block_on(runtime.execute(
            user_input(),
            Arc::new(move || ids.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        ))
        .unwrap();
        let execution = &runtime.state().nodes["child"].executions[&20];
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(14));
        assert_eq!(
            execution.attempts[0]
                .graph
                .as_ref()
                .unwrap()
                .usage
                .as_ref()
                .unwrap()
                .total,
            Some(14)
        );
        assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(14));
    }

    impl Agent for FakeAgent {
        async fn reply(
            &self,
            _: crate::agent::Input,
            _: crate::Cancellation,
        ) -> Result<(), crate::Error> {
            Err(crate::Error::new(
                "NOT_FOUND",
                "fixture has no live feedback consumer",
            ))
        }
        type Stream = stream::Iter<std::vec::IntoIter<Result<crate::event::Event, Error>>>;

        async fn run(
            &self,
            input: AgentInput,
            _stream: bool,
            _cancellation: Cancellation,
        ) -> Result<Self::Stream, Error> {
            assert_eq!(input.payload.len(), 1);
            assert_eq!(input.payload[0].r#type, "text");
            if self.mode == "error" {
                return Err(Error::new("BUSINESS", "agent failed"));
            }
            let make_message = |role| {
                let mut message = crate::Message::new(role);
                message.content.push(crate::model::content::part::Part {
                    r#type: "text".into(),
                    data: crate::components::model::content::part::object(
                        serde_json::json!({"value": "agent result"}),
                    ),
                });
                message
            };
            let first = (self.mode == "many").then(|| crate::event::Event::Complete {
                message: make_message(Role::Tool),
                usage: None,
                finish_reason: Some("tool".into()),
            });
            let finish_reason = if self.mode == "cancel" {
                "cancelled"
            } else {
                "stop"
            };
            let final_event = crate::event::Event::Complete {
                message: make_message(Role::Assistant),
                usage: None,
                finish_reason: Some(finish_reason.into()),
            };
            let events: Vec<Result<crate::event::Event, Error>> =
                first.into_iter().chain([final_event]).map(Ok).collect();
            Ok(stream::iter(events))
        }
    }

    #[test]
    fn runtime_owns_one_state_and_exposes_checkpoint_json() {
        let mut builder = WorkflowBuilder::new("runtime");
        builder.add_node("answer", crate::node::Echo).unwrap();
        builder.add_edge(START, "answer").unwrap();
        builder.add_edge("answer", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(1, workflow.clone()).unwrap();
        assert_eq!(runtime.ready_nodes(), vec!["answer"]);
        runtime
            .start_batch(vec![("answer".into(), 1, Vec::new())])
            .unwrap();
        runtime
            .complete("answer", 1, Messages::default(), vec![END.into()])
            .unwrap();
        assert!(runtime.advance_step());
        let snapshot = runtime.snapshot().unwrap();
        assert_eq!(snapshot["step"], 1);
        assert_eq!(
            snapshot["nodes"]["answer"]["executions"]["1"]["status"],
            "succeeded"
        );
        assert_eq!(snapshot["definition_id"], "runtime");
        assert_eq!(snapshot["version"], runtime.definition().version);
        let mut restored = Runtime::from_snapshot(1, workflow, snapshot).unwrap();
        assert_eq!(restored.state().step, 1);
    }

    #[test]
    fn run_executes_agent_node_and_preserves_agent_steps() {
        let mut builder = WorkflowBuilder::new("agent-node");
        builder
            .add_agent_node("agent", FakeAgent { mode: "single" })
            .unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(31, workflow).unwrap();
        let mut input = crate::Message::new(Role::User);
        input.content.push(crate::model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                serde_json::json!({"value": "hello"}),
            ),
        });
        let ids = Arc::new(|| 3100);
        let result =
            block_on(runtime.execute(Messages::new([input]), ids, Cancellation::new())).unwrap();
        assert_eq!(result.as_slice()[0].role, Role::Assistant);
        let execution = &runtime.state().nodes["agent"].executions[&3100];
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(execution.attempts[0].steps.len(), 1);
        assert_eq!(execution.attempts[0].messages.as_slice().len(), 1);
    }

    #[test]
    fn agent_node_keeps_each_complete_as_a_step_and_commits_last_assistant() {
        let mut builder = WorkflowBuilder::new("agent-many");
        builder
            .add_agent_node("agent", FakeAgent { mode: "many" })
            .unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(32, workflow).unwrap();
        let mut input = crate::Message::new(Role::User);
        input.content.push(crate::model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                serde_json::json!({"value": "hello"}),
            ),
        });
        let result = block_on(runtime.execute(
            Messages::new([input]),
            Arc::new(|| 3200),
            Cancellation::new(),
        ))
        .unwrap();
        assert_eq!(result.as_slice()[0].role, Role::Assistant);
        let attempt = &runtime.state().nodes["agent"].executions[&3200].attempts[0];
        assert_eq!(attempt.steps.len(), 2);
        assert_eq!(attempt.messages.as_slice().len(), 2);
    }

    #[test]
    fn cancelled_agent_does_not_activate_end() {
        let mut builder = WorkflowBuilder::new("agent-cancel");
        builder
            .add_agent_node("agent", FakeAgent { mode: "cancel" })
            .unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(33, workflow).unwrap();
        let mut input = crate::Message::new(Role::User);
        input.content.push(crate::model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                serde_json::json!({"value": "hello"}),
            ),
        });
        let _ = block_on(runtime.execute(
            Messages::new([input]),
            Arc::new(|| 3300),
            Cancellation::new(),
        ));
        assert_eq!(
            runtime.state().nodes["agent"].executions[&3300].status,
            node::ExecutionStatus::Cancelled
        );
        assert_ne!(runtime.state().status, Status::Succeeded);
    }

    #[test]
    fn failed_agent_is_recorded_without_successful_outgoing_relations() {
        let mut builder = WorkflowBuilder::new("agent-failure");
        builder
            .add_agent_node("agent", FakeAgent { mode: "error" })
            .unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(34, workflow).unwrap();
        let mut input = crate::Message::new(Role::User);
        input.content.push(crate::model::content::part::Part {
            r#type: "text".into(),
            data: crate::components::model::content::part::object(
                serde_json::json!({"value": "hello"}),
            ),
        });
        let _ = block_on(runtime.execute(
            Messages::new([input]),
            Arc::new(|| 3400),
            Cancellation::new(),
        ));
        let execution = &runtime.state().nodes["agent"].executions[&3400];
        assert_eq!(execution.status, node::ExecutionStatus::Failed);
        assert!(execution.to.is_empty());
        assert_eq!(
            execution.error.as_ref().map(|error| error.code.as_str()),
            Some("BUSINESS")
        );
    }

    #[test]
    fn run_executes_a_dag_and_commits_each_barrier() {
        let mut builder = WorkflowBuilder::new("run");
        builder.add_node("a", crate::node::Echo).unwrap();
        builder.add_node("b", crate::node::Echo).unwrap();
        builder.add_edge(START, "a").unwrap();
        builder.add_edge("a", "b").unwrap();
        builder.add_edge("b", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(11, workflow).unwrap();
        let initial = Messages::new([crate::Message::function(serde_json::json!({
            "query": "hello"
        }))]);
        let next = Arc::new(std::sync::atomic::AtomicI64::new(100));
        let ids = {
            let next = Arc::clone(&next);
            Arc::new(move || next.fetch_add(1, std::sync::atomic::Ordering::AcqRel))
        };
        let result = block_on(runtime.execute(initial.clone(), ids, Cancellation::new())).unwrap();
        assert_eq!(result, initial);
        assert_eq!(runtime.state().step, 2);
        assert_eq!(runtime.state().status, Status::Succeeded);
        assert_eq!(
            runtime.state().nodes["a"].executions[&100].attempts.len(),
            1
        );
        assert_eq!(
            runtime.state().nodes["b"].executions[&101].attempts.len(),
            1
        );
    }

    #[test]
    fn run_persists_a_conditional_route_in_execution_to() {
        let mut builder = WorkflowBuilder::new("conditional-run");
        for id in ["router", "left", "right"] {
            builder.add_node(id, crate::node::Echo).unwrap();
        }
        builder.add_edge(START, "router").unwrap();
        builder
            .add_conditional_edges(
                "router",
                |messages| {
                    let route = messages
                        .as_slice()
                        .last()
                        .and_then(|message| message.values.get("route"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("left");
                    Ok(vec![route.to_owned()])
                },
                ["left", "right"],
            )
            .unwrap();
        builder.add_edge("left", END).unwrap();
        builder.add_edge("right", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(14, workflow).unwrap();
        let initial = Messages::new([crate::Message::function(serde_json::json!({
            "route": "right"
        }))]);
        let next = Arc::new(std::sync::atomic::AtomicI64::new(400));
        let ids = {
            let next = Arc::clone(&next);
            Arc::new(move || next.fetch_add(1, std::sync::atomic::Ordering::AcqRel))
        };
        block_on(runtime.execute(initial, ids, Cancellation::new())).unwrap();
        assert_eq!(
            runtime.state().nodes["router"].executions[&400].to,
            vec!["right"]
        );
        assert!(runtime.state().nodes["left"].executions.is_empty());
        assert_eq!(runtime.state().status, Status::Succeeded);
    }

    #[test]
    fn runtime_can_save_and_restore_its_single_checkpoint() {
        let mut builder = WorkflowBuilder::new("checkpoint-run");
        builder.add_node("answer", crate::node::Echo).unwrap();
        builder.add_edge(START, "answer").unwrap();
        builder.add_edge("answer", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(16, workflow.clone()).unwrap();
        let ids = Arc::new(|| 600);
        block_on(runtime.execute(Messages::default(), ids, Cancellation::new())).unwrap();
        let checkpoint = crate::core::checkpoint::InMemory::new();
        block_on(runtime.save_checkpoint(&checkpoint, &Cancellation::new())).unwrap();
        let snapshot = block_on(checkpoint.load(16, &Cancellation::new())).unwrap();
        let mut restored = Runtime::from_snapshot(16, workflow, snapshot).unwrap();
        assert_eq!(restored.state().status, Status::Succeeded);
    }

    #[test]
    fn configured_checkpoint_saves_one_closeout_snapshot_after_success() {
        let mut builder = WorkflowBuilder::new("checkpoint-closeout");
        builder.add_node("answer", crate::node::Echo).unwrap();
        builder.add_edge(START, "answer").unwrap();
        builder.add_edge("answer", END).unwrap();
        let workflow = builder.compile().unwrap();
        let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
        let mut runtime = Runtime::new(160, workflow.clone())
            .unwrap()
            .with_checkpoint(checkpoint.clone());

        block_on(runtime.execute(Messages::default(), Arc::new(|| 6_000), Cancellation::new()))
            .unwrap();

        let snapshot = block_on(checkpoint.load(160, &Cancellation::new())).unwrap();
        let mut restored = Runtime::from_snapshot(160, workflow, snapshot).unwrap();
        assert_eq!(restored.state().status, Status::Succeeded);
    }

    #[test]
    fn configured_checkpoint_preserves_cancelled_state_even_when_run_is_cancelled() {
        let mut builder = WorkflowBuilder::new("checkpoint-cancel");
        builder.add_node("answer", crate::node::Echo).unwrap();
        builder.add_edge(START, "answer").unwrap();
        builder.add_edge("answer", END).unwrap();
        let workflow = builder.compile().unwrap();
        let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
        let mut runtime = Runtime::new(161, workflow)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        let cancellation = Cancellation::new();
        cancellation.cancel();

        let error =
            block_on(runtime.execute(Messages::default(), Arc::new(|| 6_001), cancellation))
                .unwrap_err();
        assert_eq!(error.code, "CANCELLED");
        let snapshot = block_on(checkpoint.load(161, &Cancellation::new())).unwrap();
        assert_eq!(snapshot["status"], "cancelled");
    }

    #[test]
    fn workflow_memory_writes_chat_graph_and_trace_state_only_at_closeout() {
        use crate::memory::short::{chat::Store as _, trace::Store as _};

        let mut builder = WorkflowBuilder::new("workflow-memory-closeout");
        builder
            .add_agent_node("agent", FakeAgent { mode: "single" })
            .unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = builder.compile().unwrap();
        let (config, chats, traces) = workflow_memory();
        let mut runtime = Runtime::new(5, workflow.clone())
            .unwrap()
            .memory(config)
            .unwrap();

        block_on(runtime.execute(workflow_input(), Arc::new(|| 5_001), Cancellation::new()))
            .unwrap();

        let chat = block_on(chats.read(5, &Cancellation::new())).unwrap();
        assert_eq!(chat.r#type, crate::memory::short::Type::Workflow);
        assert_eq!(chat.status, crate::memory::short::chat::Status::Completed);
        assert_eq!(
            chat.graph,
            Some(serde_json::to_value(runtime.definition()).unwrap())
        );
        assert_eq!(
            chat.message.as_ref().map(|message| message.role.clone()),
            Some(Role::Assistant)
        );
        let trace = block_on(traces.read(5, &Cancellation::new())).unwrap();
        assert_eq!(trace.state, runtime.snapshot().unwrap());
        assert_eq!(trace.state["chat_id"], 5);
        assert_eq!(trace.state["type"], "workflow");
        assert_eq!(trace.state["status"], "succeeded");
    }

    #[test]
    fn terminal_workflow_snapshot_is_idempotent_with_memory() {
        use crate::memory::short::{chat::Store as _, trace::Store as _};

        let mut builder = WorkflowBuilder::new("workflow-terminal-restore");
        builder
            .add_agent_node("agent", FakeAgent { mode: "single" })
            .unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = builder.compile().unwrap();
        let (config, chats, traces) = workflow_memory();
        let mut first = Runtime::new(5, workflow.clone())
            .unwrap()
            .memory(config.clone())
            .unwrap();
        block_on(first.execute(workflow_input(), Arc::new(|| 5_010), Cancellation::new())).unwrap();
        let snapshot = first.snapshot().unwrap();

        let mut restored = Runtime::from_snapshot(5, workflow, snapshot)
            .unwrap()
            .memory(config)
            .unwrap();
        block_on(restored.execute(workflow_input(), Arc::new(|| 5_011), Cancellation::new()))
            .unwrap();

        assert_eq!(
            block_on(chats.read(5, &Cancellation::new()))
                .unwrap()
                .status,
            crate::memory::short::chat::Status::Completed
        );
        assert_eq!(
            block_on(traces.read(5, &Cancellation::new()))
                .unwrap()
                .state["chat_id"],
            5
        );
    }

    #[test]
    fn workflow_snapshot_rejects_a_different_chat_id() {
        let mut builder = WorkflowBuilder::new("workflow-checkpoint-identity");
        builder.add_node("answer", crate::node::Echo).unwrap();
        builder.add_edge(START, "answer").unwrap();
        builder.add_edge("answer", END).unwrap();
        let workflow = builder.compile().unwrap();
        let runtime = Runtime::new(5, workflow.clone()).unwrap();
        let error = match Runtime::from_snapshot(6, workflow, runtime.snapshot().unwrap()) {
            Ok(_) => panic!("a checkpoint for another chat_id must be rejected"),
            Err(error) => error,
        };
        assert_eq!(error.code, "CONFLICT");
    }

    #[test]
    fn workflow_memory_keeps_waiting_chat_open_until_resume_closeout() {
        use crate::memory::short::{chat::Store as _, trace::Store as _};

        let mut builder = WorkflowBuilder::new("workflow-memory-wait");
        builder
            .add_node("approval", WaitOnce(AtomicBool::new(false)))
            .unwrap();
        builder.add_edge(START, "approval").unwrap();
        builder.add_edge("approval", END).unwrap();
        let workflow = builder.compile().unwrap();
        let (config, chats, traces) = workflow_memory();
        let mut runtime = Runtime::new(5, workflow).unwrap().memory(config).unwrap();
        let initial = workflow_input();

        assert_eq!(
            block_on(runtime.execute(initial.clone(), Arc::new(|| 5_002), Cancellation::new(),))
                .unwrap_err()
                .code,
            "WAITING"
        );
        assert_eq!(
            block_on(chats.read(5, &Cancellation::new()))
                .unwrap()
                .status,
            crate::memory::short::chat::Status::Running
        );
        assert_eq!(
            block_on(traces.read(5, &Cancellation::new()))
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );

        runtime
            .resume(
                "approval",
                Messages::new([crate::Message::function(serde_json::json!({
                    "approved": true
                }))]),
            )
            .unwrap();
        block_on(runtime.execute(initial, Arc::new(|| 5_003), Cancellation::new())).unwrap();
        assert_eq!(
            block_on(chats.read(5, &Cancellation::new()))
                .unwrap()
                .status,
            crate::memory::short::chat::Status::Completed
        );
        assert!(block_on(traces.read(5, &Cancellation::new())).is_ok());
    }

    #[test]
    fn workflow_memory_closes_failed_and_cancelled_runs_with_trace_state() {
        use crate::memory::short::{chat::Store as _, trace::Store as _};

        for (mode, expected_chat_status, expected_state) in [
            (
                "error",
                crate::memory::short::chat::Status::Failed,
                "failed",
            ),
            (
                "cancel",
                crate::memory::short::chat::Status::Stopped,
                "cancelled",
            ),
        ] {
            let mut builder = WorkflowBuilder::new(format!("workflow-memory-{mode}"));
            builder.add_agent_node("agent", FakeAgent { mode }).unwrap();
            builder.add_edge(START, "agent").unwrap();
            builder.add_edge("agent", END).unwrap();
            let workflow = builder.compile().unwrap();
            let (config, chats, traces) = workflow_memory();
            let mut runtime = Runtime::new(5, workflow).unwrap().memory(config).unwrap();
            let error = block_on(runtime.execute(
                workflow_input(),
                Arc::new(|| 5_004),
                Cancellation::new(),
            ))
            .unwrap_err();
            assert_eq!(
                error.code,
                if mode == "cancel" {
                    "CANCELLED"
                } else {
                    "BUSINESS"
                }
            );
            assert_eq!(
                block_on(chats.read(5, &Cancellation::new()))
                    .unwrap()
                    .status,
                expected_chat_status
            );
            assert_eq!(
                block_on(traces.read(5, &Cancellation::new()))
                    .unwrap()
                    .state["status"],
                expected_state
            );
        }
    }

    #[test]
    fn failed_node_can_activate_an_explicit_failure_relation() {
        let mut builder = WorkflowBuilder::new("failure-run");
        let fail = async |_: Messages, _: Cancellation| -> Result<Messages, Error> {
            Err(Error::new("BUSINESS", "failed"))
        };
        builder.add_node("fail", fail).unwrap();
        builder.add_node("handle", crate::node::Echo).unwrap();
        builder.add_edge(START, "fail").unwrap();
        builder
            .add_failure_edges(["fail"], ["handle"], "node_failed")
            .unwrap();
        builder.add_edge("handle", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(17, workflow).unwrap();
        let next = Arc::new(std::sync::atomic::AtomicI64::new(700));
        let ids = {
            let next = Arc::clone(&next);
            Arc::new(move || next.fetch_add(1, std::sync::atomic::Ordering::AcqRel))
        };
        block_on(runtime.execute(Messages::default(), ids, Cancellation::new())).unwrap();
        assert_eq!(runtime.state().nodes["fail"].status, node::Status::Failed);
        assert_eq!(
            runtime.state().nodes["handle"].status,
            node::Status::Succeeded
        );
        assert_eq!(runtime.state().status, Status::Succeeded);
    }

    #[test]
    fn run_executes_a_graph_node_and_keeps_child_state_on_attempt() {
        let mut child_builder = WorkflowBuilder::new("child-run");
        child_builder.add_node("inner", crate::node::Echo).unwrap();
        child_builder.add_edge(START, "inner").unwrap();
        child_builder.add_edge("inner", END).unwrap();
        let child = child_builder.compile().unwrap();

        let mut parent_builder = WorkflowBuilder::new("parent-run");
        parent_builder.add_graph_node("child", &child).unwrap();
        parent_builder.add_edge(START, "child").unwrap();
        parent_builder.add_edge("child", END).unwrap();
        let parent = parent_builder.compile().unwrap();
        let mut runtime = Runtime::new(15, parent).unwrap();
        let initial = Messages::new([crate::Message::function(serde_json::json!({
            "query": "nested"
        }))]);
        let next = Arc::new(std::sync::atomic::AtomicI64::new(500));
        let ids = {
            let next = Arc::clone(&next);
            Arc::new(move || next.fetch_add(1, std::sync::atomic::Ordering::AcqRel))
        };
        block_on(runtime.execute(initial.clone(), ids, Cancellation::new())).unwrap();
        let execution = &runtime.state().nodes["child"].executions[&500];
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(
            execution.attempts[0].graph.as_ref().unwrap().status,
            Status::Succeeded
        );
        assert_eq!(execution.messages, initial);
    }

    #[test]
    fn nested_graph_wait_resumes_through_the_parent_execution() {
        let mut child_builder = WorkflowBuilder::new("child-wait");
        child_builder
            .add_node("approval", WaitOnce(AtomicBool::new(false)))
            .unwrap();
        child_builder.add_edge(START, "approval").unwrap();
        child_builder.add_edge("approval", END).unwrap();
        let child = child_builder.compile().unwrap();

        let mut parent_builder = WorkflowBuilder::new("parent-wait");
        parent_builder.add_graph_node("child", &child).unwrap();
        parent_builder.add_edge(START, "child").unwrap();
        parent_builder.add_edge("child", END).unwrap();
        let parent = parent_builder.compile().unwrap();
        let mut runtime = Runtime::new(18, parent).unwrap();
        let initial = Messages::default();
        let first =
            block_on(runtime.execute(initial.clone(), Arc::new(|| 1_200), Cancellation::new()));
        assert_eq!(first.unwrap_err().code, "WAITING");
        assert_eq!(runtime.state().status, Status::Paused);
        assert_eq!(
            runtime.state().nodes["child"].executions[&1_200].status,
            node::ExecutionStatus::Waiting
        );

        runtime
            .resume(
                "child",
                Messages::new([crate::Message::function(
                    serde_json::json!({"approved": true}),
                )]),
            )
            .unwrap();
        block_on(runtime.execute(initial, Arc::new(|| 1_201), Cancellation::new())).unwrap();
        let execution = &runtime.state().nodes["child"].executions[&1_200];
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(
            execution.attempts[0].graph.as_ref().unwrap().status,
            Status::Succeeded
        );
    }

    #[test]
    fn nested_graph_checkpoint_restores_child_wait_before_parent_resume() {
        let mut child_builder = WorkflowBuilder::new("child-checkpoint-wait");
        child_builder
            .add_node("approval", WaitOnce(AtomicBool::new(false)))
            .unwrap();
        child_builder.add_edge(START, "approval").unwrap();
        child_builder.add_edge("approval", END).unwrap();
        let child = child_builder.compile().unwrap();

        let mut parent_builder = WorkflowBuilder::new("parent-checkpoint-wait");
        parent_builder.add_graph_node("child", &child).unwrap();
        parent_builder.add_edge(START, "child").unwrap();
        parent_builder.add_edge("child", END).unwrap();
        let parent = parent_builder.compile().unwrap();
        let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
        let mut runtime = Runtime::new(19, parent.clone())
            .unwrap()
            .with_checkpoint(checkpoint.clone());

        let initial = Messages::default();
        let first =
            block_on(runtime.execute(initial.clone(), Arc::new(|| 1_300), Cancellation::new()));
        assert_eq!(first.unwrap_err().code, "WAITING");

        let snapshot = block_on(checkpoint.load(19, &Cancellation::new())).unwrap();
        let child_execution = &snapshot["nodes"]["child"]["executions"]["1300"];
        assert_eq!(child_execution["status"], "waiting");
        assert_eq!(child_execution["attempts"][0]["graph"]["status"], "paused");
        assert!(
            child_execution["attempts"][0]["graph"]["waits"]
                .get("approval")
                .is_some()
        );

        let mut restored = Runtime::from_snapshot(19, parent, snapshot).unwrap();
        restored
            .resume(
                "child",
                Messages::new([crate::Message::function(
                    serde_json::json!({"approved": true}),
                )]),
            )
            .unwrap();
        block_on(restored.execute(initial, Arc::new(|| 1_301), Cancellation::new())).unwrap();

        let execution = &restored.state().nodes["child"].executions[&1_300];
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(
            execution.attempts[0].graph.as_ref().unwrap().status,
            Status::Succeeded
        );
        assert!(
            execution.attempts[0]
                .graph
                .as_ref()
                .unwrap()
                .waits
                .is_empty()
        );
    }

    #[test]
    fn checkpoint_recovery_retries_waiting_node_without_allocating_a_new_execution() {
        let mut builder = WorkflowBuilder::new("checkpoint-waiting-node");
        builder.add_node("answer", crate::node::Echo).unwrap();
        builder
            .set_retry(
                "answer",
                RetryPolicy {
                    max_retries: 1,
                    min_delay_ms: 0,
                    max_delay_ms: 0,
                },
            )
            .unwrap();
        builder.add_edge(START, "answer").unwrap();
        builder.add_edge("answer", END).unwrap();
        let workflow = builder.compile().unwrap();

        let mut runtime = Runtime::new(1_700, workflow.clone()).unwrap();
        runtime.state_mut().initial = Some(Messages::default());
        runtime
            .start_batch(vec![("answer".into(), 17_001, Vec::new())])
            .unwrap();
        let snapshot = runtime.snapshot().unwrap();
        assert_eq!(
            snapshot["nodes"]["answer"]["executions"]["17001"]["attempts"][0]["status"],
            "running"
        );

        let mut restored = Runtime::from_snapshot(1_700, workflow, snapshot).unwrap();
        let allocated = Arc::new(AtomicI64::new(0));
        let allocated_for_run = Arc::clone(&allocated);
        block_on(restored.execute(
            Messages::default(),
            Arc::new(move || allocated_for_run.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        ))
        .unwrap();

        let execution = &restored.state().nodes["answer"].executions[&17_001];
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].status, node::AttemptStatus::Succeeded);
        assert_eq!(allocated.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn checkpoint_recovery_resumes_agent_attempt_without_allocating_new_identity() {
        let mut builder = WorkflowBuilder::new("checkpoint-waiting-agent");
        builder
            .add_agent_node("agent", FakeAgent { mode: "single" })
            .unwrap();
        builder
            .set_retry(
                "agent",
                RetryPolicy {
                    max_retries: 1,
                    min_delay_ms: 0,
                    max_delay_ms: 0,
                },
            )
            .unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = builder.compile().unwrap();

        let mut runtime = Runtime::new(1_701, workflow.clone()).unwrap();
        runtime.state_mut().initial = Some(user_input());
        runtime
            .start_batch(vec![("agent".into(), 17_011, Vec::new())])
            .unwrap();
        let snapshot = runtime.snapshot().unwrap();

        let mut restored = Runtime::from_snapshot(1_701, workflow, snapshot).unwrap();
        let allocated = Arc::new(AtomicI64::new(0));
        let allocated_for_run = Arc::clone(&allocated);
        block_on(restored.execute(
            user_input(),
            Arc::new(move || allocated_for_run.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        ))
        .unwrap();

        let execution = &restored.state().nodes["agent"].executions[&17_011];
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].status, node::AttemptStatus::Succeeded);
        assert_eq!(execution.attempts[0].idx, 0);
        assert_eq!(allocated.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn embedded_agent_wait_uses_workflow_wait_and_resumes_same_execution() {
        let mut builder = WorkflowBuilder::new("embedded-agent-wait");
        builder.add_agent_node("agent", WaitingAgent).unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = builder.compile().unwrap();

        let mut runtime = Runtime::new(1_704, workflow).unwrap();
        let first =
            block_on(runtime.execute(user_input(), Arc::new(|| 17_041), Cancellation::new()));
        assert_eq!(first.unwrap_err().code, "WAITING");
        assert_eq!(runtime.state().status, Status::Paused);
        let execution = &runtime.state().nodes["agent"].executions[&17_041];
        assert_eq!(execution.status, node::ExecutionStatus::Waiting);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].status, node::AttemptStatus::Waiting);
        assert_eq!(runtime.state().waits["agent"].wait_id, "approval");

        runtime
            .resume(
                "agent",
                Messages::new([crate::Message::function(serde_json::json!({
                    "approved": true
                }))]),
            )
            .unwrap();
        block_on(runtime.execute(user_input(), Arc::new(|| 17_042), Cancellation::new())).unwrap();

        let execution = &runtime.state().nodes["agent"].executions[&17_041];
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].idx, 0);
        assert!(runtime.state().waits.is_empty());
    }

    #[test]
    fn checkpoint_recovery_passes_embedded_agent_scene_to_the_same_attempt() {
        let mut builder = WorkflowBuilder::new("checkpoint-embedded-agent-scene");
        builder.add_agent_node("agent", RestoringAgent).unwrap();
        builder.add_edge(START, "agent").unwrap();
        builder.add_edge("agent", END).unwrap();
        let workflow = builder.compile().unwrap();

        let mut runtime = Runtime::new(1_703, workflow.clone()).unwrap();
        runtime.state_mut().initial = Some(user_input());
        runtime
            .start_batch(vec![("agent".into(), 17_031, Vec::new())])
            .unwrap();
        runtime
            .state_mut()
            .nodes
            .get_mut("agent")
            .unwrap()
            .executions
            .get_mut(&17_031)
            .unwrap()
            .attempts[0]
            .agent = Some(serde_json::json!({
            "type": "agent",
            "status": "running",
            "messages": [{"role": "assistant", "content": []}]
        }));
        let snapshot = runtime.snapshot().unwrap();

        let mut restored = Runtime::from_snapshot(1_703, workflow, snapshot).unwrap();
        let allocated = Arc::new(AtomicI64::new(0));
        let allocated_for_run = Arc::clone(&allocated);
        block_on(restored.execute(
            user_input(),
            Arc::new(move || allocated_for_run.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        ))
        .unwrap();

        let execution = &restored.state().nodes["agent"].executions[&17_031];
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].idx, 0);
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(
            execution.messages.0[0].content[0].data["value"],
            serde_json::json!("restored_scene")
        );
        assert_eq!(allocated.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn checkpoint_recovery_resumes_running_node_without_retry_budget() {
        let mut builder = WorkflowBuilder::new("checkpoint-waiting-no-retry");
        builder.add_node("answer", crate::node::Echo).unwrap();
        builder.add_edge(START, "answer").unwrap();
        builder.add_edge("answer", END).unwrap();
        let workflow = builder.compile().unwrap();

        let mut runtime = Runtime::new(1_702, workflow.clone()).unwrap();
        runtime.state_mut().initial = Some(Messages::default());
        runtime
            .start_batch(vec![("answer".into(), 17_021, Vec::new())])
            .unwrap();
        let snapshot = runtime.snapshot().unwrap();

        let mut restored = Runtime::from_snapshot(1_702, workflow, snapshot).unwrap();
        let allocated = Arc::new(AtomicI64::new(0));
        let allocated_for_run = Arc::clone(&allocated);
        let result = block_on(restored.execute(
            Messages::default(),
            Arc::new(move || allocated_for_run.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        ))
        .unwrap();

        assert!(result.is_empty());
        let execution = &restored.state().nodes["answer"].executions[&17_021];
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].status, node::AttemptStatus::Succeeded);
        assert_eq!(allocated.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn run_waits_at_a_barrier_instead_of_activating_downstream() {
        let mut builder = WorkflowBuilder::new("wait");
        builder.add_node("a", crate::node::Echo).unwrap();
        builder.add_node("b", crate::node::Echo).unwrap();
        builder.add_edge(START, "a").unwrap();
        builder.add_edge("a", "b").unwrap();
        builder.add_edge("b", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(12, workflow).unwrap();
        let cancellation = Cancellation::new();
        cancellation.cancel();
        let ids = Arc::new(|| 201);
        assert_eq!(
            block_on(runtime.execute(Messages::default(), ids, cancellation))
                .unwrap_err()
                .code,
            "CANCELLED"
        );
        assert_eq!(runtime.state().status, Status::Cancelled);
    }

    #[test]
    fn resolved_wait_reuses_the_same_execution_id_and_attempt() {
        let mut builder = WorkflowBuilder::new("resume-wait");
        builder
            .add_node("approval", WaitOnce(AtomicBool::new(false)))
            .unwrap();
        builder.add_edge(START, "approval").unwrap();
        builder.add_edge("approval", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(13, workflow).unwrap();
        let mut request = crate::Message::new(crate::components::message::Role::User);
        request
            .content
            .push(crate::components::model::content::part::Part {
                r#type: "text".into(),
                data: crate::components::model::content::part::object(
                    serde_json::json!({"value":"request"}),
                ),
            });
        let initial = Messages::new([request]);
        let ids = Arc::new(AtomicI64::new(700));
        let first = block_on(runtime.execute(
            initial.clone(),
            Arc::new({
                let ids = Arc::clone(&ids);
                move || ids.fetch_add(1, Ordering::AcqRel)
            }),
            Cancellation::new(),
        ));
        assert_eq!(first.unwrap_err().code, "WAITING");
        let mut approved = crate::Message::new(crate::components::message::Role::User);
        approved
            .content
            .push(crate::components::model::content::part::Part {
                r#type: "text".into(),
                data: crate::components::model::content::part::object(
                    serde_json::json!({"value":"approved"}),
                ),
            });
        let approved = Messages::new([approved]);
        runtime.resume("approval", approved.clone()).unwrap();
        runtime.resume("approval", approved).unwrap();
        let mut changed = crate::Message::new(crate::components::message::Role::User);
        changed
            .content
            .push(crate::components::model::content::part::Part {
                r#type: "text".into(),
                data: crate::components::model::content::part::object(
                    serde_json::json!({"value":"changed"}),
                ),
            });
        assert_eq!(
            runtime
                .resume("approval", Messages::new([changed]))
                .unwrap_err()
                .code,
            "CONFLICT"
        );
        let output = block_on(runtime.execute(
            initial,
            Arc::new({
                let ids = Arc::clone(&ids);
                move || ids.fetch_add(1, Ordering::AcqRel)
            }),
            Cancellation::new(),
        ))
        .unwrap();
        assert_eq!(
            output.as_slice()[0].content[0].data,
            crate::components::model::content::part::object(
                serde_json::json!({"value":"approved"})
            )
        );
        let execution = &runtime.state().nodes["approval"].executions[&700];
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].idx, 0);
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(ids.load(Ordering::Acquire), 701);
    }

    #[test]
    fn fan_out_relation_activates_one_target_execution() {
        let mut builder = WorkflowBuilder::new("fan-out");
        builder.add_node("source", crate::node::Echo).unwrap();
        builder.add_node("target", crate::node::Echo).unwrap();
        builder.add_edge(START, "source").unwrap();
        builder.add_fan_out_edges("source", "target").unwrap();
        builder.add_edge("target", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(14, workflow).unwrap();
        let ids = Arc::new(AtomicI64::new(800));
        block_on(runtime.execute(
            Messages::default(),
            Arc::new({
                let ids = Arc::clone(&ids);
                move || ids.fetch_add(1, Ordering::AcqRel)
            }),
            Cancellation::new(),
        ))
        .unwrap();
        assert_eq!(runtime.state().nodes["target"].executions.len(), 1);
        assert_eq!(
            runtime.state().nodes["target"].executions[&801].from[0].exec_id,
            800
        );
    }

    #[test]
    fn loop_relation_uses_the_declared_condition_and_new_execution_ids() {
        let mut builder = WorkflowBuilder::new("loop");
        builder.add_node("first", crate::node::Echo).unwrap();
        builder.add_node("second", crate::node::Echo).unwrap();
        builder.add_edge(START, "first").unwrap();
        builder.add_edge("first", "second").unwrap();
        builder.add_edge("second", END).unwrap();
        let enabled = Arc::new(AtomicBool::new(true));
        let condition = Arc::clone(&enabled);
        builder
            .add_loop_edge("second", "first", move |_| {
                condition.swap(false, Ordering::AcqRel)
            })
            .unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(15, workflow).unwrap();
        let ids = Arc::new(AtomicI64::new(900));
        block_on(runtime.execute(
            Messages::default(),
            Arc::new({
                let ids = Arc::clone(&ids);
                move || ids.fetch_add(1, Ordering::AcqRel)
            }),
            Cancellation::new(),
        ))
        .unwrap();
        assert_eq!(runtime.state().nodes["first"].executions.len(), 2);
        assert_eq!(runtime.state().nodes["second"].executions.len(), 2);
        assert_eq!(
            runtime.state().nodes["first"].executions[&902].from[0].exec_id,
            901
        );
    }

    #[test]
    fn join_failure_waits_for_all_sources_and_activates_handler() {
        let mut builder = WorkflowBuilder::new("join-failure");
        builder
            .add_node("failed", |_input: Messages, _| async {
                Err::<Messages, crate::Error>(crate::Error::new("FAILED", "boom"))
            })
            .unwrap();
        builder.add_node("ok", crate::node::Echo).unwrap();
        builder.add_node("handler", crate::node::Echo).unwrap();
        builder.add_edge(START, "failed").unwrap();
        builder.add_edge(START, "ok").unwrap();
        builder.add_join(["failed", "ok"], "handler").unwrap();
        builder
            .add_failure_edges(["failed", "ok"], ["handler"], "join_failed")
            .unwrap();
        builder.add_edge("handler", END).unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(16, workflow).unwrap();
        let ids = Arc::new(AtomicI64::new(1_000));
        block_on(runtime.execute(
            Messages::default(),
            Arc::new({
                let ids = Arc::clone(&ids);
                move || ids.fetch_add(1, Ordering::AcqRel)
            }),
            Cancellation::new(),
        ))
        .unwrap();
        let handler = runtime.state().nodes["handler"]
            .executions
            .values()
            .next()
            .unwrap();
        assert_eq!(handler.from.len(), 2);
        assert_eq!(handler.status, node::ExecutionStatus::Succeeded);
    }

    #[test]
    fn runtime_retries_only_transient_node_errors() {
        let mut builder = WorkflowBuilder::new("retry");
        builder
            .add_node("retry", Flaky(AtomicBool::new(false)))
            .unwrap();
        builder.add_edge(START, "retry").unwrap();
        builder.add_edge("retry", END).unwrap();
        builder
            .set_retry(
                "retry",
                RetryPolicy {
                    max_retries: 1,
                    min_delay_ms: 0,
                    max_delay_ms: 0,
                },
            )
            .unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(17, workflow).unwrap();
        block_on(runtime.execute(Messages::default(), Arc::new(|| 1_100), Cancellation::new()))
            .unwrap();
        let execution = &runtime.state().nodes["retry"].executions[&1_100];
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(execution.attempts[0].status, node::AttemptStatus::Failed);
        assert_eq!(execution.attempts[1].status, node::AttemptStatus::Succeeded);
    }

    #[test]
    fn runtime_tool_retry_skips_successful_call_ids() {
        let first_calls = Arc::new(AtomicI64::new(0));
        let second_calls = Arc::new(AtomicI64::new(0));
        let mut builder = WorkflowBuilder::new("tool-retry-dedup");
        builder
            .add_node(
                "tools",
                crate::runtime::workflow::tool::ToolNode::new(RetryTool {
                    definition: workflow_tool_definition("first"),
                    calls: first_calls.clone(),
                    fail_first: false,
                })
                .tool(RetryTool {
                    definition: workflow_tool_definition("second"),
                    calls: second_calls.clone(),
                    fail_first: true,
                }),
            )
            .unwrap();
        builder.add_edge(START, "tools").unwrap();
        builder.add_edge("tools", END).unwrap();
        builder
            .set_retry(
                "tools",
                RetryPolicy {
                    max_retries: 1,
                    min_delay_ms: 0,
                    max_delay_ms: 0,
                },
            )
            .unwrap();
        let workflow = builder.compile().unwrap();
        let mut runtime = Runtime::new(18, workflow).unwrap();
        block_on(runtime.execute(
            workflow_tool_input(),
            Arc::new(|| 1_200),
            Cancellation::new(),
        ))
        .unwrap();

        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 2);
        let execution = &runtime.state().nodes["tools"].executions[&1_200];
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(execution.attempts[0].messages.as_slice().len(), 1);
        assert_eq!(execution.attempts[1].messages.as_slice().len(), 2);
    }

    fn retry_workflow(calls: Arc<AtomicI64>, policy: RetryPolicy) -> CompiledWorkflow {
        let mut builder = WorkflowBuilder::new("async-retry");
        builder
            .add_node("retry", move |input: Messages, _: Cancellation| {
                let calls = calls.clone();
                async move {
                    if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        Err(Error::new("TIMEOUT", "first call failed"))
                    } else {
                        Ok(input)
                    }
                }
            })
            .unwrap();
        builder.add_edge(START, "retry").unwrap();
        builder.add_edge("retry", END).unwrap();
        builder.set_retry("retry", policy).unwrap();
        builder.compile().unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn runtime_waits_for_backoff_before_starting_the_next_attempt() {
        let calls = Arc::new(AtomicI64::new(0));
        let workflow = retry_workflow(
            calls.clone(),
            RetryPolicy {
                max_retries: 1,
                min_delay_ms: 100,
                max_delay_ms: 100,
            },
        );
        let mut runtime = Runtime::new(90, workflow).unwrap();
        let mut run =
            Box::pin(runtime.execute(Messages::default(), Arc::new(|| 900), Cancellation::new()));
        assert!(futures_util::poll!(run.as_mut()).is_pending());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        tokio::time::advance(std::time::Duration::from_millis(99)).await;
        assert!(futures_util::poll!(run.as_mut()).is_pending());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        tokio::time::advance(std::time::Duration::from_millis(1)).await;
        run.await.unwrap();
        let execution = &runtime.state().nodes["retry"].executions[&900];
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(
            execution.attempts[0].error.as_ref().unwrap().code,
            "TIMEOUT"
        );
        assert_eq!(execution.attempts[1].status, node::AttemptStatus::Succeeded);
    }

    #[tokio::test(start_paused = true)]
    async fn cancellation_during_backoff_preserves_the_failed_attempt() {
        let calls = Arc::new(AtomicI64::new(0));
        let workflow = retry_workflow(
            calls.clone(),
            RetryPolicy {
                max_retries: 1,
                min_delay_ms: 100,
                max_delay_ms: 100,
            },
        );
        let mut runtime = Runtime::new(90, workflow).unwrap();
        let cancellation = Cancellation::new();
        let mut run =
            Box::pin(runtime.execute(Messages::default(), Arc::new(|| 900), cancellation.clone()));
        assert!(futures_util::poll!(run.as_mut()).is_pending());
        cancellation.cancel();
        assert_eq!(run.await.unwrap_err().code, "CANCELLED");
        let execution = &runtime.state().nodes["retry"].executions[&900];
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(execution.status, node::ExecutionStatus::Cancelled);
        assert!(execution.error.is_none());
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].status, node::AttemptStatus::Failed);
        assert_eq!(
            execution.attempts[0].error.as_ref().unwrap().code,
            "TIMEOUT"
        );
    }

    #[test]
    fn retry_without_a_timer_returns_an_error_without_replaying_the_node() {
        let calls = Arc::new(AtomicI64::new(0));
        let workflow = retry_workflow(
            calls.clone(),
            RetryPolicy {
                max_retries: 1,
                min_delay_ms: 100,
                max_delay_ms: 100,
            },
        );
        let mut runtime = Runtime::new(90, workflow).unwrap();
        let error =
            block_on(runtime.execute(Messages::default(), Arc::new(|| 900), Cancellation::new()))
                .unwrap_err();
        assert_eq!(error.code, "UNAVAILABLE");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let execution = &runtime.state().nodes["retry"].executions[&900];
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(
            execution.attempts[0].error.as_ref().unwrap().code,
            "TIMEOUT"
        );
    }

    #[test]
    fn resumed_attempt_does_not_reset_the_execution_retry_budget() {
        let calls = Arc::new(AtomicI64::new(0));
        let policy = RetryPolicy {
            max_retries: 1,
            min_delay_ms: 0,
            max_delay_ms: 0,
        };
        let workflow = retry_workflow(calls.clone(), policy.clone());
        let invocation = block_on(invoke_with_retry(
            Target::Node(workflow.executor("retry").unwrap()),
            serde_json::to_value(Messages::default()).unwrap(),
            Cancellation::new(),
            policy,
            Arc::new(|| 900),
            90,
            1,
            Messages::default(),
            None,
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(invocation.attempts.len(), 1);
        assert_eq!(invocation.attempts[0].idx, 1);
        assert!(matches!(invocation.outcome, Outcome::Failed(_)));
    }

    #[test]
    fn checkpoint_recovery_toolnode_reuses_committed_call_ids() {
        let first_calls = Arc::new(AtomicI64::new(0));
        let second_calls = Arc::new(AtomicI64::new(0));
        let mut builder = WorkflowBuilder::new("tool-checkpoint-recovery");
        builder
            .add_node(
                "tools",
                crate::runtime::workflow::tool::ToolNode::new(RetryTool {
                    definition: workflow_tool_definition("first"),
                    calls: first_calls.clone(),
                    fail_first: false,
                })
                .tool(RetryTool {
                    definition: workflow_tool_definition("second"),
                    calls: second_calls.clone(),
                    fail_first: false,
                }),
            )
            .unwrap();
        builder.add_edge(START, "tools").unwrap();
        builder.add_edge("tools", END).unwrap();
        let workflow = builder.compile().unwrap();

        let mut committed = crate::Message::function(serde_json::json!({
            "tool": "first",
            "ok": true,
        }));
        committed
            .metadata
            .insert("call_id".into(), serde_json::json!("call_a"));
        let invocation = block_on(invoke_with_retry(
            Target::Node(workflow.executor("tools").unwrap()),
            serde_json::to_value(workflow_tool_input()).unwrap(),
            Cancellation::new(),
            RetryPolicy::default(),
            Arc::new(|| 1_300),
            19,
            0,
            Messages::new([committed]),
            None,
        ));

        assert!(matches!(invocation.outcome, Outcome::Complete(_)));
        assert_eq!(first_calls.load(Ordering::SeqCst), 0);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
        assert_eq!(invocation.attempts[0].messages.as_slice().len(), 2);
    }

    fn recovery_tools(
        calls: &[Arc<AtomicI64>; 3],
        parallel: bool,
        fail_first: bool,
    ) -> tool::ToolNode {
        let node = tool::ToolNode::new(RetryTool {
            definition: workflow_tool_definition("first"),
            calls: calls[0].clone(),
            fail_first: false,
        })
        .tool(RetryTool {
            definition: workflow_tool_definition("second"),
            calls: calls[1].clone(),
            fail_first,
        })
        .tool(RetryTool {
            definition: workflow_tool_definition("third"),
            calls: calls[2].clone(),
            fail_first: false,
        });
        if parallel { node.parallel() } else { node }
    }

    fn recovery_workflow(node: tool::ToolNode) -> CompiledWorkflow {
        let mut builder = WorkflowBuilder::new("tool-process-recovery");
        builder.add_node("tools", node).unwrap();
        builder.add_edge(START, "tools").unwrap();
        builder.add_edge("tools", END).unwrap();
        builder
            .set_retry(
                "tools",
                RetryPolicy {
                    max_retries: 1,
                    min_delay_ms: 0,
                    max_delay_ms: 0,
                },
            )
            .unwrap();
        builder.compile().unwrap()
    }

    #[test]
    fn tool_checkpoint_recovers_in_a_fresh_process_without_replaying_successes() {
        const FIXTURE: &str = "HEART_TOOL_RECOVERY_FIXTURE";
        const MODE: &str = "HEART_TOOL_RECOVERY_PARALLEL";
        const RETRY: &str = "HEART_TOOL_RECOVERY_RETRY";
        let calls = std::array::from_fn(|_| Arc::new(AtomicI64::new(0)));
        if let Some(path) = std::env::var_os(FIXTURE) {
            let parallel = std::env::var_os(MODE).is_some();
            let prior_attempt = std::env::var_os(RETRY).is_some();
            let snapshot: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            let workflow = recovery_workflow(recovery_tools(&calls, parallel, !prior_attempt));
            let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
            let mut runtime = Runtime::from_snapshot(19, workflow, snapshot)
                .unwrap()
                .with_checkpoint(checkpoint.clone());
            let input = runtime.state().initial.clone().unwrap();
            let output = block_on(runtime.execute(
                input.clone(),
                Arc::new(|| panic!("recovery must reuse the existing exec_id")),
                Cancellation::new(),
            ))
            .unwrap();

            assert_eq!(calls[0].load(Ordering::SeqCst), 0);
            let retries = if prior_attempt { 1 } else { 2 };
            assert_eq!(calls[1].load(Ordering::SeqCst), retries);
            assert_eq!(calls[2].load(Ordering::SeqCst), i64::from(!parallel));
            let execution = &runtime.state().nodes["tools"].executions[&1_300];
            assert_eq!(execution.attempts.len(), 2);
            assert_eq!(execution.attempts[0].idx, 0);
            assert_eq!(execution.attempts[0].status, node::AttemptStatus::Failed);
            assert_eq!(
                execution.attempts[0].error.as_ref().unwrap().code,
                "TIMEOUT"
            );
            assert_eq!(execution.attempts[1].idx, 1);
            assert_eq!(execution.attempts[1].status, node::AttemptStatus::Succeeded);
            assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
            assert!(execution.error.is_none());
            let ids = output
                .as_slice()
                .iter()
                .map(|message| message.metadata["call_id"].as_str().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(ids, ["call_a", "call_b", "call_c"]);
            let terminal = block_on(checkpoint.load(19, &Cancellation::new())).unwrap();
            assert_eq!(terminal, runtime.snapshot().unwrap());
            // A second process-local restore of the saved terminal result is
            // a read, not another tool invocation or attempt.
            let mut terminal_runtime = Runtime::from_snapshot(
                19,
                recovery_workflow(recovery_tools(&calls, parallel, !prior_attempt)),
                terminal,
            )
            .unwrap();
            assert_eq!(
                block_on(terminal_runtime.execute(
                    input,
                    Arc::new(|| panic!("terminal restore must not allocate an id")),
                    Cancellation::new(),
                ))
                .unwrap(),
                output
            );
            assert_eq!(calls[1].load(Ordering::SeqCst), retries);
            return;
        }

        for (parallel, prior_attempt) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let calls = std::array::from_fn(|_| Arc::new(AtomicI64::new(0)));
            let tools = recovery_tools(&calls, parallel, true);
            let mut input = workflow_tool_input();
            input.0[0].content.push(crate::model::content::part::Part {
                r#type: "tool_call".into(),
                data: crate::model::content::part::object(serde_json::json!({
                    "call_id": "call_c", "name": "third", "arguments": {}
                })),
            });
            let partial = block_on(tools.call_with_recovery(
                input.clone(),
                Cancellation::new(),
                Messages::default(),
            ))
            .unwrap_err();
            assert_eq!(partial.error.code, "TIMEOUT");
            assert_eq!(partial.completed.0.len(), if parallel { 2 } else { 1 });
            let mut runtime = Runtime::new(19, recovery_workflow(tools)).unwrap();
            runtime.state_mut().initial = Some(input);
            runtime
                .start_batch(vec![("tools".into(), 1_300, vec![])])
                .unwrap();
            // Model the saved in-flight scene, not an already terminal failed
            // run: these sub-call facts are present, but normal output and
            // downstream activation have not been committed.
            runtime
                .state_mut()
                .nodes
                .get_mut("tools")
                .unwrap()
                .executions
                .get_mut(&1_300)
                .unwrap()
                .attempts[0]
                .messages = partial.completed;
            if prior_attempt {
                let execution = runtime
                    .state_mut()
                    .nodes
                    .get_mut("tools")
                    .unwrap()
                    .executions
                    .get_mut(&1_300)
                    .unwrap();
                execution.attempts[0].status = node::AttemptStatus::Failed;
                execution.attempts[0].error = Some(partial.error);
                // A crash after starting the next Attempt must not lose
                // successful calls retained only in the previous Attempt.
                execution.attempts.push(node::Attempt {
                    idx: 1,
                    ..Default::default()
                });
            }
            assert!(
                runtime.state().nodes["tools"].executions[&1_300]
                    .messages
                    .0
                    .is_empty()
            );
            let checkpoint = crate::core::checkpoint::InMemory::new();
            block_on(runtime.save_checkpoint(&checkpoint, &Cancellation::new())).unwrap();
            let saved = block_on(checkpoint.load(19, &Cancellation::new())).unwrap();
            let path = std::env::temp_dir().join(format!(
                "heart-tool-recovery-{}-{}-{parallel}.json",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::write(&path, serde_json::to_vec(&saved).unwrap()).unwrap();
            drop(runtime);
            let mut process = std::process::Command::new(std::env::current_exe().unwrap());
            process.args([
                "--exact",
                "runtime::workflow::tests::tool_checkpoint_recovers_in_a_fresh_process_without_replaying_successes",
                "--nocapture",
            ]).env(FIXTURE, &path).env_remove(MODE).env_remove(RETRY);
            if parallel {
                process.env(MODE, "1");
            }
            if prior_attempt {
                process.env(RETRY, "1");
            }
            let result = process.output();
            std::fs::remove_file(path).unwrap();
            let result = result.unwrap();
            assert!(
                result.status.success(),
                "child recovery failed: {}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}
