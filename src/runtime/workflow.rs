//! Static Workflow definition and compilation.
//!
//! Static Workflow definitions and their executable Pregel runtime. The
//! builder records an immutable graph definition and the runtime owns the
//! single mutable graph state for one `chat_id`.

mod admission;
mod events;
pub mod graph;
pub mod node;
pub mod tool;
pub mod wait;

use serde_json::Value;
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;

use crate::Error;
use crate::components::agent::{Input as AgentInput, Payload as AgentPayload};
use crate::components::message::{Messages, Role};
use crate::components::node::{Signal, View};
use crate::runtime::agent::history::{History, HistoryRun};
use crate::runtime::workflow::node::ExecutionRef;

pub use graph::{
    Builder, CompiledWorkflow, Conditional, Definition, Edge, Endpoint, FailureRelation, FanOut,
    FieldSource, GraphDefinition, GraphError, GraphReference, InputBinding, JoinEdge, Loop,
    NodeAttrs, NodeDefinition, RetryPolicy, Schema, State, Status, Wait,
};

/// Unified public construction entry for Workflow definitions.
pub mod builder {
    pub use super::graph::Builder;

    pub fn new(definition_id: impl Into<String>) -> Builder {
        Builder::new(definition_id)
    }

    pub fn from_dsl(
        document: crate::core::workflow::dsl::Document,
    ) -> Result<Builder, super::GraphError> {
        Builder::from_dsl(document)
    }
}

/// Static Native DSL parsing exposed beside the Workflow builder. The parser
/// implementation remains in Core and does not acquire Runtime dependencies.
pub mod dsl {
    pub use crate::core::workflow::dsl::*;
}

type CheckpointSave = Arc<
    dyn Fn(
            i64,
            Value,
            crate::Cancellation,
        ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'static>>
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
pub(crate) struct RecoveryReply {
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
        if max_concurrency == 0 || max_concurrency > tokio::sync::Semaphore::MAX_PERMITS {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "max_concurrency must be positive and within the supported permit limit",
            ));
        }
        self.engine_mut().max_concurrency = max_concurrency;
        Ok(self)
    }
    pub fn memory<S, L>(mut self, config: crate::memory::Config<S, L>) -> Result<Self, Error>
    where
        S: crate::memory::short::Store + 'static,
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
    /// Internal state transition used by the runtime after a `reply`.
    ///
    /// Applications submit external feedback through [`Runtime::reply`].
    #[doc(hidden)]
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
    /// Submit to the actual feedback Step. Outputs stay on the original run stream.
    /// This value scopes the reply submission; it does not cancel other Workflow or Agent runs.
    pub async fn reply(
        &self,
        response: Messages,
        cancellation: crate::Cancellation,
    ) -> Result<(), Error> {
        if response.as_slice().len() != 1 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "one feedback submission must identify one question Message",
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
    /// Start one Workflow run with its own cancellation scope.
    ///
    /// Create a fresh [`crate::Cancellation`] for each independent run. Clones
    /// may be shared with work spawned by this run, but should not be reused
    /// across independent Agent, Workflow, or Harness executions.
    pub async fn run(
        &self,
        initial_messages: Messages,
        next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: crate::Cancellation,
    ) -> Result<
        impl futures_core::Stream<Item = Result<crate::event::Event, Error>> + Send + '_,
        Error,
    > {
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
        if engine.accepts_saved_replies {
            interaction.restore_order(wait::ids(&engine.state)?);
        }
        let owner = interaction.owner();
        let limit = engine.max_concurrency;
        Ok(events::stream(
            events::Output::default(),
            engine.state.chat_id,
            async move {
                let _owner = owner;
                admission::run(
                    limit,
                    events::with_interaction(
                        interaction,
                        engine.execute(initial_messages, next_exec_id, cancellation),
                    ),
                )
                .await
            },
        ))
    }
}

mod execution;
use execution::Engine;
#[cfg(test)]
pub(super) use execution::{Outcome, Target, invoke_with_retry};
pub(super) use execution::{attempt_question, messages_to_agent_input};

#[cfg(test)]
#[path = "../../tests/unit/workflow_progress.rs"]
mod progress_tests;

#[cfg(test)]
#[path = "../../tests/unit/workflow_runtime.rs"]
mod tests;
