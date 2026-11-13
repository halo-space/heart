//! Opinionated high-level Agent assembly built from Core components.
//!
//! Harness supplies long-task defaults and uses a configured Workspace. It
//! delegates execution to the Runtime ReAct Agent and Core components; it does
//! not introduce a second execution loop or a Workflow lifecycle.
//!
//! The directory modules keep Harness policy and context defaults separate
//! from the assembly builder. Context compression, result offloading and
//! external task control belong here as they are added; their execution still
//! delegates to existing Runtime and Core contracts.

pub mod context;
pub mod observe;
pub mod policy;
pub mod state;
mod state_store;

use crate::Cancellation;
use crate::components::agent::{Error, Input};
use crate::components::evaluation::Evaluator;
use crate::components::model::chat::Chat;
use crate::components::workspace::Workspace;
use crate::core::dispatch::Dispatch;
use crate::runtime::agent::{react, subagent::SubAgent};
use std::sync::Arc;

pub use context::DEFAULT_SYSTEM_PROMPT;

/// Start building the long-task Harness preset.
pub fn builder() -> Builder {
    let state = Arc::new(std::sync::Mutex::new(None));
    let subagents = Arc::new(std::sync::Mutex::new(Vec::new()));
    Builder {
        inner: react::builder().system_prompt(context::system_prompt(None)),
        workspace: None,
        policy: policy::Policy::default(),
        state,
        subagents,
    }
}

/// High-level assembly builder. Core Agent tools remain individually
/// configurable; the Harness only adds its long-task defaults and composition.
pub struct Builder<M = ()> {
    inner: react::Builder<M>,
    workspace: Option<Dispatch>,
    policy: policy::Policy,
    state: Arc<std::sync::Mutex<Option<Arc<state::Store>>>>,
    subagents: Arc<std::sync::Mutex<Vec<String>>>,
}

impl<M> Builder<M> {
    pub fn model<N>(self, model: N) -> Builder<context::ContextChat<N>> {
        Builder {
            inner: self.inner.model(context::ContextChat::new(
                model,
                Arc::clone(&self.state),
                Arc::clone(&self.subagents),
            )),
            workspace: self.workspace,
            policy: self.policy,
            state: self.state,
            subagents: self.subagents,
        }
    }

    /// Replace the Harness long-task prompt with an application prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        self.inner = self
            .inner
            .system_prompt(context::system_prompt(Some(&prompt)));
        self
    }

    /// Use a Workspace with its configured tools and backend-specific tools.
    /// Local file adapters and Server database tools keep their own schemas;
    /// Harness does not require file methods or construct a second dispatcher.
    #[allow(private_bounds)]
    pub fn workspace<W>(mut self, workspace: W) -> Self
    where
        W: Workspace + 'static,
    {
        self.workspace = Some(Dispatch::workspace(workspace));
        self
    }

    pub fn subagents(mut self, subagents: impl IntoIterator<Item = SubAgent>) -> Self {
        let subagents = subagents.into_iter().collect::<Vec<_>>();
        *self
            .subagents
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = subagents
            .iter()
            .map(|subagent| subagent.name().to_owned())
            .collect();
        self.inner = self.inner.subagents(subagents);
        self
    }

    pub fn middleware<H: crate::middleware::Middleware + 'static>(mut self, middleware: H) -> Self {
        self.inner = self.inner.middleware(middleware);
        self
    }

    pub fn permission<P: crate::permission::Permission + 'static>(mut self, permission: P) -> Self {
        self.inner = self.inner.permission(permission);
        self
    }

    pub fn evaluator<E: Evaluator + 'static>(mut self, evaluator: E) -> Self {
        self.inner = self.inner.evaluator(evaluator);
        self
    }

    pub fn checkpoint<C: crate::checkpoint::Checkpoint + 'static>(mut self, checkpoint: C) -> Self {
        self.inner = self.inner.checkpoint(checkpoint);
        self
    }

    pub fn memory<S, L>(mut self, config: crate::memory::Config<S, L>) -> Self
    where
        S: crate::memory::short::Store + 'static,
        L: crate::memory::Memory + 'static,
    {
        self.inner = self.inner.memory(config);
        self
    }

    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.policy.max_iters = max_iters;
        self.inner = self.inner.with_max_iters(max_iters);
        self
    }

    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.policy.max_concurrency = max_concurrency;
        self.inner = self.inner.with_max_concurrency(max_concurrency);
        self
    }

    pub fn with_id_fn(mut self, id_fn: fn() -> i64) -> Self {
        self.inner = self.inner.with_id_fn(id_fn);
        self
    }

    /// Attach the durable Harness state backend. Runtime Checkpoints and
    /// Memory remain separate stores with their existing responsibilities.
    pub fn state(self, store: state::Store) -> Self {
        *self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Arc::new(store));
        self
    }
}

impl<M> Builder<M>
where
    M: Chat + 'static,
{
    /// Finish the long-task preset. This only assembles the configured Core
    /// components and the existing ReAct runtime; it does not start a run.
    /// The returned Agent keeps `run` and `reply` semantics identical to the
    /// direct `agent::react()` builder.
    pub fn build(self) -> Result<Agent<M>, Error> {
        self.policy.validate()?;
        let dispatch = Dispatch::new().merge(self.workspace.unwrap_or_default())?;
        let inner = self.inner.with_dispatch(dispatch).build()?;
        Ok(Agent {
            inner,
            state: self.state,
        })
    }
}

/// Long-task Harness Agent. All execution semantics remain those of the
/// wrapped Runtime ReAct Agent and its configured Core components.
pub struct Agent<M> {
    inner: react::Agent<M>,
    state: Arc<std::sync::Mutex<Option<Arc<state::Store>>>>,
}

impl<M> Agent<M> {
    /// Load the durable Harness state attached to this Agent.
    pub async fn load_state(
        &self,
        cancellation: &Cancellation,
    ) -> Result<Option<state::State>, Error> {
        let store = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| Error::new("NOT_CONFIGURED", "Harness state is not configured"))?;
        store.load(cancellation).await
    }

    /// Persist a new Harness state version through the configured backend.
    pub async fn save_state(
        &self,
        value: &state::State,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        let store = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| Error::new("NOT_CONFIGURED", "Harness state is not configured"))?;
        store.save(value, cancellation).await
    }

    /// Apply one accepted Refinement and persist the next Harness version.
    pub async fn refine(
        &self,
        item: state::Item,
        change: state::Change,
        cancellation: &Cancellation,
    ) -> Result<state::State, Error> {
        let store = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| Error::new("NOT_CONFIGURED", "Harness state is not configured"))?;
        store.refine(item, change, cancellation).await
    }

    /// Update the persistent long-task Goal.
    pub async fn update_goal(
        &self,
        goal: state::Goal,
        change: state::Change,
        cancellation: &Cancellation,
    ) -> Result<state::State, Error> {
        let store = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| Error::new("NOT_CONFIGURED", "Harness state is not configured"))?;
        store.update_goal(goal, change, cancellation).await
    }
}

impl<M> crate::components::agent::Agent for Agent<M>
where
    M: Chat + 'static,
{
    type Stream = <react::Agent<M> as crate::components::agent::Agent>::Stream;

    async fn run(
        &self,
        input: Input,
        stream: bool,
        cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        self.inner.run(input, stream, cancellation).await
    }

    async fn reply(&self, input: Input, cancellation: Cancellation) -> Result<(), Error> {
        self.inner.reply(input, cancellation).await
    }
}
