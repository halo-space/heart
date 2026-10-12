//! Static Workflow definition and compilation.
//!
//! Static Workflow definitions and their executable Pregel runtime. The
//! builder records an immutable graph definition and the runtime owns the
//! single mutable graph state for one `chat_id`.

pub mod graph;
pub mod node;
pub mod tool;

use serde_json::Value;
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::{StreamExt, stream};

use crate::Error;
use crate::agent::{Input as AgentInput, Payload as AgentPayload};
use crate::components::message::{Messages, Role};
use crate::components::node::Signal;
use crate::harness::memory::{History, Run};
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
    workflow: CompiledWorkflow,
    state: State,
    max_concurrency: usize,
    checkpoint: Option<CheckpointSave>,
    memory: Option<History>,
    history_run: Option<Run>,
    history_closed: bool,
}

impl Runtime {
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
        };
        if runtime.state.mark_waiting() {
            runtime.state.status = graph::Status::Running;
        }
        runtime.state.recompute_usage();
        Ok(runtime)
    }

    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Result<Self, Error> {
        if max_concurrency == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "max_concurrency must be positive",
            ));
        }
        self.max_concurrency = max_concurrency;
        Ok(self)
    }

    /// Configures the existing short-memory Chat/Trace stores for one
    /// Workflow run. They are touched only at terminal closeout; waiting runs
    /// remain open and can resume without creating a second Chat.
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
        self.memory = Some(history);
        Ok(self)
    }

    /// Configures the optional Checkpoint backend. The configured backend is
    /// used by `run` for one closeout save after the run returns; it is never
    /// written after individual Node executions.
    pub fn with_checkpoint<C>(mut self, checkpoint: C) -> Self
    where
        C: crate::checkpoint::Checkpoint + 'static,
    {
        let checkpoint = Arc::new(checkpoint);
        self.checkpoint = Some(Arc::new(move |id, state, cancellation| {
            let checkpoint = Arc::clone(&checkpoint);
            Box::pin(async move { checkpoint.save(id, state, &cancellation).await })
        }));
        self
    }

    pub fn chat_id(&self) -> i64 {
        self.state.chat_id
    }

    pub fn definition(&self) -> &Definition {
        self.workflow.definition()
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
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

    pub fn advance_step(&mut self) -> bool {
        self.state.advance_step()
    }

    /// Produces the exact graph-state JSON accepted by Checkpoint.
    pub fn snapshot(&self) -> Result<Value, Error> {
        serde_json::to_value(&self.state)
            .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))
    }

    pub async fn save_checkpoint<C: crate::checkpoint::Checkpoint + ?Sized>(
        &self,
        checkpoint: &C,
        cancellation: &crate::Cancellation,
    ) -> Result<(), Error> {
        checkpoint
            .save(self.state.chat_id, self.snapshot()?, cancellation)
            .await
    }

    /// Executes the Workflow and, when a Checkpoint backend is configured,
    /// persists one complete current snapshot after the run returns. The
    /// snapshot is written for every return state (success, failure, waiting
    /// and cancellation), rather than after individual Node executions. A
    /// fresh cancellation token is intentionally used for this closeout so a
    /// cancelled run can still preserve the state required for later recovery.
    pub async fn run(
        &mut self,
        initial_messages: Messages,
        next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: crate::Cancellation,
    ) -> Result<Messages, Error> {
        self.state.check_identity(self.state.chat_id)?;
        if self
            .state
            .initial
            .as_ref()
            .is_some_and(|initial| initial != &initial_messages)
        {
            return Err(Error::new(
                "CONFLICT",
                "workflow initial Messages cannot change after activation",
            ));
        }
        if let Some(chat_id) = messages_to_agent_input(&initial_messages)
            .metadata
            .get("chat_id")
            && chat_id.as_i64() != Some(self.state.chat_id)
        {
            return Err(Error::new(
                "CONFLICT",
                "input chat_id does not match this Workflow run",
            ));
        }
        // A terminal snapshot is already closed. Re-running it is a read of
        // the committed result and must not create another Chat/Trace record
        // or invoke any node again.
        if matches!(
            self.state.status,
            graph::Status::Succeeded | graph::Status::Failed | graph::Status::Cancelled
        ) {
            return self
                .run_inner(initial_messages, next_exec_id, cancellation)
                .await;
        }
        self.begin_history(&initial_messages, &cancellation).await?;
        let result = self
            .run_inner(initial_messages, next_exec_id, cancellation)
            .await;
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
                let target = if let Some(workflow) = self.workflow.graph_node(&node_id) {
                    let child_state = self.state.nodes[&node_id]
                        .executions
                        .get(&exec_id)
                        .and_then(|execution| execution.attempts.last())
                        .and_then(|attempt| attempt.graph.clone());
                    Target::Graph {
                        workflow: Arc::new(workflow.clone()),
                        state: child_state,
                    }
                } else if let Some(agent) = self.workflow.agent(&node_id) {
                    Target::Agent(
                        agent,
                        self.state.nodes[&node_id]
                            .executions
                            .get(&exec_id)
                            .and_then(|execution| execution.attempts.last())
                            .and_then(|attempt| attempt.agent.clone()),
                    )
                } else {
                    Target::Node(self.workflow.executor(&node_id).ok_or_else(|| {
                        Error::new("UNSUPPORTED", "workflow node has no executor")
                    })?)
                };
                let retry = self.definition().nodes[&node_id].retry.clone();
                let start_idx = self.state.nodes[&node_id].executions[&exec_id]
                    .attempts
                    .last()
                    .map(|attempt| attempt.idx)
                    .ok_or_else(|| Error::new("INVALID_STATE", "execution has no attempt"))?;
                let invocation = invoke_with_retry(
                    target,
                    serde_json::to_value(response)
                        .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))?,
                    cancellation.clone(),
                    retry,
                    next_exec_id.clone(),
                    self.state.chat_id,
                    start_idx,
                    self.state.nodes[&node_id].executions[&exec_id]
                        .attempts
                        .last()
                        .map(|attempt| attempt.messages.clone())
                        .unwrap_or_default(),
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
                continue;
            }
            if self.resume_waiting(&next_exec_id, &cancellation).await? {
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
            let results = stream::iter(specs.into_iter().map(
                move |(node_id, exec_id, from, input, target, retry)| {
                    let cancellation = task_cancellation.clone();
                    let next_exec_id = Arc::clone(&task_next_exec_id);
                    async move {
                        let result = invoke_with_retry(
                            target,
                            input,
                            cancellation,
                            retry,
                            next_exec_id,
                            chat_id,
                            0,
                            Messages::default(),
                        )
                        .await;
                        (node_id, exec_id, from, result)
                    }
                },
            ))
            .buffer_unordered(limit)
            .collect::<Vec<_>>()
            .await;

            for (node_id, exec_id, _from, result) in results {
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
        let Some((node_id, exec_id)) = self.state.waiting_execution() else {
            return Ok(false);
        };
        let execution = self.state.nodes[&node_id].executions[&exec_id].clone();
        let previous_idx = execution
            .attempts
            .last()
            .map(|attempt| attempt.idx)
            .ok_or_else(|| Error::new("INVALID_STATE", "waiting execution has no attempt"))?;
        let retry = self.definition().nodes[&node_id].retry.clone();
        if previous_idx >= retry.max_retries {
            self.state.fail_waiting(
                &node_id,
                exec_id,
                Error::new(
                    "RETRY_EXHAUSTED",
                    "workflow execution has no remaining retry budget",
                ),
            )?;
            self.state.recompute_usage();
            return Ok(true);
        }
        let target = self.target_for(&node_id, exec_id)?;
        let input = self.input_for(&node_id, &execution.from)?;
        let invocation = invoke_with_retry(
            target,
            input,
            cancellation.clone(),
            retry,
            Arc::clone(next_exec_id),
            self.state.chat_id,
            previous_idx.saturating_add(1),
            execution
                .attempts
                .last()
                .map(|attempt| attempt.messages.clone())
                .unwrap_or_default(),
        )
        .await;
        let invocation_graph = invocation.graph;
        self.append_recovered_attempts(&node_id, exec_id, invocation.attempts)?;
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
        Ok(true)
    }

    fn target_for(&self, node_id: &str, exec_id: i64) -> Result<Target, Error> {
        if let Some(workflow) = self.workflow.graph_node(node_id) {
            let state = self.state.nodes[node_id]
                .executions
                .get(&exec_id)
                .and_then(|execution| execution.attempts.last())
                .and_then(|attempt| attempt.graph.clone());
            return Ok(Target::Graph {
                workflow: Arc::new(workflow.clone()),
                state,
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
        attempts: Vec<node::Attempt>,
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
        execution.attempts.pop();
        execution.attempts.extend(attempts);
        execution.usage = execution.attempts.iter().fold(None, |total, attempt| {
            crate::components::model::token::merge(total, attempt.usage.clone())
        });
        Ok(())
    }

    fn append_recovered_attempts(
        &mut self,
        node_id: &str,
        exec_id: i64,
        attempts: Vec<node::Attempt>,
    ) -> Result<(), Error> {
        let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
        let expected = u32::try_from(execution.attempts.len())
            .map_err(|_| Error::new("INVALID_STATE", "too many attempts"))?;
        if attempts.is_empty() || attempts[0].idx != expected {
            return Err(Error::new(
                "INVALID_STATE",
                "invalid recovered attempt history",
            ));
        }
        for (offset, attempt) in attempts.iter().enumerate() {
            let expected = u32::try_from(execution.attempts.len() + offset)
                .map_err(|_| Error::new("INVALID_STATE", "too many attempts"))?;
            if attempt.idx != expected {
                return Err(Error::new(
                    "INVALID_STATE",
                    "recovered attempt indexes are not contiguous",
                ));
            }
        }
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
    },
}

// The final argument is the already committed ToolNode context restored from
// Checkpoint; keeping it alongside the retry inputs makes recovery explicit.
#[allow(clippy::too_many_arguments)]
async fn invoke_with_retry(
    target: Target,
    input: Value,
    cancellation: crate::Cancellation,
    retry: graph::RetryPolicy,
    next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
    chat_id: i64,
    attempt_start: u32,
    committed: Messages,
) -> Invocation {
    let mut attempts = Vec::new();
    // ToolNode-only committed call facts. They remain private to this one
    // logical execution and are rebuilt from the current Attempt on retries;
    // ordinary Nodes keep the unchanged single-call path.
    let mut committed_tool_messages = committed;
    let retry_policy = retry.clone();
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
                            if let Ok(messages) = serde_json::from_value::<Messages>(value.clone())
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
                let agent_input = match serde_json::from_value(input.clone()) {
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
                let run = executor
                    .call(agent_input, cancellation.clone(), restore.clone())
                    .await;
                attempt.steps = run.steps;
                attempt.messages = run.messages;
                attempt.usage = run.usage;
                attempt.agent = run.scene.map(|mut scene| {
                    // An embedded Agent does not own a top-level checkpoint,
                    // so the Workflow supplies the single run identity.
                    scene["chat_id"] = Value::from(chat_id);
                    scene
                });
                run.result.and_then(|messages| {
                    serde_json::to_value(messages)
                        .map(Signal::Complete)
                        .map_err(|error| Error::new("SERIALIZATION_ERROR", error.to_string()))
                })
            }
            Target::Graph { workflow, state } => {
                match run_subgraph(
                    Arc::clone(workflow),
                    input.clone(),
                    cancellation.clone(),
                    Arc::clone(&next_exec_id),
                    chat_id,
                    state.clone(),
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

type SubgraphFuture = std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<(Value, graph::State), SubgraphFailure>> + 'static>,
>;

fn run_subgraph(
    workflow: Arc<CompiledWorkflow>,
    input: Value,
    cancellation: crate::Cancellation,
    next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
    chat_id: i64,
    snapshot: Option<Box<graph::State>>,
) -> SubgraphFuture {
    Box::pin(async move {
        let messages: Messages =
            serde_json::from_value(input).map_err(|error| SubgraphFailure {
                error: Error::new("INVALID_ARGUMENTS", error.to_string()),
                state: None,
            })?;
        let mut runtime = match snapshot {
            Some(snapshot) => Runtime::from_snapshot(
                chat_id,
                (*workflow).clone(),
                serde_json::to_value(*snapshot).map_err(|error| SubgraphFailure {
                    error: Error::new("SERIALIZATION_ERROR", error.to_string()),
                    state: None,
                })?,
            )
            .map_err(|error| SubgraphFailure { error, state: None })?,
            None => Runtime::new(chat_id, (*workflow).clone())
                .map_err(|error| SubgraphFailure { error, state: None })?,
        };
        let run_input = runtime.state.initial.clone().unwrap_or(messages);
        let output = runtime
            .run(run_input, next_exec_id, cancellation)
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

    struct MeteredAgent {
        calls: Arc<AtomicI64>,
        fail_once: bool,
    }

    impl Agent for MeteredAgent {
        type Stream = stream::Iter<std::vec::IntoIter<Result<crate::event::Event, Error>>>;

        async fn run(
            &self,
            _input: AgentInput,
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
        block_on(runtime.run(user_input(), Arc::new(|| 12), Cancellation::new())).unwrap();
        let execution = &runtime.state().nodes["agent"].executions[&12];
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(28));
        let total = runtime.state().usage.as_ref().unwrap();
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
        let restored = Runtime::from_snapshot(91, workflow, snapshot).unwrap();
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
        block_on(runtime.run(
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
        type Stream = stream::Iter<std::vec::IntoIter<Result<crate::event::Event, Error>>>;

        async fn run(
            &self,
            input: AgentInput,
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
        let restored = Runtime::from_snapshot(1, workflow, snapshot).unwrap();
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
            block_on(runtime.run(Messages::new([input]), ids, Cancellation::new())).unwrap();
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
        let result = block_on(runtime.run(
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
        let _ = block_on(runtime.run(
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
        let _ = block_on(runtime.run(
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
        let result = block_on(runtime.run(initial.clone(), ids, Cancellation::new())).unwrap();
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
        block_on(runtime.run(initial, ids, Cancellation::new())).unwrap();
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
        block_on(runtime.run(Messages::default(), ids, Cancellation::new())).unwrap();
        let checkpoint = crate::core::checkpoint::InMemory::new();
        block_on(runtime.save_checkpoint(&checkpoint, &Cancellation::new())).unwrap();
        let snapshot = block_on(checkpoint.load(16, &Cancellation::new())).unwrap();
        let restored = Runtime::from_snapshot(16, workflow, snapshot).unwrap();
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

        block_on(runtime.run(Messages::default(), Arc::new(|| 6_000), Cancellation::new()))
            .unwrap();

        let snapshot = block_on(checkpoint.load(160, &Cancellation::new())).unwrap();
        let restored = Runtime::from_snapshot(160, workflow, snapshot).unwrap();
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

        let error = block_on(runtime.run(Messages::default(), Arc::new(|| 6_001), cancellation))
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

        block_on(runtime.run(workflow_input(), Arc::new(|| 5_001), Cancellation::new())).unwrap();

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
        block_on(first.run(workflow_input(), Arc::new(|| 5_010), Cancellation::new())).unwrap();
        let snapshot = first.snapshot().unwrap();

        let mut restored = Runtime::from_snapshot(5, workflow, snapshot)
            .unwrap()
            .memory(config)
            .unwrap();
        block_on(restored.run(workflow_input(), Arc::new(|| 5_011), Cancellation::new())).unwrap();

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
            block_on(runtime.run(initial.clone(), Arc::new(|| 5_002), Cancellation::new(),))
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
        block_on(runtime.run(initial, Arc::new(|| 5_003), Cancellation::new())).unwrap();
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
            let error =
                block_on(runtime.run(workflow_input(), Arc::new(|| 5_004), Cancellation::new()))
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
        block_on(runtime.run(Messages::default(), ids, Cancellation::new())).unwrap();
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
        block_on(runtime.run(initial.clone(), ids, Cancellation::new())).unwrap();
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
        let first = block_on(runtime.run(initial.clone(), Arc::new(|| 1_200), Cancellation::new()));
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
        block_on(runtime.run(initial, Arc::new(|| 1_201), Cancellation::new())).unwrap();
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
        let first = block_on(runtime.run(initial.clone(), Arc::new(|| 1_300), Cancellation::new()));
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
        block_on(restored.run(initial, Arc::new(|| 1_301), Cancellation::new())).unwrap();

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
        block_on(restored.run(
            Messages::default(),
            Arc::new(move || allocated_for_run.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        ))
        .unwrap();

        let execution = &restored.state().nodes["answer"].executions[&17_001];
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(execution.attempts[0].status, node::AttemptStatus::Waiting);
        assert_eq!(execution.attempts[1].status, node::AttemptStatus::Succeeded);
        assert_eq!(allocated.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn checkpoint_recovery_retries_waiting_agent_without_allocating_a_new_execution() {
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
        block_on(restored.run(
            user_input(),
            Arc::new(move || allocated_for_run.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        ))
        .unwrap();

        let execution = &restored.state().nodes["agent"].executions[&17_011];
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(execution.attempts.len(), 2);
        assert_eq!(execution.attempts[0].status, node::AttemptStatus::Waiting);
        assert_eq!(execution.attempts[1].status, node::AttemptStatus::Succeeded);
        assert_eq!(execution.attempts[1].steps.len(), 1);
        assert_eq!(allocated.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn checkpoint_recovery_does_not_replay_waiting_node_without_retry_budget() {
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
        let error = block_on(restored.run(
            Messages::default(),
            Arc::new(move || allocated_for_run.fetch_add(1, Ordering::SeqCst)),
            Cancellation::new(),
        ))
        .unwrap_err();

        assert_eq!(error.code, "RETRY_EXHAUSTED");
        let execution = &restored.state().nodes["answer"].executions[&17_021];
        assert_eq!(execution.status, node::ExecutionStatus::Failed);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].status, node::AttemptStatus::Waiting);
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
            block_on(runtime.run(Messages::default(), ids, cancellation))
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
        let first = block_on(runtime.run(
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
        let output = block_on(runtime.run(
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
        block_on(runtime.run(
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
        block_on(runtime.run(
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
        block_on(runtime.run(
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
        block_on(runtime.run(Messages::default(), Arc::new(|| 1_100), Cancellation::new()))
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
        block_on(runtime.run(
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
            Box::pin(runtime.run(Messages::default(), Arc::new(|| 900), Cancellation::new()));
        assert!(futures_util::poll!(run.as_mut()).is_pending());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        tokio::time::advance(std::time::Duration::from_millis(99)).await;
        assert!(futures_util::poll!(run.as_mut()).is_pending());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        tokio::time::advance(std::time::Duration::from_millis(1)).await;
        run.await.unwrap();
        let execution = &runtime.state.nodes["retry"].executions[&900];
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
            Box::pin(runtime.run(Messages::default(), Arc::new(|| 900), cancellation.clone()));
        assert!(futures_util::poll!(run.as_mut()).is_pending());
        cancellation.cancel();
        assert_eq!(run.await.unwrap_err().code, "CANCELLED");
        let execution = &runtime.state.nodes["retry"].executions[&900];
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
            block_on(runtime.run(Messages::default(), Arc::new(|| 900), Cancellation::new()))
                .unwrap_err();
        assert_eq!(error.code, "UNAVAILABLE");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let execution = &runtime.state.nodes["retry"].executions[&900];
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
        ));

        assert!(matches!(invocation.outcome, Outcome::Complete(_)));
        assert_eq!(first_calls.load(Ordering::SeqCst), 0);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
        assert_eq!(invocation.attempts[0].messages.as_slice().len(), 2);
    }
}
