//! Workflow execution engine and nested graph invocation.
//!
//! This module coordinates mutable Workflow execution. Retry, history, and
//! nested graph concerns live in focused child modules; the parent exposes only
//! the Runtime facade and graph contracts.

use super::*;

mod facts;
mod history;
mod retry;
mod subgraph;

use facts::{
    agent_scene_for_reply_path, attempt_feedback, attempt_waiting, commit_feedback,
    prepend_agent_observations, reply_duplicate,
};
pub(crate) use facts::{attempt_question, messages_to_agent_input};
pub use retry::invoke_with_retry;
use retry::{cancelled_observations, committed_tool_facts, terminal_observations};
use subgraph::outgoing;

pub(super) struct Engine {
    pub(super) workflow: CompiledWorkflow,
    pub(super) state: State,
    pub(super) max_concurrency: usize,
    pub(super) checkpoint: Option<CheckpointSave>,
    pub(super) memory: Option<History>,
    pub(super) history_execute: Option<Execute>,
    pub(super) history_closed: bool,
    // Derived from the parent's earlier Attempt.graph, not separate truth.
    // It supplies inherited scenes when a failed nested Graph Node is retried.
    pub(super) retry_source: Option<Box<graph::State>>,
    // Derived from the saved Agent Step and carried only until the existing
    // execution consumes it. This is transient routing, not saved Runtime state.
    pub(super) recovery_reply: Option<RecoveryReply>,
    // Saved replies are accepted only by a Runtime explicitly reconstructed
    // from Checkpoint, never by an abandoned live Runtime.
    pub(super) accepts_saved_replies: bool,
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    async fn accept_progress(
        &mut self,
        node_id: &str,
        exec_id: i64,
        mut attempt: node::Attempt,
        event: Option<crate::event::Event>,
        entered: bool,
        cancellation: &crate::Cancellation,
        base: Option<&node::Attempt>,
    ) -> Result<(), Error> {
        if entered {
            let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
            if !matches!(
                execution.status,
                node::status::execution::Status::Pending
                    | node::status::execution::Status::Running
                    | node::status::execution::Status::Waiting
            ) {
                return Err(Error::new(
                    "INVALID_STATE",
                    "admission target is no longer active",
                ));
            }
            let current = execution
                .attempts
                .last_mut()
                .filter(|current| current.idx == attempt.idx)
                .ok_or_else(|| Error::new("INVALID_STATE", "admission target Attempt changed"))?;
            execution.status = node::status::execution::Status::Running;
            current.status = node::status::attempt::Status::Running;
            // In-memory handoff only: neither a new Event nor a Checkpoint.
            return Box::pin(self.publish(None, false)).await;
        }
        let delta = matches!(&event, Some(crate::event::Event::Delta { .. }));
        let question = event.as_ref().and_then(|event| match event {
            crate::event::Event::Complete { message, .. } => {
                message.metadata.get("reply_id").and_then(Value::as_i64)
            }
            _ => None,
        });
        let feedback = if delta {
            false
        } else {
            let previous = self
                .state
                .nodes
                .get(node_id)
                .and_then(|node| node.executions.get(&exec_id))
                .and_then(|execution| execution.attempts.last());
            let before = previous.map(attempt_feedback).unwrap_or_default();
            attempt_feedback(&attempt)
                .iter()
                .any(|(id, count)| *count > before.get(id).copied().unwrap_or(0))
        };
        if let Some(base) = base.filter(|_| attempt.agent.is_some()) {
            prepend_agent_observations(&mut attempt, base);
        }
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
                node::status::execution::Status::Waiting
            } else {
                node::status::execution::Status::Running
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
        if !delta {
            wait::update(&mut self.state, node_id, exec_id)?;
        }
        if !delta && let Some(run) = events::interaction() {
            let attempt = self.state.nodes[node_id].executions[&exec_id]
                .attempts
                .last()
                .expect("committed Workflow progress");
            commit_feedback(attempt, &run)?;
        }
        if question.is_some() || feedback {
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
                    "Workflow cancelled while committing feedback",
                ));
            }
        }
        Box::pin(self.publish(event, delta)).await
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
                        node::status::attempt::Status::Waiting
                    } else {
                        node::status::attempt::Status::Running
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
            history_execute: None,
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
            .map_err(|error| Error::new("INVALID_STATE", error.to_string()))?;
        state.check_identity(chat_id)?;
        wait::validate(&state)?;
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
            history_execute: None,
            history_closed: false,
            retry_source: None,
            recovery_reply: None,
            accepts_saved_replies: true,
        };
        runtime.validate_tool_scenes()?;
        if runtime.state.mark_waiting() {
            runtime.state.status = graph::Status::Running;
        }
        runtime.state.recompute_usage();
        Ok(runtime)
    }

    pub fn definition(&self) -> &Definition {
        self.workflow.definition()
    }

    fn validate_tool_scenes(&self) -> Result<(), Error> {
        for (id, node) in &self.state.nodes {
            for execution in node.executions.values() {
                if let Some(child) = self.workflow.graph_node(id) {
                    for attempt in &execution.attempts {
                        if let Some(state) = &attempt.graph {
                            Self::from_snapshot(
                                self.state.chat_id,
                                child.clone(),
                                serde_json::to_value(state).map_err(|_| {
                                    Error::new("INVALID_STATE", "invalid child Tool scene")
                                })?,
                            )?;
                        }
                    }
                } else if execution
                    .attempts
                    .iter()
                    .any(|attempt| !attempt.steps.is_empty())
                    && self.workflow.agent(id).is_none()
                {
                    let executor = self
                        .workflow
                        .executor(id)
                        .ok_or_else(|| Error::new("INVALID_STATE", "ToolNode owner is missing"))?;
                    let input = self.input_for(id, &execution.from)?;
                    for attempt in &execution.attempts {
                        executor.validate_recovery(input.clone(), attempt)?;
                    }
                    executor.validate_recovery(input, &committed_tool_facts(execution))?;
                }
            }
        }
        Ok(())
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

    pub(super) fn reply_saved(
        &mut self,
        response: Messages,
        input: Input,
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
        if reply_duplicate(&self.state, &input)? {
            return Ok(());
        }
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
        if !crate::runtime::agent::reply::reply_duplicate(scene, &input)?
            && wait::ids(&self.state)?.first() != Some(&reply_id)
        {
            return Err(Error::new(
                "CONFLICT",
                "replies must be submitted in the saved Workflow question order",
            ));
        }
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
            .pause(&path[0].node_id)?
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
            .map_err(|error| Error::new("SERIALIZATION", error.to_string()))
    }

    pub(super) fn validate_run_input(&self, initial_messages: &Messages) -> Result<(), Error> {
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
    pub(super) async fn execute(
        &mut self,
        initial_messages: Messages,
        next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: crate::Cancellation,
    ) -> Result<Messages, Error> {
        admission::inherit(
            self.max_concurrency,
            Box::pin(self.execute_inner(initial_messages, next_exec_id, cancellation)),
        )
        .await
    }

    async fn execute_inner(
        &mut self,
        initial_messages: Messages,
        next_exec_id: Arc<dyn Fn() -> i64 + Send + Sync>,
        cancellation: crate::Cancellation,
    ) -> Result<Messages, Error> {
        self.validate_run_input(&initial_messages)?;
        // A terminal snapshot is already closed. Re-running it is a read of
        // the committed result and must not create another Chat record
        // or invoke any node again.
        if matches!(
            self.state.status,
            graph::Status::Succeeded | graph::Status::Failed | graph::Status::Cancelled
        ) {
            let result =
                Box::pin(self.run_inner(initial_messages, next_exec_id, cancellation)).await;
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
        let result = Box::pin(self.run_inner(initial_messages, next_exec_id, cancellation)).await;
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
                                    node::status::execution::Status::Pending
                                        | node::status::execution::Status::Running
                                        | node::status::execution::Status::Waiting
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

    /// Executes the compiled graph until it reaches END, waiting, failure or
    /// cancellation. The caller's ID source supplies one id for each new
    /// logical activation; retries keep the original id.
    pub(super) async fn run_inner(
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
                            .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?
                    }
                    Target::Graph { .. } => self.input_for(
                        &node_id,
                        &self.state.nodes[&node_id].executions[&exec_id].from,
                    )?,
                    Target::Node(_) => serde_json::to_value(response)
                        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?,
                };
                let retry = self.definition().nodes[&node_id].retry.clone();
                let start_idx = self.state.nodes[&node_id].executions[&exec_id]
                    .attempts
                    .last()
                    .map(|attempt| attempt.idx)
                    .ok_or_else(|| Error::new("INVALID_STATE", "execution has no attempt"))?;
                let base = self.state.nodes[&node_id].executions[&exec_id]
                    .attempts
                    .last()
                    .expect("validated Attempt")
                    .clone();
                let view = self.view_for(&node_id, exec_id, input.clone(), Some(start_idx))?;
                let invocation = events::with_source(
                    self.state.chat_id,
                    &node_id,
                    exec_id,
                    invoke_with_retry(
                        target,
                        input,
                        view,
                        cancellation.clone(),
                        retry,
                        next_exec_id.clone(),
                        self.state.chat_id,
                        start_idx,
                        committed_tool_facts(&self.state.nodes[&node_id].executions[&exec_id]),
                        None,
                    ),
                );
                let invocation =
                    self.continue_invocation(&node_id, invocation, &cancellation, &base);
                let invocation = Box::pin(invocation).await?;
                let invocation_graph = invocation.graph;
                self.replace_resumed_attempts(&node_id, exec_id, invocation.attempts, &base)?;
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
                    Outcome::Waiting => self.wait_call(&node_id, exec_id)?,
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
            if Box::pin(self.resume_waiting(&next_exec_id, &cancellation)).await? {
                // Reenter each saved invocation once per run. A method that
                // waits again must yield its scene, not spin on the same wait.
                if !self.state.advance_step() {
                    self.state.status = graph::Status::Paused;
                    return Err(Error::new(
                        "WAITING",
                        "workflow batch did not reach a barrier",
                    ));
                }
                continue;
            }
            let ready = self.ready_nodes();
            if ready.is_empty() {
                if !self.state.waits.is_empty() || !self.state.pauses().is_empty() {
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
            let specs = specs
                .into_iter()
                .map(|(node_id, exec_id, from, input, target, retry)| {
                    let view = self.view_for(&node_id, exec_id, input.clone(), Some(0))?;
                    Ok((node_id, exec_id, from, input, target, retry, view))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            for (node_id, exec_id, _, _, target, _, _) in &specs {
                if !matches!(target, Target::Graph { .. }) {
                    self.state
                        .node_mut(node_id)?
                        .execution_mut(*exec_id)?
                        .status = node::status::execution::Status::Pending;
                }
            }
            // Capture allocated child executions before polling their work,
            // including pending leaves. This is not a persistence trigger.
            Box::pin(self.publish(None, false)).await?;

            let task_cancellation = cancellation.clone();
            let task_next_exec_id = Arc::clone(&next_exec_id);
            let chat_id = self.state.chat_id;
            let relay = events::Relay::default();
            let slots = admission::slots();
            let changed = Arc::new(futures_util::task::AtomicWaker::new());
            let interaction = events::interaction();
            let mut active = Vec::new();
            let mut results = specs
                .into_iter()
                .map(|(node_id, exec_id, from, input, target, retry, view)| {
                    let cancellation = task_cancellation.clone();
                    let next_exec_id = Arc::clone(&task_next_exec_id);
                    let relay = relay.clone();
                    // Graphs are containers, not admitted work. Their leaves
                    // use this same outer semaphore, including after restore.
                    let slot = (!matches!(&target, Target::Graph { .. })).then(|| {
                        crate::runtime::agent::reply::Slot::waiting(slots.clone(), changed.clone())
                    });
                    let run = interaction.as_ref().map(|run| match &slot {
                        Some(slot) => run.branch(slot.clone()),
                        None => run.child(),
                    });
                    active.push((node_id.clone(), exec_id, run.clone(), slot.clone()));
                    async move {
                        let owner = run.as_ref().map(|run| run.owner());
                        let entering = match &run {
                            Some(run) => run.continue_execution(&cancellation).await,
                            None => match &slot {
                                Some(slot) => slot.acquire(&cancellation).await,
                                None => Ok(()),
                            },
                        };
                        if let Err(error) = entering {
                            return (
                                node_id,
                                exec_id,
                                from,
                                Invocation {
                                    attempts: vec![node::Attempt {
                                        status: node::status::attempt::Status::Cancelled,
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
                                        view,
                                        cancellation,
                                        retry,
                                        next_exec_id,
                                        chat_id,
                                        0,
                                        node::Attempt::default(),
                                        None,
                                    ),
                                ),
                            ),
                        )
                        .await;
                        drop(owner);
                        if let Some(slot) = slot {
                            slot.release();
                        }
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
                            .is_some_and(|attempt| {
                                attempt.status == node::status::attempt::Status::Waiting
                            })
                            && let Some(run) = run
                            && run.children_blocked()
                        {
                            run.suspend();
                        }
                    }
                    if active
                        .iter()
                        .all(|(_, _, _, slot)| slot.as_ref().is_none_or(|slot| !slot.held()))
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
                            entered,
                            ack,
                        } = *progress;
                        let result = Box::pin(self.accept_progress(
                            &node_id,
                            exec_id,
                            attempt,
                            event,
                            entered,
                            &cancellation,
                            None,
                        ))
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
                    Outcome::Waiting => self.wait_call(&node_id, exec_id)?,
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
            if self.state.waits.is_empty() && self.state.pauses().is_empty() {
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
                        && execution.status == node::status::execution::Status::Cancelled
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
                execution.status == node::status::execution::Status::Succeeded
                    && execution.to.iter().any(|target| target == graph::END)
            })
        })
    }

    fn end_messages(&self) -> Messages {
        let mut messages = Messages::default();
        for node_id in self.definition().nodes.keys() {
            if let Some(state) = self.state.nodes.get(node_id) {
                for execution in state.executions.values() {
                    if execution.status == node::status::execution::Status::Succeeded
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
                            node::status::execution::Status::Pending
                                | node::status::execution::Status::Running
                                | node::status::execution::Status::Waiting
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
                return serde_json::to_value(Input::default())
                    .map_err(|error| Error::new("SERIALIZATION", error.to_string()));
            }
            if self.workflow.input_is_messages(node_id) {
                return serde_json::to_value(Messages::default())
                    .map_err(|error| Error::new("SERIALIZATION", error.to_string()));
            }
            return Ok(Value::Object(serde_json::Map::new()));
        }
        if self.definition().nodes[node_id].agent {
            return serde_json::to_value(self.agent_input_for(node_id, from)?)
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()));
        }
        if !self.definition().nodes[node_id].attrs.input.is_empty() {
            return self.bound_values(node_id, from).map(Value::Object);
        }
        if from.is_empty() {
            if self.workflow.input_is_messages(node_id) {
                return serde_json::to_value(self.state.initial.as_ref().ok_or_else(|| {
                    Error::new("INVALID_STATE", "workflow initial Messages are missing")
                })?)
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()));
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
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()));
        }
        self.bound_values(node_id, from).map(Value::Object)
    }

    /// Snapshot only the current node's local execution facts for an
    /// invocation. This deliberately does not expose the owning `State` or
    /// graph mutators; the Node/Agent receives a clone through `View`.
    fn view_for(
        &self,
        node_id: &str,
        exec_id: i64,
        input: Value,
        attempt_idx: Option<u32>,
    ) -> Result<View, Error> {
        let execution = self
            .state
            .nodes
            .get(node_id)
            .and_then(|state| state.executions.get(&exec_id));
        let mut view = View {
            node_id: node_id.to_owned(),
            exec_id: Some(exec_id),
            attempt_idx,
            status: execution
                .map(|execution| {
                    serde_json::to_value(&execution.status)
                        .ok()
                        .and_then(|value| value.as_str().map(str::to_owned))
                        .unwrap_or_else(|| "pending".into())
                })
                .unwrap_or_else(|| "pending".into()),
            input,
            ..View::default()
        };
        if let Some(execution) = execution {
            view.result = (!execution.messages.is_empty())
                .then(|| serde_json::to_value(&execution.messages))
                .transpose()
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
            view.attempts = execution
                .attempts
                .iter()
                .map(|attempt| {
                    serde_json::to_value(attempt)
                        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))
                })
                .collect::<Result<_, _>>()?;
            view.error = execution
                .error
                .as_ref()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
            view.waiting = execution
                .waits
                .values()
                .map(|wait| {
                    serde_json::to_value(wait)
                        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))
                })
                .collect::<Result<_, _>>()?;
            view.usage = execution
                .usage
                .as_ref()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
            if let Some(attempt) = execution.attempts.last() {
                if let Some(scene) = &attempt.agent {
                    view.children.extend(
                        scene
                            .get("nodes")
                            .and_then(Value::as_object)
                            .into_iter()
                            .flat_map(|nodes| {
                                nodes.iter().map(|(child_id, state)| {
                                    serde_json::json!({
                                        "node_id": child_id,
                                        "status": state.get("status"),
                                        "messages": state.get("messages"),
                                        "waits": state.get("waits"),
                                        "usage": state.get("usage"),
                                        "error": state.get("error"),
                                    })
                                })
                            }),
                    );
                }
                if let Some(graph) = &attempt.graph {
                    let nodes = graph
                        .nodes
                        .iter()
                        .map(|(child_id, state)| {
                            serde_json::json!({
                                "node_id": child_id,
                                "status": state.status,
                                "executions": state.executions.len(),
                            })
                        })
                        .collect::<Vec<_>>();
                    view.children.push(serde_json::json!({
                        "type": "workflow",
                        "status": graph.status,
                        "step": graph.step,
                        "nodes": nodes,
                    }));
                }
            }
            for source in &execution.from {
                if let Some(source_execution) = self
                    .state
                    .nodes
                    .get(&source.node_id)
                    .and_then(|state| state.executions.get(&source.exec_id))
                {
                    let key = format!("{}#{}", source.node_id, source.exec_id);
                    view.sources.insert(
                        key,
                        serde_json::json!({
                            "node_id": source.node_id,
                            "exec_id": source.exec_id,
                            "status": source_execution.status,
                            "messages": source_execution.messages,
                        }),
                    );
                }
            }
            let mut bindings = serde_json::Map::new();
            for (target, binding) in &self.definition().nodes[node_id].attrs.input {
                let source = execution
                    .from
                    .iter()
                    .find(|source| source.node_id == binding.from.node_id);
                let value = source
                    .and_then(|source| {
                        self.state
                            .nodes
                            .get(&source.node_id)
                            .and_then(|state| state.executions.get(&source.exec_id))
                    })
                    .and_then(|source| {
                        source
                            .messages
                            .as_slice()
                            .iter()
                            .rev()
                            .find_map(|message| message.values.get(&binding.from.field))
                    })
                    .or_else(|| {
                        (binding.from.node_id == graph::START)
                            .then(|| {
                                self.state
                                    .initial
                                    .as_ref()?
                                    .as_slice()
                                    .iter()
                                    .rev()
                                    .find_map(|message| message.values.get(&binding.from.field))
                            })
                            .flatten()
                    })
                    .cloned()
                    .unwrap_or(Value::Null);
                bindings.insert(
                    target.clone(),
                    serde_json::json!({
                        "node_id": binding.from.node_id,
                        "field": binding.from.field,
                        "value": value,
                    }),
                );
            }
            if !bindings.is_empty() {
                view.sources
                    .insert("attrs.input".into(), Value::Object(bindings));
            }
        }
        Ok(view)
    }

    fn agent_input_for(&self, node_id: &str, from: &[ExecutionRef]) -> Result<Input, Error> {
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
            if source_id == graph::START {
                return self.state.initial.as_ref().ok_or_else(|| {
                    Error::new("INVALID_STATE", "workflow initial Messages are missing")
                });
            }
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
            if execution.status != node::status::execution::Status::Succeeded {
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
                                execution.status == node::status::execution::Status::Failed
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
        let mut recovery_reply = self.recovery_reply.take();
        if let Some(reply) = &recovery_reply {
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
                            node::status::execution::Status::Running
                                | node::status::execution::Status::Waiting
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
        }
        let targets = self
            .state
            .nodes
            .iter()
            .flat_map(|(node_id, node)| {
                node.executions
                    .values()
                    .filter(|execution| {
                        execution.idx == self.state.step
                            && execution.waits.is_empty()
                            && matches!(
                                execution.status,
                                node::status::execution::Status::Pending
                                    | node::status::execution::Status::Running
                                    | node::status::execution::Status::Waiting
                            )
                            && execution.attempts.last().is_some_and(|attempt| {
                                attempt.status == node::status::attempt::Status::Waiting
                            })
                    })
                    .map(move |execution| (node_id.clone(), execution.exec_id))
            })
            .collect::<Vec<_>>();
        if targets.is_empty() {
            return Ok(false);
        }
        let mut specs = Vec::new();
        let mut bases = std::collections::BTreeMap::new();
        for (node_id, exec_id) in targets {
            let execution = self.state.nodes[&node_id].executions[&exec_id].clone();
            let previous_idx = execution
                .attempts
                .last()
                .map(|attempt| attempt.idx)
                .ok_or_else(|| Error::new("INVALID_STATE", "waiting execution has no attempt"))?;
            let retry = self.definition().nodes[&node_id].retry.clone();
            let target = self.target_for(&node_id, exec_id)?;
            let mut input = self.input_for(&node_id, &execution.from)?;
            let mut nested_reply = None;
            if recovery_reply.as_ref().is_some_and(|reply| {
                reply
                    .path
                    .first()
                    .is_some_and(|target| target.node_id == node_id && target.exec_id == exec_id)
            }) {
                let mut reply = recovery_reply.take().expect("matched recovery reply");
                let target = reply.path.remove(0);
                if target.node_id != node_id || target.exec_id != exec_id {
                    return Err(Error::new("INVALID_STATE", "saved reply target changed"));
                }
                match self.target_for(&node_id, exec_id)? {
                    Target::Agent(_, _) if reply.path.is_empty() => {
                        let mut agent_input: Input = serde_json::from_value(input)
                            .map_err(|error| Error::new("INVALID_STATE", error.to_string()))?;
                        agent_input.payload = reply.input.payload;
                        agent_input.metadata.extend(reply.input.metadata);
                        input = serde_json::to_value(agent_input)
                            .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
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
            let base = execution
                .attempts
                .last()
                .expect("validated current Attempt")
                .clone();
            let view = self.view_for(&node_id, exec_id, input.clone(), Some(previous_idx))?;
            bases.insert((node_id.clone(), exec_id), base);
            specs.push((
                node_id,
                exec_id,
                target,
                input,
                retry,
                previous_idx,
                committed_tool_facts(&execution),
                nested_reply,
                view,
            ));
        }
        if recovery_reply.is_some() {
            return Err(Error::new(
                "INVALID_STATE",
                "saved reply has no active continuation",
            ));
        }
        // Rebuild all current consumers concurrently under the same admission
        // limit. Awaiting one Agent to completion first can strand a FIFO head
        // whose owner is another saved branch.
        let relay = events::Relay::default();
        let slots = admission::slots();
        let changed = Arc::new(futures_util::task::AtomicWaker::new());
        let interaction = events::interaction();
        let chat_id = self.state.chat_id;
        let mut active = Vec::new();
        let mut results = specs
            .into_iter()
            .map(
                |(node_id, exec_id, target, input, retry, idx, messages, reply, view)| {
                    let cancellation = cancellation.clone();
                    let next_exec_id = Arc::clone(next_exec_id);
                    let relay = relay.clone();
                    let slot = (!matches!(&target, Target::Graph { .. })).then(|| {
                        crate::runtime::agent::reply::Slot::waiting(slots.clone(), changed.clone())
                    });
                    let run = interaction.as_ref().map(|run| match &slot {
                        Some(slot) => run.branch(slot.clone()),
                        None => run.child(),
                    });
                    active.push((node_id.clone(), exec_id, run.clone(), slot.clone()));
                    async move {
                        let owner = run.as_ref().map(|run| run.owner());
                        let entering = match &run {
                            Some(run) => run.continue_execution(&cancellation).await,
                            None => match &slot {
                                Some(slot) => slot.acquire(&cancellation).await,
                                None => Ok(()),
                            },
                        };
                        let invocation = match entering {
                            Ok(()) => {
                                events::with_optional_interaction(
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
                                                view,
                                                cancellation,
                                                retry,
                                                next_exec_id,
                                                chat_id,
                                                idx,
                                                messages,
                                                reply,
                                            ),
                                        ),
                                    ),
                                )
                                .await
                            }
                            Err(error) => Invocation {
                                attempts: Vec::new(),
                                outcome: if error.code == "CANCELLED" {
                                    Outcome::Cancelled
                                } else {
                                    Outcome::Failed(error)
                                },
                                graph: None,
                            },
                        };
                        drop(owner);
                        if let Some(slot) = slot {
                            slot.release();
                        }
                        (node_id, exec_id, invocation)
                    }
                },
            )
            .collect::<futures_util::stream::FuturesUnordered<_>>();
        enum Next<T> {
            Facts(Box<events::Progress>),
            Finished(Box<T>),
        }
        loop {
            let next = std::future::poll_fn(|cx| {
                changed.register(cx.waker());
                match results.poll_next_unpin(cx) {
                    std::task::Poll::Ready(Some(result)) => {
                        return std::task::Poll::Ready(Some(Next::Finished(Box::new(result))));
                    }
                    std::task::Poll::Ready(None) => return std::task::Poll::Ready(None),
                    std::task::Poll::Pending => {}
                }
                if let std::task::Poll::Ready(facts) = relay.poll(cx) {
                    return std::task::Poll::Ready(Some(Next::Facts(Box::new(facts))));
                }
                for (node_id, exec_id, run, _) in &active {
                    if self.state.nodes[node_id].executions[exec_id]
                        .attempts
                        .last()
                        .is_some_and(|attempt| {
                            attempt.status == node::status::attempt::Status::Waiting
                        })
                        && let Some(run) = run
                        && run.children_blocked()
                    {
                        run.suspend();
                    }
                }
                if active
                    .iter()
                    .all(|(_, _, _, slot)| slot.as_ref().is_none_or(|slot| !slot.held()))
                    && let Some(run) = &interaction
                {
                    run.suspend();
                }
                std::task::Poll::Pending
            })
            .await;
            let Some(next) = next else { break };
            let (node_id, exec_id, invocation) = match next {
                Next::Facts(progress) => {
                    let events::Progress {
                        node_id,
                        exec_id,
                        attempt,
                        event,
                        entered,
                        ack,
                    } = *progress;
                    let result = Box::pin(self.accept_progress(
                        &node_id,
                        exec_id,
                        attempt,
                        event,
                        entered,
                        cancellation,
                        bases.get(&(node_id.clone(), exec_id)),
                    ))
                    .await;
                    let _ = ack.send(result.clone());
                    result?;
                    continue;
                }
                Next::Finished(result) => *result,
            };
            let base = bases
                .get(&(node_id.clone(), exec_id))
                .expect("saved continuation baseline");
            // Admission can be cancelled before invoking any component. Keep
            // the entire original scene; there is no replacement Attempt.
            if invocation.attempts.is_empty() {
                match invocation.outcome {
                    Outcome::Cancelled => self.state.cancel_execution(&node_id, exec_id)?,
                    Outcome::Failed(error) => {
                        self.state.fail_execution(&node_id, exec_id, error)?
                    }
                    _ => return Err(Error::new("INVALID_STATE", "continuation has no Attempt")),
                }
                self.state.recompute_usage();
                self.observe_commit(&node_id, exec_id).await?;
                continue;
            }
            let invocation_graph = invocation.graph;
            self.replace_resumed_attempts(&node_id, exec_id, invocation.attempts, base)?;
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
                Outcome::Waiting => self.wait_call(&node_id, exec_id)?,
                Outcome::Cancelled => self.state.cancel_execution(&node_id, exec_id)?,
                Outcome::Failed(error) => self.state.fail_execution(&node_id, exec_id, error)?,
            }
            self.state.recompute_usage();
            self.observe_commit(&node_id, exec_id).await?;
        }
        Ok(true)
    }

    /// Restored invocations need the same backpressured native handoff as
    /// fresh work. This is private execution plumbing, not a second run API.
    async fn continue_invocation<F: Future<Output = Invocation>>(
        &mut self,
        node_id: &str,
        invocation: F,
        cancellation: &crate::Cancellation,
        base: &node::Attempt,
    ) -> Result<Invocation, Error> {
        let relay = events::Relay::default();
        let changed = Arc::new(futures_util::task::AtomicWaker::new());
        let slot = self.workflow.graph_node(node_id).is_none().then(|| {
            crate::runtime::agent::reply::Slot::waiting(admission::slots(), changed.clone())
        });
        let run = events::interaction().map(|run| match &slot {
            Some(slot) => run.branch(slot.clone()),
            None => run.child(),
        });
        let _owner = run.as_ref().map(|run| run.owner());
        match &run {
            Some(run) => run.continue_execution(cancellation).await?,
            None => {
                if let Some(slot) = &slot {
                    slot.acquire(cancellation).await?;
                }
            }
        }
        let mut invocation = Box::pin(events::with_optional_interaction(
            run.clone(),
            events::with_relay(relay.clone(), invocation),
        ));
        loop {
            enum Next {
                Facts(Box<events::Progress>),
                Finished(Box<Invocation>),
            }
            let next = std::future::poll_fn(|cx| {
                changed.register(cx.waker());
                if let std::task::Poll::Ready(result) = invocation.as_mut().poll(cx) {
                    return std::task::Poll::Ready(Next::Finished(Box::new(result)));
                }
                if let std::task::Poll::Ready(facts) = relay.poll(cx) {
                    return std::task::Poll::Ready(Next::Facts(Box::new(facts)));
                }
                if self.state.nodes[node_id]
                    .executions
                    .values()
                    .any(|execution| {
                        execution.attempts.last().is_some_and(|attempt| {
                            attempt.status == node::status::attempt::Status::Waiting
                        })
                    })
                    && let Some(run) = &run
                    && run.children_blocked()
                {
                    run.suspend();
                }
                std::task::Poll::Pending
            })
            .await;
            match next {
                Next::Finished(result) => {
                    if let Some(slot) = &slot {
                        slot.release();
                    }
                    return Ok(*result);
                }
                Next::Facts(facts) => {
                    let events::Progress {
                        node_id,
                        exec_id,
                        attempt,
                        event,
                        entered,
                        ack,
                    } = *facts;
                    let result = Box::pin(self.accept_progress(
                        &node_id,
                        exec_id,
                        attempt,
                        event,
                        entered,
                        cancellation,
                        Some(base),
                    ))
                    .await;
                    let _ = ack.send(result.clone());
                    result?;
                }
            }
        }
    }

    async fn observe_commit(&self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        let definition = &self.definition().nodes[node_id];
        if definition.agent || definition.workflow.is_some() {
            // Agent calls already forward their real stream, and Graph Node
            // containers must not duplicate the child's observations.
            // Their final committed status still belongs in the parent scene,
            // including while a sibling keeps this graph waiting.
            Box::pin(self.publish(None, false)).await?;
            return Ok(());
        }
        let execution = &self.state.nodes[node_id].executions[&exec_id];
        let messages = if execution.status == node::status::execution::Status::Succeeded {
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
            Box::pin(self.publish(Some(event), false)).await?;
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

    // A Tool lifecycle wait has no fabricated human question or response.
    fn wait_call(&mut self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        self.state.node_mut(node_id)?.execution_mut(exec_id)?.status =
            node::status::execution::Status::Waiting;
        Ok(())
    }

    fn replace_resumed_attempts(
        &mut self,
        node_id: &str,
        exec_id: i64,
        mut attempts: Vec<node::Attempt>,
        base: &node::Attempt,
    ) -> Result<(), Error> {
        let execution = self.state.node_mut(node_id)?.execution_mut(exec_id)?;
        // Native progress may already have appended a retry. Replace only
        // the suffix that this continuation owns, using its original base,
        // not the now-advanced last Attempt. Earlier history stays immutable.
        let previous = base.idx as usize;
        if execution
            .attempts
            .get(previous)
            .is_none_or(|attempt| attempt.idx != base.idx)
            || attempts.is_empty()
            || attempts
                .iter()
                .enumerate()
                .any(|(offset, attempt)| attempt.idx as usize != previous + offset)
        {
            return Err(Error::new(
                "INVALID_STATE",
                "invalid resumed attempt history",
            ));
        }
        let original = base;
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
            prepend_agent_observations(&mut attempts[0], original);
        }
        execution.attempts.truncate(previous);
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
#[doc(hidden)]
pub struct Invocation {
    pub attempts: Vec<node::Attempt>,
    pub outcome: Outcome,
    pub graph: Option<Box<graph::State>>,
}

#[derive(Debug)]
#[doc(hidden)]
pub enum Outcome {
    Complete(Value),
    Wait(String),
    Waiting,
    Failed(Error),
    Cancelled,
}

#[doc(hidden)]
pub enum Target {
    Node(Arc<dyn graph::ErasedNode + 'static>),
    Agent(Arc<dyn graph::ErasedAgent + 'static>, Option<Value>),
    Graph {
        workflow: Arc<CompiledWorkflow>,
        state: Option<Box<graph::State>>,
        history: Option<Box<graph::State>>,
    },
}
