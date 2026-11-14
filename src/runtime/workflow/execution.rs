//! Workflow execution engine and nested graph invocation.
//!
//! This module coordinates mutable Workflow execution. Retry, history, and
//! nested graph concerns live in focused child modules; the parent exposes only
//! the Runtime facade and graph contracts.

use super::*;

mod facts;
mod history;
mod input;
mod retry;
mod run;
mod subgraph;
mod types;

use facts::{
    agent_scene_for_reply_path, attempt_feedback, attempt_waiting, commit_feedback,
    prepend_agent_observations, reply_duplicate,
};
pub(crate) use facts::{attempt_question, messages_to_agent_input};
pub(crate) use retry::invoke_with_retry;
use retry::{cancelled_observations, committed_tool_facts, terminal_observations};
use subgraph::outgoing;
pub(crate) use types::{Invocation, Outcome, Target};

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
    pub(super) async fn accept_progress(
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
    pub(super) async fn observe_commit(&self, node_id: &str, exec_id: i64) -> Result<(), Error> {
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

    pub(super) fn target_for(&self, node_id: &str, exec_id: i64) -> Result<Target, Error> {
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

    pub(super) fn inherited_graph(&self, node_id: &str) -> Option<Box<graph::State>> {
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

    pub(super) fn replace_attempts(
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
    pub(super) fn wait_call(&mut self, node_id: &str, exec_id: i64) -> Result<(), Error> {
        self.state.node_mut(node_id)?.execution_mut(exec_id)?.status =
            node::status::execution::Status::Waiting;
        Ok(())
    }

    pub(super) fn replace_resumed_attempts(
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

    pub(super) fn attach_graph(
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
