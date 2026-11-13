use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::task::Poll;

use futures_core::Stream;
use futures_util::{
    StreamExt,
    future::poll_fn,
    stream::{FuturesUnordered, unfold},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::Cancellation;
use crate::cancellation::cancellable;
use crate::components::agent::{Error, Input};
use crate::components::evaluation::{Evaluation, Evaluator};
use crate::components::event::Event;
use crate::components::message::{Message, Messages, Role};
use crate::components::model::chat::{Chat, Request, ToolChoice};
use crate::components::model::content::part::{Part, object as part_data};
use crate::components::tool::{Call, Evaluation as ToolEvaluation, ToolDefinition, validate_call};
use crate::components::{mcp::Server, skill::Skill};
use crate::core::dispatch::{Dispatch, ToolInputs};
use crate::runtime::agent::access::Access;
use crate::runtime::agent::checkpoint::{Backend, invalid as invalid_checkpoint};
use crate::runtime::agent::graph::Graph;
use crate::runtime::agent::history::{History, HistoryRun};
use crate::runtime::agent::hooks::Hooks;
use crate::runtime::agent::node::Step;
use crate::runtime::agent::subagent::SubAgent;
use crate::runtime::agent::subagent::{Failure as SubAgentFailure, Progress as SubAgentProgress};
use crate::runtime::agent::{State as AgentState, graph};
use crate::runtime::trace::fastrace::{Root as TraceRoot, SpanScope};

mod execution;
mod tools;
use super::lifecycle::Failure;
use execution::TaskProgress;

type EvaluationFuture = Pin<Box<dyn Future<Output = Result<Evaluation, Error>> + Send + 'static>>;
type EvaluatorHandle =
    std::sync::Arc<dyn Fn(Input, Messages, Cancellation) -> EvaluationFuture + Send + Sync>;
type CheckpointSave = Backend;
type ProgressFuture<M> = Pin<
    Box<
        dyn Future<Output = Option<(Option<Result<Event, Error>>, ReActState<M>)>> + Send + 'static,
    >,
>;

fn plan_definition() -> ToolDefinition {
    ToolDefinition {
        name: "plan".into(),
        description: "Create or update a complete task plan for the original user query. Choose this when the task or system instructions call for planning. The plan describes tasks; the Agent decides how to use its configured Tools, MCP servers and Skills while executing them. This creates a plan, not execution results.".into(),
        parameters: serde_json::from_value(json!({"type":"object","properties":{},"additionalProperties":false})).expect("static plan Schema is an object"),
    }
}

/// Public builder for the first runnable ReAct execution loop.
pub struct Builder<M = ()> {
    model: Option<M>,
    system_prompt: Option<String>,
    dispatch: Dispatch,
    subagents: Vec<SubAgent>,
    handoff_targets: Vec<String>,
    max_iters: usize,
    max_concurrency: usize,
    hooks: Hooks,
    access: Access,
    history: History,
    id_fn: Option<fn() -> i64>,
    evaluator: Option<EvaluatorHandle>,
    checkpoint: Option<CheckpointSave>,
    team_mode: bool,
}

pub fn builder() -> Builder {
    Builder {
        model: None,
        system_prompt: None,
        dispatch: Dispatch::new(),
        subagents: Vec::new(),
        handoff_targets: Vec::new(),
        max_iters: 20,
        max_concurrency: 100,
        hooks: Hooks::default(),
        access: Access::default(),
        history: History::default(),
        id_fn: None,
        evaluator: None,
        checkpoint: None,
        team_mode: false,
    }
}

impl<M> Builder<M> {
    pub fn model<N>(self, model: N) -> Builder<N> {
        Builder {
            model: Some(model),
            system_prompt: self.system_prompt,
            dispatch: self.dispatch,
            subagents: self.subagents,
            handoff_targets: self.handoff_targets,
            max_iters: self.max_iters,
            max_concurrency: self.max_concurrency,
            hooks: self.hooks,
            access: self.access,
            history: self.history,
            id_fn: self.id_fn,
            evaluator: self.evaluator,
            checkpoint: self.checkpoint,
            team_mode: self.team_mode,
        }
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub(crate) fn with_dispatch(mut self, dispatch: Dispatch) -> Self {
        self.dispatch = dispatch;
        self
    }

    pub(crate) fn with_team_members(mut self, subagents: Vec<SubAgent>) -> Self {
        self.subagents = subagents;
        self.team_mode = true;
        self
    }

    pub(crate) fn with_team_handoffs(mut self, targets: Vec<String>) -> Self {
        self.handoff_targets = targets;
        self
    }

    /// Register application Tools directly on this Agent.
    #[allow(private_bounds)]
    pub fn tools<I>(mut self, tools: I) -> Self
    where
        I: ToolInputs,
    {
        self.dispatch = self.dispatch.tools(tools);
        self
    }

    /// Register already connected and discovered MCP servers on this Agent.
    #[allow(private_bounds)]
    pub fn mcps<S>(mut self, servers: impl IntoIterator<Item = S>) -> Self
    where
        S: Server + 'static,
    {
        self.dispatch = self.dispatch.mcps(servers);
        self
    }

    /// Register in-process Skills on this Agent.
    #[allow(private_bounds)]
    pub fn skills<S>(mut self, skills: impl IntoIterator<Item = S>) -> Result<Self, Error>
    where
        S: Skill + 'static,
    {
        self.dispatch = self.dispatch.skills(skills)?;
        Ok(self)
    }

    /// Register the heterogeneous Agent set available to the built-in
    /// `agent_spawn` delegation path. SubAgents are not tool dispatch entries;
    /// Team adds its private `team_assign`/`agent_handoff` controls separately.
    pub fn subagents(mut self, subagents: impl IntoIterator<Item = SubAgent>) -> Self {
        self.subagents.extend(subagents);
        self
    }

    /// Attach an Agent-local middleware; it is not a Tool lifecycle hook.
    pub fn middleware<H: crate::middleware::Middleware + 'static>(mut self, middleware: H) -> Self {
        self.hooks.add(middleware);
        self
    }

    /// Configure the policy checked before local or discovered MCP Tool calls.
    /// An absent policy does not activate permission middleware stages.
    pub fn permission<P: crate::permission::Permission + 'static>(mut self, permission: P) -> Self {
        self.access.set(permission);
        self
    }

    /// Attach an optional Agent-level business evaluator. It is called only
    /// for a model response that contains no tool calls; Tool::eval remains a
    /// separate lifecycle. With an executing Plan and an application ID
    /// function, rejection invokes the built-in Planner and executes its
    /// complete replacement; otherwise feedback goes to the next ordinary
    /// model request.
    pub fn evaluator<E: Evaluator + 'static>(mut self, evaluator: E) -> Self {
        let evaluator = std::sync::Arc::new(evaluator);
        self.evaluator = Some(std::sync::Arc::new(move |input, result, cancellation| {
            let evaluator = evaluator.clone();
            Box::pin(async move { evaluator.evaluate(&input, &result, &cancellation).await })
        }));
        self
    }

    /// Attach the Agent checkpoint backend. A configured backend is written
    /// for the Agent runtime whether or not this run uses an execution graph.
    /// Graph fields are included only when a plan has been accepted; an
    /// ordinary run still persists its messages, usage and terminal status.
    pub fn checkpoint<C: crate::checkpoint::Checkpoint + 'static>(mut self, checkpoint: C) -> Self {
        self.checkpoint = Some(Backend::new(checkpoint));
        self
    }

    pub fn memory<S, L>(mut self, config: crate::memory::Config<S, L>) -> Self
    where
        S: crate::memory::short::Store + 'static,
        L: crate::memory::Memory + 'static,
    {
        self.history.set(config);
        self
    }

    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Supply application-generated execution/question IDs independently of Memory.
    /// Configuration and build do not invoke this function. The application
    /// owns uniqueness across concurrent runs and restored executions.
    ///
    /// A fixed ID is not a function:
    /// ```compile_fail
    /// use halo::agent;
    /// agent::react().with_id_fn(123_i64);
    /// ```
    pub fn with_id_fn(mut self, id_fn: fn() -> i64) -> Self {
        self.id_fn = Some(id_fn);
        self
    }

    /// Bound concurrent Tool lifecycles within each model-selected batch.
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }
}

impl<M> Builder<M>
where
    M: Chat,
{
    pub fn build(self) -> Result<Agent<M>, Error> {
        let model = self
            .model
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "react agent requires a model"))?;
        if self.max_iters == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "react max_iters must be positive",
            ));
        }
        if self.max_concurrency == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "react max_concurrency must be positive",
            ));
        }
        self.history.validate()?;
        if self.dispatch.names().contains("plan") {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "plan is reserved for the built-in Planner",
            ));
        }
        validate_subagents(&self.subagents)?;
        Ok(Agent {
            replies: Default::default(),
            model: std::sync::Arc::new(model),
            system_prompt: self.system_prompt,
            dispatch: std::sync::Arc::new(self.dispatch),
            subagents: std::sync::Arc::new(self.subagents),
            handoff_targets: std::sync::Arc::new(self.handoff_targets),
            max_iters: self.max_iters,
            max_concurrency: self.max_concurrency,
            hooks: self.hooks,
            access: self.access,
            history: self.history,
            id_fn: self.id_fn,
            evaluator: self.evaluator,
            checkpoint: self.checkpoint,
            team_mode: self.team_mode,
        })
    }
}

pub struct Agent<M> {
    replies: crate::runtime::agent::reply::Hub,
    model: std::sync::Arc<M>,
    system_prompt: Option<String>,
    dispatch: std::sync::Arc<Dispatch>,
    subagents: std::sync::Arc<Vec<SubAgent>>,
    handoff_targets: std::sync::Arc<Vec<String>>,
    max_iters: usize,
    max_concurrency: usize,
    hooks: Hooks,
    access: Access,
    history: History,
    id_fn: Option<fn() -> i64>,
    evaluator: Option<EvaluatorHandle>,
    checkpoint: Option<CheckpointSave>,
    team_mode: bool,
}

fn validate_subagents(subagents: &[SubAgent]) -> Result<(), Error> {
    let mut names = BTreeSet::new();
    for subagent in subagents {
        if !names.insert(subagent.name()) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "subagent names must be unique",
            ));
        }
    }
    Ok(())
}

impl<M> Agent<M> {
    pub fn max_iters(&self) -> usize {
        self.max_iters
    }

    pub fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }
}

impl<M> crate::components::agent::Agent for Agent<M>
where
    M: Chat + 'static,
{
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send + 'static>>;

    async fn reply(&self, input: Input, cancellation: Cancellation) -> Result<(), Error> {
        self.replies.submit(input, &cancellation)
    }

    async fn run(
        &self,
        input: Input,
        mut stream: bool,
        cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "react agent run cancelled"));
        }
        let embedded = crate::runtime::agent::scope::active();
        let embedded_restoring = embedded
            && crate::runtime::agent::scope::interaction().is_some_and(|run| run.is_restoring());
        if !embedded {
            self.replies.ensure_idle(&input.metadata)?;
        }
        let checkpoint = self.checkpoint.clone().filter(|_| !embedded);
        let history = if embedded {
            History::default()
        } else {
            self.history.clone()
        };
        let checkpoint_id = checkpoint
            .as_ref()
            .map(|_| checkpoint_id(&input.metadata))
            .transpose()?;
        let restored = match crate::runtime::agent::scope::restore() {
            Some(snapshot) => Some(snapshot),
            None => match (&checkpoint, checkpoint_id) {
                (Some(checkpoint), Some(id)) => checkpoint.load(id, cancellation.clone()).await?,
                _ => None,
            },
        };
        let restoring_waiting = restored.as_ref().is_some_and(is_waiting_snapshot);
        let external_wait = restored.as_ref().is_some_and(is_external_wait_snapshot);
        let mut restored_approvals = None;
        if let Some(snapshot) = &restored {
            if !checkpoint_identity_matches(snapshot, &input.metadata) {
                return Err(Error::new(
                    "CONFLICT",
                    "checkpoint belongs to a different Agent identity",
                ));
            }
            if !restoring_waiting
                && snapshot
                    .get("query")
                    .is_some_and(|query| serde_json::to_value(&input).ok().as_ref() != Some(query))
                && !crate::runtime::agent::reply::reply_duplicate(snapshot, &input)?
            {
                return Err(Error::new(
                    "CONFLICT",
                    "checkpoint belongs to a different Agent input",
                ));
            }
            if embedded {
                if embedded_restoring {
                    crate::runtime::agent::wait::validate(snapshot)?;
                } else {
                    crate::runtime::agent::checkpoint::check_waits(snapshot)?;
                }
            } else {
                restored_approvals =
                    crate::runtime::agent::reply::prepare_restore(snapshot, &input)?;
            }
            if let Some(result) = crate::runtime::agent::checkpoint::terminal(snapshot)? {
                let mut result = result;
                if let Ok(event) = &mut result {
                    crate::runtime::observation::associate(
                        event,
                        input.metadata.get("chat_id").and_then(Value::as_i64),
                        None,
                    );
                }
                return Ok(Box::pin(futures_util::stream::iter([result])));
            }
            stream = crate::runtime::agent::checkpoint::stream_mode(snapshot)?;
            crate::runtime::agent::model_steps::validate_scene(snapshot, self.hooks.len())?;
            super::access::validate_scene(snapshot, self.access.is_configured())?;
        }
        let original = restored
            .as_ref()
            .and_then(|snapshot| snapshot.get("query"))
            .map(|value| {
                serde_json::from_value(value.clone())
                    .map_err(|_| invalid_checkpoint("invalid original Agent query"))
            })
            .transpose()?
            .unwrap_or_else(|| input.clone());
        let discovered_dispatch = self.dispatch.with_discovered_skills(&cancellation).await;
        let mut preparation = match &restored {
            Some(snapshot) => crate::runtime::agent::preparation::State::restore(snapshot)?,
            None => Some(crate::runtime::agent::preparation::State::new(
                original.clone(),
                discovered_dispatch.as_ref().map_or_else(
                    |_| self.system_prompt.clone(),
                    |dispatch| dispatch.system_prompt(self.system_prompt.clone()),
                ),
            )),
        };
        if let Some(state) = &preparation {
            state.validate(self.hooks.len(), &original)?;
        }
        let interaction = if embedded {
            Some(match crate::runtime::agent::scope::interaction() {
                Some(run) => run,
                None => self.replies.start(original.metadata.clone())?,
            })
        } else {
            Some(self.replies.start(original.metadata.clone())?)
        };
        let targeted_restore = restored_approvals.is_some() || embedded_restoring;
        if let Some(restore) = restored_approvals {
            interaction
                .as_ref()
                .expect("root restore interaction")
                .restore_approvals(restore);
        }
        let preparing = preparation.as_ref().is_some_and(|state| {
            state.stage != crate::runtime::agent::preparation::Stage::Complete
        });
        let mut run = HistoryRun::default();
        let mut messages = Messages::default();
        let prepared = async {
            if let Some(state) = preparation.as_mut() {
                Box::pin(super::preparation::advance(
                    state,
                    &original,
                    &mut messages,
                    &mut run,
                    &history,
                    &self.hooks,
                    &cancellation,
                ))
                .await?;
            } else {
                run = history.resume(original.clone(), &cancellation).await?;
            }
            let dispatch = discovered_dispatch
                .as_ref()
                .map_err(Clone::clone)?
                .prepared(&cancellation)
                .await?;
            if dispatch.names().contains("plan") {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "plan is reserved for the built-in Planner",
                ));
            }
            Ok::<_, Error>(dispatch)
        }
        .await;
        let (dispatch, preparation_error) = match prepared {
            Ok(dispatch) => (dispatch, None),
            Err(error) => (Dispatch::new(), Some(error)),
        };
        let prepared_messages = messages.clone();
        let metadata = preparation
            .as_ref()
            .map(|state| state.input.metadata.clone())
            .unwrap_or_else(|| original.metadata.clone());
        let mut state = ReActState {
            completion: None,
            restored_output: None,
            replayed_completion: false,
            preparation,
            _interaction_owner: interaction
                .as_ref()
                .map(crate::runtime::agent::reply::Run::owner),
            interaction,
            waits: Vec::new(),
            stream,
            model: std::sync::Arc::clone(&self.model),
            dispatch: std::sync::Arc::new(dispatch),
            subagents: self.subagents.clone(),
            handoff_targets: self.handoff_targets.clone(),
            team_mode: self.team_mode,
            max_iters: self.max_iters,
            max_concurrency: self.max_concurrency,
            cancellation,
            task_base: messages.clone(),
            messages,
            hooks: self.hooks.clone(),
            access: self.access.clone(),
            evaluator: self.evaluator.clone(),
            metadata,
            iteration: Arc::new(AtomicUsize::new(0)),
            provider: None,
            query: original,
            graph: None,
            resume_nodes: BTreeSet::new(),
            external_resume_nodes: BTreeSet::new(),
            // Retain assembly configuration in this run, not in Input,
            // Messages, Plan or a serialized Graph. Plan execution is enabled
            // only when the caller supplied an ID function; a planning
            // confirmation or ordinary model/Tool call never allocates one.
            id_fn: crate::runtime::agent::scope::id_fn().or(self.id_fn),
            checkpoint,
            checkpoint_id,
            plan_execution: false,
            active_tasks: FuturesUnordered::new(),
            task_runs: BTreeMap::new(),
            task_slots: Arc::new(tokio::sync::Semaphore::new(self.max_concurrency)),
            task_ready: Arc::new(futures_util::task::AtomicWaker::new()),
            task_error: None,
            dispatch_failed: false,
            is_plan_task: false,
            tool_slots: Arc::new(tokio::sync::Semaphore::new(self.max_concurrency)),
            plan_finalizing: false,
            pending_plans: VecDeque::new(),
            planning: None,
            pending_replan: None,
            plan_result: None,
            planning_error: preparation_error,
            resume_input: None,
            pending_calls: VecDeque::new(),
            active_tools: FuturesUnordered::new(),
            tool_progress: VecDeque::new(),
            child_events: VecDeque::new(),
            child_relay: crate::runtime::agent::subagent::Relay::default(),
            pending_subagents: BTreeMap::new(),
            tool_results: BTreeMap::new(),
            evaluations: BTreeMap::new(),
            steps: Vec::new(),
            child_nodes: BTreeMap::new(),
            child_nodes_dirty: false,
            feedback_dirty: false,
            phase_progress: false,
            root: crate::runtime::agent::subagent::Root::default(),
            partial: Message::new(Role::Assistant),
            done: false,
            run,
            model_pending: false,
            model_settled: false,
            input_tokens: 0,
            request: None,
            trace: TraceRoot::start("agent.react"),
            model_span: None,
        };
        // Record the original public input once. The Agent graph and
        // checkpoint continue to own mutable execution state.
        state.trace.property_json("agent.input", &state.query);
        if let Some(chat_id) = state.metadata.get("chat_id").and_then(Value::as_i64) {
            state.trace.property("chat_id", chat_id.to_string());
        }
        if let Some(snapshot) = restored {
            restore_state(&mut state, snapshot)?;
            if preparing {
                state.messages = prepared_messages.clone();
                state.task_base = prepared_messages;
            }
            if external_wait && !targeted_restore && !preparing {
                // Until the saved question-to-Step association is available,
                // reject ambiguous feedback rather than append it to every
                // waiting task. Match fill_ready's possible input consumers;
                // saved Tool phases keep their existing continuation path.
                let ambiguous = state.graph.as_ref().is_some_and(|graph| {
                    state
                        .external_resume_nodes
                        .iter()
                        .filter(|node_id| {
                            !graph.nodes[*node_id]
                                .execution
                                .as_ref()
                                .and_then(|execution| execution.attempts.last())
                                .is_some_and(|attempt| {
                                    tools::pending(&attempt.steps)
                                        || crate::runtime::agent::model_steps::continuing(
                                            &attempt.steps,
                                        )
                                        || attempt.completion.as_ref().is_some_and(
                                            crate::runtime::agent::completion::State::pending,
                                        )
                                })
                        })
                        .take(2)
                        .count()
                        > 1
                });
                if ambiguous {
                    return Err(Error::new(
                        "CONFLICT",
                        "feedback has multiple possible waiting consumers",
                    ));
                }
                state.resume_input = Some(input);
                if state.graph.is_none()
                    && let Some(resume_input) = state.resume_input.clone()
                {
                    let message = crate::components::agent_input::user_message(resume_input);
                    state.messages.push(message.clone());
                    state.task_base.push(message);
                }
            }
        }
        if state
            .planning_error
            .as_ref()
            .is_some_and(|error| !matches!(error.code.as_str(), "WAITING" | "CANCELLED"))
        {
            // Preserve direct preparation errors, but close the same native
            // scene/Chat.state as an error emitted later from the run stream.
            let (result, _) = Box::pin(next_react_event(state))
                .await
                .expect("preparation error closeout");
            return Err(result.expect_err("preparation must not execute a Model"));
        }
        let chat_id = state.query.metadata.get("chat_id").and_then(Value::as_i64);
        let trace = state.trace.clone();
        Ok(Box::pin(
            unfold(state, next_react_event::<M>)
                .filter(move |result| {
                    std::future::ready(stream || !matches!(result, Ok(Event::Delta { .. })))
                })
                .map(move |mut result| {
                    if let Ok(event) = &mut result {
                        crate::runtime::observation::associate(event, chat_id, None);
                    }
                    crate::runtime::observation::observe_trace(&trace, &result);
                    result
                }),
        ))
    }
}

mod runner;
pub(crate) use runner::*;

#[cfg(test)]
#[path = "../../../tests/unit/agent_progress.rs"]
mod progress_tests;

#[cfg(test)]
#[path = "../../../tests/unit/agent_react.rs"]
mod tests;
