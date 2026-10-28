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
use crate::runtime::agent::history::{History, Run};
use crate::runtime::agent::hooks::{Hooks, cancellable};
use crate::runtime::agent::node::Step;
use crate::runtime::agent::subagent::SubAgent;
use crate::runtime::agent::subagent::{Failure as SubAgentFailure, Progress as SubAgentProgress};
use crate::runtime::agent::{State as AgentState, graph};
use crate::runtime::trace::fastrace::{Root as TraceRoot, SpanScope};

mod execution;
mod tools;
use super::lifecycle::Failure;
use execution::TaskProgress;

type EvaluationFuture = Pin<Box<dyn Future<Output = Result<Evaluation, Error>> + 'static>>;
type EvaluatorHandle =
    std::sync::Arc<dyn Fn(Input, Messages, Cancellation) -> EvaluationFuture + Send + Sync>;
type CheckpointSave = Backend;
type ProgressFuture<M> =
    Pin<Box<dyn Future<Output = Option<(Option<Result<Event, Error>>, ReActState<M>)>> + 'static>>;

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
    max_iters: usize,
    max_concurrency: usize,
    hooks: Hooks,
    access: Access,
    history: History,
    id_fn: Option<fn() -> i64>,
    evaluator: Option<EvaluatorHandle>,
    checkpoint: Option<CheckpointSave>,
}

pub fn builder() -> Builder {
    Builder {
        model: None,
        system_prompt: None,
        dispatch: Dispatch::new(),
        subagents: Vec::new(),
        max_iters: 20,
        max_concurrency: 100,
        hooks: Hooks::default(),
        access: Access::default(),
        history: History::default(),
        id_fn: None,
        evaluator: None,
        checkpoint: None,
    }
}

impl<M> Builder<M> {
    pub fn model<N>(self, model: N) -> Builder<N> {
        Builder {
            model: Some(model),
            system_prompt: self.system_prompt,
            dispatch: self.dispatch,
            subagents: self.subagents,
            max_iters: self.max_iters,
            max_concurrency: self.max_concurrency,
            hooks: self.hooks,
            access: self.access,
            history: self.history,
            id_fn: self.id_fn,
            evaluator: self.evaluator,
            checkpoint: self.checkpoint,
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
    /// `agent_spawn` delegation path. SubAgents are not tool dispatch entries.
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
            max_iters: self.max_iters,
            max_concurrency: self.max_concurrency,
            hooks: self.hooks,
            access: self.access,
            history: self.history,
            id_fn: self.id_fn,
            evaluator: self.evaluator,
            checkpoint: self.checkpoint,
        })
    }
}

pub struct Agent<M> {
    replies: crate::runtime::agent::reply::Hub,
    model: std::sync::Arc<M>,
    system_prompt: Option<String>,
    dispatch: std::sync::Arc<Dispatch>,
    subagents: std::sync::Arc<Vec<SubAgent>>,
    max_iters: usize,
    max_concurrency: usize,
    hooks: Hooks,
    access: Access,
    history: History,
    id_fn: Option<fn() -> i64>,
    evaluator: Option<EvaluatorHandle>,
    checkpoint: Option<CheckpointSave>,
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
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + 'static>>;

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
                && !crate::runtime::agent::reply::saved_duplicate(snapshot, &input)?
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
        let mut run = Run::default();
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

struct ReActState<M: Chat> {
    completion: Option<crate::runtime::agent::completion::State>,
    restored_output: Option<Event>,
    replayed_completion: bool,
    preparation: Option<crate::runtime::agent::preparation::State>,
    _interaction_owner: Option<crate::runtime::agent::reply::Owner>,
    interaction: Option<crate::runtime::agent::reply::Run>,
    waits: Vec<crate::runtime::agent::wait::Wait>,
    stream: bool,
    model: std::sync::Arc<M>,
    dispatch: std::sync::Arc<Dispatch>,
    subagents: std::sync::Arc<Vec<SubAgent>>,
    max_iters: usize,
    max_concurrency: usize,
    cancellation: Cancellation,
    messages: Messages,
    // Each planned task has private context but shares this run's model-call
    // budget. Atomic reservation keeps the budget safe across worker threads.
    iteration: Arc<AtomicUsize>,
    task_base: Messages,
    provider: Option<Pin<Box<M::Stream>>>,
    query: Input,
    graph: Option<Graph>,
    resume_nodes: BTreeSet<String>,
    external_resume_nodes: BTreeSet<String>,
    id_fn: Option<fn() -> i64>,
    checkpoint: Option<CheckpointSave>,
    checkpoint_id: Option<i64>,
    plan_execution: bool,
    active_tasks: FuturesUnordered<Pin<Box<dyn Future<Output = TaskProgress<M>> + 'static>>>,
    /// Handles to the actual task accounting, never another serialized scene.
    task_runs: BTreeMap<String, execution::Active>,
    task_slots: Arc<tokio::sync::Semaphore>,
    task_ready: Arc<futures_util::task::AtomicWaker>,
    task_error: Option<Error>,
    dispatch_failed: bool,
    is_plan_task: bool,
    tool_slots: Arc<tokio::sync::Semaphore>,
    plan_finalizing: bool,
    pending_plans: VecDeque<(usize, Call)>,
    planning: Option<Planning>,
    pending_replan: Option<Evaluation>,
    plan_result: Option<Message>,
    planning_error: Option<Error>,
    resume_input: Option<Input>,
    pending_calls: VecDeque<(usize, Call)>,
    active_tools: FuturesUnordered<Pin<Box<dyn Future<Output = ToolProgress> + 'static>>>,
    tool_progress: VecDeque<ToolProgress>,
    child_events: VecDeque<Result<Event, Error>>,
    child_relay: crate::runtime::agent::subagent::Relay,
    /// Child calls which have been handed off but have not yet committed a
    /// Function message into the parent context. This queue is derived from
    /// canonical Steps and child nodes, never serialized as another scene.
    pending_subagents: BTreeMap<String, PendingSubAgent>,
    tool_results: BTreeMap<usize, Message>,
    evaluations: BTreeMap<String, ToolEvaluation>,
    steps: Vec<Step>,
    // Planned tasks publish these only into their owning Attempt.nodes.
    child_nodes: BTreeMap<String, crate::runtime::agent::node::State>,
    child_nodes_dirty: bool,
    feedback_dirty: bool,
    // Transient eventless handoff marker; phases never add persistence triggers.
    phase_progress: bool,
    root: crate::runtime::agent::subagent::Root,
    partial: Message,
    done: bool,
    hooks: Hooks,
    access: Access,
    evaluator: Option<EvaluatorHandle>,
    metadata: Map<String, Value>,
    run: Run,
    model_pending: bool,
    model_settled: bool,
    input_tokens: u64,
    request: Option<Box<Request>>,
    trace: TraceRoot,
    model_span: Option<SpanScope>,
}

// Distinguish model-selected control from business evaluation. Replanning is
// not a Tool invocation and must not invent a Call or acknowledge a fake one.
#[derive(Clone, Serialize, Deserialize)]
enum Planning {
    Selected(usize, Call),
    Evaluation,
}

enum ToolProgress {
    Interaction(crate::runtime::agent::reply::Change),
    Observation(Box<crate::runtime::agent::subagent::Observation>),
    Called(
        usize,
        Call,
        Box<Result<(Call, Message), Failure>>,
        Option<crate::runtime::agent::reply::Slot>,
    ),
    Evaluated(String, Result<ToolEvaluation, Error>),
    SubAgent(
        usize,
        Call,
        Box<Result<SubAgentProgress, Box<SubAgentFailure>>>,
        Option<tokio::sync::OwnedSemaphorePermit>,
    ),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct PendingSubAgent {
    index: usize,
    call: Call,
}

// Private per-call dependencies, not a new component Context or Runtime state.
#[derive(Clone)]
struct ToolExecution {
    interaction: Option<crate::runtime::agent::reply::Run>,
    id_fn: Option<fn() -> i64>,
    dispatch: std::sync::Arc<Dispatch>,
    hooks: Hooks,
    access: Access,
    metadata: Map<String, Value>,
    cancellation: Cancellation,
    trace: TraceRoot,
}

async fn prepare_tool(state: &ToolExecution, call: &Call, saved: &Step) -> Result<Call, Failure> {
    Box::pin(super::lifecycle::tool_prepare(
        &state.dispatch,
        &state.access,
        &state.hooks,
        &state.metadata,
        state.interaction.as_ref(),
        state.id_fn,
        saved,
        call,
        |effective| async move {
            if let Some(run) = &state.interaction {
                run.tool_stage(effective, None, &state.cancellation).await?;
            }
            Ok(())
        },
        |permission| async move {
            if let Some(run) = &state.interaction {
                let call = permission.call()?;
                run.permission_stage(call, permission, &state.cancellation)
                    .await?;
            }
            Ok(())
        },
        &state.cancellation,
    ))
    .await
}

async fn execute_prepared_tool(
    state: &ToolExecution,
    call: Call,
    saved: Step,
) -> Result<(Call, Message), Error> {
    super::lifecycle::tool_call(
        &state.dispatch,
        &saved,
        call,
        state.interaction.clone(),
        state.id_fn,
        |call, message| async move {
            if let Some(run) = &state.interaction {
                run.tool_stage(call, Some(message), &state.cancellation)
                    .await?;
            }
            Ok(())
        },
        &state.cancellation,
    )
    .await
}

fn decode_subagent_input(
    call: &Call,
    parent: &Map<String, Value>,
) -> Result<(String, Input), Error> {
    let name = call
        .arguments
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "agent_spawn name is missing"))?
        .to_owned();
    let input = call
        .arguments
        .get("input")
        .cloned()
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "agent_spawn input is missing"))?;
    let mut input: Input = serde_json::from_value(input).map_err(|_| {
        Error::new(
            "INVALID_ARGUMENTS",
            "agent_spawn input must be agent::Input",
        )
    })?;
    crate::runtime::agent::subagent::inherit_identity(&mut input, parent)?;
    let depth = parent.get("depth").and_then(Value::as_u64).unwrap_or(1);
    if depth >= 2 {
        return Err(Error::new("INVALID_ARGUMENTS", "SubAgent depth limit is 2"));
    }
    input
        .metadata
        .insert("depth".into(), Value::from(depth + 1));
    Ok((name, input))
}

async fn execute_subagent(
    subagents: Arc<Vec<SubAgent>>,
    call: Call,
    parent: Map<String, Value>,
    restore: Option<Value>,
    cancellation: Cancellation,
    id_fn: Option<fn() -> i64>,
    observer: Option<crate::runtime::agent::subagent::Observer>,
) -> Result<SubAgentProgress, Box<SubAgentFailure>> {
    let (name, input) = decode_subagent_input(&call, &parent).map_err(SubAgentFailure::new)?;
    let subagent = subagents
        .iter()
        .find(|subagent| subagent.has_name(&name))
        .ok_or_else(|| {
            SubAgentFailure::new(Error::new(
                "NOT_FOUND",
                "agent_spawn target is not registered",
            ))
        })?;
    crate::runtime::agent::subagent::run(
        subagent,
        input,
        &call,
        restore,
        cancellation,
        id_fn,
        observer,
    )
    .await
}

fn subagent_error_message(call: &Call, error: &Error) -> Message {
    let value = json!({
        "code": error.code,
        "message": error.message,
        "details": error.details,
        "call_id": call.call_id,
        "name": call.name,
    });
    Message {
        role: Role::Function,
        content: vec![Part {
            r#type: "text".into(),
            data: part_data(json!({"value": value.to_string()})),
        }],
        values: Map::from_iter([(String::from("error"), value)]),
        metadata: Map::new(),
        usage: None,
    }
}

async fn next_react_event<M: Chat + 'static>(
    mut state: ReActState<M>,
) -> Option<(Result<Event, Error>, ReActState<M>)> {
    loop {
        let (event, next) = next_react_progress(state).await?;
        state = next;
        if let Some(event) = event {
            return Some((event, state));
        }
    }
}

// Private state handoff: None carries native task progress, never an Event.
fn next_react_progress<M: Chat + 'static>(state: ReActState<M>) -> ProgressFuture<M> {
    Box::pin(async move {
        // Keep the complete invocation machine out of each forwarding/task frame.
        // Deeply nested native scenes still use the same executor and state owner.
        let (result, state) = Box::pin(next_react_event_inner(Box::new(state))).await?;
        let mut state = *state;
        let Some(result) = result else {
            return Some((None, state));
        };
        if result.is_err() && state.model_pending && !state.model_settled {
            let usage = super::lifecycle::model_unfinished(&mut state.steps, &state.partial);
            if let Err(error) = &result {
                crate::runtime::agent::model_steps::stopped(&mut state.steps, error);
            }
            state.run.settle(usage);
            state.model_settled = true;
        }
        let result = if state.done {
            if result.as_ref().is_err_and(|error| error.code != "WAITING")
                && !state.waits.is_empty()
            {
                state.waits.clear();
                crate::runtime::agent::tool_steps::cancel_pending(&mut state.steps);
                execution::cancel_active(&mut state);
            }
            if let Some(run) = &state.interaction {
                run.close();
            }
            // Serialize the actual terminal scene once for Checkpoint and Chat.state.
            // An I/O failure is not an execution failure and must not produce a
            // second, rewritten runtime scene.
            let snapshot = checkpoint_snapshot(&state, &result);
            let saved = save_checkpoint(&state, snapshot.clone()).await;
            let graph = state
                .graph
                .as_ref()
                .and_then(|graph| serde_json::to_value(&graph.plan).ok());
            // Drop all pending futures even if the consumer retains the terminal
            // stream. Local resources are released; external effects are not undone.
            state.active_tools.clear();
            state.tool_progress.clear();
            state.active_tasks.clear();
            state.pending_calls.clear();
            state.pending_plans.clear();
            state.pending_replan = None;
            state.provider = None;
            match saved {
                Err(error) => Err(error),
                Ok(()) => {
                    state
                        .run
                        .close_with_state(result, graph, Some(snapshot))
                        .await
                }
            }
        } else {
            result
        };
        if !state.done
            && !state.is_plan_task
            && matches!(&result, Ok(Event::Complete { .. }))
            && crate::runtime::agent::scope::active()
        {
            crate::runtime::agent::scope::capture(if state.waits.is_empty() {
                running_snapshot(&state)
            } else {
                waiting_snapshot(&state)
            });
        }
        Some((Some(result), state))
    })
}

fn emitted<T>(
    (event, state): (Result<Event, Error>, T),
) -> Option<(Option<Result<Event, Error>>, T)> {
    Some((Some(event), state))
}

fn result_status(result: &Result<Event, Error>) -> &'static str {
    match result {
        Ok(Event::Complete {
            finish_reason: Some(reason),
            ..
        }) if reason == "cancelled" => "stopped",
        Ok(Event::Complete { .. }) => "succeeded",
        Err(error) if error.code == "CANCELLED" => "stopped",
        Err(error) if error.code == "WAITING" => "waiting",
        Err(_) => "failed",
        Ok(Event::Delta { .. }) => "running",
    }
}

/// Serialize the canonical Agent Runtime scene. Checkpoint and Chat.state
/// store independent copies; neither a graph nor configured storage is required.
fn checkpoint_snapshot<M: Chat>(state: &ReActState<M>, result: &Result<Event, Error>) -> Value {
    let (graph, nodes, history) = state
        .graph
        .as_ref()
        .map(|graph| {
            (
                Some(graph::State {
                    plan: graph.plan.clone(),
                }),
                graph.nodes.clone(),
                graph.history.clone(),
            )
        })
        .unwrap_or((None, BTreeMap::new(), Vec::new()));
    let mut snapshot = serde_json::to_value(AgentState {
        completion: state.completion.clone(),
        preparation: state.preparation.clone(),
        chat_id: state.query.metadata.get("chat_id").and_then(Value::as_i64),
        r#type: "agent".into(),
        status: result_status(result).into(),
        stream: state.stream,
        usage: execution::usage(state),
        execution: state.root.snapshot(
            result_status(result),
            execution::usage(state),
            result.as_ref().err().cloned(),
        ),
        graph,
        nodes,
        history,
        messages: state.messages.clone(),
        steps: state.steps.clone(),
        waits: state.waits.clone(),
        error: result.as_ref().err().cloned(),
        final_message: match result {
            Ok(Event::Complete { message, .. }) => {
                Some(crate::runtime::observation::original_message(message))
            }
            _ => None,
        },
        metadata: state.query.metadata.clone().into_iter().collect(),
    })
    .unwrap_or_else(|_| json!({}));
    snapshot["query"] = serde_json::to_value(&state.query).unwrap_or(Value::Null);
    snapshot["task_base"] = serde_json::to_value(&state.task_base).unwrap_or(Value::Null);
    snapshot["iteration"] = json!(state.iteration.load(Ordering::Relaxed));
    snapshot["plan_execution"] = json!(state.plan_execution);
    snapshot["plan_finalizing"] = json!(state.plan_finalizing);
    snapshot["tool_results"] = serde_json::to_value(&state.tool_results).unwrap_or(Value::Null);
    snapshot["planning"] = serde_json::to_value(&state.planning).expect("planning control");
    crate::runtime::agent::checkpoint::record_result(&mut snapshot, result);
    snapshot
}

fn checkpoint_id(metadata: &Map<String, Value>) -> Result<i64, Error> {
    let id = metadata
        .get("chat_id")
        .and_then(Value::as_i64)
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "checkpoint requires chat_id"))?;
    if id <= 0 {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "checkpoint chat_id must be positive",
        ));
    }
    Ok(id)
}

fn save_checkpoint<M: Chat>(
    state: &ReActState<M>,
    snapshot: Value,
) -> impl Future<Output = Result<(), Error>> + 'static {
    // Own just the save handle and ID, not a borrow of the whole stream state.
    // Send event streams need not also be Sync.
    let save = state.checkpoint.clone();
    let id = state.checkpoint_id;
    async move {
        if crate::runtime::agent::scope::capture(snapshot.clone()) {
            Ok(())
        } else if let (Some(save), Some(id)) = (save, id) {
            save.save(id, snapshot, Cancellation::new()).await
        } else {
            Ok(())
        }
    }
}

fn restore_output<M: Chat>(state: &mut ReActState<M>) -> Result<(), Error> {
    crate::runtime::agent::model_steps::validate_owner(
        &state.steps,
        state.completion.as_ref(),
        state.hooks.len(),
    )?;
    state.restored_output =
        if let Some(step) = crate::runtime::agent::model_steps::pending(&state.steps) {
            Some(crate::runtime::agent::model_steps::event(step)?)
        } else {
            state
                .completion
                .as_ref()
                .filter(|completion| completion.pending())
                .map(|saved| Event::Complete {
                    message: saved.message.clone(),
                    usage: saved.message.usage.clone(),
                    finish_reason: saved.finish_reason.clone(),
                })
        };
    if let Some(Event::Complete { message, .. }) = &state.restored_output {
        state.partial = message.clone();
        state.model_settled = true;
        state.model_pending = true;
        state.replayed_completion = true;
        state.input_tokens = state
            .steps
            .iter()
            .rev()
            .find(|step| step.kind == "model")
            .and_then(|step| step.request.as_deref())
            .map(crate::runtime::agent::chat::estimate_input_tokens)
            .unwrap_or(0);
    }
    Ok(())
}

fn restore_state<M: Chat>(state: &mut ReActState<M>, snapshot: Value) -> Result<(), Error> {
    fn field<T: serde::de::DeserializeOwned>(snapshot: &Value, name: &str) -> Result<T, Error> {
        serde_json::from_value(snapshot[name].clone())
            .map_err(|_| invalid_checkpoint(&format!("invalid Agent checkpoint field: {name}")))
    }
    state.messages = field(&snapshot, "messages")?;
    state.task_base = field(&snapshot, "task_base")?;
    state.query = field(&snapshot, "query")?;
    state.iteration = Arc::new(AtomicUsize::new(field(&snapshot, "iteration")?));
    state.plan_execution = field(&snapshot, "plan_execution")?;
    state.plan_finalizing = field(&snapshot, "plan_finalizing")?;
    state.root = crate::runtime::agent::subagent::Root::restore(&snapshot)?;
    state.tool_results = field(&snapshot, "tool_results")?;
    state.steps = snapshot
        .get("steps")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| invalid_checkpoint("invalid Agent checkpoint steps"))?
        .unwrap_or_default();
    state.completion = snapshot
        .get("completion")
        .filter(|value| !value.is_null())
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| invalid_checkpoint("invalid Agent completion"))?;
    state.planning = snapshot
        .get("planning")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| invalid_checkpoint("invalid Model planning control"))?
        .flatten();
    restore_output(state)?;
    state.waits = snapshot
        .get("waits")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| invalid_checkpoint("invalid saved waiting locators"))?
        .unwrap_or_default();
    tools::restore(state)?;
    // A parent waiting on a child is resumed by replaying only the registered
    // child hand-off.  The parent model is not called again for that Call,
    // and a committed tool result always wins over a stale pending entry.
    let mut pending = state.root.resume_calls(&state.steps, &mut state.messages)?;
    for (index, call) in &pending {
        state.pending_subagents.insert(
            call.call_id.clone(),
            PendingSubAgent {
                index: *index,
                call: call.clone(),
            },
        );
    }
    pending.sort_by_key(|(index, _)| *index);
    state.pending_calls.extend(pending);
    state.run.settle(field(&snapshot, "usage")?);
    if let Some(graph) = snapshot.get("graph").filter(|graph| !graph.is_null()) {
        let mut graph: Graph = serde_json::from_value(json!({
            "plan": graph["plan"],
            "history": snapshot.get("history").cloned().unwrap_or_else(|| json!([])),
            "nodes": snapshot["nodes"],
        }))
        .map_err(|_| invalid_checkpoint("invalid Agent checkpoint graph"))?;
        graph.plan.validate_structure()?;
        if graph.plan.nodes.keys().ne(graph.nodes.keys()) {
            return Err(invalid_checkpoint(
                "checkpoint graph nodes do not match the plan",
            ));
        }
        use crate::runtime::agent::node::Status;
        let waiting_nodes = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.status, Status::Running | Status::Waiting))
            .map(|(node_id, _)| node_id.clone())
            .collect::<Vec<_>>();
        let external_nodes = graph
            .nodes
            .iter()
            .filter(|(_, node)| node.status == Status::Waiting)
            .map(|(node_id, _)| node_id.clone())
            .collect::<BTreeSet<_>>();
        if matches!(
            snapshot.get("status").and_then(Value::as_str),
            Some("waiting" | "paused")
        ) && external_nodes.is_empty()
            && state.pending_subagents.is_empty()
            && state.waits.is_empty()
            && state.restored_output.is_none()
            && crate::runtime::agent::model_steps::pending_input(&state.steps).is_none()
        {
            return Err(invalid_checkpoint(
                "waiting Agent checkpoint has no waiting node",
            ));
        }
        for node in graph.nodes.values_mut() {
            if node.status == Status::Running {
                node.status = Status::Waiting;
                if let Some(attempt) = node
                    .execution
                    .as_mut()
                    .and_then(|execution| execution.attempts.last_mut())
                {
                    attempt.status = Status::Waiting;
                }
            } else if node.status == Status::Succeeded {
                if node.messages.is_none()
                    || node
                        .execution
                        .as_ref()
                        .and_then(|execution| execution.attempts.last())
                        .is_none_or(|attempt| attempt.status != Status::Succeeded)
                {
                    return Err(invalid_checkpoint(
                        "successful Agent node has no committed result",
                    ));
                }
            } else if node.status == Status::Failed {
                state.task_error = node
                    .execution
                    .as_ref()
                    .and_then(|execution| execution.error.clone())
                    .or_else(|| Some(Error::new("EXECUTION_FAILED", "restored Agent node failed")));
            }
        }
        graph.activate_ready()?;
        for node_id in waiting_nodes {
            graph.resume(&node_id)?;
            state.resume_nodes.insert(node_id);
        }
        state.external_resume_nodes = external_nodes;
        state.graph = Some(graph);
    }
    // A waiting snapshot without a plan graph is a valid external wait.  The
    // next call with the same chat_id supplies the continuation payload; no
    // internal node or wait ID is required in the public input.
    Ok(())
}

fn is_waiting_snapshot(snapshot: &Value) -> bool {
    matches!(
        snapshot.get("status").and_then(Value::as_str),
        Some("running" | "waiting" | "paused")
    ) || snapshot
        .get("nodes")
        .and_then(Value::as_object)
        .is_some_and(|nodes| {
            nodes.values().any(|node| {
                matches!(
                    node.get("status").and_then(Value::as_str),
                    Some("running" | "waiting")
                )
            })
        })
}

fn is_external_wait_snapshot(snapshot: &Value) -> bool {
    if snapshot
        .get("completion")
        .and_then(|state| state.get("stage"))
        .and_then(Value::as_str)
        .is_some_and(|stage| stage != "committed")
        || snapshot
            .get("steps")
            .and_then(Value::as_array)
            .is_some_and(|steps| {
                steps.iter().any(|step| {
                    step["kind"] == "model"
                        && (matches!(step["stage"].as_str(), Some("before" | "after"))
                            || (step["stage"] == "call"
                                && step["attempts"]
                                    .as_array()
                                    .and_then(|attempts| attempts.last())
                                    .is_none_or(|attempt| attempt["status"] != "waiting")))
                })
            })
    {
        return false;
    }
    let waiting_child =
        crate::runtime::agent::subagent::Root::restore(snapshot).is_ok_and(|root| root.pending());
    let waiting_tool = snapshot
        .get("steps")
        .and_then(Value::as_array)
        .is_some_and(|steps| {
            steps.iter().any(|step| {
                step.get("kind").and_then(Value::as_str) == Some("tool")
                    && matches!(
                        step.get("stage").and_then(Value::as_str),
                        Some("before" | "call" | "after" | "eval" | "complete")
                    )
            })
        });
    !waiting_child
        && !waiting_tool
        && (snapshot.get("status").and_then(Value::as_str) == Some("waiting")
            || snapshot
                .get("nodes")
                .and_then(Value::as_object)
                .is_some_and(|nodes| {
                    nodes
                        .values()
                        .any(|node| node.get("status").and_then(Value::as_str) == Some("waiting"))
                }))
}

fn checkpoint_identity_matches(snapshot: &Value, metadata: &Map<String, Value>) -> bool {
    crate::runtime::agent::checkpoint::identity_matches(snapshot, metadata)
}

fn running_snapshot<M: Chat>(state: &ReActState<M>) -> Value {
    checkpoint_snapshot(
        state,
        &Ok(Event::Delta {
            message: state.partial.clone(),
        }),
    )
}

fn waiting_snapshot<M: Chat>(state: &ReActState<M>) -> Value {
    let mut snapshot = checkpoint_snapshot(
        state,
        &Ok(Event::Delta {
            message: state.partial.clone(),
        }),
    );
    snapshot["status"] = Value::String("waiting".into());
    if let Some(execution) = state
        .root
        .snapshot("waiting", execution::usage(state), None)
    {
        snapshot["execution"] =
            serde_json::to_value(execution).expect("serializable root execution");
    }
    snapshot
}

// Cancellation-only work must not enlarge the ordinary nested poll frame.
#[inline(never)]
fn cancel_children<M: Chat + 'static>(
    state: &mut ReActState<M>,
    cx: &mut std::task::Context<'_>,
) -> Result<Vec<Event>, Error> {
    let relay = state.child_relay.clone();
    let (mut completed, events) = super::subagent::cancel_ready(
        cx,
        &relay,
        |cx| state.active_tools.poll_next_unpin(cx),
        |observation| {
            if state.is_plan_task {
                state.child_nodes_dirty |= observation.has_facts();
                super::subagent::observe(&mut state.child_nodes, observation)?;
            } else {
                state.root.observe(observation)?;
            }
            state.run.settle_child(observation.usage.clone());
            Ok(())
        },
    )?;
    completed.extend(state.tool_progress.drain(..));
    for progress in completed {
        if let ToolProgress::SubAgent(_, call, result, _) = progress {
            let message = result
                .as_ref()
                .as_ref()
                .ok()
                .map(|progress| progress.message.clone());
            if state.is_plan_task {
                super::subagent::complete(
                    &mut state.child_nodes,
                    &mut state.steps,
                    &call,
                    &result,
                    message,
                )?;
            } else {
                state
                    .root
                    .complete(&mut state.steps, &call, &result, message)?;
            }
            state.run.settle_child(match &*result {
                Ok(progress) => progress.usage.clone(),
                Err(failure) => failure.usage.clone(),
            });
        }
    }
    if state.is_plan_task {
        super::subagent::cancel_pending(&mut state.child_nodes);
    } else {
        state.root.cancel_pending();
    }
    state.active_tools.clear();
    // No live question may still point into a child just made terminal.
    // Historical question records remain on their actual Tool Steps.
    state.waits.clear();
    Ok(events)
}

async fn next_react_event_inner<M: Chat + 'static>(
    // A planned parent may poll a delegated ReAct's own planned tasks. Heap
    // ownership avoids moving full run state through each nested poll frame.
    mut state: Box<ReActState<M>>,
) -> Option<(Option<Result<Event, Error>>, Box<ReActState<M>>)> {
    loop {
        if state.done {
            return None;
        }

        if let Some(event) = state.child_events.pop_front() {
            return emitted((event, state));
        }

        if state.cancellation.is_cancelled() && state.active_tasks.is_empty() {
            let events = match poll_fn(|cx| Poll::Ready(cancel_children(&mut state, cx))).await {
                Ok(events) => events,
                Err(error) => {
                    state.done = true;
                    return emitted((Err(error), state));
                }
            };
            state.child_events.extend(events.into_iter().map(Ok));
            if let Some(event) = state.child_events.pop_front() {
                return emitted((event, state));
            }
            state.done = true;
            return emitted((Ok(cancelled_event(&mut state)), state));
        }

        if let Some(error) = state.planning_error.take() {
            state.done = true;
            return emitted((Err(error), state));
        }

        if let Some(message) = state.plan_result.take() {
            return emitted((
                Ok(Event::Complete {
                    message,
                    usage: None,
                    finish_reason: Some("tool".into()),
                }),
                state,
            ));
        }

        // Root ordinary-feedback consumers release Tool admission while waiting.
        // Planned tasks retain their existing task-level suspension policy;
        // admitting their queued calls here would bypass another ready task.
        while state.active_tools.len().saturating_sub(
            state
                .steps
                .iter()
                .filter(|step| {
                    !state.is_plan_task
                        && step.status == "waiting"
                        && step
                            .waits
                            .values()
                            .any(|wait| wait.kind == "input" && wait.status == "waiting")
                        && step.call_id.as_ref().is_some_and(|id| {
                            !state
                                .pending_calls
                                .iter()
                                .any(|(_, call)| &call.call_id == id)
                        })
                })
                .count()
                + if state.is_plan_task {
                    0
                } else {
                    state.root.feedback_waiters()
                },
        ) < state.max_concurrency
        {
            let Some((index, call)) = state.pending_calls.pop_front() else {
                break;
            };
            if call.name == "plan" && !state.is_plan_task {
                if let Err(error) = validate_call(&plan_definition(), &call) {
                    let message = tool_error_message(&call, error);
                    state.tool_results.insert(index, message.clone());
                    return emitted((
                        Ok(Event::Complete {
                            message,
                            usage: None,
                            finish_reason: Some("tool".into()),
                        }),
                        state,
                    ));
                }
                state.pending_plans.push_back((index, call));
                continue;
            }
            if call.name == "agent_spawn" && !state.subagents.is_empty() {
                let cached = if state.is_plan_task {
                    crate::runtime::agent::subagent::message(
                        &state.child_nodes,
                        &state.steps,
                        &call,
                    )
                } else {
                    state.root.message(&state.steps, &call)
                };
                if let Some(message) = cached {
                    state.pending_subagents.remove(&call.call_id);
                    state.tool_results.insert(index, message.clone());
                    return emitted((
                        Ok(Event::Complete {
                            message,
                            usage: None,
                            finish_reason: Some("agent".into()),
                        }),
                        state,
                    ));
                }
                let prepared = if state.is_plan_task {
                    crate::runtime::agent::subagent::prepare(
                        &mut state.child_nodes,
                        &mut state.steps,
                        &call,
                        state.id_fn,
                    )
                } else {
                    state.root.prepare(&mut state.steps, &call, state.id_fn)
                };
                let restore = match prepared {
                    Ok(restore) => restore,
                    Err(error) => {
                        state.done = true;
                        return emitted((Err(error), state));
                    }
                };
                state.child_nodes_dirty |= state.is_plan_task;
                state
                    .pending_subagents
                    .entry(call.call_id.clone())
                    .or_insert_with(|| PendingSubAgent {
                        index,
                        call: call.clone(),
                    });
                // Persist the hand-off before the child starts.  If the
                // process disappears after this point, restoration can
                // resubmit this exact child Call without replaying the parent
                // model.  No new public resume API or child ID is introduced.
                if let Err(error) = save_checkpoint(&state, waiting_snapshot(&state)).await {
                    state.done = true;
                    return emitted((Err(error), state));
                }
                let slots = state.tool_slots.clone();
                let source = if state.is_plan_task {
                    crate::runtime::agent::subagent::source(&state.child_nodes, &state.steps, &call)
                } else {
                    state.root.source(&state.steps, &call)
                };
                let source = match source {
                    Ok(source) => source,
                    Err(error) => {
                        state.done = true;
                        return emitted((Err(error), state));
                    }
                };
                let observer = state.child_relay.observer(
                    state.metadata.get("chat_id").and_then(Value::as_i64),
                    source.clone(),
                );
                let admission =
                    crate::runtime::agent::reply::Slot::waiting(slots, state.task_ready.clone());
                let interaction = state
                    .interaction
                    .as_ref()
                    .map(|run| run.branch(admission.clone()));
                if let (Some(parent), Some(child)) = (&state.interaction, &interaction) {
                    parent.restore_consumer(&source, child);
                }
                let observer = observer.with_interaction(interaction.clone());
                let subagents = state.subagents.clone();
                let parent = state.metadata.clone();
                let id_fn = state.id_fn;
                let cancellation = state.cancellation.clone();
                state.active_tools.push(Box::pin(async move {
                    let entering = match &interaction {
                        Some(run) => run.continue_execution(&cancellation).await,
                        None => admission.acquire(&cancellation).await,
                    };
                    if let Err(error) = entering {
                        return ToolProgress::SubAgent(
                            index,
                            call,
                            Box::new(Err(SubAgentFailure::new(error))),
                            None,
                        );
                    }
                    let result = execute_subagent(
                        subagents,
                        call.clone(),
                        parent,
                        restore,
                        cancellation,
                        id_fn,
                        Some(observer),
                    )
                    .await;
                    admission.release();
                    ToolProgress::SubAgent(index, call, Box::new(result), None)
                }));
                continue;
            }
            match tools::continue_step(&mut state, index, &call) {
                Ok(true) => continue,
                Ok(false) => {}
                Err(error) => {
                    state.done = true;
                    return emitted((Err(error), state));
                }
            }
            let execution = ToolExecution {
                interaction: state.interaction.clone(),
                id_fn: state.id_fn,
                dispatch: state.dispatch.clone(),
                hooks: state.hooks.clone(),
                access: state.access.clone(),
                metadata: state.metadata.clone(),
                cancellation: state.cancellation.clone(),
                trace: state.trace.clone(),
            };
            let saved = crate::runtime::agent::tool_steps::current(&mut state.steps, &call.call_id)
                .expect("selected Tool Step")
                .clone();
            let slots = state.tool_slots.clone();
            state.active_tools.push(Box::pin(async move {
                // Human permission waits must not retain Tool execution slots.
                let effective = match prepare_tool(&execution, &call, &saved).await {
                    Ok(effective) => effective,
                    Err(error) => {
                        return ToolProgress::Called(index, call, Box::new(Err(error)), None);
                    }
                };
                if let Some(run) = &execution.interaction
                    && let Err(error) = run.continue_execution(&execution.cancellation).await
                {
                    return ToolProgress::Called(
                        index,
                        call,
                        Box::new(Err(Failure::Hook(error))),
                        None,
                    );
                }
                let permit = match cancellable(
                    async {
                        slots
                            .clone()
                            .acquire_owned()
                            .await
                            .map_err(|_| Error::new("CANCELLED", "tool capacity closed"))
                    },
                    &execution.cancellation,
                )
                .await
                {
                    Ok(permit) => permit,
                    Err(error) => {
                        return ToolProgress::Called(
                            index,
                            call,
                            Box::new(Err(Failure::Hook(error))),
                            None,
                        );
                    }
                };
                let permit = crate::runtime::agent::reply::Slot::with_permit(
                    slots,
                    Default::default(),
                    permit,
                );
                let tool_span = execution.trace.child("agent.tool");
                tool_span.property("tool", effective.name.clone());
                let tool_execution = execution.clone();
                let result = crate::runtime::agent::reply::feedback::with_slot(
                    permit.clone(),
                    tool_span.in_span("agent.tool.call", async move {
                        execute_prepared_tool(&tool_execution, effective, saved).await
                    }),
                )
                .await
                .map_err(Failure::Tool);
                ToolProgress::Called(index, call, Box::new(result), Some(permit))
            }));
        }

        if !state.active_tools.is_empty() || !state.tool_progress.is_empty() {
            let progress = poll_fn(|context| {
                state.cancellation.register(context.waker());
                if state.cancellation.is_cancelled() {
                    return Poll::Ready(None);
                }
                if state.tool_progress.is_empty()
                    && let Poll::Ready(Some(progress)) = state.active_tools.poll_next_unpin(context)
                {
                    state.tool_progress.push_back(progress);
                }
                if let Some(run) = &state.interaction
                    && let Poll::Ready(change) = run.poll_change(context)
                {
                    return Poll::Ready(Some(ToolProgress::Interaction(change)));
                }
                // Successful observations already produced by a sibling must
                // be delivered/applied before a waiting/error closes the run.
                if let Poll::Ready(observation) = state.child_relay.poll_event(context) {
                    return Poll::Ready(Some(ToolProgress::Observation(Box::new(observation))));
                }
                if let Some(progress) = state.tool_progress.pop_front() {
                    Poll::Ready(Some(progress))
                } else {
                    if execution::waiting_on_feedback(&state)
                        && let Some(run) = &state.interaction
                    {
                        run.suspend();
                    }
                    Poll::Pending
                }
            })
            .await;
            if state.cancellation.is_cancelled() {
                // This await may have started before cancellation (notably
                // when non-stream delivery hides the child's Delta). Rejoin
                // the same owner closeout rather than dropping active children.
                continue;
            }
            match progress.expect("nonempty Tool futures remain until polled") {
                ToolProgress::Interaction(change) => {
                    let lifecycle = matches!(
                        &change,
                        crate::runtime::agent::reply::Change::ToolStage { .. }
                            | crate::runtime::agent::reply::Change::PermissionStage { .. }
                    );
                    let feedback = matches!(
                        &change,
                        crate::runtime::agent::reply::Change::Feedback { .. }
                    );
                    let run = state.interaction.clone().expect("interaction owner");
                    match crate::runtime::agent::reply::apply(
                        change,
                        &mut state.steps,
                        &mut state.waits,
                        &run,
                    ) {
                        Ok(Some(question)) => {
                            let snapshot = checkpoint_snapshot(
                                &state,
                                &Err(Error::new("WAITING", "Agent is waiting for feedback")),
                            );
                            if let Err(error) = save_checkpoint(&state, snapshot).await {
                                state.done = true;
                                return emitted((Err(error), state));
                            }
                            if state.cancellation.is_cancelled() {
                                continue;
                            }
                            return emitted((Ok(question), state));
                        }
                        Ok(None) => {
                            if (feedback || lifecycle) && state.is_plan_task {
                                state.phase_progress = lifecycle;
                                // Yield the actual task state to its owner so
                                // partial collection can reach the outer scene
                                // without manufacturing an observation Event.
                                return Some((None, state));
                            }
                            if lifecycle {
                                crate::runtime::agent::scope::capture(running_snapshot(&state));
                            }
                            if feedback
                                && (!state.waits.is_empty()
                                    || crate::runtime::agent::scope::active())
                                && let Err(error) = save_checkpoint(
                                    &state,
                                    if state.waits.is_empty() {
                                        running_snapshot(&state)
                                    } else {
                                        waiting_snapshot(&state)
                                    },
                                )
                                .await
                            {
                                state.done = true;
                                return emitted((Err(error), state));
                            }
                            continue;
                        }
                        Err(error) => {
                            state.done = true;
                            return emitted((Err(error), state));
                        }
                    }
                }
                ToolProgress::Observation(observation) => {
                    let question = match observation.question() {
                        Ok(question) => question,
                        Err(error) => {
                            state.done = true;
                            return emitted((Err(error), state));
                        }
                    };
                    state.child_nodes_dirty |= state.is_plan_task && observation.has_facts();
                    state.feedback_dirty |= state.is_plan_task && observation.feedback;
                    let result = if state.is_plan_task {
                        crate::runtime::agent::subagent::observe(
                            &mut state.child_nodes,
                            &observation,
                        )
                        .and_then(|_| {
                            crate::runtime::agent::subagent::sync_waits(
                                &state.child_nodes,
                                &mut state.waits,
                                &observation,
                            )
                        })
                    } else {
                        state
                            .root
                            .observe(&observation)
                            .and_then(|_| state.root.sync_waits(&mut state.waits, &observation))
                    };
                    if let Err(error) = result.and_then(|_| {
                        if state.is_plan_task {
                            Ok(())
                        } else {
                            observation.commit_feedback(state.interaction.as_ref())
                        }
                    }) {
                        state.done = true;
                        return emitted((Err(error), state));
                    }
                    state.run.settle_child(observation.usage);
                    if state.is_plan_task && observation.event.is_none() {
                        return Some((None, state));
                    }
                    if question || observation.event.is_none() || observation.feedback {
                        if let Err(error) = save_checkpoint(
                            &state,
                            if state.waits.is_empty() {
                                running_snapshot(&state)
                            } else {
                                waiting_snapshot(&state)
                            },
                        )
                        .await
                        {
                            state.done = true;
                            return emitted((Err(error), state));
                        }
                        if state.cancellation.is_cancelled() {
                            continue;
                        }
                    }
                    if let Some(event) = observation.event {
                        return emitted((Ok(event), state));
                    }
                    continue;
                }
                ToolProgress::Evaluated(call_id, Ok(evaluation)) => {
                    tools::evaluated(&mut state.steps, &call_id, &evaluation);
                    // Business feedback is not a technical model repair error.
                    state.evaluations.insert(call_id, evaluation);
                    continue;
                }
                ToolProgress::Evaluated(call_id, Err(error)) => {
                    tools::stopped(&mut state.steps, &call_id, &error);
                    // Default zero retries: never repeat a successful call to
                    // recover its eval. The emitted result remains unchanged.
                    state.done = true;
                    return emitted((Err(error), state));
                }
                ToolProgress::Called(index, call, result, permit) => {
                    let message = match *result {
                        Ok((effective_call, business)) => {
                            tools::called(&mut state.steps, &effective_call, &business);
                            let dispatch = state.dispatch.clone();
                            let cancellation = state.cancellation.clone();
                            let evaluation_message = business.clone();
                            // Keep this lifecycle's slot occupied through eval,
                            // but publish the real call/after result first.
                            state.active_tools.push(Box::pin(async move {
                                let result = dispatch
                                    .evaluate(&effective_call, &evaluation_message, &cancellation)
                                    .await;
                                drop(permit);
                                ToolProgress::Evaluated(effective_call.call_id, result)
                            }));
                            tool_success_message(&call, business)
                        }
                        Err(Failure::Policy(error, effective)) => {
                            if let Some(step) = crate::runtime::agent::tool_steps::current(
                                &mut state.steps,
                                &call.call_id,
                            ) {
                                step.call = Some(*effective);
                            }
                            tools::stopped(&mut state.steps, &call.call_id, &error);
                            if error.code == "WAITING" {
                                state.done = true;
                                return emitted((Err(error), state));
                            }
                            let message = tool_error_message(&call, error);
                            tools::failed_message(&mut state.steps, &call.call_id, &message);
                            message
                        }
                        Err(Failure::Tool(error)) => {
                            tools::stopped(&mut state.steps, &call.call_id, &error);
                            if error.code == "WAITING" {
                                state.done = true;
                                return emitted((Err(error), state));
                            }
                            let message = tool_error_message(&call, error);
                            tools::failed_message(&mut state.steps, &call.call_id, &message);
                            message
                        }
                        Err(Failure::Hook(error)) => {
                            tools::stopped(&mut state.steps, &call.call_id, &error);
                            state.done = true;
                            return emitted((Err(error), state));
                        }
                    };
                    state.tool_results.insert(index, message.clone());
                    return emitted((
                        Ok(Event::Complete {
                            message,
                            usage: None,
                            finish_reason: Some("tool".into()),
                        }),
                        state,
                    ));
                }
                ToolProgress::SubAgent(index, call, result, permit) => {
                    drop(permit);
                    let message = match &*result {
                        Ok(progress) => Some(progress.message.clone()),
                        Err(failure) if failure.error.code == "WAITING" => None,
                        Err(failure) => Some(subagent_error_message(&call, &failure.error)),
                    };
                    let completed = if state.is_plan_task {
                        state.child_nodes_dirty = true;
                        crate::runtime::agent::subagent::complete(
                            &mut state.child_nodes,
                            &mut state.steps,
                            &call,
                            &result,
                            message,
                        )
                    } else {
                        state
                            .root
                            .complete(&mut state.steps, &call, &result, message)
                    };
                    if let Err(error) = completed {
                        state.done = true;
                        return emitted((Err(error), state));
                    }
                    state.run.settle_child(match &*result {
                        Ok(progress) => progress.usage.clone(),
                        Err(failure) => failure.usage.clone(),
                    });
                    if let Err(failure) = &*result
                        && failure.error.code == "WAITING"
                    {
                        if let Err(error) = save_checkpoint(&state, waiting_snapshot(&state)).await
                        {
                            state.done = true;
                            return emitted((Err(error), state));
                        }
                        state.done = true;
                        return emitted((Err(failure.error.clone()), state));
                    }
                    state.pending_subagents.remove(&call.call_id);
                    let message = match *result {
                        Ok(progress) => progress.message,
                        Err(failure) => subagent_error_message(&call, &failure.error),
                    };
                    state.tool_results.insert(index, message.clone());
                    if let Err(error) = save_checkpoint(&state, running_snapshot(&state)).await {
                        state.done = true;
                        return emitted((Err(error), state));
                    }
                    state.child_events.push_back(Ok(Event::Complete {
                        message,
                        usage: None,
                        finish_reason: Some("agent".into()),
                    }));
                    if let Some(event) = state.child_events.pop_front() {
                        return emitted((event, state));
                    }
                }
            }
        }

        // Events may arrive in completion order. Model context stays in the
        // original Call order, and no next inference sees an incomplete batch.
        if state.pending_plans.is_empty() && state.planning.is_none() {
            for (_, message) in std::mem::take(&mut state.tool_results) {
                let call_id = message
                    .metadata
                    .get("call_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                state.messages.push(message);
                if let Some(call_id) = call_id
                    && let Some(evaluation) = state.evaluations.remove(&call_id)
                    && state.is_plan_task
                    && !evaluation.passed
                    && let Some(feedback) = evaluation.feedback
                {
                    // Business evaluation feedback is part of the current
                    // Agent node's next model input. It follows the actual
                    // Tool Message, remains ordinary Messages, and never
                    // becomes a technical error or a tool_result envelope.
                    state.messages.0.extend(feedback.0);
                }
            }
            tools::committed(&mut state.steps);
        }

        if state.provider.is_some() || state.restored_output.is_some() {
            let restoring = state.restored_output.is_some();
            let resuming_completion = restoring
                && state
                    .completion
                    .as_ref()
                    .is_some_and(crate::runtime::agent::completion::State::pending);
            let item = if let Some(event) = state.restored_output.take() {
                Some(Ok(event))
            } else {
                let provider = state.provider.as_mut().expect("active provider");
                poll_fn(|context| {
                    state.cancellation.register(context.waker());
                    if state.cancellation.is_cancelled() {
                        return Poll::Ready(None);
                    }
                    provider.as_mut().poll_next(context)
                })
                .await
            };
            if state.cancellation.is_cancelled() {
                state.model_span = None;
                state.done = true;
                return emitted((Ok(cancelled_event(&mut state)), state));
            }
            let Some(item) = item else {
                state.model_span = None;
                state.done = true;
                return emitted((
                    Err(Error::new(
                        "INVALID_RESPONSE",
                        "model did not return Complete",
                    )),
                    state,
                ));
            };
            match item {
                Err(error) => {
                    state.model_span = None;
                    crate::runtime::agent::model_steps::stopped(&mut state.steps, &error);
                    state.done = true;
                    return emitted((
                        Err(Error::new(error.code, error.message)
                            .with_optional_details(error.details)),
                        state,
                    ));
                }
                Ok(Event::Delta { mut message }) => {
                    crate::runtime::observation::clear_source(&mut message);
                    collect_delta(&mut state.partial, &message);
                    return emitted((Ok(Event::Delta { message }), state));
                }
                Ok(Event::Complete {
                    mut message,
                    usage,
                    finish_reason,
                }) => {
                    state.provider = None;
                    state.model_span = None;
                    crate::runtime::observation::clear_source(&mut message);
                    let usage = usage.or_else(|| message.usage.clone());
                    if !restoring {
                        state.run.settle(usage.clone());
                        state.replayed_completion = false;
                    }
                    state.model_settled = true;
                    message.usage = usage.clone();
                    state.partial = message.clone();
                    if !restoring {
                        crate::runtime::agent::model_steps::returned(
                            &mut state.steps,
                            state.request.take(),
                            message.clone(),
                            usage.clone(),
                            finish_reason.clone(),
                        );
                    }
                    let model_after =
                        crate::runtime::agent::model_steps::pending(&state.steps).cloned();
                    let after_result = if let Some(saved) = model_after {
                        let hooks = state.hooks.clone();
                        let cancellation = state.cancellation.clone();
                        Box::pin(super::completion::model(
                            &hooks,
                            saved,
                            |saved| {
                                let step = state
                                    .steps
                                    .iter_mut()
                                    .find(|step| step.id == saved.id)
                                    .expect("actual Model Step");
                                *step = saved;
                                Ok(())
                            },
                            &cancellation,
                        ))
                        .await
                    } else {
                        Ok(Event::Complete {
                            message,
                            usage: usage.clone(),
                            finish_reason: finish_reason.clone(),
                        })
                    };
                    message = match after_result {
                        Ok(Event::Complete { message, .. }) => message,
                        Ok(Event::Delta { .. }) => unreachable!(),
                        Err(error) => {
                            state.done = true;
                            return emitted((
                                if state.cancellation.is_cancelled() {
                                    Ok(cancelled_event(&mut state))
                                } else {
                                    Err(error)
                                },
                                state,
                            ));
                        }
                    };
                    if let Some(planning) = state.planning.take() {
                        if finish_reason.as_deref() == Some("cancelled") {
                            state.done = true;
                            return emitted((Ok(cancelled_event(&mut state)), state));
                        }
                        let updated = crate::core::plan::decode_plan(&message).and_then(|plan| {
                            if let Some(graph) = &mut state.graph {
                                graph.update(plan)
                            } else {
                                state.graph = Some(Graph::from_plan(plan)?);
                                Ok(())
                            }
                        });
                        if let Err(error) = updated {
                            // Forward the actual Complete even when its Plan
                            // is invalid; fail next, without acknowledging it.
                            state.planning_error = Some(error);
                            state.model_pending = false;
                            return emitted((
                                Ok(Event::Complete {
                                    message,
                                    usage,
                                    finish_reason,
                                }),
                                state,
                            ));
                        }
                        if state.id_fn.is_some() {
                            state.plan_execution = true;
                            state.plan_finalizing = false;
                            state.task_error = None;
                            state.dispatch_failed = false;
                        }
                        match planning {
                            Planning::Selected(index, call) => {
                                let plan =
                                    &state.graph.as_ref().expect("validated plan accepted").plan;
                                let result = tool_success_message(
                                    &call,
                                    Message::function(json!({"plan": plan})),
                                );
                                state.tool_results.insert(index, result.clone());
                                state.plan_result = Some(result);
                            }
                            Planning::Evaluation => {
                                // Old results remain graph history, not live model context.
                                // The new plan is executed from the original run input.
                                state.messages = state.task_base.clone();
                            }
                        }
                        state.model_pending = false;
                        return emitted((
                            Ok(Event::Complete {
                                message,
                                usage,
                                finish_reason,
                            }),
                            state,
                        ));
                    }
                    if !resuming_completion {
                        state.messages.push(message.clone());
                    }
                    let calls = match message
                        .content
                        .iter()
                        .filter(|part| part.r#type == "tool_call")
                        .map(decode_call)
                        .collect::<Result<Vec<_>, _>>()
                    {
                        Ok(calls) => calls,
                        Err(error) => {
                            state.done = true;
                            return emitted((Err(error), state));
                        }
                    };
                    if calls.is_empty() {
                        if !resuming_completion {
                            state.completion = Some(crate::runtime::agent::completion::State::new(
                                message,
                                finish_reason.clone(),
                            ));
                        }
                        let saved = state.completion.clone().expect("actual Agent result");
                        let hooks = state.hooks.clone();
                        let cancellation = state.cancellation.clone();
                        message = match Box::pin(super::completion::agent(
                            &hooks,
                            saved,
                            |saved| {
                                state.completion = Some(saved);
                                Ok(())
                            },
                            &cancellation,
                        ))
                        .await
                        {
                            Ok(Event::Complete { message, .. }) => message,
                            Ok(Event::Delta { .. }) => unreachable!(),
                            Err(error) => {
                                state.done = true;
                                return emitted((
                                    if state.cancellation.is_cancelled() {
                                        Ok(cancelled_event(&mut state))
                                    } else {
                                        Err(error)
                                    },
                                    state,
                                ));
                            }
                        };
                        if let Some(evaluator) = state.evaluator.clone() {
                            let evaluation = cancellable(
                                evaluator(
                                    state.query.clone(),
                                    Messages::new([message.clone()]),
                                    state.cancellation.clone(),
                                ),
                                &state.cancellation,
                            )
                            .await;
                            match evaluation {
                                Err(error) => {
                                    state.done = true;
                                    return emitted((
                                        if state.cancellation.is_cancelled() {
                                            Ok(cancelled_event(&mut state))
                                        } else {
                                            Err(error)
                                        },
                                        state,
                                    ));
                                }
                                Ok(Evaluation {
                                    passed: true,
                                    feedback: _,
                                }) => state.done = true,
                                Ok(evaluation) if state.plan_execution && state.graph.is_some() => {
                                    // Only the final answer is evaluated; task children have no
                                    // Agent Evaluator. All graph work is terminal at this point.
                                    state.pending_replan = Some(evaluation);
                                    state.done = false;
                                }
                                Ok(Evaluation {
                                    passed: false,
                                    feedback: Some(feedback),
                                }) => {
                                    for feedback_message in feedback.as_slice() {
                                        state.messages.push(feedback_message.clone());
                                    }
                                    // The rejected final response has already
                                    // been emitted as this model call's
                                    // Complete. A later model call can now
                                    // repair it using the ordinary feedback
                                    // Messages; no Tool error envelope is made.
                                    state.done = false;
                                }
                                Ok(Evaluation {
                                    passed: false,
                                    feedback: None,
                                }) => state.done = true,
                            }
                        } else {
                            state.done = true;
                        }
                        state.completion.as_mut().expect("Agent result").stage = "committed".into();
                    } else {
                        if let Err(error) = tools::selected(&mut state.steps, &calls) {
                            state.done = true;
                            return emitted((Err(error), state));
                        }
                        state.pending_calls = calls.into_iter().enumerate().collect();
                    }
                    state.model_pending = false;
                    return emitted((
                        Ok(Event::Complete {
                            message,
                            usage,
                            finish_reason,
                        }),
                        state,
                    ));
                }
            }
        }

        if state.plan_execution
            && !state.plan_finalizing
            && state.pending_plans.is_empty()
            && state.planning.is_none()
            && state.pending_replan.is_none()
        {
            if let Err(error) = execution::fill_ready(&mut state) {
                // Preserve running siblings and settle their actual calls
                // before reporting a dispatch error.
                state.task_error.get_or_insert(error);
                state.dispatch_failed = true;
            }
            if !state.active_tasks.is_empty() {
                let progress = poll_fn(|cx| {
                    state.task_ready.register(cx.waker());
                    match state.active_tasks.poll_next_unpin(cx) {
                        Poll::Ready(progress) => Poll::Ready(progress),
                        Poll::Pending
                            if state.task_slots.available_permits() > 0
                                && (!state.resume_nodes.is_empty()
                                    || state
                                        .graph
                                        .as_ref()
                                        .is_some_and(|graph| !graph.ready_nodes().is_empty())) =>
                        {
                            Poll::Ready(None)
                        }
                        Poll::Pending => {
                            if state.task_runs.values().all(|active| !active.slot_held())
                                && let Some(run) = &state.interaction
                            {
                                run.suspend();
                            }
                            Poll::Pending
                        }
                    }
                })
                .await;
                let Some(progress) = progress else { continue };
                match execution::commit(&mut state, progress) {
                    Ok((event, save)) => {
                        if !save && event.is_none() {
                            crate::runtime::agent::scope::capture(running_snapshot(&state));
                        }
                        let question = event.as_ref().is_some_and(|event| match event {
                            Event::Complete { message, .. } => {
                                message.metadata.contains_key("reply_id")
                            }
                            _ => false,
                        });
                        if save
                            && let Err(error) = save_checkpoint(
                                &state,
                                if state.waits.is_empty() {
                                    running_snapshot(&state)
                                } else {
                                    waiting_snapshot(&state)
                                },
                            )
                            .await
                        {
                            state.done = true;
                            return emitted((Err(error), state));
                        }
                        if question && state.cancellation.is_cancelled() {
                            // Never present an actionable question after its
                            // owner was cancelled during Checkpoint saving.
                            continue;
                        }
                        if let Some(event) = event {
                            return emitted((Ok(event), state));
                        }
                        continue;
                    }
                    Err(error) => {
                        state.done = true;
                        return emitted((Err(error), state));
                    }
                }
            }
            if let Some(error) = state.task_error.take() {
                state.done = true;
                return emitted((Err(error), state));
            }
            if let Err(error) = execution::finalize(&mut state) {
                state.done = true;
                return emitted((Err(error), state));
            }
        }

        // A delegated plan may have released its enclosing admission while
        // all of its tasks awaited feedback. Its own final inference is work
        // too, and must not bypass capacity after the last task completes.
        if let Some(run) = &state.interaction
            && let Err(error) = run.continue_execution(&state.cancellation).await
        {
            state.done = true;
            return emitted((
                if state.cancellation.is_cancelled() {
                    Ok(cancelled_event(&mut state))
                } else {
                    Err(error)
                },
                state,
            ));
        }
        if (crate::runtime::agent::model_steps::pending_input(&state.steps).is_none()
            || crate::runtime::agent::model_steps::external_wait(&state.steps))
            && state
                .iteration
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                    (used < state.max_iters).then_some(used + 1)
                })
                .is_err()
        {
            state.done = true;
            return emitted((
                Err(Error::new(
                    "MAX_ITERS",
                    "react agent reached max_iters before a final assistant response",
                )),
                state,
            ));
        }

        let request = if let Some(saved) =
            crate::runtime::agent::model_steps::pending_input(&state.steps)
            && !crate::runtime::agent::model_steps::external_wait(&state.steps)
        {
            saved
                .request
                .as_deref()
                .expect("validated Model input")
                .clone()
        } else if let Some(evaluation) = state.pending_replan.take() {
            let request = match crate::core::plan::stream_request(
                &state.query,
                state.graph.as_ref().map(|graph| &graph.plan),
                evaluation.feedback.as_ref(),
                state.stream,
            ) {
                Ok(request) => request,
                Err(error) => {
                    state.done = true;
                    return emitted((Err(error), state));
                }
            };
            state.planning = Some(Planning::Evaluation);
            request
        } else if let Some((index, call)) = state.pending_plans.pop_front() {
            let request = match crate::core::plan::stream_request(
                &state.query,
                state.graph.as_ref().map(|graph| &graph.plan),
                None,
                state.stream,
            ) {
                Ok(request) => request,
                Err(error) => {
                    state.done = true;
                    return emitted((Err(error), state));
                }
            };
            state.planning = Some(Planning::Selected(index, call));
            request
        } else {
            let mut tools = if state.is_plan_task {
                state.dispatch.tool_catalog()
            } else {
                let mut tools = state.dispatch.tool_catalog();
                tools.push(plan_definition());
                if !state.subagents.is_empty() {
                    tools.push(state.subagents[0].definition());
                }
                tools
            };
            if state.is_plan_task && !state.subagents.is_empty() {
                tools.push(state.subagents[0].definition());
            }
            if state.plan_finalizing {
                tools.clear();
            }
            Request {
                messages: state.messages.clone(),
                tool_choice: (!tools.is_empty()).then_some(ToolChoice::Auto),
                tools,
                stream: state.stream,
                ..Request::default()
            }
        };
        state.completion = None;
        state.partial = Message::new(Role::Assistant);
        let saved = crate::runtime::agent::model_steps::input(&mut state.steps, request);
        let hooks = state.hooks.clone();
        let cancellation = state.cancellation.clone();
        let request = match Box::pin(super::lifecycle::model_prepare(
            &hooks,
            saved,
            |saved| {
                let owner = state
                    .steps
                    .iter_mut()
                    .find(|step| step.id == saved.id)
                    .expect("actual Model input owner");
                *owner = saved;
                Ok(())
            },
            &cancellation,
        ))
        .await
        {
            Ok(request) => request,
            Err(error) => {
                state.done = true;
                return emitted((
                    if state.cancellation.is_cancelled() {
                        Ok(cancelled_event(&mut state))
                    } else {
                        Err(error)
                    },
                    state,
                ));
            }
        };
        state.input_tokens = crate::runtime::agent::chat::estimate_input_tokens(&request);
        state.completion = None;
        state.request = Some(Box::new(request.clone()));
        crate::runtime::agent::model_steps::calling(&mut state.steps);
        state.model_pending = true;
        state.model_settled = false;
        state.partial = Message::new(Role::Assistant);
        match cancellable(state.model.stream(request), &state.cancellation).await {
            Ok(provider) => {
                state.provider = Some(Box::pin(provider));
                state.model_span = Some(state.trace.child("agent.model"));
                state.partial = Message::new(Role::Assistant);
            }
            Err(error) => {
                crate::runtime::agent::model_steps::stopped(&mut state.steps, &error);
                state.done = true;
                if state.cancellation.is_cancelled() {
                    return emitted((Ok(cancelled_event(&mut state)), state));
                }
                return emitted((
                    Err(Error::new(error.code, error.message).with_optional_details(error.details)),
                    state,
                ));
            }
        }
    }
}

fn cancelled_event<M: Chat>(state: &mut ReActState<M>) -> Event {
    crate::runtime::agent::model_steps::stopped(
        &mut state.steps,
        &Error::new("CANCELLED", "Agent cancelled"),
    );
    if state.model_pending
        && state.model_settled
        && let Some(message) = super::completion::partial(state.completion.as_ref(), &state.steps)
    {
        state.partial = message;
    }
    state.waits.clear();
    crate::runtime::agent::tool_steps::cancel_pending(&mut state.steps);
    if !state.model_pending {
        // The prior Complete was already settled. Cancellation while executing
        // a Tool/eval/hook must not bill that completed model call a second time.
        let mut message = state.partial.clone();
        message.usage = None;
        return Event::Complete {
            message,
            usage: None,
            finish_reason: Some("cancelled".into()),
        };
    }
    let output = super::chat::estimate_output_tokens(&state.partial);
    let mut usage = state.partial.usage.clone().unwrap_or_default();
    usage.input.get_or_insert(state.input_tokens);
    usage.output.get_or_insert(output);
    usage.total.get_or_insert(
        usage
            .input
            .unwrap_or(0)
            .saturating_add(usage.output.unwrap_or(0)),
    );
    let mut message = state.partial.clone();
    message.usage = Some(usage.clone());
    if state.replayed_completion && state.model_settled {
        state.run.refine_saved(state.partial.usage.as_ref(), &usage);
    } else if state.model_settled {
        state.run.refine_last(Some(usage.clone()));
    } else {
        state.run.settle(Some(usage.clone()));
        crate::runtime::agent::model_steps::unfinished(&mut state.steps, &message, &usage);
    }
    if state.model_settled {
        crate::runtime::agent::model_steps::refine(&mut state.steps, &usage);
        if let Some(completion) = &mut state.completion {
            completion.message.usage = Some(usage.clone());
        }
    }
    state.model_pending = false;
    Event::Complete {
        message,
        usage: Some(usage),
        finish_reason: Some("cancelled".into()),
    }
}

fn decode_call(part: &Part) -> Result<Call, Error> {
    let call_id = part
        .data
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "tool call_id is missing"))?;
    let name = part
        .data
        .get("name")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "tool name is missing"))?;
    let arguments = match part.data.get("arguments") {
        Some(Value::Object(arguments)) => arguments.clone(),
        Some(Value::String(raw)) => serde_json::from_str(raw).map_err(|error| {
            Error::new("INVALID_ARGUMENTS", "tool arguments are not a JSON object")
                .with_details(format!("raw_arguments={raw}; {error}"))
        })?,
        Some(_) => {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "tool arguments must be a JSON object",
            ));
        }
        None => Map::new(),
    };
    Ok(Call {
        call_id: call_id.to_owned(),
        name: name.to_owned(),
        arguments,
    })
}

fn tool_success_message(call: &Call, mut message: Message) -> Message {
    // Keep business content/values, but not execution/question observation
    // metadata that must not leak into the next model input.
    crate::runtime::observation::clear_source(&mut message);
    message
        .metadata
        .insert("call_id".into(), Value::String(call.call_id.clone()));
    message
        .metadata
        .insert("name".into(), Value::String(call.name.clone()));
    message
}

/// An Agent-internal Tool failure is model context, not a successful Tool result.
/// Keep the original common error fields in `Message.values` and put the same
/// direct error object in visible text so the next model call can repair the
/// call without inventing a result envelope.
fn tool_error_message(call: &Call, error: Error) -> Message {
    let error_value = json!({
        "code": error.code,
        "message": error.message,
        "details": error.details,
    });
    let value = serde_json::to_string(&error_value).unwrap_or_else(|_| error_value.to_string());
    Message {
        role: Role::Tool,
        content: vec![Part {
            r#type: "text".into(),
            data: part_data(json!({"value": value})),
        }],
        values: Map::from_iter([
            ("error".into(), error_value),
            ("call_id".into(), Value::String(call.call_id.clone())),
            ("name".into(), Value::String(call.name.clone())),
        ]),
        metadata: Map::new(),
        usage: None,
    }
}

fn collect_delta(partial: &mut Message, delta: &Message) {
    if delta.usage.is_some() {
        partial.usage = delta.usage.clone();
    }
    for part in &delta.content {
        let last = partial.content.last_mut();
        if matches!(part.r#type.as_str(), "text" | "think")
            && let Some(last) = last
            && last.r#type == part.r#type
            && let (Some(left), Some(right)) = (
                last.data.get("value").and_then(Value::as_str),
                part.data.get("value").and_then(Value::as_str),
            )
        {
            last.data["value"] = Value::String(format!("{left}{right}"));
        } else {
            partial.content.push(part.clone());
        }
    }
    partial.values.extend(delta.values.clone());
    partial.metadata.extend(delta.metadata.clone());
}

trait ErrorDetails {
    fn with_optional_details(self, details: Option<String>) -> Self;
}

impl ErrorDetails for Error {
    fn with_optional_details(mut self, details: Option<String>) -> Self {
        self.details = details;
        self
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/agent_progress.rs"]
mod progress_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::{Evaluation, Evaluator};
    use crate::plan::{Edge, Plan, PlanNode};
    use futures_executor::block_on;

    struct Accept;

    impl Evaluator for Accept {
        async fn evaluate(
            &self,
            _input: &Input,
            _result: &Messages,
            _cancellation: &Cancellation,
        ) -> Result<Evaluation, Error> {
            Ok(Evaluation {
                passed: true,
                feedback: None,
            })
        }
    }

    #[test]
    fn evaluator_returns_final_result_judgment_without_node_identity() {
        let evaluation = block_on(Accept.evaluate(
            &Input::default(),
            &Messages::default(),
            &Cancellation::new(),
        ))
        .unwrap();
        assert!(evaluation.passed);
        assert!(evaluation.feedback.is_none());
    }

    fn node(id: &str, name: &str) -> PlanNode {
        PlanNode {
            node_id: id.into(),
            name: name.into(),
            objective: format!("run {name}"),
        }
    }

    #[test]
    fn plan_operations_are_atomic_and_preserve_dag_rules() {
        let mut plan = Plan::new(1).unwrap();
        plan.add(node("a", "search"), vec![], vec![]).unwrap();
        plan.add(node("b", "summarize"), vec!["a".into()], vec![])
            .unwrap();
        let before = plan.clone();
        plan.add(node("c", "any task"), vec!["b".into()], vec![])
            .unwrap();
        assert_ne!(plan, before);

        plan.insert("a", node("d", "review"), "b").unwrap();
        assert!(plan.edges.contains(&Edge {
            from: "a".into(),
            to: "d".into()
        }));
        assert!(plan.edges.contains(&Edge {
            from: "d".into(),
            to: "b".into()
        }));
    }

    #[test]
    fn delete_removes_the_node_and_all_successors() {
        let mut plan = Plan::new(1).unwrap();
        plan.add(node("a", "search"), vec![], vec![]).unwrap();
        plan.add(node("b", "summarize"), vec!["a".into()], vec![])
            .unwrap();
        plan.add(node("c", "review"), vec!["b".into()], vec![])
            .unwrap();
        plan.delete("b").unwrap();
        assert_eq!(plan.nodes.keys().collect::<Vec<_>>(), vec![&"a".to_owned()]);
        assert!(plan.edges.is_empty());
    }

    #[test]
    fn validation_rejects_cycle_and_unknown_edges() {
        let mut plan = Plan::new(1).unwrap();
        plan.add(node("a", "search"), vec![], vec![]).unwrap();
        plan.add(node("b", "summarize"), vec!["a".into()], vec![])
            .unwrap();
        let before = plan.clone();
        assert!(
            plan.add(node("c", "review"), vec!["missing".into()], vec![])
                .is_err()
        );
        assert_eq!(plan, before);
        plan.edges.push(Edge {
            from: "b".into(),
            to: "a".into(),
        });
        assert!(plan.validate().is_err());
    }

    #[test]
    fn plan_json_uses_node_map_keys_as_logical_ids() {
        let mut plan = Plan::new(2).unwrap();
        plan.add(node("search_a", "search"), vec![], vec![])
            .unwrap();
        let value = serde_json::to_value(&plan).unwrap();
        assert_eq!(value["version"], 2);
        assert_eq!(value["nodes"]["search_a"]["name"], "search");
        assert!(value["nodes"]["search_a"].get("node_id").is_none());
        let round_trip: Plan = serde_json::from_value(value).unwrap();
        assert_eq!(round_trip.nodes["search_a"].node_id, "search_a");
    }

    fn application_id() -> i64 {
        panic!("the application ID function must not run during assembly")
    }

    fn uncalled_provider_chat() -> impl Chat {
        crate::core::provider::vllm::Vllm::builder()
            .api_key("compile-only")
            .build()
            .expect("construct provider without sending a request")
            .chat(crate::model::chat::Config::new("compile-only"))
            .expect("construct Provider Chat")
    }

    #[test]
    fn id_function_is_optional_and_is_not_called_by_build() {
        let agent = crate::agent::react()
            .model(uncalled_provider_chat())
            .build()
            .expect("react agent without an ID source builds");
        assert!(agent.id_fn.is_none());

        let configured = crate::agent::react()
            .with_id_fn(application_id)
            .model(uncalled_provider_chat())
            .build()
            .expect("react agent with an application ID source builds");
        assert_eq!(
            configured.id_fn.map(|id| id as usize),
            Some(application_id as *const () as usize)
        );
    }

    #[test]
    fn id_function_survives_model_rebinding_without_memory() {
        let configured = crate::agent::react()
            .model(uncalled_provider_chat())
            .with_id_fn(application_id)
            .model(uncalled_provider_chat())
            .build()
            .expect("model rebinding keeps ID configuration");
        assert_eq!(
            configured.id_fn.map(|id| id as usize),
            Some(application_id as *const () as usize)
        );
    }
}
