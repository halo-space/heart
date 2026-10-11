use std::collections::{BTreeMap, VecDeque};
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
use serde_json::{Map, Value, json};

use crate::components::agent::toolkit::Toolkit;
use crate::components::agent::{Error, Input};
use crate::components::evaluation::{Evaluation, Evaluator};
use crate::components::event::Event;
use crate::components::message::{Message, Messages, Role};
use crate::components::model::chat::{Chat, Request, ToolChoice};
use crate::components::model::content::part::{Part, object as part_data};
use crate::components::tool::{Call, Evaluation as ToolEvaluation, ToolDefinition, validate_call};
use crate::harness::access::Access;
use crate::harness::checkpoint::{Backend, invalid as invalid_checkpoint};
use crate::harness::hooks::{Hooks, cancellable};
use crate::harness::memory::{History, Run};
use crate::runtime::agent::graph::Graph;
use crate::runtime::agent::{GraphState, State as AgentState};
use crate::runtime::cancellation::Cancellation;

mod execution;
use execution::TaskProgress;

type EvaluationFuture = Pin<Box<dyn Future<Output = Result<Evaluation, Error>> + 'static>>;
type EvaluatorHandle =
    std::sync::Arc<dyn Fn(Input, Messages, Cancellation) -> EvaluationFuture + Send + Sync>;
type CheckpointSave = Backend;

fn plan_definition() -> ToolDefinition {
    ToolDefinition {
        name: "plan".into(),
        description: "Create or update a complete task plan for the original user query. Choose this when the task or system instructions call for planning. The plan describes tasks; the Agent decides how to use its Toolkit while executing them. This creates a plan, not execution results.".into(),
        parameters: serde_json::from_value(json!({"type":"object","properties":{},"additionalProperties":false})).expect("static plan Schema is an object"),
    }
}

/// Public builder for the first runnable ReAct execution loop.
pub struct Builder<M = ()> {
    model: Option<M>,
    system_prompt: Option<String>,
    toolkit: Toolkit,
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
        toolkit: Toolkit::new(),
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
            toolkit: self.toolkit,
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

    pub fn toolkit(mut self, toolkit: Toolkit) -> Self {
        self.toolkit = toolkit;
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

    pub fn memory<S, C, T, L>(mut self, config: crate::memory::Config<S, C, T, L>) -> Self
    where
        S: crate::memory::short::session::Store + 'static,
        C: crate::memory::short::chat::Store + 'static,
        T: crate::memory::short::trace::Store + 'static,
        L: crate::memory::Memory + 'static,
    {
        self.history.set(config);
        self
    }

    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Supply application-generated execution IDs independently of Memory.
    /// Configuration and build do not invoke this function. The application
    /// owns uniqueness across concurrent runs and restored executions.
    ///
    /// A fixed ID is not a function:
    /// ```compile_fail
    /// use halo_agents::agent;
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
        if self.toolkit.capability_names().contains("plan") {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "plan is reserved for the built-in Planner",
            ));
        }
        Ok(Agent {
            model: std::sync::Arc::new(model),
            system_prompt: self.system_prompt,
            toolkit: std::sync::Arc::new(self.toolkit),
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
    model: std::sync::Arc<M>,
    system_prompt: Option<String>,
    toolkit: std::sync::Arc<Toolkit>,
    max_iters: usize,
    max_concurrency: usize,
    hooks: Hooks,
    access: Access,
    history: History,
    id_fn: Option<fn() -> i64>,
    evaluator: Option<EvaluatorHandle>,
    checkpoint: Option<CheckpointSave>,
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

    async fn run(&self, input: Input, cancellation: Cancellation) -> Result<Self::Stream, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "react agent run cancelled"));
        }
        let checkpoint_id = self
            .checkpoint
            .as_ref()
            .map(|_| checkpoint_id(&input.metadata))
            .transpose()?;
        let restored = match (&self.checkpoint, checkpoint_id) {
            (Some(checkpoint), Some(id)) => checkpoint.load(id, cancellation.clone()).await?,
            _ => None,
        };
        if let Some(snapshot) = &restored {
            if snapshot
                .get("query")
                .is_some_and(|query| serde_json::to_value(&input).ok().as_ref() != Some(query))
            {
                return Err(Error::new(
                    "CONFLICT",
                    "checkpoint belongs to a different Agent input",
                ));
            }
            if let Some(result) = crate::harness::checkpoint::terminal(snapshot)? {
                return Ok(Box::pin(futures_util::stream::iter([result])));
            }
        }
        let original = input.clone();
        let input = if restored.is_some() {
            input
        } else {
            self.hooks.before_execute(input, &cancellation).await?
        };
        self.history.check_identity(&original, &input)?;
        let prompt = if restored.is_some() {
            None
        } else {
            self.hooks
                .prompt(self.system_prompt.clone(), &cancellation)
                .await?
        };
        let metadata = input.metadata.clone();
        let history = if restored.is_some() {
            Vec::new()
        } else {
            self.history
                .load(metadata.clone(), self.hooks.clone(), &cancellation)
                .await?
        };
        let run = if restored.is_some() {
            self.history.resume(original.clone(), &cancellation).await?
        } else {
            self.history.begin(original.clone(), &cancellation).await?
        };
        let toolkit = match self.toolkit.prepared(&cancellation).await {
            Ok(toolkit) => toolkit,
            Err(error) => return Err(run.fail(error).await),
        };
        if toolkit.capability_names().contains("plan") {
            return Err(run
                .fail(Error::new(
                    "INVALID_ARGUMENTS",
                    "plan is reserved for the built-in Planner",
                ))
                .await);
        }
        let messages =
            crate::components::agent::chat::initial_messages(input, prompt.as_deref(), history);
        let mut state = ReActState {
            model: std::sync::Arc::clone(&self.model),
            toolkit: std::sync::Arc::new(toolkit),
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
            // Retain assembly configuration in this run, not in Input,
            // Messages, Plan or a serialized Graph. Plan execution is enabled
            // only when the caller supplied an ID function; a planning
            // confirmation or ordinary model/Tool call never allocates one.
            id_fn: self.id_fn,
            checkpoint: self.checkpoint.clone(),
            checkpoint_id,
            plan_execution: false,
            active_tasks: FuturesUnordered::new(),
            task_error: None,
            dispatch_failed: false,
            is_plan_task: false,
            tool_slots: Arc::new(tokio::sync::Semaphore::new(self.max_concurrency)),
            plan_finalizing: false,
            pending_plans: VecDeque::new(),
            planning: None,
            pending_replan: None,
            plan_result: None,
            planning_error: None,
            pending_calls: VecDeque::new(),
            active_tools: FuturesUnordered::new(),
            tool_results: BTreeMap::new(),
            evaluations: BTreeMap::new(),
            partial: Message::new(Role::Assistant),
            done: false,
            run,
            model_pending: false,
            model_settled: false,
            input_tokens: 0,
        };
        if let Some(snapshot) = restored {
            restore_state(&mut state, snapshot)?;
        }
        Ok(Box::pin(unfold(state, next_react_event::<M>)))
    }
}

struct ReActState<M: Chat> {
    model: std::sync::Arc<M>,
    toolkit: std::sync::Arc<Toolkit>,
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
    id_fn: Option<fn() -> i64>,
    checkpoint: Option<CheckpointSave>,
    checkpoint_id: Option<i64>,
    plan_execution: bool,
    active_tasks: FuturesUnordered<Pin<Box<dyn Future<Output = TaskProgress<M>> + 'static>>>,
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
    pending_calls: VecDeque<(usize, Call)>,
    active_tools: FuturesUnordered<Pin<Box<dyn Future<Output = ToolProgress> + 'static>>>,
    tool_results: BTreeMap<usize, Message>,
    evaluations: BTreeMap<String, ToolEvaluation>,
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
}

enum PermissionFailure {
    Policy(Error),
    Hook(Error),
    Tool(Error),
}

// Distinguish model-selected control from business evaluation. Replanning is
// not a Tool invocation and must not invent a Call or acknowledge a fake one.
enum Planning {
    Selected(usize, Call),
    Evaluation,
}

enum ToolProgress {
    Called(
        usize,
        Call,
        Box<Result<(Call, Message), PermissionFailure>>,
        Option<tokio::sync::OwnedSemaphorePermit>,
    ),
    Evaluated(String, Result<ToolEvaluation, Error>),
}

// Private per-call dependencies, not a new component Context or Runtime state.
#[derive(Clone)]
struct ToolExecution {
    toolkit: std::sync::Arc<Toolkit>,
    hooks: Hooks,
    access: Access,
    metadata: Map<String, Value>,
    cancellation: Cancellation,
}

async fn check_tool_permission(
    state: &ToolExecution,
    call: &Call,
) -> Result<(), PermissionFailure> {
    let mut metadata = state.metadata.clone();
    metadata.insert(
        "call".into(),
        serde_json::to_value(call).map_err(|_| {
            PermissionFailure::Hook(Error::new(
                "INVALID_ARGUMENTS",
                "cannot encode permission call",
            ))
        })?,
    );
    let request = crate::permission::Request {
        action: "call".into(),
        resource: call.name.clone(),
        metadata,
    };
    let request = state
        .hooks
        .before_permission(request, &state.cancellation)
        .await
        .map_err(PermissionFailure::Hook)?;
    state
        .access
        .check(request.clone(), &state.cancellation)
        .await
        .map_err(PermissionFailure::Policy)?;
    state
        .hooks
        .after_permission(request, &state.cancellation)
        .await
        .map_err(PermissionFailure::Hook)
}

async fn call_with_permission(
    state: &ToolExecution,
    call: &Call,
) -> Result<(Call, Message), PermissionFailure> {
    state
        .toolkit
        .validate(call)
        .map_err(PermissionFailure::Tool)?;
    check_tool_permission(state, call).await?;
    let effective = state
        .toolkit
        .prepare_call(call.clone(), &state.cancellation)
        .await
        .map_err(PermissionFailure::Tool)?;
    if effective.arguments != call.arguments {
        check_tool_permission(state, &effective).await?;
    }
    state
        .toolkit
        .call_prepared(effective, &state.cancellation)
        .await
        .map_err(PermissionFailure::Tool)
}

async fn next_react_event<M: Chat + 'static>(
    state: ReActState<M>,
) -> Option<(Result<Event, Error>, ReActState<M>)> {
    let (result, mut state) = next_react_event_inner(state).await?;
    if result.is_err() && state.model_pending && !state.model_settled {
        state.run.settle(state.partial.usage.clone());
    }
    let result = if state.done {
        let mut terminal = result;
        if let Err(error) = save_checkpoint(&state, checkpoint_snapshot(&state, &terminal)).await {
            terminal = Err(error);
        }
        let (graph, state_json) = closeout_state(&state, &terminal);
        // Drop all pending futures even if the consumer retains the terminal
        // stream. Local resources are released; external effects are not undone.
        state.active_tools.clear();
        state.active_tasks.clear();
        state.pending_calls.clear();
        state.pending_plans.clear();
        state.pending_replan = None;
        state.provider = None;
        state
            .run
            .close_with_state(terminal, graph, state_json)
            .await
    } else {
        result
    };
    Some((result, state))
}

/// Build the single final Agent state projection used by Chat.graph and
/// Trace.state. This is a closeout snapshot, not a second runtime model;
/// the values are serialized from the live execution graph and node states.
fn closeout_state<M: Chat>(
    state: &ReActState<M>,
    result: &Result<Event, Error>,
) -> (Option<Value>, Option<Value>) {
    let status = match result {
        Ok(Event::Complete {
            finish_reason: Some(reason),
            ..
        }) if reason == "cancelled" => "stopped",
        Ok(Event::Complete { .. }) => "succeeded",
        Err(error) if error.code == "CANCELLED" => "stopped",
        Err(_) => "failed",
        Ok(Event::Delta { .. }) => "running",
    };
    let graph_value = state
        .graph
        .as_ref()
        .and_then(|graph| serde_json::to_value(&graph.plan).ok());
    let (graph_state, nodes, history) = state
        .graph
        .as_ref()
        .map(|graph| {
            (
                Some(GraphState {
                    plan: graph.plan.clone(),
                }),
                graph.nodes.clone(),
                graph.history.clone(),
            )
        })
        .unwrap_or((None, BTreeMap::new(), Vec::new()));
    let state_json = serde_json::to_value(AgentState {
        chat_id: state.query.metadata.get("chat_id").and_then(Value::as_i64),
        r#type: "agent".into(),
        status: status.into(),
        usage: state.run.usage(),
        graph: graph_state,
        nodes,
        history,
        messages: state.messages.clone(),
        error: result.as_ref().err().cloned(),
        final_message: match result {
            Ok(Event::Complete { message, .. }) => Some(message.clone()),
            _ => None,
        },
        metadata: BTreeMap::new(),
    })
    .unwrap_or_else(|_| json!({}));
    (graph_value, Some(state_json))
}

fn result_status(result: &Result<Event, Error>) -> &'static str {
    match result {
        Ok(Event::Complete {
            finish_reason: Some(reason),
            ..
        }) if reason == "cancelled" => "stopped",
        Ok(Event::Complete { .. }) => "succeeded",
        Err(error) if error.code == "CANCELLED" => "stopped",
        Err(_) => "failed",
        Ok(Event::Delta { .. }) => "running",
    }
}

/// Build the Agent Runtime checkpoint independently from Chat.graph and
/// Trace.state. A graph enriches the snapshot when present, but it is not
/// a precondition for checkpointing an Agent run.
fn checkpoint_snapshot<M: Chat>(state: &ReActState<M>, result: &Result<Event, Error>) -> Value {
    let (graph, nodes, history) = state
        .graph
        .as_ref()
        .map(|graph| {
            (
                Some(GraphState {
                    plan: graph.plan.clone(),
                }),
                graph.nodes.clone(),
                graph.history.clone(),
            )
        })
        .unwrap_or((None, BTreeMap::new(), Vec::new()));
    let mut snapshot = serde_json::to_value(AgentState {
        chat_id: state.checkpoint_id,
        r#type: "agent".into(),
        status: result_status(result).into(),
        usage: state.run.usage(),
        graph,
        nodes,
        history,
        messages: state.messages.clone(),
        error: result.as_ref().err().cloned(),
        final_message: match result {
            Ok(Event::Complete { message, .. }) => Some(message.clone()),
            _ => None,
        },
        metadata: BTreeMap::new(),
    })
    .unwrap_or_else(|_| json!({}));
    snapshot["query"] = serde_json::to_value(&state.query).unwrap_or(Value::Null);
    snapshot["task_base"] = serde_json::to_value(&state.task_base).unwrap_or(Value::Null);
    snapshot["iteration"] = json!(state.iteration.load(Ordering::Relaxed));
    snapshot["plan_execution"] = json!(state.plan_execution);
    snapshot["plan_finalizing"] = json!(state.plan_finalizing);
    snapshot["tool_results"] = serde_json::to_value(&state.tool_results).unwrap_or(Value::Null);
    crate::harness::checkpoint::record_result(&mut snapshot, result);
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
        if let (Some(save), Some(id)) = (save, id) {
            save.save(id, snapshot, Cancellation::new()).await
        } else {
            Ok(())
        }
    }
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
    state.tool_results = field(&snapshot, "tool_results")?;
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
        let mut interrupted = false;
        let mut waiting = false;
        for node in graph.nodes.values_mut() {
            if node.status == Status::Running {
                node.status = Status::Interrupted;
                node.set_attempt_result(
                    Status::Interrupted,
                    None,
                    Some(Error::new(
                        "INTERRUPTED",
                        "Agent execution interrupted before checkpoint recovery",
                    )),
                )?;
                interrupted = true;
            } else if matches!(node.status, Status::Waiting | Status::Interrupted) {
                waiting |= node.status == Status::Waiting;
                interrupted |= node.status == Status::Interrupted;
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
        state.graph = Some(graph);
        if waiting || snapshot["status"] == "waiting" || snapshot["status"] == "paused" {
            // No response protocol is inferred from ordinary Input. Keep the
            // saved wait intact until explicit resume integration is provided.
            return Err(Error::new(
                "WAITING",
                "Agent checkpoint requires an external resume response",
            ));
        }
        if interrupted {
            state.planning_error = Some(Error::new(
                "INTERRUPTED",
                "unfinished Agent calls require an explicit retry policy",
            ));
        }
    } else {
        // A live provider/Tool call cannot be resumed from messages alone.
        state.planning_error = Some(Error::new(
            "INTERRUPTED",
            "unfinished Agent call cannot be replayed from message history",
        ));
    }
    Ok(())
}

fn running_snapshot<M: Chat>(state: &ReActState<M>) -> Value {
    checkpoint_snapshot(
        state,
        &Ok(Event::Delta {
            message: state.partial.clone(),
        }),
    )
}

async fn next_react_event_inner<M: Chat + 'static>(
    mut state: ReActState<M>,
) -> Option<(Result<Event, Error>, ReActState<M>)> {
    loop {
        if state.done {
            return None;
        }

        if state.cancellation.is_cancelled() && state.active_tasks.is_empty() {
            state.done = true;
            return Some((Ok(cancelled_event(&mut state)), state));
        }

        if let Some(error) = state.planning_error.take() {
            state.done = true;
            return Some((Err(error), state));
        }

        if let Some(message) = state.plan_result.take() {
            return Some((
                Ok(Event::Complete {
                    message,
                    usage: None,
                    finish_reason: Some("tool".into()),
                }),
                state,
            ));
        }

        while state.active_tools.len() < state.max_concurrency {
            let Some((index, call)) = state.pending_calls.pop_front() else {
                break;
            };
            if call.name == "plan" && !state.is_plan_task {
                if let Err(error) = validate_call(&plan_definition(), &call) {
                    let message = tool_error_message(&call, error);
                    state.tool_results.insert(index, message.clone());
                    return Some((
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
            let execution = ToolExecution {
                toolkit: state.toolkit.clone(),
                hooks: state.hooks.clone(),
                access: state.access.clone(),
                metadata: state.metadata.clone(),
                cancellation: state.cancellation.clone(),
            };
            let slots = state.tool_slots.clone();
            state.active_tools.push(Box::pin(async move {
                let permit = match cancellable(
                    async {
                        slots
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
                            Box::new(Err(PermissionFailure::Hook(error))),
                            None,
                        );
                    }
                };
                let result = if execution.access.is_configured() {
                    call_with_permission(&execution, &call).await
                } else {
                    execution
                        .toolkit
                        .call(call.clone(), &execution.cancellation)
                        .await
                        .map_err(PermissionFailure::Tool)
                };
                ToolProgress::Called(index, call, Box::new(result), Some(permit))
            }));
        }

        if !state.active_tools.is_empty() {
            let progress = poll_fn(|context| {
                state.cancellation.register(context.waker());
                if state.cancellation.is_cancelled() {
                    return Poll::Ready(None);
                }
                state.active_tools.poll_next_unpin(context)
            })
            .await;
            if state.cancellation.is_cancelled() {
                state.done = true;
                return Some((Ok(cancelled_event(&mut state)), state));
            }
            match progress.expect("nonempty Tool futures remain until polled") {
                ToolProgress::Evaluated(call_id, Ok(evaluation)) => {
                    // Business feedback is not a technical model repair error.
                    state.evaluations.insert(call_id, evaluation);
                    continue;
                }
                ToolProgress::Evaluated(_, Err(error)) => {
                    // Default zero retries: never repeat a successful call to
                    // recover its eval. The emitted result remains unchanged.
                    state.done = true;
                    return Some((Err(error), state));
                }
                ToolProgress::Called(index, call, result, permit) => {
                    let message = match *result {
                        Ok((effective_call, business)) => {
                            let toolkit = state.toolkit.clone();
                            let cancellation = state.cancellation.clone();
                            let evaluation_message = business.clone();
                            // Keep this lifecycle's slot occupied through eval,
                            // but publish the real call/after result first.
                            state.active_tools.push(Box::pin(async move {
                                let result = toolkit
                                    .evaluate(&effective_call, &evaluation_message, &cancellation)
                                    .await;
                                drop(permit);
                                ToolProgress::Evaluated(effective_call.call_id, result)
                            }));
                            tool_success_message(&call, business)
                        }
                        Err(PermissionFailure::Policy(error) | PermissionFailure::Tool(error)) => {
                            tool_error_message(&call, error)
                        }
                        Err(PermissionFailure::Hook(error)) => {
                            state.done = true;
                            return Some((Err(error), state));
                        }
                    };
                    state.tool_results.insert(index, message.clone());
                    return Some((
                        Ok(Event::Complete {
                            message,
                            usage: None,
                            finish_reason: Some("tool".into()),
                        }),
                        state,
                    ));
                }
            }
        }

        // Events may arrive in completion order. Model context stays in the
        // original Call order, and no next inference sees an incomplete batch.
        if state.pending_plans.is_empty() && state.planning.is_none() {
            for (_, message) in std::mem::take(&mut state.tool_results) {
                state.messages.push(message);
            }
        }

        if let Some(provider) = state.provider.as_mut() {
            let item = poll_fn(|context| {
                state.cancellation.register(context.waker());
                if state.cancellation.is_cancelled() {
                    return Poll::Ready(None);
                }
                provider.as_mut().poll_next(context)
            })
            .await;
            if state.cancellation.is_cancelled() {
                state.done = true;
                return Some((Ok(cancelled_event(&mut state)), state));
            }
            let Some(item) = item else {
                state.done = true;
                return Some((
                    Err(Error::new(
                        "INVALID_RESPONSE",
                        "model did not return Complete",
                    )),
                    state,
                ));
            };
            match item {
                Err(error) => {
                    state.done = true;
                    return Some((
                        Err(Error::new(error.code, error.message)
                            .with_optional_details(error.details)),
                        state,
                    ));
                }
                Ok(Event::Delta { message }) => {
                    collect_delta(&mut state.partial, &message);
                    return Some((Ok(Event::Delta { message }), state));
                }
                Ok(Event::Complete {
                    mut message,
                    usage,
                    finish_reason,
                }) => {
                    state.provider = None;
                    let usage = usage.or_else(|| message.usage.clone());
                    state.run.settle(usage.clone());
                    state.model_settled = true;
                    message.usage = usage.clone();
                    state.partial = message.clone();
                    message = match state.hooks.after_model(message, &state.cancellation).await {
                        Ok(message) => message,
                        Err(error) => {
                            state.done = true;
                            return Some((
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
                            return Some((Ok(cancelled_event(&mut state)), state));
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
                            return Some((
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
                        return Some((
                            Ok(Event::Complete {
                                message,
                                usage,
                                finish_reason,
                            }),
                            state,
                        ));
                    }
                    state.messages.push(message.clone());
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
                            return Some((Err(error), state));
                        }
                    };
                    if calls.is_empty() {
                        message = match state
                            .hooks
                            .after_execute(message, &state.cancellation)
                            .await
                        {
                            Ok(message) => message,
                            Err(error) => {
                                state.done = true;
                                return Some((
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
                                    return Some((
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
                    } else {
                        state.pending_calls = calls.into_iter().enumerate().collect();
                    }
                    state.model_pending = false;
                    return Some((
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
                let progress = state.active_tasks.next().await.expect("active task");
                match execution::commit(&mut state, progress) {
                    Ok((event, terminal)) => {
                        if terminal
                            && let Err(error) =
                                save_checkpoint(&state, running_snapshot(&state)).await
                        {
                            state.done = true;
                            return Some((Err(error), state));
                        }
                        if let Some(event) = event {
                            return Some((Ok(event), state));
                        }
                        continue;
                    }
                    Err(error) => {
                        state.done = true;
                        return Some((Err(error), state));
                    }
                }
            }
            if let Some(error) = state.task_error.take() {
                state.done = true;
                return Some((Err(error), state));
            }
            if let Err(error) = execution::finalize(&mut state) {
                state.done = true;
                return Some((Err(error), state));
            }
        }

        if state
            .iteration
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                (used < state.max_iters).then_some(used + 1)
            })
            .is_err()
        {
            state.done = true;
            return Some((
                Err(Error::new(
                    "MAX_ITERS",
                    "react agent reached max_iters before a final assistant response",
                )),
                state,
            ));
        }

        let request = if let Some(evaluation) = state.pending_replan.take() {
            let request = match crate::core::plan::stream_request(
                &state.query,
                state.graph.as_ref().map(|graph| &graph.plan),
                evaluation.feedback.as_ref(),
            ) {
                Ok(request) => request,
                Err(error) => {
                    state.done = true;
                    return Some((Err(error), state));
                }
            };
            state.planning = Some(Planning::Evaluation);
            request
        } else if let Some((index, call)) = state.pending_plans.pop_front() {
            let request = match crate::core::plan::stream_request(
                &state.query,
                state.graph.as_ref().map(|graph| &graph.plan),
                None,
            ) {
                Ok(request) => request,
                Err(error) => {
                    state.done = true;
                    return Some((Err(error), state));
                }
            };
            state.planning = Some(Planning::Selected(index, call));
            request
        } else {
            let mut tools = if state.is_plan_task {
                state.toolkit.definitions()
            } else {
                let mut tools = state.toolkit.definitions();
                tools.push(plan_definition());
                tools
            };
            if state.plan_finalizing {
                tools.clear();
            }
            Request {
                messages: state.messages.clone(),
                tools,
                tool_choice: Some(ToolChoice::Auto),
                stream: true,
                ..Request::default()
            }
        };
        let request = match state.hooks.before_model(request, &state.cancellation).await {
            Ok(request) => request,
            Err(error) => {
                state.done = true;
                return Some((
                    if state.cancellation.is_cancelled() {
                        Ok(cancelled_event(&mut state))
                    } else {
                        Err(error)
                    },
                    state,
                ));
            }
        };
        state.input_tokens = crate::components::agent::chat::estimate_input_tokens(&request);
        state.model_pending = true;
        state.model_settled = false;
        state.partial = Message::new(Role::Assistant);
        match cancellable(state.model.stream(request), &state.cancellation).await {
            Ok(provider) => {
                state.provider = Some(Box::pin(provider));
                state.partial = Message::new(Role::Assistant);
            }
            Err(error) => {
                state.done = true;
                if state.cancellation.is_cancelled() {
                    return Some((Ok(cancelled_event(&mut state)), state));
                }
                return Some((
                    Err(Error::new(error.code, error.message).with_optional_details(error.details)),
                    state,
                ));
            }
        }
    }
}

fn cancelled_event<M: Chat>(state: &mut ReActState<M>) -> Event {
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
    let output = estimate_message_tokens(&state.partial);
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
    if state.model_settled {
        state.run.refine_last(Some(usage.clone()));
    } else {
        state.run.settle(Some(usage.clone()));
    }
    state.model_pending = false;
    Event::Complete {
        message,
        usage: Some(usage),
        finish_reason: Some("cancelled".into()),
    }
}

fn estimate_message_tokens(message: &Message) -> u64 {
    serde_json::to_vec(message)
        .map(|bytes| bytes.len() as u64 / 4)
        .unwrap_or(0)
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
    // Keep the Tool's own Message untouched as the business result. The
    // invocation identity is correlation metadata, not a result envelope.
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
mod tests {
    use super::*;
    use crate::components::agent::Agent as AgentContract;
    use crate::components::tool::{Tool, ToolDefinition};
    use crate::evaluation::{Evaluation, Evaluator};
    use crate::plan::{Edge, Plan, PlanNode};
    use futures_executor::block_on;
    use futures_util::StreamExt;
    use std::sync::{Arc, Mutex};

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

    struct RepairChat {
        calls: Arc<Mutex<usize>>,
    }

    impl Chat for RepairChat {
        type Stream = futures_util::stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

        async fn stream(&self, _request: Request) -> Result<Self::Stream, Error> {
            let mut calls = self.calls.lock().expect("repair chat lock poisoned");
            let text = if *calls == 0 { "bad" } else { "good" };
            *calls += 1;
            let mut message = Message::new(Role::Assistant);
            message.content.push(Part {
                r#type: "text".into(),
                data: part_data(json!({"value": text})),
            });
            Ok(futures_util::stream::iter(vec![Ok(Event::Complete {
                message,
                usage: None,
                finish_reason: Some("stop".into()),
            })]))
        }
    }

    #[test]
    fn optional_agent_evaluator_feeds_business_feedback_to_next_model_call() {
        let calls = Arc::new(Mutex::new(0));
        let evaluator = |_: Input, result: Messages, _: Cancellation| async move {
            let value = result
                .as_slice()
                .first()
                .and_then(|message| message.content.first())
                .and_then(|part| part.data.get("value"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(Evaluation {
                passed: value == "good",
                feedback: (value == "bad").then(|| {
                    Messages::new([Message::function(json!({
                        "feedback": "please repair the answer"
                    }))])
                }),
            })
        };
        let agent = crate::agent::react()
            .model(RepairChat {
                calls: calls.clone(),
            })
            .evaluator(evaluator)
            .build()
            .unwrap();
        let stream = block_on(agent.run(Input::default(), Cancellation::new())).unwrap();
        let events = block_on(stream.collect::<Vec<_>>());
        assert_eq!(*calls.lock().unwrap(), 2);
        let complete = events
            .into_iter()
            .filter_map(Result::ok)
            .filter_map(|event| match event {
                Event::Complete { message, .. } => Some(message),
                Event::Delta { .. } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(complete.len(), 2);
        assert_eq!(
            complete.last().unwrap().content[0].data["value"],
            Value::String("good".into())
        );
    }

    #[test]
    fn terminal_checkpoint_is_idempotently_returned_without_replaying_model() {
        let calls = Arc::new(Mutex::new(0));
        let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
        let agent = crate::agent::react()
            .model(RepairChat {
                calls: calls.clone(),
            })
            .checkpoint(checkpoint)
            .build()
            .unwrap();
        let input = Input {
            metadata: Map::from_iter([("chat_id".into(), json!(901))]),
            payload: Vec::new(),
        };
        let first = block_on(agent.run(input.clone(), Cancellation::new())).unwrap();
        let first_events = block_on(first.collect::<Vec<_>>());
        assert!(first_events.iter().any(|event| event.is_ok()));
        assert_eq!(*calls.lock().unwrap(), 1);

        let restored = block_on(agent.run(input, Cancellation::new())).unwrap();
        let restored_events = block_on(restored.collect::<Vec<_>>());
        assert!(matches!(
            restored_events.as_slice(),
            [Ok(Event::Complete { finish_reason: Some(reason), .. })]
                if reason == "stop"
        ));
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    struct CheckpointChat {
        requests: Arc<Mutex<Vec<Request>>>,
        replies: Mutex<VecDeque<Vec<Result<Event, Error>>>>,
    }

    impl Chat for CheckpointChat {
        type Stream = futures_util::stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

        async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
            self.requests.lock().unwrap().push(request);
            Ok(futures_util::stream::iter(
                self.replies
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("unexpected model replay"),
            ))
        }
    }

    fn checkpoint_input() -> Input {
        Input {
            metadata: Map::from_iter([("chat_id".into(), json!(902))]),
            payload: Vec::new(),
        }
    }

    fn checkpoint_message(text: &str) -> Message {
        let mut message = Message::new(Role::Assistant);
        message.values.insert("answer".into(), json!(text));
        message.content.push(Part {
            r#type: "text".into(),
            data: part_data(json!({"value": text})),
        });
        message
    }

    fn checkpoint_complete(text: &str) -> Result<Event, Error> {
        Ok(Event::Complete {
            message: checkpoint_message(text),
            usage: Some(crate::model::token::Usage {
                input: Some(10),
                output: Some(5),
                total: Some(15),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })
    }

    /// A committed running scene; live futures are intentionally not persisted.
    fn checkpoint_fixture(graph: Option<&Graph>) -> Value {
        let messages = Messages::new([checkpoint_message("saved context")]);
        let mut snapshot = serde_json::to_value(AgentState {
            chat_id: Some(902),
            r#type: "agent".into(),
            status: "running".into(),
            graph: graph.map(|graph| GraphState {
                plan: graph.plan.clone(),
            }),
            nodes: graph.map(|graph| graph.nodes.clone()).unwrap_or_default(),
            history: graph.map(|graph| graph.history.clone()).unwrap_or_default(),
            messages: messages.clone(),
            ..Default::default()
        })
        .unwrap();
        snapshot["query"] = serde_json::to_value(checkpoint_input()).unwrap();
        snapshot["task_base"] = serde_json::to_value(messages).unwrap();
        snapshot["iteration"] = json!(1);
        snapshot["plan_execution"] = json!(graph.is_some());
        snapshot["plan_finalizing"] = json!(false);
        snapshot["tool_results"] = json!({});
        snapshot
    }

    #[test]
    fn no_graph_checkpoint_accepts_missing_or_null_graph_without_replaying_calls() {
        use crate::checkpoint::Checkpoint as _;
        block_on(async {
            for null_graph in [false, true] {
                let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
                let mut snapshot = checkpoint_fixture(None);
                if null_graph {
                    snapshot["graph"] = Value::Null;
                }
                let saved_messages = snapshot["messages"].clone();
                checkpoint
                    .save(902, snapshot, &Cancellation::new())
                    .await
                    .unwrap();
                let requests = Arc::new(Mutex::new(Vec::new()));
                let agent = crate::agent::react()
                    .model(CheckpointChat {
                        requests: requests.clone(),
                        replies: Mutex::new(VecDeque::new()),
                    })
                    .checkpoint(checkpoint.clone())
                    .build()
                    .unwrap();
                let events = agent
                    .run(checkpoint_input(), Cancellation::new())
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                assert!(matches!(events.as_slice(), [Err(error)] if error.code == "INTERRUPTED"));
                assert!(requests.lock().unwrap().is_empty());
                let saved = checkpoint.load(902, &Cancellation::new()).await.unwrap();
                assert_eq!(saved["messages"], saved_messages);
                assert_eq!(saved["status"], "failed");
                assert!(saved.get("graph").is_none());
                assert!(saved.get("nodes").is_none());
                let typed: AgentState = serde_json::from_value(saved).unwrap();
                assert_eq!(
                    typed,
                    serde_json::from_value(serde_json::to_value(&typed).unwrap()).unwrap()
                );
            }
        });
    }

    #[test]
    fn restored_graph_reuses_success_and_executes_only_ready_successors() {
        use crate::checkpoint::Checkpoint as _;
        block_on(async {
            let mut plan = Plan::new(1).unwrap();
            plan.add(node("a", "research"), vec![], vec![]).unwrap();
            plan.add(node("b", "summarize"), vec!["a".into()], vec![])
                .unwrap();
            let mut graph = Graph::new(plan).unwrap();
            graph.start("a", 9001).unwrap();
            graph
                .succeed(
                    "a",
                    Messages::new([checkpoint_message("saved source")]),
                    None,
                )
                .unwrap();
            // Replan history remains top-level and is not mixed with the current graph.
            graph.history.push(crate::runtime::agent::graph::History {
                plan: graph.plan.clone(),
                nodes: graph.nodes.clone(),
            });
            let source = graph.nodes["a"].clone();
            let history = graph.history.clone();
            let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
            checkpoint
                .save(902, checkpoint_fixture(Some(&graph)), &Cancellation::new())
                .await
                .unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let agent = crate::agent::react()
                .model(CheckpointChat {
                    requests: requests.clone(),
                    replies: Mutex::new(VecDeque::from([
                        vec![checkpoint_complete("successor")],
                        vec![checkpoint_complete("final")],
                    ])),
                })
                .with_id_fn(|| 9002)
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap();
            let events = agent
                .run(checkpoint_input(), Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert!(events.iter().all(Result::is_ok), "{events:?}");
            let (request_count, input_values) = {
                let requests = requests.lock().unwrap();
                (
                    requests.len(),
                    requests[0]
                        .messages
                        .as_slice()
                        .last()
                        .unwrap()
                        .values
                        .clone(),
                )
            };
            assert_eq!(request_count, 2); // Successor and final summary; no source replay.
            assert_eq!(input_values["plan_node"], "b");
            assert_eq!(
                input_values["sources"]["a"][0]["values"]["answer"],
                "saved source"
            );
            let saved = checkpoint.load(902, &Cancellation::new()).await.unwrap();
            let typed: AgentState = serde_json::from_value(saved).unwrap();
            assert_eq!(typed.nodes["a"], source);
            assert_eq!(typed.history, history);
            assert_eq!(
                typed.nodes["b"].status,
                crate::runtime::agent::node::Status::Succeeded
            );
            assert_eq!(typed.nodes["b"].execution.as_ref().unwrap().exec_id, 9002);
            assert_eq!(
                typed,
                serde_json::from_value(serde_json::to_value(&typed).unwrap()).unwrap()
            );
        });
    }

    #[test]
    fn interrupted_graph_preserves_attempts_without_automatically_replaying() {
        use crate::checkpoint::Checkpoint as _;
        use crate::runtime::agent::node::Status;
        block_on(async {
            let mut plan = Plan::new(1).unwrap();
            plan.add(node("a", "research"), vec![], vec![]).unwrap();
            let mut graph = Graph::new(plan).unwrap();
            graph.start("a", 9001).unwrap();
            let created_time =
                graph.nodes["a"].execution.as_ref().unwrap().attempts[0].created_time;
            let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
            checkpoint
                .save(902, checkpoint_fixture(Some(&graph)), &Cancellation::new())
                .await
                .unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let agent = crate::agent::react()
                .model(CheckpointChat {
                    requests: requests.clone(),
                    replies: Mutex::new(VecDeque::new()),
                })
                .with_id_fn(|| panic!("must not allocate another execution identity"))
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap();
            let events = agent
                .run(checkpoint_input(), Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert!(matches!(events.as_slice(), [Err(error)] if error.code == "INTERRUPTED"));
            assert!(requests.lock().unwrap().is_empty());
            let saved = checkpoint.load(902, &Cancellation::new()).await.unwrap();
            let typed: AgentState = serde_json::from_value(saved).unwrap();
            let execution = typed.nodes["a"].execution.as_ref().unwrap();
            assert_eq!(execution.exec_id, 9001);
            assert_eq!(execution.attempts.len(), 1);
            assert_eq!(execution.attempts[0].created_time, created_time);
            assert_eq!(execution.attempts[0].status, Status::Interrupted);
            assert_eq!(
                execution.attempts[0].error.as_ref().unwrap().code,
                "INTERRUPTED"
            );
        });
    }

    #[test]
    fn terminal_snapshots_round_trip_success_failure_and_cancellation_without_audit_reads() {
        use crate::checkpoint::Checkpoint as _;
        block_on(async {
            for with_graph in [false, true] {
                for status in ["succeeded", "failed", "stopped"] {
                    let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
                    if with_graph {
                        let mut plan = Plan::new(1).unwrap();
                        plan.add(node("a", "research"), vec![], vec![]).unwrap();
                        let mut graph = Graph::new(plan).unwrap();
                        graph.start("a", 9001).unwrap();
                        graph
                            .succeed(
                                "a",
                                Messages::new([checkpoint_message("saved source")]),
                                None,
                            )
                            .unwrap();
                        let mut snapshot = checkpoint_fixture(Some(&graph));
                        snapshot["plan_finalizing"] = json!(true);
                        checkpoint
                            .save(902, snapshot, &Cancellation::new())
                            .await
                            .unwrap();
                    }
                    let requests = Arc::new(Mutex::new(Vec::new()));
                    let delta = Ok(Event::Delta {
                        message: checkpoint_message("partial"),
                    });
                    let replies = match status {
                        "succeeded" => vec![checkpoint_complete("final")],
                        "failed" => vec![
                            delta,
                            Err(Error::new("UNAVAILABLE", "network interrupted")
                                .with_details("connection closed")),
                        ],
                        _ => vec![
                            delta,
                            checkpoint_complete("must not be read after cancellation"),
                        ],
                    };
                    let agent = crate::agent::react()
                        .model(CheckpointChat {
                            requests: requests.clone(),
                            replies: Mutex::new(VecDeque::from([replies])),
                        })
                        .checkpoint(checkpoint.clone())
                        .build()
                        .unwrap();
                    let cancellation = Cancellation::new();
                    let mut stream = agent
                        .run(checkpoint_input(), cancellation.clone())
                        .await
                        .unwrap();
                    let mut events = Vec::new();
                    while let Some(event) = stream.next().await {
                        if status == "stopped" && matches!(&event, Ok(Event::Delta { .. })) {
                            cancellation.cancel();
                        }
                        events.push(event);
                    }
                    let terminal = events.last().unwrap().clone();
                    let saved = checkpoint.load(902, &Cancellation::new()).await.unwrap();
                    assert_eq!(saved["status"], status);
                    let typed: AgentState = serde_json::from_value(saved.clone()).unwrap();
                    assert_eq!(typed.graph.is_some(), with_graph);
                    assert_eq!(
                        typed,
                        serde_json::from_value(serde_json::to_value(&typed).unwrap()).unwrap()
                    );
                    if status == "failed" {
                        assert_eq!(
                            typed.error.as_ref().unwrap().details.as_deref(),
                            Some("connection closed")
                        );
                    }
                    if status == "stopped" {
                        assert_eq!(
                            typed.final_message.as_ref().unwrap().content[0].data["value"],
                            "partial"
                        );
                        assert!(typed.usage.as_ref().unwrap().input.is_some());
                    }
                    // Empty Memory backends would fail if recovery read Session/Chat/Trace.
                    // A terminal Checkpoint must return before any audit/history I/O.
                    let memory = crate::memory::Config::new(
                        crate::core::memory::short::session::InMemory::new(),
                        crate::core::memory::short::chat::InMemory::new(),
                        crate::core::memory::short::trace::InMemory::new(),
                        crate::core::memory::InMemory::new(),
                    );
                    let restored = crate::agent::react()
                        .model(CheckpointChat {
                            requests: requests.clone(),
                            replies: Mutex::new(VecDeque::new()),
                        })
                        .memory(memory)
                        .checkpoint(checkpoint.clone())
                        .build()
                        .unwrap();
                    let replay = restored
                        .run(checkpoint_input(), Cancellation::new())
                        .await
                        .unwrap()
                        .collect::<Vec<_>>()
                        .await;
                    assert_eq!(replay, vec![terminal]);
                    assert_eq!(requests.lock().unwrap().len(), 1);
                    assert_eq!(
                        checkpoint.load(902, &Cancellation::new()).await.unwrap(),
                        saved
                    );
                }
            }
        });
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

    struct SequenceChat {
        calls: Arc<Mutex<usize>>,
        require_tool_error: bool,
    }

    impl Chat for SequenceChat {
        type Stream = futures_util::stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

        async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
            assert!(request.stream);
            let mut calls = self.calls.lock().expect("sequence chat lock poisoned");
            let index = *calls;
            *calls += 1;
            drop(calls);

            if index == 1 && self.require_tool_error {
                assert!(request.messages.as_slice().iter().any(|message| {
                    message.values.get("error").is_some_and(|error| {
                        error.get("code") == Some(&Value::String("TIMEOUT".into()))
                    })
                }));
            }

            let message = if index == 0 {
                let mut message = Message::new(Role::Assistant);
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: part_data(json!({
                        "call_id": "call_weather_1",
                        "name": "weather",
                        "arguments": {"city": "Beijing"}
                    })),
                });
                message
            } else {
                let mut message = Message::new(Role::Assistant);
                message.content.push(Part {
                    r#type: "text".into(),
                    data: part_data(json!({"value": "北京今天晴天。"})),
                });
                message
            };

            Ok(futures_util::stream::iter(vec![Ok(Event::Complete {
                message,
                usage: None,
                finish_reason: Some("stop".into()),
            })]))
        }
    }

    struct WeatherTool {
        calls: Arc<Mutex<usize>>,
        definition: ToolDefinition,
    }

    struct FailingTool {
        definition: ToolDefinition,
    }

    struct EvalFailingTool {
        inner: WeatherTool,
    }
    impl Tool for EvalFailingTool {
        fn definition(&self) -> &ToolDefinition {
            self.inner.definition()
        }
        async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
            self.inner.call(call, cancellation).await
        }
        async fn eval(
            &self,
            _call: &Call,
            _message: &Message,
            _cancellation: &Cancellation,
        ) -> Result<crate::tool::Evaluation, Error> {
            Err(Error::new("TIMEOUT", "evaluation service timed out"))
        }
    }

    #[test]
    fn eval_failure_preserves_successful_tool_complete_and_does_not_repeat_call() {
        let model_calls = Arc::new(Mutex::new(0));
        let tool_calls = Arc::new(Mutex::new(0));
        let agent = crate::agent::react()
            .model(SequenceChat {
                calls: model_calls.clone(),
                require_tool_error: false,
            })
            .toolkit(Toolkit::new().tool(EvalFailingTool {
                inner: WeatherTool {
                    calls: tool_calls.clone(),
                    definition: ToolDefinition {
                        name: "weather".into(),
                        description: "weather".into(),
                        parameters: serde_json::from_value(json!({"type": "object"})).unwrap(),
                    },
                },
            }))
            .build()
            .unwrap();
        let mut events = block_on(agent.run(Input::default(), Cancellation::new())).unwrap();
        assert!(matches!(
            block_on(events.next()),
            Some(Ok(Event::Complete { .. }))
        ));
        let Some(Ok(Event::Complete { message, .. })) = block_on(events.next()) else {
            panic!("successful Tool result must be delivered independently of eval");
        };
        assert_eq!(message.values["temperature"], 22);
        assert!(!message.values.contains_key("error"));
        let Some(Err(error)) = block_on(events.next()) else {
            panic!("eval failure must remain a separate error");
        };
        assert_eq!(error.code, "TIMEOUT");
        assert!(block_on(events.next()).is_none());
        assert_eq!(*model_calls.lock().unwrap(), 1);
        assert_eq!(*tool_calls.lock().unwrap(), 1);
    }

    impl Tool for FailingTool {
        fn definition(&self) -> &ToolDefinition {
            &self.definition
        }

        async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
            Err(Error::new("TIMEOUT", "weather service timed out"))
        }
    }

    impl Tool for WeatherTool {
        fn definition(&self) -> &ToolDefinition {
            &self.definition
        }

        async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
            *self.calls.lock().expect("weather tool lock poisoned") += 1;
            Ok(Message::function(json!({"temperature": 22})))
        }
    }

    #[test]
    fn public_react_entry_runs_tool_then_final_model_call() {
        let model_calls = Arc::new(Mutex::new(0));
        let tool_calls = Arc::new(Mutex::new(0));
        let toolkit = Toolkit::new().tool(WeatherTool {
            calls: tool_calls.clone(),
            definition: ToolDefinition {
                name: "weather".into(),
                description: "lookup weather".into(),
                parameters: serde_json::from_value(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                }))
                .expect("tool schema"),
            },
        });
        let agent = crate::agent::react()
            .model(SequenceChat {
                calls: model_calls.clone(),
                require_tool_error: false,
            })
            .toolkit(toolkit)
            .with_max_iters(2)
            .build()
            .expect("react agent build");

        let mut events =
            block_on(agent.run(Input::default(), Cancellation::new())).expect("react agent run");
        assert!(matches!(
            block_on(events.next()),
            Some(Ok(Event::Complete { .. }))
        ));
        assert!(
            matches!(block_on(events.next()), Some(Ok(Event::Complete { message, .. })) if message.role == Role::Function)
        );
        assert!(
            matches!(block_on(events.next()), Some(Ok(Event::Complete { message, .. })) if message.content[0].data["value"] == "北京今天晴天。")
        );
        assert!(block_on(events.next()).is_none());
        assert_eq!(*model_calls.lock().expect("model calls lock poisoned"), 2);
        assert_eq!(*tool_calls.lock().expect("tool calls lock poisoned"), 1);
    }

    #[test]
    fn react_rejects_zero_max_iters_and_reports_exhaustion() {
        let error = crate::agent::react()
            .model(SequenceChat {
                calls: Arc::new(Mutex::new(0)),
                require_tool_error: false,
            })
            .with_max_iters(0)
            .build()
            .err()
            .expect("zero max_iters must be rejected");
        assert_eq!(error.code, "INVALID_ARGUMENTS");

        let agent = crate::agent::react()
            .model(SequenceChat {
                calls: Arc::new(Mutex::new(0)),
                require_tool_error: false,
            })
            .with_max_iters(1)
            .toolkit(
                Toolkit::new().tool(WeatherTool {
                    calls: Arc::new(Mutex::new(0)),
                    definition: ToolDefinition {
                        name: "weather".into(),
                        description: "lookup weather".into(),
                        parameters: serde_json::from_value(json!({
                            "type": "object",
                            "properties": {}
                        }))
                        .expect("tool schema"),
                    },
                }),
            )
            .build()
            .expect("react agent build");
        let mut events =
            block_on(agent.run(Input::default(), Cancellation::new())).expect("react agent run");
        let mut error = None;
        while let Some(item) = block_on(events.next()) {
            if let Err(value) = item {
                error = Some(value);
            }
        }
        let error = error.expect("tool-only iteration must exhaust");
        assert_eq!(error.code, "MAX_ITERS");
    }

    #[test]
    fn react_writes_direct_tool_error_into_next_model_context() {
        let agent = crate::agent::react()
            .model(SequenceChat {
                calls: Arc::new(Mutex::new(0)),
                require_tool_error: true,
            })
            .toolkit(
                Toolkit::new().tool(FailingTool {
                    definition: ToolDefinition {
                        name: "weather".into(),
                        description: "lookup weather".into(),
                        parameters: serde_json::from_value(json!({
                            "type": "object",
                            "properties": {"city": {"type": "string"}}
                        }))
                        .expect("tool schema"),
                    },
                }),
            )
            .with_max_iters(2)
            .build()
            .expect("react agent build");

        let mut events =
            block_on(agent.run(Input::default(), Cancellation::new())).expect("react agent run");
        let _assistant = block_on(events.next()).expect("assistant event");
        let tool_error = block_on(events.next())
            .expect("tool error event")
            .expect("event ok");
        match tool_error {
            Event::Complete { message, .. } => {
                assert_eq!(message.role, Role::Tool);
                assert_eq!(message.content[0].r#type, "text");
                assert!(
                    message.content[0].data["value"]
                        .as_str()
                        .is_some_and(|value| { value.contains("TIMEOUT") })
                );
                assert_eq!(message.values["error"]["code"], "TIMEOUT");
                assert!(
                    message
                        .content
                        .iter()
                        .all(|part| part.r#type != "tool_result")
                );
            }
            other => panic!("unexpected event: {other:?}"),
        }
        assert!(matches!(
            block_on(events.next()),
            Some(Ok(Event::Complete { .. }))
        ));
    }

    fn application_id() -> i64 {
        panic!("the application ID function must not run during assembly")
    }

    #[test]
    fn id_function_is_optional_and_is_not_called_by_build() {
        let agent = crate::agent::react()
            .model(SequenceChat {
                calls: Arc::new(Mutex::new(0)),
                require_tool_error: false,
            })
            .build()
            .expect("react agent without an ID source builds");
        assert!(agent.id_fn.is_none());

        let configured = crate::agent::react()
            .with_id_fn(application_id)
            .model(SequenceChat {
                calls: Arc::new(Mutex::new(0)),
                require_tool_error: false,
            })
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
            .model(SequenceChat {
                calls: Arc::new(Mutex::new(0)),
                require_tool_error: false,
            })
            .with_id_fn(application_id)
            .model(SequenceChat {
                calls: Arc::new(Mutex::new(0)),
                require_tool_error: false,
            })
            .build()
            .expect("model rebinding keeps ID configuration");
        assert_eq!(
            configured.id_fn.map(|id| id as usize),
            Some(application_id as *const () as usize)
        );
    }
}
