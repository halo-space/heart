use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::task::{Context, Poll};

use futures_core::Stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::components::agent::toolkit::Toolkit;
use crate::components::agent::{Agent as AgentContract, Error, Input, SubAgent};
use crate::components::evaluation::{Evaluation, Evaluator};
use crate::components::event::Event;
use crate::components::message::{Message, Messages, Role};
use crate::components::model::chat::{Chat, Request};
use crate::components::model::content::part::{Part, object as part_data};
use crate::components::tool::Call;
use crate::core::agent::access::Access;
use crate::core::agent::hooks::{Hooks, cancellable};
use crate::core::agent::input::user_message;
use crate::core::agent::subagent::{
    Failure as SubAgentFailure, Progress as SubAgentProgress, Root,
};
use crate::core::memory::history::{History, Run};
use crate::runtime::agent::checkpoint::Backend;
use crate::runtime::agent::node::Step;
use crate::runtime::agent::tool_steps;
use crate::runtime::cancellation::Cancellation;

pub struct Builder<M = ()> {
    model: Option<M>,
    system_prompt: Option<String>,
    toolkit: Option<Toolkit>,
    subagents: Vec<SubAgent>,
    hooks: Hooks,
    access: Access,
    evaluator: Option<EvaluatorHandle>,
    checkpoint: Option<CheckpointSave>,
    history: History,
    max_concurrency: usize,
    id_fn: Option<fn() -> i64>,
}

pub fn builder() -> Builder {
    Builder {
        model: None,
        system_prompt: None,
        toolkit: None,
        subagents: Vec::new(),
        hooks: Hooks::default(),
        access: Access::default(),
        evaluator: None,
        checkpoint: None,
        history: History::default(),
        max_concurrency: 100,
        id_fn: None,
    }
}

impl<M> Builder<M> {
    pub fn model<N>(self, model: N) -> Builder<N> {
        Builder {
            model: Some(model),
            system_prompt: self.system_prompt,
            toolkit: self.toolkit,
            subagents: self.subagents,
            hooks: self.hooks,
            access: self.access,
            evaluator: self.evaluator,
            checkpoint: self.checkpoint,
            history: self.history,
            max_concurrency: self.max_concurrency,
            id_fn: self.id_fn,
        }
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn toolkit(mut self, toolkit: Toolkit) -> Self {
        self.toolkit = Some(toolkit);
        self
    }

    /// Register the heterogeneous Agent set available to the built-in
    /// `agent_spawn` delegation path. SubAgents are not Toolkit entries.
    pub fn subagents(mut self, subagents: impl IntoIterator<Item = SubAgent>) -> Self {
        self.subagents.extend(subagents);
        self
    }

    /// Attach the permission policy used for every actual Tool/MCP call.
    /// Permission is shared by Chat and ReAct; it is not checked merely when
    /// tool definitions are sent to the model.
    pub fn permission<P: crate::permission::Permission + 'static>(mut self, permission: P) -> Self {
        self.access.set(permission);
        self
    }

    /// Attach an optional final-result evaluator. Chat has no Planner loop:
    /// a rejected final result closes the run with an evaluation error.
    pub fn evaluator<E: Evaluator + 'static>(mut self, evaluator: E) -> Self {
        let evaluator = Arc::new(evaluator);
        self.evaluator = Some(Arc::new(move |input, result, cancellation| {
            let evaluator = Arc::clone(&evaluator);
            Box::pin(async move { evaluator.evaluate(&input, &result, &cancellation).await })
        }));
        self
    }

    /// Attach the Agent checkpoint backend. Chat checkpoints contain the
    /// current messages, usage and terminal status; they do not invent a plan.
    pub fn checkpoint<C: crate::checkpoint::Checkpoint + 'static>(mut self, checkpoint: C) -> Self {
        self.checkpoint = Some(Backend::new(checkpoint));
        self
    }

    /// Attach an Agent-local middleware. Repeated calls preserve registration order.
    pub fn middleware<H: crate::middleware::Middleware + 'static>(mut self, middleware: H) -> Self {
        self.hooks.add(middleware);
        self
    }

    /// Compose short history with explicitly selected long recall scopes.
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

    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Application-owned execution/question identity function, independent of Memory.
    /// Construction does not invoke it; embedded children inherit their owner.
    pub fn with_id_fn(mut self, id_fn: fn() -> i64) -> Self {
        self.id_fn = Some(id_fn);
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
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "chat agent requires a model"))?;
        if self.max_concurrency == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "agent limits must be positive",
            ));
        }
        self.history.validate()?;
        validate_subagents(&self.subagents)?;
        Ok(Agent {
            replies: Default::default(),
            model: Arc::new(model),
            system_prompt: self.system_prompt,
            toolkit: self.toolkit,
            subagents: self.subagents,
            hooks: self.hooks,
            access: self.access,
            evaluator: self.evaluator,
            checkpoint: self.checkpoint,
            history: self.history,
            max_concurrency: self.max_concurrency,
            id_fn: self.id_fn,
        })
    }
}

pub struct Agent<M> {
    replies: crate::runtime::agent::reply::Hub,
    model: Arc<M>,
    system_prompt: Option<String>,
    toolkit: Option<Toolkit>,
    subagents: Vec<SubAgent>,
    hooks: Hooks,
    access: Access,
    evaluator: Option<EvaluatorHandle>,
    checkpoint: Option<CheckpointSave>,
    history: History,
    max_concurrency: usize,
    id_fn: Option<fn() -> i64>,
}

fn validate_subagents(subagents: &[SubAgent]) -> Result<(), Error> {
    let mut names = std::collections::BTreeSet::new();
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

type CompletionFuture = Pin<Box<dyn Future<Output = Result<Event, Error>> + 'static>>;
type CheckpointFuture = Pin<Box<dyn Future<Output = Result<(), Error>> + 'static>>;
type ProviderFuture<S> = Pin<Box<dyn Future<Output = Result<S, Error>> + 'static>>;
type ToolBatch = (Vec<Message>, Option<crate::model::token::Usage>);
type ToolBatchFuture =
    Pin<Box<dyn Future<Output = Result<ToolBatch, Box<ToolBatchFailure>>> + 'static>>;
type ProviderFactory<S> =
    Arc<dyn Fn(Messages, Cancellation, Arc<AtomicU64>) -> ProviderFuture<S> + Send + Sync>;
type EvaluatorHandle = Arc<dyn Fn(Input, Messages, Cancellation) -> EvaluationFuture + Send + Sync>;
type EvaluationFuture = Pin<Box<dyn Future<Output = Result<Evaluation, Error>> + 'static>>;
type CheckpointSave = Backend;

struct ToolBatchConfig {
    interaction: Option<crate::runtime::agent::reply::Run>,
    relay: Option<crate::core::agent::subagent::Relay>,
    steps: Arc<Mutex<Vec<Step>>>,
    toolkit: Arc<Toolkit>,
    subagents: Vec<SubAgent>,
    access: Access,
    hooks: Hooks,
    metadata: Map<String, Value>,
    cancellation: Cancellation,
    max_concurrency: usize,
    root: Root,
    id_fn: Option<fn() -> i64>,
}

// Private per-call dependencies; not a public Tool context or saved state.
struct ToolExecution {
    toolkit: Arc<Toolkit>,
    access: Access,
    hooks: Hooks,
    metadata: Map<String, Value>,
    cancellation: Cancellation,
    steps: Arc<Mutex<Vec<Step>>>,
    interaction: Option<crate::runtime::agent::reply::Run>,
    id_fn: Option<fn() -> i64>,
}

struct EventsConfig<S> {
    interaction: Option<crate::runtime::agent::reply::Run>,
    steps: Vec<Step>,
    provider_factory: ProviderFactory<S>,
    toolkit: Option<Toolkit>,
    subagents: Vec<SubAgent>,
    access: Access,
    evaluator: Option<EvaluatorHandle>,
    checkpoint: Option<CheckpointSave>,
    input: Input,
    checkpoint_id: Option<i64>,
    metadata: Map<String, Value>,
    messages: Messages,
    max_concurrency: usize,
    pending_subagents: std::collections::BTreeMap<String, PendingSubAgent>,
    root: Root,
    id_fn: Option<fn() -> i64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct PendingSubAgent {
    index: usize,
    call: Call,
}

#[derive(Debug)]
struct ToolBatchFailure {
    error: Error,
    usage: Option<crate::model::token::Usage>,
}

/// The concrete stream returned by this Chat Agent. Its items remain the
/// shared Event contract; this type only owns cancellation and partial state.
pub struct Events<S> {
    _interaction_owner: Option<crate::runtime::agent::reply::Owner>,
    interaction: Option<crate::runtime::agent::reply::Run>,
    waits: Vec<crate::runtime::agent::wait::Wait>,
    questions: VecDeque<Event>,
    stream: bool,
    steps: Arc<Mutex<Vec<Step>>>,
    root: Root,
    provider: Option<Pin<Box<S>>>,
    provider_factory: Option<ProviderFactory<S>>,
    cancellation: Cancellation,
    partial: Message,
    input_tokens: u64,
    pending_input_tokens: Arc<AtomicU64>,
    finished: bool,
    hooks: Hooks,
    toolkit: Option<Arc<Toolkit>>,
    subagents: Vec<SubAgent>,
    access: Access,
    evaluator: Option<EvaluatorHandle>,
    checkpoint: Option<CheckpointSave>,
    input: Option<Input>,
    checkpoint_id: Option<i64>,
    metadata: Map<String, Value>,
    messages: Messages,
    max_concurrency: usize,
    id_fn: Option<fn() -> i64>,
    pending_tools: Option<ToolBatchFuture>,
    pending_provider: Option<ProviderFuture<S>>,
    tool_results: VecDeque<Message>,
    child_events: crate::core::agent::subagent::Relay,
    pending_subagents: std::collections::BTreeMap<String, PendingSubAgent>,
    pending_checkpoint: Option<CheckpointFuture>,
    pending: Option<CompletionFuture>,
    run: Run,
    settled: bool,
    closing: bool,
}

impl<S> std::fmt::Debug for Events<S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Events")
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl<S> Events<S> {
    fn new(provider: S, cancellation: Cancellation, input_tokens: u64, hooks: Hooks) -> Self {
        let mut events = Self::empty(cancellation, hooks);
        events.provider = Some(Box::pin(provider));
        events.input_tokens = input_tokens;
        events
    }

    fn empty(cancellation: Cancellation, hooks: Hooks) -> Self {
        Self {
            _interaction_owner: None,
            interaction: None,
            waits: Vec::new(),
            questions: VecDeque::new(),
            stream: true,
            steps: Default::default(),
            root: Root::default(),
            provider: None,
            provider_factory: None,
            cancellation,
            partial: Message::new(Role::Assistant),
            input_tokens: 0,
            pending_input_tokens: Arc::new(AtomicU64::new(0)),
            finished: false,
            hooks,
            toolkit: None,
            subagents: Vec::new(),
            access: Access::default(),
            evaluator: None,
            checkpoint: None,
            input: None,
            checkpoint_id: None,
            metadata: Map::new(),
            messages: Messages::default(),
            max_concurrency: 100,
            id_fn: None,
            pending_tools: None,
            pending_provider: None,
            tool_results: VecDeque::new(),
            child_events: crate::core::agent::subagent::Relay::default(),
            pending_subagents: std::collections::BTreeMap::new(),
            pending_checkpoint: None,
            pending: None,
            run: Run::default(),
            settled: false,
            closing: false,
        }
    }

    fn configure_tools(mut self, config: EventsConfig<S>) -> Self {
        self._interaction_owner = config
            .interaction
            .as_ref()
            .map(crate::runtime::agent::reply::Run::owner);
        self.interaction = config.interaction;
        self.steps = Arc::new(Mutex::new(config.steps));
        self.provider_factory = Some(config.provider_factory);
        self.toolkit = config.toolkit.map(Arc::new);
        self.subagents = config.subagents;
        self.access = config.access;
        self.evaluator = config.evaluator;
        self.checkpoint = config.checkpoint;
        self.input = Some(config.input);
        self.checkpoint_id = config.checkpoint_id;
        self.metadata = config.metadata;
        self.messages = config.messages;
        self.max_concurrency = config.max_concurrency;
        self.id_fn = config.id_fn;
        self.pending_subagents = config.pending_subagents;
        self.root = config.root;
        self
    }

    fn cancelled(&mut self) -> Event {
        let message = std::mem::replace(&mut self.partial, Message::new(Role::Assistant));
        let usage = {
            let output = estimate_output_tokens(&message);
            let mut usage = message.usage.clone().unwrap_or_default();
            usage
                .input
                .get_or_insert(if self.pending_provider.is_some() {
                    self.pending_input_tokens.load(Ordering::Relaxed)
                } else {
                    self.input_tokens
                });
            usage.output.get_or_insert(output);
            usage.total.get_or_insert(
                usage
                    .input
                    .unwrap_or(0)
                    .saturating_add(usage.output.unwrap_or(0)),
            );
            if self.settled {
                self.run.refine_last(Some(usage.clone()));
            } else {
                self.run.settle(Some(usage.clone()));
            }
            self.settled = true;
            Some(usage)
        };
        let mut message = message;
        message.usage = usage.clone();
        Event::Complete {
            message,
            usage,
            finish_reason: Some("cancelled".into()),
        }
    }

    fn close(&mut self, result: Result<Event, Error>) {
        if let Some(run) = &self.interaction {
            run.close();
        }
        // Stop opaque in-flight work before capturing the final consistent
        // scene; do not let a pending Provider future block closeout polling.
        self.pending_provider = None;
        self.pending_tools = None;
        self.questions.clear();
        self.provider = None;
        if matches!(&result, Ok(Event::Complete { finish_reason:Some(reason), .. }) if reason=="cancelled")
            || matches!(&result, Err(error) if error.code!="WAITING")
        {
            self.waits.clear();
            tool_steps::cancel_pending(&mut self.steps.lock().expect("Tool steps lock poisoned"));
        }
        if result.is_err() && !self.settled {
            // Keep already reported consumption, but never estimate an error
            // stream as though it had delivered a successful Complete.
            self.run.settle(self.partial.usage.clone());
            self.settled = true;
        }
        let run = self.run.clone();
        let checkpoint = self.checkpoint.clone();
        let checkpoint_id = self.checkpoint_id;
        let mut state = checkpoint_state(
            &self.messages,
            self.run.usage(),
            self.input.as_ref(),
            &result,
            &self.root,
            &self.steps.lock().expect("Tool steps lock poisoned"),
            self.stream,
        );
        state["waits"] = serde_json::to_value(&self.waits).expect("waits serialization");
        self.closing = true;
        self.pending = Some(Box::pin(async move {
            if crate::runtime::agent::scope::capture(state.clone()) {
                // The outer Workflow owns the embedded Agent checkpoint.
            } else if let (Some(checkpoint), Some(id)) = (checkpoint, checkpoint_id) {
                checkpoint
                    .save(id, state.clone(), Cancellation::new())
                    .await?;
            }
            // Separate persistence copies of the same scene. Trace is written
            // only at terminal closeout and is never read for restoration.
            run.close_with_state(result, None, Some(state)).await
        }));
    }
}

impl<S> Stream for Events<S>
where
    S: Stream<Item = Result<Event, Error>> + 'static,
{
    type Item = Result<Event, Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            let mut result = this.poll_event(cx);
            if !this.stream && matches!(result, Poll::Ready(Some(Ok(Event::Delta { .. })))) {
                continue;
            }
            if let Poll::Ready(Some(Ok(event))) = &mut result {
                if matches!(event, Event::Complete { .. }) && !this.closing {
                    this.capture_live();
                }
                crate::runtime::observation::associate(
                    event,
                    this.metadata.get("chat_id").and_then(Value::as_i64),
                    None,
                );
            }
            return result;
        }
    }
}

impl<S: Stream<Item = Result<Event, Error>> + 'static> Events<S> {
    fn capture_live(&self) {
        if !crate::runtime::agent::scope::active() {
            return;
        }
        let steps = self.steps.lock().expect("Tool steps lock poisoned");
        let mut snapshot = if self.waits.is_empty() {
            checkpoint_state(
                &self.messages,
                self.run.usage(),
                self.input.as_ref(),
                &Ok(Event::Delta {
                    message: self.partial.clone(),
                }),
                &self.root,
                &steps,
                self.stream,
            )
        } else {
            waiting_checkpoint_state(
                &self.messages,
                self.run.usage(),
                self.input.as_ref(),
                &self.root,
                &steps,
                self.stream,
            )
        };
        snapshot["waits"] = serde_json::to_value(&self.waits).expect("waits serialization");
        crate::runtime::agent::scope::capture(snapshot);
    }
    // Finish state/history mutations before decorating the outgoing copy.
    fn poll_event(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<Event, Error>>> {
        let this = self;
        if this.finished {
            return Poll::Ready(None);
        }
        if !this.closing {
            this.cancellation.register(cx.waker());
        }
        if this.cancellation.is_cancelled() && !this.closing {
            let result = this.cancelled();
            this.close(Ok(result));
        }
        if let Some(checkpoint) = this.pending_checkpoint.as_mut() {
            match checkpoint.as_mut().poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(())) => {
                    this.pending_checkpoint = None;
                    return Pin::new(this).poll_next(cx);
                }
                Poll::Ready(Err(error)) => {
                    this.pending_checkpoint = None;
                    this.close(Err(error));
                    return Pin::new(this).poll_next(cx);
                }
            }
        }
        if let Some(question) = this.questions.pop_front() {
            return Poll::Ready(Some(Ok(question)));
        }
        if let Some(provider) = this.pending_provider.as_mut() {
            match provider.as_mut().poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(provider)) => {
                    this.input_tokens = this.pending_input_tokens.load(Ordering::Relaxed);
                    this.provider = Some(Box::pin(provider));
                    this.pending_provider = None;
                    return Pin::new(this).poll_next(cx);
                }
                Poll::Ready(Err(error)) => {
                    this.pending_provider = None;
                    this.close(Err(error));
                    return Pin::new(this).poll_next(cx);
                }
            }
        }
        if let Some(tools) = this.pending_tools.as_mut() {
            match tools.as_mut().poll(cx) {
                Poll::Pending => {
                    if let Some(run) = this.interaction.clone()
                        && let Poll::Ready(change) = run.poll_change(cx)
                    {
                        let applied = crate::runtime::agent::reply::apply(
                            change,
                            &mut this.steps.lock().expect("Tool steps lock poisoned"),
                            &mut this.waits,
                            &run,
                        );
                        match applied {
                            Ok(Some(question)) => {
                                let mut snapshot = waiting_checkpoint_state(
                                    &this.messages,
                                    this.run.usage(),
                                    this.input.as_ref(),
                                    &this.root,
                                    &this.steps.lock().expect("Tool steps lock poisoned"),
                                    this.stream,
                                );
                                snapshot["waits"] =
                                    serde_json::to_value(&this.waits).expect("waits serialization");
                                this.questions.push_back(question);
                                this.pending_checkpoint = Some(checkpoint_future(
                                    this.checkpoint.clone(),
                                    this.checkpoint_id,
                                    snapshot,
                                ));
                            }
                            Ok(None) => this.capture_live(),
                            Err(error) => this.close(Err(error)),
                        }
                        return Pin::new(this).poll_next(cx);
                    }
                    return match this.child_events.poll_event(cx) {
                        Poll::Pending => {
                            if !this.waits.is_empty()
                                && let Some(run) = &this.interaction
                                && !this
                                    .steps
                                    .lock()
                                    .expect("Tool steps lock poisoned")
                                    .iter()
                                    .any(|step| {
                                        step.status == "running"
                                            && !(step.kind == "agent" && run.children_blocked())
                                    })
                            {
                                run.suspend();
                            }
                            Poll::Pending
                        }
                        Poll::Ready(observation) => {
                            let question = match observation.question() {
                                Ok(question) => question,
                                Err(error) => {
                                    this.close(Err(error));
                                    return Pin::new(this).poll_next(cx);
                                }
                            };
                            if let Err(error) = this
                                .root
                                .observe(&observation)
                                .and_then(|_| this.root.sync_waits(&mut this.waits, &observation))
                            {
                                this.close(Err(error));
                                return Pin::new(this).poll_next(cx);
                            }
                            this.run.settle_child(observation.usage);
                            if question {
                                let mut snapshot = waiting_checkpoint_state(
                                    &this.messages,
                                    this.run.usage(),
                                    this.input.as_ref(),
                                    &this.root,
                                    &this.steps.lock().expect("Tool steps lock poisoned"),
                                    this.stream,
                                );
                                snapshot["waits"] =
                                    serde_json::to_value(&this.waits).expect("waits serialization");
                                this.questions.push_back(observation.event);
                                this.pending_checkpoint = Some(checkpoint_future(
                                    this.checkpoint.clone(),
                                    this.checkpoint_id,
                                    snapshot,
                                ));
                                return Pin::new(this).poll_next(cx);
                            }
                            Poll::Ready(Some(Ok(observation.event)))
                        }
                    };
                }
                Poll::Ready(Ok(messages)) => {
                    this.pending_tools = None;
                    this.run.settle_child(messages.1);
                    this.pending_subagents.clear();
                    this.tool_results.extend(messages.0);
                    this.pending_checkpoint = Some(checkpoint_future(
                        this.checkpoint.clone(),
                        this.checkpoint_id,
                        checkpoint_state(
                            &this.messages,
                            this.run.usage(),
                            this.input.as_ref(),
                            &Ok(Event::Delta {
                                message: this.partial.clone(),
                            }),
                            &this.root,
                            &this.steps.lock().expect("Tool steps lock poisoned"),
                            this.stream,
                        ),
                    ));
                    return Pin::new(this).poll_next(cx);
                }
                Poll::Ready(Err(error)) => {
                    this.pending_tools = None;
                    this.run.settle_child(error.usage);
                    this.close(Err(error.error));
                    return Pin::new(this).poll_next(cx);
                }
            }
        }
        if let Some(message) = this.tool_results.pop_front() {
            this.messages.push(message.clone());
            let event = Event::Complete {
                message,
                usage: None,
                finish_reason: Some("tool".into()),
            };
            if this.tool_results.is_empty()
                && let Some(factory) = this.provider_factory.clone()
            {
                tool_steps::committed(&mut this.steps.lock().expect("Tool steps lock poisoned"));
                let messages = this.messages.clone();
                let cancellation = this.cancellation.clone();
                this.partial = Message::new(Role::Assistant);
                this.settled = false;
                this.input_tokens = 0;
                this.pending_input_tokens.store(0, Ordering::Relaxed);
                let provider = factory(
                    messages,
                    cancellation.clone(),
                    this.pending_input_tokens.clone(),
                );
                let interaction = this.interaction.clone();
                this.pending_provider = Some(Box::pin(async move {
                    if let Some(run) = interaction {
                        run.continue_execution(&cancellation).await?;
                    }
                    provider.await
                }));
                // The tool result is observable before the next model call.
            }
            return Poll::Ready(Some(Ok(event)));
        }
        if let Some(pending) = this.pending.as_mut() {
            let result = pending.as_mut().poll(cx);
            if this.cancellation.is_cancelled() && !this.closing {
                let result = this.cancelled();
                this.close(Ok(result));
                return Pin::new(this).poll_next(cx);
            }
            return match result {
                Poll::Pending => Poll::Pending,
                Poll::Ready(result) => {
                    this.pending = None;
                    if this.closing {
                        this.finished = true;
                        return Poll::Ready(Some(result));
                    }
                    match result {
                        Ok(event) => {
                            let Event::Complete { message, .. } = &event else {
                                return Poll::Ready(Some(Ok(event)));
                            };
                            let calls = message
                                .content
                                .iter()
                                .filter(|part| part.r#type == "tool_call")
                                .map(decode_call)
                                .collect::<Result<Vec<_>, _>>();
                            let calls = match calls {
                                Ok(calls) => calls,
                                Err(error) => {
                                    this.close(Err(error));
                                    return Pin::new(this).poll_next(cx);
                                }
                            };
                            this.messages.push(message.clone());
                            if calls.is_empty() {
                                this.close(Ok(event));
                                return Pin::new(this).poll_next(cx);
                            }
                            if this.toolkit.is_some() || !this.subagents.is_empty() {
                                let selected = tool_steps::selected(
                                    &mut this.steps.lock().expect("Tool steps lock poisoned"),
                                    &calls,
                                );
                                if let Err(error) = selected {
                                    this.close(Err(error));
                                    return Pin::new(this).poll_next(cx);
                                }
                                let pending_subagents = calls
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, call)| call.name == "agent_spawn")
                                    .map(|(index, call)| {
                                        (
                                            call.call_id.clone(),
                                            PendingSubAgent {
                                                index,
                                                call: call.clone(),
                                            },
                                        )
                                    })
                                    .collect::<std::collections::BTreeMap<_, _>>();
                                if !pending_subagents.is_empty() {
                                    this.pending_subagents = pending_subagents;
                                    let calls = this
                                        .pending_subagents
                                        .values()
                                        .map(|pending| pending.call.clone())
                                        .collect::<Vec<_>>();
                                    for call in calls {
                                        let prepared = this.root.prepare(
                                            &mut this
                                                .steps
                                                .lock()
                                                .expect("Tool steps lock poisoned"),
                                            &call,
                                            this.id_fn,
                                        );
                                        if let Err(error) = prepared {
                                            this.close(Err(error));
                                            return Pin::new(this).poll_next(cx);
                                        }
                                    }
                                    this.pending_checkpoint = Some(checkpoint_future(
                                        this.checkpoint.clone(),
                                        this.checkpoint_id,
                                        waiting_checkpoint_state(
                                            &this.messages,
                                            this.run.usage(),
                                            this.input.as_ref(),
                                            &this.root,
                                            &this.steps.lock().expect("Tool steps lock poisoned"),
                                            this.stream,
                                        ),
                                    ));
                                }
                                let toolkit = this
                                    .toolkit
                                    .clone()
                                    .unwrap_or_else(|| Arc::new(Toolkit::new()));
                                let access = this.access.clone();
                                let hooks = this.hooks.clone();
                                let metadata = this.metadata.clone();
                                let cancellation = this.cancellation.clone();
                                let max_concurrency = this.max_concurrency;
                                this.pending_tools = Some(Box::pin(execute_tool_batch(
                                    ToolBatchConfig {
                                        interaction: this.interaction.clone(),
                                        relay: Some(this.child_events.clone()),
                                        steps: this.steps.clone(),
                                        toolkit,
                                        subagents: this.subagents.clone(),
                                        access,
                                        hooks,
                                        metadata,
                                        cancellation,
                                        max_concurrency,
                                        id_fn: this.id_fn,
                                        root: this.root.clone(),
                                    },
                                    calls,
                                )));
                                // The model Complete is emitted before its tool calls.
                                return Poll::Ready(Some(Ok(event)));
                            }
                            this.close(Err(Error::new(
                                "INVALID_STATE",
                                "model returned tool calls without a configured Toolkit",
                            )));
                            Pin::new(this).poll_next(cx)
                        }
                        Err(error) => {
                            this.close(Err(error));
                            Pin::new(this).poll_next(cx)
                        }
                    }
                }
            };
        }
        let Some(provider) = this.provider.as_mut() else {
            this.close(Err(Error::new(
                "INVALID_STATE",
                "Chat provider is unavailable",
            )));
            return Pin::new(this).poll_next(cx);
        };
        match provider.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(Event::Delta { mut message }))) => {
                crate::runtime::observation::clear_source(&mut message);
                collect_delta(&mut this.partial, &message);
                Poll::Ready(Some(Ok(Event::Delta { message })))
            }
            Poll::Ready(Some(Ok(Event::Complete {
                mut message,
                usage,
                finish_reason,
            }))) => {
                crate::runtime::observation::clear_source(&mut message);
                let usage = usage.or_else(|| message.usage.clone());
                message.usage = usage.clone();
                this.run.settle(usage.clone());
                this.settled = true;
                this.partial = message.clone();
                let hooks = this.hooks.clone();
                let cancellation = this.cancellation.clone();
                let evaluator = this.evaluator.clone();
                let input = this.input.clone();
                this.pending = Some(Box::pin(async move {
                    let message = hooks.after_model(message, &cancellation).await?;
                    let message = hooks.after_execute(message, &cancellation).await?;
                    let has_tool_calls = message
                        .content
                        .iter()
                        .any(|part| part.r#type == "tool_call");
                    if !has_tool_calls && let (Some(evaluator), Some(input)) = (evaluator, input) {
                        let evaluation = evaluator(
                            input,
                            Messages::new([message.clone()]),
                            cancellation.clone(),
                        )
                        .await?;
                        if !evaluation.passed {
                            let details = evaluation
                                .feedback
                                .and_then(|feedback| serde_json::to_string(&feedback).ok())
                                .unwrap_or_else(|| "final result was rejected".into());
                            return Err(Error::new(
                                "EVALUATION_FAILED",
                                "agent final result was rejected",
                            )
                            .with_details(details));
                        }
                    }
                    Ok(Event::Complete {
                        message,
                        usage,
                        finish_reason,
                    })
                }));
                // Poll the owned hook future immediately; no extra Delta/Complete.
                Pin::new(this).poll_next(cx)
            }
            Poll::Ready(Some(Err(error))) => {
                this.close(Err(error));
                Pin::new(this).poll_next(cx)
            }
            Poll::Ready(None) => {
                this.close(Err(Error::new(
                    "INVALID_RESPONSE",
                    "model stream ended without Complete",
                )));
                Pin::new(this).poll_next(cx)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<M> Agent<M> {
    pub fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }
}

impl<M> AgentContract for Agent<M>
where
    M: Chat + 'static,
{
    type Stream = Events<M::Stream>;

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
            return Err(Error::new("CANCELLED", "agent run cancelled"));
        }
        let embedded_snapshot = crate::runtime::agent::scope::restore();
        let embedded = crate::runtime::agent::scope::active();
        if !embedded {
            self.replies.ensure_idle(&input.metadata)?;
        }
        let id_fn = crate::runtime::agent::scope::id_fn().or(self.id_fn);
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
        let mut restored_messages = None;
        let mut restored_waiting = false;
        let mut restored_usage = None;
        let mut restored_query = None;
        let mut restored_steps: Vec<Step> = Vec::new();
        let mut restored_pending_subagents = std::collections::BTreeMap::new();
        let mut restored_root = Root::default();
        let mut restored_approvals = None;
        let mut embedded_waits = Vec::new();
        if let Some(snapshot) = embedded_snapshot {
            if !checkpoint_identity_matches(&snapshot, &input.metadata) {
                return Err(Error::new(
                    "CONFLICT",
                    "checkpoint belongs to a different Agent identity",
                ));
            }
            if crate::runtime::agent::scope::interaction().is_some_and(|run| run.is_restoring()) {
                crate::runtime::agent::wait::validate(&snapshot)?;
                embedded_waits = snapshot
                    .get("waits")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|_| Error::new("INVALID_STATE", "invalid embedded waiting locators"))?
                    .unwrap_or_default();
            } else {
                crate::runtime::agent::checkpoint::check_waits(&snapshot)?;
            }
            restored_waiting = is_waiting_snapshot(&snapshot);
            if let Some(result) = crate::runtime::agent::checkpoint::terminal(&snapshot)? {
                let mut events = Events::empty(cancellation, self.hooks.clone());
                events.metadata = input.metadata.clone();
                events.closing = true;
                events.pending = Some(Box::pin(async move { result }));
                return Ok(events);
            }
            stream = crate::runtime::agent::checkpoint::stream_mode(&snapshot)?;
            restored_messages = Some(checkpoint_messages(&snapshot)?);
            restored_usage = serde_json::from_value(snapshot["usage"].clone()).ok();
            restored_pending_subagents = checkpoint_pending_subagents(&snapshot)?;
            restored_query = Some(checkpoint_query(&snapshot)?);
            restored_steps = checkpoint_steps(&snapshot)?;
            restored_root = Root::restore(&snapshot)?;
        }
        if let (Some(checkpoint), Some(id)) = (&checkpoint, checkpoint_id)
            && let Some(snapshot) = checkpoint.load(id, cancellation.clone()).await?
        {
            if !checkpoint_identity_matches(&snapshot, &input.metadata) {
                return Err(Error::new(
                    "CONFLICT",
                    "checkpoint belongs to a different Agent identity",
                ));
            }
            restored_approvals = crate::runtime::agent::reply::prepare_restore(&snapshot, &input)?;
            restored_waiting = is_waiting_snapshot(&snapshot);
            if !restored_waiting
                && snapshot
                    .get("query")
                    .is_some_and(|query| serde_json::to_value(&input).ok().as_ref() != Some(query))
                && !crate::runtime::agent::reply::saved_duplicate(&snapshot, &input)?
            {
                return Err(Error::new(
                    "CONFLICT",
                    "checkpoint belongs to a different Agent input",
                ));
            }
            if let Some(result) = crate::runtime::agent::checkpoint::terminal(&snapshot)? {
                let mut events = Events::empty(cancellation, self.hooks.clone());
                events.closing = true;
                events.metadata = input.metadata.clone();
                events.pending = Some(Box::pin(async move { result }));
                return Ok(events);
            }
            stream = crate::runtime::agent::checkpoint::stream_mode(&snapshot)?;
            restored_messages = Some(checkpoint_messages(&snapshot)?);
            restored_usage = serde_json::from_value(snapshot["usage"].clone()).ok();
            restored_pending_subagents = checkpoint_pending_subagents(&snapshot)?;
            restored_query = Some(checkpoint_query(&snapshot)?);
            restored_steps = checkpoint_steps(&snapshot)?;
            restored_root = Root::restore(&snapshot)?;
        }
        let restored_calls = if let Some(messages) = restored_messages.as_mut() {
            restored_root.resume_calls(&restored_steps, messages)?;
            tool_steps::resume_calls(&restored_steps, messages)?
        } else {
            Vec::new()
        };
        let original = input.clone();
        let interaction = if embedded {
            crate::runtime::agent::scope::interaction()
        } else {
            Some(self.replies.start(original.metadata.clone())?)
        };
        let restored_waits = restored_approvals
            .map(|restore| {
                interaction
                    .as_ref()
                    .expect("root restore interaction")
                    .restore_approvals(restore)
            })
            .unwrap_or(embedded_waits);
        let restoring = restored_messages.is_some();
        let input = if restoring {
            input
        } else {
            self.hooks.before_execute(input, &cancellation).await?
        };
        history.check_identity(&original, &input)?;
        let prompt = if restoring {
            None
        } else {
            self.hooks
                .prompt(self.system_prompt.clone(), &cancellation)
                .await?
        };
        let recalled = if restoring {
            Vec::new()
        } else {
            history
                .load(input.metadata.clone(), self.hooks.clone(), &cancellation)
                .await?
        };
        let run = if restored_messages.is_some() {
            history.resume(original.clone(), &cancellation).await?
        } else {
            history.begin(original.clone(), &cancellation).await?
        };
        // Preparation belongs to this run too. Retain its constructed input
        // before Toolkit preparation, model hooks or Provider dispatch stop it.
        let mut preparation_messages = Messages::default();
        let mut preparation_input_tokens = 0;
        let mut restore_validation_error = false;
        let prepared = async {
            let mut messages = restored_messages
                .clone()
                .unwrap_or_else(|| initial_messages(input.clone(), prompt.as_deref(), recalled));
            if restored_waiting
                && restored_pending_subagents.is_empty()
                && restored_calls.is_empty()
            {
                messages.push(user_message(input.clone()));
            }
            preparation_messages = messages.clone();
            let toolkit = match &self.toolkit {
                Some(toolkit) => Some(toolkit.prepared(&cancellation).await?),
                None => None,
            };
            for step in restored_steps
                .iter()
                .filter(|step| step.stage.as_deref() == Some("eval"))
            {
                let checked = toolkit
                    .as_ref()
                    .ok_or_else(|| {
                        Error::new("INVALID_STATE", "saved eval requires a configured Toolkit")
                    })
                    .and_then(|toolkit| {
                        toolkit.validate(step.call.as_ref().expect("validated saved Tool Call"))
                    });
                if let Err(error) = checked {
                    restore_validation_error = true;
                    return Err(Error::new(
                        "INVALID_STATE",
                        "saved eval does not match configured Tool",
                    )
                    .with_details(format!("{}: {}", error.code, error.message)));
                }
            }
            let tool_definitions = toolkit
                .as_ref()
                .map(Toolkit::definitions)
                .unwrap_or_default();
            let mut tool_definitions = tool_definitions;
            if !self.subagents.is_empty() {
                tool_definitions.push(self.subagents[0].definition());
            }
            let model = Arc::clone(&self.model);
            // A child-wait checkpoint already contains the exact child Call.
            // Do not invoke the parent model before restoring that hand-off.
            let (provider, input_tokens) =
                if restored_pending_subagents.is_empty() && restored_calls.is_empty() {
                    let request = Request {
                        messages: messages.clone(),
                        stream,
                        tools: tool_definitions.clone(),
                        ..Request::default()
                    };
                    let request = self.hooks.before_model(request, &cancellation).await?;
                    let input_tokens = estimate_input_tokens(&request);
                    preparation_input_tokens = input_tokens;
                    let provider = cancellable(model.stream(request), &cancellation).await?;
                    (Some(provider), input_tokens)
                } else {
                    (None, 0)
                };
            let hooks = self.hooks.clone();
            let factory: ProviderFactory<M::Stream> =
                Arc::new(move |messages, cancellation, input_tokens| {
                    let model = Arc::clone(&model);
                    let hooks = hooks.clone();
                    let tools = tool_definitions.clone();
                    Box::pin(async move {
                        let request = Request {
                            messages,
                            stream,
                            tools,
                            ..Request::default()
                        };
                        let request = hooks.before_model(request, &cancellation).await?;
                        input_tokens.store(estimate_input_tokens(&request), Ordering::Relaxed);
                        model.stream(request).await
                    })
                });
            let mut events = match provider {
                Some(provider) => Events::new(
                    provider,
                    cancellation.clone(),
                    input_tokens,
                    self.hooks.clone(),
                ),
                None => Events::empty(cancellation.clone(), self.hooks.clone()),
            }
            .configure_tools(EventsConfig {
                interaction: interaction.clone(),
                steps: restored_steps.clone(),
                provider_factory: factory,
                toolkit,
                subagents: self.subagents.clone(),
                access: self.access.clone(),
                evaluator: self.evaluator.clone(),
                checkpoint: checkpoint.clone(),
                input: restored_query.clone().unwrap_or_else(|| original.clone()),
                checkpoint_id,
                metadata: input.metadata.clone(),
                messages: messages.clone(),
                max_concurrency: self.max_concurrency,
                pending_subagents: restored_pending_subagents.clone(),
                root: restored_root.clone(),
                id_fn,
            });
            events.stream = stream;
            events.waits = restored_waits.clone();
            if !restored_pending_subagents.is_empty() || !restored_calls.is_empty() {
                let calls = restored_pending_subagents
                    .values()
                    .map(|pending| (pending.index, pending.call.clone()))
                    .collect::<Vec<_>>();
                let mut calls = calls;
                calls.extend(restored_calls.clone());
                calls.sort_by_key(|(index, _)| *index);
                let calls = calls.into_iter().map(|(_, call)| call).collect::<Vec<_>>();
                let toolkit = events
                    .toolkit
                    .clone()
                    .unwrap_or_else(|| Arc::new(Toolkit::new()));
                events.provider = None;
                events.pending_tools = Some(Box::pin(execute_tool_batch(
                    ToolBatchConfig {
                        interaction: events.interaction.clone(),
                        relay: Some(events.child_events.clone()),
                        steps: events.steps.clone(),
                        toolkit,
                        subagents: events.subagents.clone(),
                        access: events.access.clone(),
                        hooks: events.hooks.clone(),
                        metadata: events.metadata.clone(),
                        cancellation: events.cancellation.clone(),
                        max_concurrency: events.max_concurrency,
                        id_fn,
                        root: events.root.clone(),
                    },
                    calls,
                )));
            }
            Ok::<_, Error>(events)
        }
        .await;
        match prepared {
            Ok(mut events) => {
                events.run = run;
                if let Some(usage) = restored_usage {
                    events.run.settle(Some(usage));
                }
                Ok(events)
            }
            Err(error) => {
                if restore_validation_error {
                    return Err(error);
                }
                let mut events = Events::<M::Stream>::empty(cancellation, self.hooks.clone());
                events.stream = stream;
                events.checkpoint = checkpoint;
                events.checkpoint_id = checkpoint_id;
                events.input = Some(restored_query.unwrap_or(original));
                events.metadata = input.metadata;
                events.messages = preparation_messages;
                events.pending_subagents = restored_pending_subagents;
                events.root = restored_root;
                events.steps = Arc::new(Mutex::new(restored_steps));
                events.run = run;
                events.input_tokens = preparation_input_tokens;
                if let Some(usage) = restored_usage {
                    events.run.settle(Some(usage));
                }
                if error.code == "CANCELLED" {
                    // Cancellation needs a terminal Complete even if dispatch
                    // has not yielded a stream. A bare stopped error cannot be
                    // replayed by the frozen terminal checkpoint contract.
                    let result = events.cancelled();
                    events.close(Ok(result));
                    Ok(events)
                } else if error.code == "WAITING" {
                    // Waiting is a saved, nonterminal run. Closing through the
                    // existing stream keeps Memory and embedded Workflow
                    // scenes consistent with waiting during Provider polling.
                    events.close(Err(error));
                    Ok(events)
                } else {
                    // Preserve preparation-error return behavior, but save
                    // the same scene and terminal Trace as an in-stream error.
                    events.close(Err(error.clone()));
                    if let Some(close) = events.pending.take()
                        && let Err(error) = close.await
                    {
                        return Err(error);
                    }
                    Err(error)
                }
            }
        }
    }
}

async fn execute_tool_batch(
    config: ToolBatchConfig,
    calls: Vec<Call>,
) -> Result<ToolBatch, Box<ToolBatchFailure>> {
    let ToolBatchConfig {
        interaction,
        relay,
        steps,
        toolkit,
        subagents,
        access,
        hooks,
        metadata,
        cancellation,
        max_concurrency,
        root,
        id_fn,
    } = config;
    let tasks = calls.into_iter().enumerate().map(|(index, call)| {
        let toolkit = Arc::clone(&toolkit);
        let steps = steps.clone();
        let interaction = interaction.clone();
        let subagents = subagents.clone();
        let access = access.clone();
        let hooks = hooks.clone();
        let metadata = metadata.clone();
        let cancellation = cancellation.clone();
        let root = root.clone();
        let relay = relay.clone();
        async move {
            let result = if call.name == "agent_spawn" && !subagents.is_empty() {
                if let Some(run) = &interaction
                    && let Err(error) = run.continue_execution(&cancellation).await
                {
                    return (
                        index,
                        call,
                        Err(Box::new(ToolBatchFailure { error, usage: None })),
                    );
                }
                if let Some(message) =
                    root.message(&steps.lock().expect("Tool steps lock poisoned"), &call)
                {
                    return (index, call, Ok((vec![message], None)));
                }
                let prepared = root.prepare(
                    &mut steps.lock().expect("Tool steps lock poisoned"),
                    &call,
                    id_fn,
                );
                let restore = match prepared {
                    Ok(restore) => restore,
                    Err(error) => {
                        return (
                            index,
                            call,
                            Err(Box::new(ToolBatchFailure { error, usage: None })),
                        );
                    }
                };
                let observer = if let Some(relay) = relay {
                    let source =
                        root.source(&steps.lock().expect("Tool steps lock poisoned"), &call);
                    match source {
                        Ok(source) => {
                            let child = interaction
                                .as_ref()
                                .map(crate::runtime::agent::reply::Run::child);
                            if let (Some(parent), Some(child)) = (&interaction, &child) {
                                parent.restore_consumer(&source, child);
                            }
                            Some(
                                relay
                                    .observer(
                                        metadata.get("chat_id").and_then(Value::as_i64),
                                        source,
                                    )
                                    .with_interaction(child),
                            )
                        }
                        Err(error) => {
                            return (
                                index,
                                call,
                                Err(Box::new(ToolBatchFailure { error, usage: None })),
                            );
                        }
                    }
                } else {
                    None
                };
                let progress = execute_subagent(
                    subagents,
                    metadata,
                    call.clone(),
                    restore,
                    cancellation,
                    id_fn,
                    observer,
                )
                .await;
                let message = match &progress {
                    Ok(progress) => Some(progress.message.clone()),
                    Err(failure)
                        if matches!(failure.error.code.as_str(), "WAITING" | "CANCELLED") =>
                    {
                        None
                    }
                    Err(failure) => Some(subagent_error_message(&call, &failure.error)),
                };
                let completed = root.complete(
                    &mut steps.lock().expect("Tool steps lock poisoned"),
                    &call,
                    &progress,
                    message,
                );
                if let Err(error) = completed {
                    return (
                        index,
                        call,
                        Err(Box::new(ToolBatchFailure { error, usage: None })),
                    );
                }
                match progress {
                    Ok(progress) => Ok((vec![progress.message], progress.usage)),
                    Err(failure) if failure.error.code == "CANCELLED" => {
                        Err(Box::new(ToolBatchFailure {
                            error: failure.error,
                            usage: failure.usage,
                        }))
                    }
                    Err(failure) if failure.error.code == "WAITING" => {
                        Err(Box::new(ToolBatchFailure {
                            error: failure.error,
                            usage: failure.usage,
                        }))
                    }
                    Err(failure) => Ok((
                        vec![subagent_error_message(&call, &failure.error)],
                        failure.usage,
                    )),
                }
            } else {
                execute_tool(
                    ToolExecution {
                        toolkit,
                        access,
                        hooks,
                        metadata,
                        cancellation,
                        steps,
                        interaction,
                        id_fn,
                    },
                    call.clone(),
                )
                .await
                .map(|messages| (messages, None))
                .map_err(|error| Box::new(ToolBatchFailure { error, usage: None }))
            };
            (index, call, result)
        }
    });
    let mut results = futures_util::stream::iter(tasks)
        .buffer_unordered(max_concurrency.max(1))
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Vec<_>>();
    results.sort_by_key(|(index, _, _)| *index);
    let mut failures = Vec::new();
    let mut messages = Vec::new();
    let mut usage = None;
    for (index, _call, result) in results {
        match result {
            Ok((batch, child_usage)) => {
                usage = crate::model::token::merge(usage, child_usage);
                messages.extend(batch);
            }
            Err(mut failure) => {
                usage = crate::model::token::merge(usage, failure.usage.take());
                failures.push((index, failure));
            }
        }
    }
    if let Some((_, failure)) = failures.into_iter().min_by_key(|(index, _)| *index) {
        return Err(Box::new(ToolBatchFailure {
            error: failure.error,
            usage,
        }));
    }
    Ok((messages, usage))
}

async fn execute_subagent(
    subagents: Vec<SubAgent>,
    parent: Map<String, Value>,
    call: Call,
    restore: Option<Value>,
    cancellation: Cancellation,
    id_fn: Option<fn() -> i64>,
    observer: Option<crate::core::agent::subagent::Observer>,
) -> Result<SubAgentProgress, Box<SubAgentFailure>> {
    let name = call
        .arguments
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .ok_or_else(|| {
            SubAgentFailure::new(Error::new(
                "INVALID_ARGUMENTS",
                "agent_spawn name is missing",
            ))
        })?;
    let input = call.arguments.get("input").cloned().ok_or_else(|| {
        SubAgentFailure::new(Error::new(
            "INVALID_ARGUMENTS",
            "agent_spawn input is missing",
        ))
    })?;
    let mut input: Input = serde_json::from_value(input).map_err(|_| {
        SubAgentFailure::new(Error::new(
            "INVALID_ARGUMENTS",
            "agent_spawn input must be agent::Input",
        ))
    })?;
    crate::core::agent::subagent::inherit_identity(&mut input, &parent)
        .map_err(SubAgentFailure::new)?;
    let depth = parent.get("depth").and_then(Value::as_u64).unwrap_or(1);
    if depth >= 2 {
        return Err(SubAgentFailure::new(Error::new(
            "INVALID_ARGUMENTS",
            "SubAgent depth limit is 2",
        )));
    }
    input
        .metadata
        .insert("depth".into(), Value::from(depth + 1));
    let subagent = subagents
        .iter()
        .find(|subagent| subagent.has_name(name))
        .ok_or_else(|| {
            SubAgentFailure::new(Error::new(
                "NOT_FOUND",
                "agent_spawn target is not registered",
            ))
        })?;
    crate::core::agent::subagent::run(
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

fn checkpoint_state(
    messages: &Messages,
    usage: Option<crate::model::token::Usage>,
    input: Option<&Input>,
    result: &Result<Event, Error>,
    root: &Root,
    steps: &[Step],
    stream: bool,
) -> Value {
    let status = match result {
        Ok(Event::Complete {
            finish_reason: Some(reason),
            ..
        }) if reason == "cancelled" => "stopped",
        Ok(Event::Complete { .. }) => "succeeded",
        Err(error) if error.code == "CANCELLED" => "stopped",
        Err(error) if error.code == "WAITING" => "waiting",
        Err(_) => "failed",
        Ok(Event::Delta { .. }) => "running",
    };
    let mut state = json!({
        "chat_id": input.and_then(|input| input.metadata.get("chat_id")).and_then(Value::as_i64),
        "type": "agent",
        "status": status,
        "stream": stream,
        "messages": messages,
        "usage": usage,
        "query": input,
        "metadata": input.map(|input| input.metadata.clone()),
        "execution": root.snapshot(status, usage.clone(), result.as_ref().err().cloned()),
        "steps": steps,
        "waits": [],
    });
    crate::runtime::agent::checkpoint::record_result(&mut state, result);
    state
}

fn waiting_checkpoint_state(
    messages: &Messages,
    usage: Option<crate::model::token::Usage>,
    input: Option<&Input>,
    root: &Root,
    steps: &[Step],
    stream: bool,
) -> Value {
    let mut state = checkpoint_state(
        messages,
        usage.clone(),
        input,
        &Ok(Event::Delta {
            message: Message::new(Role::Assistant),
        }),
        root,
        steps,
        stream,
    );
    state["status"] = Value::String("waiting".into());
    state["execution"] = serde_json::to_value(root.snapshot("waiting", usage, None))
        .expect("serializable root execution");
    state
}

fn checkpoint_future(
    checkpoint: Option<CheckpointSave>,
    checkpoint_id: Option<i64>,
    state: Value,
) -> CheckpointFuture {
    Box::pin(async move {
        if crate::runtime::agent::scope::capture(state.clone()) {
            Ok(())
        } else if let (Some(checkpoint), Some(id)) = (checkpoint, checkpoint_id) {
            checkpoint.save(id, state, Cancellation::new()).await
        } else {
            Ok(())
        }
    })
}

fn checkpoint_pending_subagents(
    snapshot: &Value,
) -> Result<std::collections::BTreeMap<String, PendingSubAgent>, Error> {
    let root = Root::restore(snapshot)?;
    let mut messages = checkpoint_messages(snapshot)?;
    let steps = checkpoint_steps(snapshot)?;
    Ok(root
        .resume_calls(&steps, &mut messages)?
        .into_iter()
        .map(|(index, call)| (call.call_id.clone(), PendingSubAgent { index, call }))
        .collect())
}

fn checkpoint_messages(snapshot: &Value) -> Result<Messages, Error> {
    serde_json::from_value(snapshot["messages"].clone())
        .map_err(|_| Error::new("INVALID_STATE", "checkpoint has invalid Agent messages"))
}

fn checkpoint_steps(snapshot: &Value) -> Result<Vec<Step>, Error> {
    snapshot
        .get("steps")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| Error::new("INVALID_STATE", "checkpoint has invalid Tool steps"))
        .map(|steps| steps.unwrap_or_default())
}

fn checkpoint_query(snapshot: &Value) -> Result<Input, Error> {
    serde_json::from_value(snapshot["query"].clone()).map_err(|_| {
        Error::new(
            "INVALID_STATE",
            "checkpoint has invalid original Agent input",
        )
    })
}

fn is_waiting_snapshot(snapshot: &Value) -> bool {
    matches!(
        snapshot.get("status").and_then(Value::as_str),
        Some("running" | "waiting" | "paused")
    )
}

fn checkpoint_identity_matches(snapshot: &Value, metadata: &Map<String, Value>) -> bool {
    crate::runtime::agent::checkpoint::identity_matches(snapshot, metadata)
}

async fn execute_tool(execution: ToolExecution, call: Call) -> Result<Vec<Message>, Error> {
    let ToolExecution {
        toolkit,
        access,
        hooks,
        metadata,
        cancellation,
        steps,
        interaction,
        id_fn,
    } = execution;
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "tool call cancelled"));
    }
    let saved = {
        let mut steps = steps.lock().expect("Tool steps lock poisoned");
        if tool_steps::current(&mut steps, &call.call_id).is_none() {
            tool_steps::new_step(&mut steps, &call);
        }
        tool_steps::current(&mut steps, &call.call_id)
            .expect("selected Tool step")
            .clone()
    };
    if saved.stage.as_deref() == Some("complete") {
        return step_messages(&call, &saved);
    }
    let approval =
        crate::runtime::agent::reply::consume_restore(interaction.as_ref(), &saved, &cancellation)
            .await;
    let resumed = if saved.stage.as_deref() == Some("eval") {
        Some((
            saved
                .call
                .ok_or_else(|| Error::new("INVALID_STATE", "eval step has no actual Call"))?,
            saved
                .message
                .ok_or_else(|| Error::new("INVALID_STATE", "eval step has no business result"))?,
        ))
    } else {
        None
    };
    let mut permission_hook_failed = false;
    let result = if let Err(error) = &approval {
        Err(error.clone())
    } else if let Some(result) = resumed {
        Ok(result)
    } else if access.is_configured() {
        async {
            toolkit.validate(&call)?;
            crate::core::agent::access::check_tool(
                &access,
                &hooks,
                metadata.clone(),
                &call,
                &cancellation,
                interaction.as_ref(),
                id_fn,
                approval.as_ref().ok().and_then(Option::as_ref),
            )
            .await
            .map_err(|failure| match failure {
                crate::core::agent::access::Failure::Policy(error) => error,
                crate::core::agent::access::Failure::Hook(error) => {
                    permission_hook_failed = true;
                    error
                }
            })?;
            let effective = toolkit.prepare_call(call.clone(), &cancellation).await?;
            if let Some(step) = tool_steps::current(
                &mut steps.lock().expect("Tool steps lock poisoned"),
                &call.call_id,
            ) {
                step.call = Some(effective.clone());
            }
            if effective.arguments != call.arguments {
                crate::core::agent::access::check_tool(
                    &access,
                    &hooks,
                    metadata.clone(),
                    &effective,
                    &cancellation,
                    interaction.as_ref(),
                    id_fn,
                    approval.as_ref().ok().and_then(Option::as_ref),
                )
                .await
                .map_err(|failure| match failure {
                    crate::core::agent::access::Failure::Policy(error) => error,
                    crate::core::agent::access::Failure::Hook(error) => {
                        permission_hook_failed = true;
                        error
                    }
                })?;
            }
            if let Some(run) = &interaction {
                run.continue_execution(&cancellation).await?;
            }
            toolkit.call_prepared(effective, &cancellation).await
        }
        .await
    } else {
        async {
            if let Some(run) = &interaction {
                run.continue_execution(&cancellation).await?;
            }
            toolkit.call(call.clone(), &cancellation).await
        }
        .await
    };
    let (effective_call, business) = match result {
        Ok(result) => result,
        Err(error) => {
            let mut steps = steps.lock().expect("Tool steps lock poisoned");
            tool_steps::stopped(&mut steps, &call.call_id, &error);
            if permission_hook_failed || matches!(error.code.as_str(), "CANCELLED" | "WAITING") {
                return Err(error);
            }
            let message = tool_error_message(&call, error);
            tool_steps::failed_message(&mut steps, &call.call_id, &message);
            return Ok(vec![message]);
        }
    };
    tool_steps::called(
        &mut steps.lock().expect("Tool steps lock poisoned"),
        &effective_call,
        &business,
    );
    let evaluation = match toolkit
        .evaluate(&effective_call, &business, &cancellation)
        .await
    {
        Ok(evaluation) => evaluation,
        Err(error) => {
            tool_steps::stopped(
                &mut steps.lock().expect("Tool steps lock poisoned"),
                &call.call_id,
                &error,
            );
            return Err(error);
        }
    };
    let mut steps = steps.lock().expect("Tool steps lock poisoned");
    tool_steps::evaluated(&mut steps, &call.call_id, &evaluation);
    step_messages(
        &call,
        tool_steps::current(&mut steps, &call.call_id).expect("completed Tool step"),
    )
}

fn step_messages(call: &Call, step: &Step) -> Result<Vec<Message>, Error> {
    let business = step
        .message
        .clone()
        .ok_or_else(|| Error::new("INVALID_STATE", "Tool step lost its business result"))?;
    let mut messages = vec![if step.status == "failed" {
        business
    } else {
        tool_success_message(call, business)
    }];
    if let Some(evaluation) = &step.eval
        && !evaluation.passed
        && let Some(feedback) = &evaluation.feedback
    {
        messages.extend(feedback.0.iter().cloned());
    }
    Ok(messages)
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
    crate::runtime::observation::clear_source(&mut message);
    message
        .metadata
        .insert("call_id".into(), Value::String(call.call_id.clone()));
    message
        .metadata
        .insert("name".into(), Value::String(call.name.clone()));
    message
}

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

fn checkpoint_id(metadata: &Map<String, Value>) -> Result<i64, Error> {
    let id = metadata
        .get("chat_id")
        .and_then(Value::as_i64)
        .filter(|value| *value > 0)
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "checkpoint requires chat_id"))?;
    Ok(id)
}

pub(crate) fn initial_messages(
    input: Input,
    system_prompt: Option<&str>,
    history: Vec<Message>,
) -> Messages {
    let mut messages = Vec::new();
    if let Some(prompt) = system_prompt {
        let mut message = Message::new(Role::System);
        message.content.push(Part {
            r#type: "text".into(),
            data: part_data(serde_json::json!({ "value": prompt })),
        });
        messages.push(message);
    }
    messages.extend(history);
    messages.push(user_message(input));
    Messages::new(messages)
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
                last.data.get("value").and_then(serde_json::Value::as_str),
                part.data.get("value").and_then(serde_json::Value::as_str),
            )
        {
            let joined = format!("{left}{right}");
            last.data["value"] = serde_json::Value::String(joined);
        } else {
            partial.content.push(part.clone());
        }
    }
    partial.values.extend(delta.values.clone());
    partial.metadata.extend(delta.metadata.clone());
}

pub(crate) fn estimate_input_tokens(request: &Request) -> u64 {
    let message_bytes: usize = request
        .messages
        .as_slice()
        .iter()
        .map(|message| {
            let role_bytes = format!("{:?}", message.role).len();
            role_bytes + message.content.iter().map(part_bytes).sum::<usize>()
        })
        .sum();
    let tool_bytes: usize = request
        .tools
        .iter()
        .map(|tool| {
            tool.name.len()
                + tool.description.len()
                + serde_json::Value::Object(tool.parameters.clone())
                    .to_string()
                    .len()
        })
        .sum();
    estimate_tokens(message_bytes.saturating_add(tool_bytes))
}

fn estimate_output_tokens(message: &Message) -> u64 {
    estimate_tokens(message.content.iter().map(part_bytes).sum())
}

fn part_bytes(part: &Part) -> usize {
    let body = part
        .data
        .get("value")
        .and_then(serde_json::Value::as_str)
        .map(str::len)
        .unwrap_or_else(|| {
            serde_json::Value::Object(part.data.clone())
                .to_string()
                .len()
        });
    part.r#type.len().saturating_add(body)
}

fn estimate_tokens(bytes: usize) -> u64 {
    u64::try_from(bytes.saturating_add(3) / 4).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    fn test_id() -> i64 {
        static NEXT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(600000);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
    use crate::components::model::token::Usage;
    use std::collections::VecDeque;
    use std::pin::Pin;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::task::{Context, Poll, Wake, Waker};

    use super::*;
    use crate::components::agent::Agent as AgentContract;
    use crate::components::event::Event;
    use crate::components::message::Role;
    use crate::components::permission::{Permission, Request as PermissionRequest};
    use crate::components::tool::{Call, Tool, ToolDefinition};
    use futures_core::Stream;
    use futures_executor::block_on;
    use serde_json::json;

    #[derive(Debug)]
    struct FakeStream {
        events: VecDeque<Result<Event, Error>>,
        pending: bool,
    }

    impl Stream for FakeStream {
        type Item = Result<Event, Error>;

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            if self.pending {
                return Poll::Pending;
            }
            Poll::Ready(self.events.pop_front())
        }
    }

    #[derive(Clone)]
    struct FakeChat {
        called: Arc<AtomicBool>,
        failure: Option<Error>,
        include_complete: bool,
        pending: bool,
    }

    impl Chat for FakeChat {
        type Stream = FakeStream;

        async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
            self.called.store(true, Ordering::Release);
            assert!(request.stream);
            if let Some(error) = &self.failure {
                return Err(error.clone());
            }
            let mut delta = Message::new(Role::Assistant);
            delta.content.push(Part {
                r#type: "text".into(),
                data: crate::components::model::content::part::object(json!({"value": "hel"})),
            });
            let mut complete = Message::new(Role::Assistant);
            complete.content.push(Part {
                r#type: "text".into(),
                data: crate::components::model::content::part::object(json!({"value": "hello"})),
            });
            let mut events = VecDeque::from([Ok(Event::Delta { message: delta })]);
            if self.include_complete {
                events.push_back(Ok(Event::Complete {
                    message: complete,
                    usage: Some(Usage {
                        input: Some(1),
                        output: Some(2),
                        total: Some(3),
                        input_details: None,
                        output_details: None,
                        details: Map::new(),
                    }),
                    finish_reason: Some("stop".into()),
                }));
            }
            Ok(FakeStream {
                events,
                pending: self.pending,
            })
        }
    }

    struct SequenceChat {
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Chat for SequenceChat {
        type Stream = FakeStream;

        async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
            let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            assert!(request.stream);
            let mut message = Message::new(Role::Assistant);
            if call == 0 {
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: part_data(json!({
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": {"value": "hello"}
                    })),
                });
            } else {
                message.content.push(Part {
                    r#type: "text".into(),
                    data: part_data(json!({"value": "done"})),
                });
            }
            Ok(FakeStream {
                events: VecDeque::from([Ok(Event::Complete {
                    message,
                    usage: None,
                    finish_reason: Some("stop".into()),
                })]),
                pending: false,
            })
        }
    }

    struct WaitingSubAgent;

    struct CountingSubAgent {
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl AgentContract for WaitingSubAgent {
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
        type Stream = futures_util::stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

        async fn run(
            &self,
            _input: Input,
            _stream: bool,
            _cancellation: Cancellation,
        ) -> Result<Self::Stream, Error> {
            if crate::runtime::agent::scope::restore().is_none() {
                assert!(crate::runtime::agent::scope::capture(json!({
                    "owner": "chat-child"
                })));
                return Err(Error::new("WAITING", "child is waiting for approval"));
            }
            let mut message = Message::new(Role::Assistant);
            message.content.push(Part {
                r#type: "text".into(),
                data: part_data(json!({"value": "child approved"})),
            });
            Ok(futures_util::stream::iter(vec![Ok(Event::Complete {
                message,
                usage: None,
                finish_reason: Some("stop".into()),
            })]))
        }
    }

    impl AgentContract for CountingSubAgent {
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
        type Stream = futures_util::stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

        async fn run(
            &self,
            _input: Input,
            _stream: bool,
            _cancellation: Cancellation,
        ) -> Result<Self::Stream, Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut message = Message::new(Role::Assistant);
            message.content.push(Part {
                r#type: "text".into(),
                data: part_data(json!({"value": "done child"})),
            });
            Ok(futures_util::stream::iter(vec![Ok(Event::Complete {
                message,
                usage: None,
                finish_reason: Some("stop".into()),
            })]))
        }
    }

    struct EchoTool {
        definition: ToolDefinition,
        calls: Arc<AtomicBool>,
    }

    impl Tool for EchoTool {
        fn definition(&self) -> &ToolDefinition {
            &self.definition
        }

        async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
            self.calls.store(true, Ordering::SeqCst);
            Ok(Message {
                role: Role::Tool,
                content: Vec::new(),
                values: Map::from_iter([(String::from("value"), json!("hello"))]),
                metadata: Map::new(),
                usage: None,
            })
        }
    }

    fn echo_tool(calls: Arc<AtomicBool>) -> EchoTool {
        EchoTool {
            definition: ToolDefinition {
                name: "echo".into(),
                description: "Echo a value".into(),
                parameters: serde_json::from_value(json!({
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"]
                }))
                .expect("valid schema"),
            },
            calls,
        }
    }

    struct DenyPermission;

    impl Permission for DenyPermission {
        async fn check(
            &self,
            _request: &PermissionRequest,
            _cancellation: &Cancellation,
        ) -> Result<crate::permission::Decision, Error> {
            Ok(crate::permission::Decision {
                behavior: crate::permission::Behavior::Deny,
                reason: Some("tool call denied".into()),
            })
        }
    }

    fn next_now<S: Stream<Item = Result<Event, Error>> + Unpin>(
        stream: &mut S,
    ) -> Option<Result<Event, Error>> {
        let waker: &Waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        match Pin::new(stream).poll_next(&mut context) {
            Poll::Ready(value) => value,
            Poll::Pending => None,
        }
    }

    #[test]
    fn builder_uses_the_frozen_defaults_and_maps_payload_parts() {
        let config = builder();
        assert_eq!(config.max_concurrency, 100);

        let message = user_message(Input {
            metadata: Map::from_iter([
                (String::from("chat_id"), json!(101)),
                (String::from("language"), json!("zh")),
            ]),
            payload: vec![crate::components::agent::Payload {
                r#type: "text".into(),
                data: json!("hello"),
                mime_type: Some("text/plain".into()),
            }],
        });
        assert!(!message.metadata.contains_key("chat_id"));
        assert_eq!(message.metadata["language"], "zh");
        assert_eq!(message.content[0].r#type, "text");
        assert_eq!(message.content[0].data["value"], "hello");
        assert_eq!(message.content[0].data["mime_type"], "text/plain");
    }

    #[test]
    fn public_chat_builder_keeps_system_prompt_before_user_payload() {
        let messages = initial_messages(Input::default(), Some("你是助手"), Vec::new());
        assert_eq!(messages.as_slice()[0].role, Role::System);
        assert_eq!(messages.as_slice()[0].content[0].data["value"], "你是助手");
        assert_eq!(messages.as_slice()[1].role, Role::User);
    }

    #[test]
    fn run_forwards_provider_events_and_maps_provider_errors() {
        let called = Arc::new(AtomicBool::new(false));
        let agent = builder()
            .model(FakeChat {
                called: called.clone(),
                failure: None,
                include_complete: true,
                pending: false,
            })
            .build()
            .expect("agent build");
        let mut stream = block_on(agent.run(Input::default(), true, Cancellation::new()))
            .expect("provider stream");
        assert!(matches!(
            next_now(&mut stream),
            Some(Ok(Event::Delta { .. }))
        ));
        assert!(matches!(
            next_now(&mut stream),
            Some(Ok(Event::Complete { .. }))
        ));
        assert!(called.load(Ordering::Acquire));

        let provider_error = Error::new("PROVIDER", "unavailable").with_details("retryable");
        let agent = builder()
            .model(FakeChat {
                called: Arc::new(AtomicBool::new(false)),
                failure: Some(provider_error),
                include_complete: true,
                pending: false,
            })
            .build()
            .expect("agent build");
        let error = block_on(agent.run(Input::default(), true, Cancellation::new())).unwrap_err();
        assert_eq!(error.code, "PROVIDER");
        assert_eq!(error.details.as_deref(), Some("retryable"));
    }

    #[test]
    fn stream_item_errors_and_normal_completion_are_terminal_once() {
        let mut errored = Events::new(
            FakeStream {
                events: VecDeque::from([Err(Error::new("PROVIDER", "stream failed"))]),
                pending: false,
            },
            Cancellation::new(),
            0,
            Hooks::default(),
        );
        assert_eq!(
            next_now(&mut errored)
                .expect("stream error")
                .unwrap_err()
                .code,
            "PROVIDER"
        );
        assert!(next_now(&mut errored).is_none());

        let cancellation = Cancellation::new();
        let mut completed = Events::new(
            FakeStream {
                events: VecDeque::from([Ok(Event::Complete {
                    message: Message::new(Role::Assistant),
                    usage: None,
                    finish_reason: Some("stop".into()),
                })]),
                pending: false,
            },
            cancellation.clone(),
            0,
            Hooks::default(),
        );
        assert!(matches!(
            next_now(&mut completed),
            Some(Ok(Event::Complete { .. }))
        ));
        cancellation.cancel();
        assert!(next_now(&mut completed).is_none());
    }

    #[test]
    fn cancelled_run_does_not_invoke_provider() {
        let called = Arc::new(AtomicBool::new(false));
        let agent = builder()
            .model(FakeChat {
                called: called.clone(),
                failure: None,
                include_complete: true,
                pending: false,
            })
            .build()
            .expect("agent build");
        let cancellation = Cancellation::new();
        cancellation.cancel();
        let error = block_on(agent.run(Input::default(), true, cancellation)).unwrap_err();
        assert_eq!(error.code, "CANCELLED");
        assert!(!called.load(Ordering::Acquire));
    }

    #[test]
    fn cancellation_after_delta_emits_one_partial_complete() {
        let called = Arc::new(AtomicBool::new(false));
        let agent = builder()
            .model(FakeChat {
                called,
                failure: None,
                include_complete: true,
                pending: false,
            })
            .build()
            .expect("agent build");
        let cancellation = Cancellation::new();
        let mut stream = block_on(agent.run(Input::default(), true, cancellation.clone()))
            .expect("provider stream");
        assert!(matches!(
            next_now(&mut stream),
            Some(Ok(Event::Delta { .. }))
        ));
        cancellation.cancel();
        let event = next_now(&mut stream)
            .expect("cancel event")
            .expect("event ok");
        match event {
            Event::Complete {
                message,
                usage: Some(usage),
                finish_reason: Some(reason),
            } => {
                assert_eq!(reason, "cancelled");
                assert_eq!(message.content[0].data["value"], "hel");
                assert!(usage.total.unwrap_or_default() > 0);
            }
            other => panic!("unexpected event: {other:?}"),
        }
        assert!(next_now(&mut stream).is_none());
    }

    #[test]
    fn missing_provider_complete_is_an_invalid_response() {
        let agent = builder()
            .model(FakeChat {
                called: Arc::new(AtomicBool::new(false)),
                failure: None,
                include_complete: false,
                pending: false,
            })
            .build()
            .expect("agent build");
        let mut stream = block_on(agent.run(Input::default(), true, Cancellation::new()))
            .expect("provider stream");
        assert!(matches!(
            next_now(&mut stream),
            Some(Ok(Event::Delta { .. }))
        ));
        let error = next_now(&mut stream)
            .expect("stream error")
            .expect_err("missing Complete must fail");
        assert_eq!(error.code, "INVALID_RESPONSE");
        assert!(next_now(&mut stream).is_none());
    }

    #[test]
    fn cancellation_wakes_a_pending_agent_stream() {
        struct WakeFlag(AtomicBool);

        impl Wake for WakeFlag {
            fn wake(self: Arc<Self>) {
                self.0.store(true, Ordering::Release);
            }

            fn wake_by_ref(self: &Arc<Self>) {
                self.0.store(true, Ordering::Release);
            }
        }

        let agent = builder()
            .model(FakeChat {
                called: Arc::new(AtomicBool::new(false)),
                failure: None,
                include_complete: false,
                pending: true,
            })
            .build()
            .expect("agent build");
        let cancellation = Cancellation::new();
        let mut stream = block_on(agent.run(Input::default(), true, cancellation.clone()))
            .expect("provider stream");
        let flag = Arc::new(WakeFlag(AtomicBool::new(false)));
        let waker: Waker = flag.clone().into();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(
            Pin::new(&mut stream).poll_next(&mut context),
            Poll::Pending
        ));
        cancellation.cancel();
        assert!(flag.0.load(Ordering::Acquire));
        assert!(matches!(
            Pin::new(&mut stream).poll_next(&mut context),
            Poll::Ready(Some(Ok(Event::Complete {
                finish_reason: Some(reason),
                ..
            }))) if reason == "cancelled"
        ));
    }

    #[test]
    fn chat_executes_model_tool_calls_and_continues_with_tool_message() {
        let model_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tool_called = Arc::new(AtomicBool::new(false));
        let agent = builder()
            .model(SequenceChat {
                calls: Arc::clone(&model_calls),
            })
            .toolkit(Toolkit::new().tool(echo_tool(Arc::clone(&tool_called))))
            .build()
            .expect("agent build");
        let mut stream =
            block_on(agent.run(Input::default(), true, Cancellation::new())).expect("agent stream");
        let events = block_on(async {
            let mut events = Vec::new();
            while let Some(event) = stream.next().await {
                events.push(event.expect("event"));
            }
            events
        });

        assert_eq!(model_calls.load(Ordering::SeqCst), 2);
        assert!(tool_called.load(Ordering::SeqCst));
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], Event::Complete { .. }));
        assert!(
            matches!(events[1], Event::Complete { ref message, finish_reason: Some(ref reason), .. } if message.role == Role::Tool && reason == "tool")
        );
        assert!(
            matches!(events[2], Event::Complete { ref message, .. } if message.role == Role::Assistant)
        );
    }

    #[test]
    fn chat_waiting_child_checkpoint_does_not_call_parent_before_child_restore() {
        use crate::checkpoint::Checkpoint as _;

        struct CountingChat {
            calls: Arc<std::sync::atomic::AtomicUsize>,
        }

        impl Chat for CountingChat {
            type Stream = FakeStream;

            async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
                assert!(request.stream);
                self.calls.fetch_add(1, Ordering::SeqCst);
                let mut message = Message::new(Role::Assistant);
                message.content.push(Part {
                    r#type: "text".into(),
                    data: part_data(json!({"value": "child complete"})),
                });
                Ok(FakeStream {
                    events: VecDeque::from([Ok(Event::Complete {
                        message,
                        usage: None,
                        finish_reason: Some("stop".into()),
                    })]),
                    pending: false,
                })
            }
        }

        block_on(async {
            let checkpoint = Arc::new(crate::core::checkpoint::InMemory::new());
            checkpoint
                .save(
                    991,
                    crate::core::agent::subagent::waiting_fixture(
                        json!({
                            "type": "agent",
                            "chat_id": 991,
                            "status": "waiting",
                            "stream": true,
                            "messages": [],
                            "usage": null,
                            "query": {"metadata": {"chat_id": 991}, "payload": []},
                            "metadata": {"chat_id": 991},
                        }),
                        Call {
                            call_id: "spawn_1".into(),
                            name: "agent_spawn".into(),
                            arguments: serde_json::from_value(
                                json!({"name":"child", "input":{"payload":[]}}),
                            )
                            .unwrap(),
                        },
                        test_id,
                    ),
                    &Cancellation::new(),
                )
                .await
                .unwrap();
            let parent_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let child_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let child = builder()
                .model(CountingChat {
                    calls: child_calls.clone(),
                })
                .build()
                .unwrap();
            let parent = builder()
                .model(CountingChat {
                    calls: parent_calls.clone(),
                })
                .subagents(vec![SubAgent::new("child", child).unwrap()])
                .with_id_fn(test_id)
                .checkpoint(checkpoint)
                .build()
                .unwrap();
            let input = Input {
                metadata: Map::from_iter([("chat_id".into(), json!(991))]),
                payload: Vec::new(),
            };
            let events = parent
                .run(input, true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert!(events.iter().all(Result::is_ok), "{events:?}");
            assert_eq!(child_calls.load(Ordering::SeqCst), 1);
            assert_eq!(parent_calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn chat_subagent_waiting_scene_is_resumed_for_the_same_call() {
        block_on(async {
            let child = SubAgent::new("reviewer", WaitingSubAgent).unwrap();
            let call = Call {
                call_id: "spawn_1".into(),
                name: "agent_spawn".into(),
                arguments: serde_json::from_value(json!({
                    "name": "reviewer",
                    "input": {"payload": []}
                }))
                .unwrap(),
            };
            let failure = execute_subagent(
                vec![child.clone()],
                Map::new(),
                call.clone(),
                None,
                Cancellation::new(),
                None,
                None,
            )
            .await
            .expect_err("child should wait before approval");
            assert_eq!(failure.error.code, "WAITING");
            assert_eq!(failure.scene, Some(json!({"owner": "chat-child"})));

            let progress = execute_subagent(
                vec![child],
                Map::new(),
                call,
                failure.scene,
                Cancellation::new(),
                None,
                None,
            )
            .await
            .expect("saved child scene should resume");
            assert_eq!(progress.message.role, Role::Function);
            assert_eq!(progress.message.content[0].data["value"], "child approved");
        });
    }

    #[test]
    fn chat_parallel_subagents_keep_completed_siblings_when_one_waits() {
        block_on(async {
            let root = Root::default();
            let steps: Arc<Mutex<Vec<Step>>> = Default::default();
            let completed_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let subagents = vec![
                SubAgent::new(
                    "done",
                    CountingSubAgent {
                        calls: completed_calls.clone(),
                    },
                )
                .unwrap(),
                SubAgent::new("waiting", WaitingSubAgent).unwrap(),
            ];
            let calls = vec![
                Call {
                    call_id: "spawn_done".into(),
                    name: "agent_spawn".into(),
                    arguments: serde_json::from_value(json!({
                        "name": "done",
                        "input": {"payload": []}
                    }))
                    .unwrap(),
                },
                Call {
                    call_id: "spawn_waiting".into(),
                    name: "agent_spawn".into(),
                    arguments: serde_json::from_value(json!({
                        "name": "waiting",
                        "input": {"payload": []}
                    }))
                    .unwrap(),
                },
            ];
            let first = execute_tool_batch(
                ToolBatchConfig {
                    interaction: None,
                    relay: None,
                    toolkit: Arc::new(Toolkit::new()),
                    steps: steps.clone(),
                    subagents: subagents.clone(),
                    access: Access::default(),
                    hooks: Hooks::default(),
                    metadata: Map::new(),
                    cancellation: Cancellation::new(),
                    max_concurrency: 2,
                    root: root.clone(),
                    id_fn: Some(test_id),
                },
                calls.clone(),
            )
            .await
            .expect_err("one child should wait");
            assert_eq!(first.error.code, "WAITING");
            assert!(root.message(&steps.lock().unwrap(), &calls[0]).is_some());
            let resumed = execute_tool_batch(
                ToolBatchConfig {
                    interaction: None,
                    relay: None,
                    toolkit: Arc::new(Toolkit::new()),
                    steps,
                    subagents,
                    access: Access::default(),
                    hooks: Hooks::default(),
                    metadata: Map::new(),
                    cancellation: Cancellation::new(),
                    max_concurrency: 2,
                    root,
                    id_fn: None,
                },
                calls,
            )
            .await
            .expect("waiting child should resume with completed sibling");
            assert_eq!(resumed.0.len(), 2);
            assert_eq!(completed_calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn denied_chat_tool_call_becomes_direct_tool_error_for_next_model_turn() {
        let model_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tool_called = Arc::new(AtomicBool::new(false));
        let agent = builder()
            .model(SequenceChat {
                calls: Arc::clone(&model_calls),
            })
            .toolkit(Toolkit::new().tool(echo_tool(Arc::clone(&tool_called))))
            .permission(DenyPermission)
            .build()
            .expect("agent build");
        let mut stream =
            block_on(agent.run(Input::default(), true, Cancellation::new())).expect("agent stream");
        let events = block_on(async {
            let mut events = Vec::new();
            while let Some(event) = stream.next().await {
                events.push(event.expect("event"));
            }
            events
        });

        assert_eq!(model_calls.load(Ordering::SeqCst), 2);
        assert!(!tool_called.load(Ordering::SeqCst));
        assert!(
            matches!(events[1], Event::Complete { ref message, .. } if message.role == Role::Tool && message.values.get("error").is_some())
        );
        assert!(
            matches!(events.last(), Some(Event::Complete { message, .. }) if message.role == Role::Assistant)
        );
    }
}
