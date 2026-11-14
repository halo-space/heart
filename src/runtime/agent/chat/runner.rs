//! Chat Agent stream execution and tool lifecycle.
//!
//! The parent module owns public construction and the Agent contract. This
//! module owns polling, model/provider transitions, tool batches, checkpoint
//! snapshots, and event closeout so those runtime concerns do not leak into
//! the public builder.

use super::*;

mod checkpoint;
mod tools;

use checkpoint::{checkpoint_future, checkpoint_state, waiting_checkpoint_state};
pub(crate) use checkpoint::{
    checkpoint_identity_matches, checkpoint_messages, checkpoint_pending_subagents,
    checkpoint_query, checkpoint_steps, is_waiting_snapshot,
};
use tools::decode_call;
#[doc(hidden)]
pub use tools::execute_subagent;
#[doc(hidden)]
pub use tools::execute_tool_batch;

pub(crate) type CompletionFuture =
    Pin<Box<dyn Future<Output = Result<Event, Error>> + Send + 'static>>;
pub(crate) type CheckpointFuture =
    Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'static>>;
pub(crate) type ProviderFuture<S> =
    Pin<Box<dyn Future<Output = Result<S, Error>> + Send + 'static>>;
pub(crate) type ToolBatch = (Vec<Message>, Option<crate::model::token::Usage>);
pub(crate) type ToolBatchFuture =
    Pin<Box<dyn Future<Output = Result<ToolBatch, Box<ToolBatchFailure>>> + Send + 'static>>;
pub(crate) type ProviderFactory<S> = Arc<
    dyn Fn(Messages, Cancellation, Arc<AtomicU64>, Arc<Mutex<Vec<Step>>>) -> ProviderFuture<S>
        + Send
        + Sync,
>;
pub(crate) type EvaluatorHandle =
    Arc<dyn Fn(Input, Messages, Cancellation) -> EvaluationFuture + Send + Sync>;
pub(crate) type EvaluationFuture =
    Pin<Box<dyn Future<Output = Result<Evaluation, Error>> + Send + 'static>>;
pub(crate) type CheckpointSave = Backend;

#[doc(hidden)]
pub struct ToolBatchConfig {
    pub interaction: Option<crate::runtime::agent::reply::Handle>,
    pub relay: Option<crate::runtime::agent::subagent::Relay>,
    pub steps: Arc<Mutex<Vec<Step>>>,
    pub dispatch: Arc<Dispatch>,
    pub subagents: Vec<SubAgent>,
    pub access: Access,
    pub hooks: Hooks,
    pub metadata: Map<String, Value>,
    pub cancellation: Cancellation,
    pub max_concurrency: usize,
    pub root: Root,
    pub id_fn: Option<fn() -> i64>,
    pub trace: TraceRoot,
}

// Private per-call dependencies; not a public Tool context or saved state.
struct ToolExecution {
    dispatch: Arc<Dispatch>,
    access: Access,
    hooks: Hooks,
    metadata: Map<String, Value>,
    cancellation: Cancellation,
    steps: Arc<Mutex<Vec<Step>>>,
    interaction: Option<crate::runtime::agent::reply::Handle>,
    id_fn: Option<fn() -> i64>,
    trace: TraceRoot,
}

pub(crate) struct EventsConfig<S> {
    pub(super) completion: Option<crate::runtime::agent::completion::State>,
    pub(super) interaction: Option<crate::runtime::agent::reply::Handle>,
    pub(super) steps: Arc<Mutex<Vec<Step>>>,
    pub(super) provider_factory: ProviderFactory<S>,
    pub(super) dispatch: Option<Dispatch>,
    pub(super) subagents: Vec<SubAgent>,
    pub(super) access: Access,
    pub(super) evaluator: Option<EvaluatorHandle>,
    pub(super) checkpoint: Option<CheckpointSave>,
    pub(super) input: Input,
    pub(super) checkpoint_id: Option<i64>,
    pub(super) metadata: Map<String, Value>,
    pub(super) messages: Messages,
    pub(super) max_concurrency: usize,
    pub(super) pending_subagents: std::collections::BTreeMap<String, PendingSubAgent>,
    pub(super) root: Root,
    pub(super) id_fn: Option<fn() -> i64>,
    pub(super) trace: TraceRoot,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct PendingSubAgent {
    pub(super) index: usize,
    pub(super) call: Call,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct ToolBatchFailure {
    pub error: Error,
    pub usage: Option<crate::model::token::Usage>,
}

/// The concrete stream returned by this Chat Agent. Its items remain the
/// shared Event contract; this type only owns cancellation and partial state.
pub struct Events<S> {
    pub(super) completion: Arc<Mutex<Option<crate::runtime::agent::completion::State>>>,
    pub(super) preparation: Option<crate::runtime::agent::preparation::State>,
    pub(super) _interaction_owner: Option<crate::runtime::agent::reply::Owner>,
    pub(super) interaction: Option<crate::runtime::agent::reply::Handle>,
    pub(super) waits: Vec<crate::runtime::agent::wait::Wait>,
    pub(super) questions: VecDeque<Event>,
    pub(super) stream: bool,
    pub(super) steps: Arc<Mutex<Vec<Step>>>,
    pub(super) root: Root,
    pub(super) provider: Option<Pin<Box<S>>>,
    pub(super) provider_factory: Option<ProviderFactory<S>>,
    pub(super) cancellation: Cancellation,
    pub(super) cancellation_waiter: Option<crate::cancellation::Registration>,
    pub(super) partial: Message,
    pub(super) input_tokens: u64,
    pub(super) pending_input_tokens: Arc<AtomicU64>,
    pub(super) request: Arc<Mutex<Option<Box<Request>>>>,
    pub(super) replayed_completion: bool,
    pub(super) finished: bool,
    pub(super) hooks: Hooks,
    pub(super) dispatch: Option<Arc<Dispatch>>,
    pub(super) subagents: Vec<SubAgent>,
    pub(super) access: Access,
    pub(super) evaluator: Option<EvaluatorHandle>,
    pub(super) checkpoint: Option<CheckpointSave>,
    pub(super) input: Option<Input>,
    pub(super) checkpoint_id: Option<i64>,
    pub(super) metadata: Map<String, Value>,
    pub(super) messages: Messages,
    pub(super) max_concurrency: usize,
    pub(super) id_fn: Option<fn() -> i64>,
    pub(super) pending_tools: Option<ToolBatchFuture>,
    pub(super) pending_provider: Option<ProviderFuture<S>>,
    pub(super) tool_results: VecDeque<Message>,
    pub(super) child_events: crate::runtime::agent::subagent::Relay,
    pub(super) pending_subagents: std::collections::BTreeMap<String, PendingSubAgent>,
    pub(super) pending_checkpoint: Option<CheckpointFuture>,
    pub(super) pending: Option<CompletionFuture>,
    pub(super) run: Execute,
    pub(super) trace: TraceRoot,
    pub(super) model_span: Option<SpanScope>,
    pub(super) polling: bool,
    pub(super) settled: bool,
    pub(super) closing: bool,
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
    #[doc(hidden)]
    pub fn new(provider: S, cancellation: Cancellation, input_tokens: u64, hooks: Hooks) -> Self {
        Self::new_with_trace(
            provider,
            cancellation,
            input_tokens,
            hooks,
            TraceRoot::start("agent.chat"),
        )
    }

    pub(super) fn new_with_trace(
        provider: S,
        cancellation: Cancellation,
        input_tokens: u64,
        hooks: Hooks,
        trace: TraceRoot,
    ) -> Self {
        let mut events = Self::empty_with_trace(cancellation, hooks, trace);
        events.provider = Some(Box::pin(provider));
        events.input_tokens = input_tokens;
        events
    }

    pub(super) fn empty(cancellation: Cancellation, hooks: Hooks) -> Self {
        Self::empty_with_trace(cancellation, hooks, TraceRoot::noop())
    }

    pub(super) fn empty_with_trace(
        cancellation: Cancellation,
        hooks: Hooks,
        trace: TraceRoot,
    ) -> Self {
        Self {
            completion: Default::default(),
            preparation: None,
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
            cancellation_waiter: None,
            partial: Message::new(Role::Assistant),
            input_tokens: 0,
            pending_input_tokens: Arc::new(AtomicU64::new(0)),
            request: Default::default(),
            replayed_completion: false,
            finished: false,
            hooks,
            dispatch: None,
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
            child_events: crate::runtime::agent::subagent::Relay::default(),
            pending_subagents: std::collections::BTreeMap::new(),
            pending_checkpoint: None,
            pending: None,
            run: Execute::default(),
            trace,
            model_span: None,
            polling: false,
            settled: false,
            closing: false,
        }
    }

    pub(super) fn configure_tools(mut self, config: EventsConfig<S>) -> Self {
        self.completion = Arc::new(Mutex::new(config.completion));
        self._interaction_owner = config
            .interaction
            .as_ref()
            .map(crate::runtime::agent::reply::Handle::owner);
        self.interaction = config.interaction;
        self.steps = config.steps;
        self.provider_factory = Some(config.provider_factory);
        self.dispatch = config.dispatch.map(Arc::new);
        self.subagents = config.subagents;
        self.access = config.access;
        self.evaluator = config.evaluator;
        self.checkpoint = config.checkpoint;
        self.input = Some(config.input);
        self.checkpoint_id = config.checkpoint_id;
        self.trace = config.trace;
        self.model_span = self
            .provider
            .as_ref()
            .map(|_| self.trace.child("agent.model"));
        self.metadata = config.metadata;
        if let Some(chat_id) = self.metadata.get("chat_id").and_then(Value::as_i64) {
            self.trace.property("chat_id", chat_id.to_string());
        }
        self.messages = config.messages;
        self.max_concurrency = config.max_concurrency;
        self.id_fn = config.id_fn;
        self.pending_subagents = config.pending_subagents;
        self.root = config.root;
        self
    }

    pub(super) fn cancelled(&mut self) -> Event {
        if self.settled
            && let Some(message) = crate::runtime::agent::completion::partial(
                self.completion.lock().expect("completion lock").as_ref(),
                &self.steps.lock().expect("Agent steps lock"),
            )
        {
            self.partial = message;
        }
        let message = std::mem::replace(&mut self.partial, Message::new(Role::Assistant));
        if self
            .preparation
            .as_ref()
            .is_some_and(|state| state.stage != crate::runtime::agent::preparation::Stage::Complete)
            || crate::runtime::agent::model_steps::pending_input(
                &self.steps.lock().expect("Model steps lock"),
            )
            .is_some_and(|step| step.stage.as_deref() == Some("before"))
        {
            // No Model request exists yet: preserve stopped output without
            // inventing token consumption for a cancelled preparation Hook.
            return Event::Complete {
                message,
                usage: None,
                finish_reason: Some("cancelled".into()),
            };
        }
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
            if self.replayed_completion && self.settled {
                self.run.refine_saved(message.usage.as_ref(), &usage);
            } else if self.settled {
                self.run.refine_last(Some(usage.clone()));
            } else {
                self.run.settle(Some(usage.clone()));
                crate::runtime::agent::model_steps::unfinished(
                    &mut self.steps.lock().expect("Model steps lock"),
                    &message,
                    &usage,
                );
            }
            if self.settled
                && let Ok(mut steps) = self.steps.lock()
            {
                crate::runtime::agent::model_steps::refine(&mut steps, &usage);
                if let Some(completion) = self.completion.lock().expect("completion lock").as_mut()
                {
                    completion.message.usage = Some(usage.clone());
                }
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

    pub(super) fn close(&mut self, result: Result<Event, Error>) {
        if let Err(error) = &result {
            crate::runtime::agent::model_steps::stopped(
                &mut self.steps.lock().expect("Model steps lock"),
                error,
            );
        } else if matches!(&result, Ok(Event::Complete { finish_reason:Some(reason), .. }) if reason == "cancelled")
        {
            crate::runtime::agent::model_steps::stopped(
                &mut self.steps.lock().expect("Model steps lock"),
                &Error::new("CANCELLED", "Agent cancelled"),
            );
        }
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
            step::cancel_pending(&mut self.steps.lock().expect("Tool steps lock poisoned"));
        }
        if result.is_err() && !self.settled {
            let usage = crate::runtime::agent::lifecycle::model_unfinished(
                &mut self.steps.lock().expect("Model steps lock"),
                &self.partial,
            );
            self.run.settle(usage);
            self.settled = true;
        }
        let run = self.run.clone();
        let checkpoint = self.checkpoint.clone();
        let checkpoint_id = self.checkpoint_id;
        let mut state = checkpoint_state(
            &self.messages,
            self.run.usage(),
            (
                self.input.as_ref(),
                self.preparation.as_ref(),
                self.completion.lock().expect("completion lock").as_ref(),
            ),
            &result,
            &self.root,
            &self.steps.lock().expect("Tool steps lock poisoned"),
            self.stream,
        );
        state["waits"] = serde_json::to_value(&self.waits).expect("waits serialization");
        self.closing = true;
        let embedded = crate::runtime::agent::scope::capture(state.clone());
        self.pending = Some(Box::pin(async move {
            if embedded {
                // The outer Workflow owns the embedded Agent checkpoint.
            } else if let (Some(checkpoint), Some(id)) = (checkpoint, checkpoint_id) {
                checkpoint
                    .save(id, state.clone(), Cancellation::new())
                    .await?;
            }
            // Chat.state is written only at terminal closeout and is never
            // read for restoration.
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
        let outer_poll = !this.polling;
        if outer_poll {
            this.polling = true;
        }
        loop {
            if outer_poll {
                this.polling = true;
            }
            let mut result = this.poll_event(cx);
            if !this.stream && matches!(result, Poll::Ready(Some(Ok(Event::Delta { .. })))) {
                if outer_poll {
                    this.polling = false;
                }
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
            if outer_poll {
                this.polling = false;
            }
            if outer_poll && let Poll::Ready(Some(item)) = &result {
                crate::runtime::observation::observe_trace(&this.trace, item);
            }
            return result;
        }
    }
}

impl<S: Stream<Item = Result<Event, Error>> + 'static> Events<S> {
    pub(super) fn postprocess(&self, model: Option<Step>) -> CompletionFuture {
        let hooks = self.hooks.clone();
        let steps = self.steps.clone();
        let completion = self.completion.clone();
        let cancellation = self.cancellation.clone();
        let evaluator = self.evaluator.clone();
        let input = self.input.clone();
        Box::pin(async move {
            let event = if let Some(saved) = model {
                Box::pin(crate::runtime::agent::completion::model(
                    &hooks,
                    saved,
                    |saved| {
                        let mut steps = steps.lock().expect("model steps lock");
                        let step = steps
                            .iter_mut()
                            .find(|step| step.id == saved.id)
                            .ok_or_else(|| {
                                Error::new("INVALID_STATE", "Model Step owner is missing")
                            })?;
                        *step = saved;
                        Ok(())
                    },
                    &cancellation,
                ))
                .await?
            } else {
                let saved = completion
                    .lock()
                    .expect("completion lock")
                    .clone()
                    .ok_or_else(|| Error::new("INVALID_STATE", "Agent result is missing"))?;
                Event::Complete {
                    usage: saved.message.usage.clone(),
                    message: saved.message,
                    finish_reason: saved.finish_reason,
                }
            };
            let Event::Complete {
                message,
                finish_reason,
                ..
            } = event
            else {
                unreachable!()
            };
            if crate::runtime::agent::completion::has_calls(&message) {
                return Ok(Event::Complete {
                    usage: message.usage.clone(),
                    message,
                    finish_reason,
                });
            }
            let saved = {
                let mut owner = completion.lock().expect("completion lock");
                if owner.as_ref().is_none_or(|state| !state.pending()) {
                    *owner = Some(crate::runtime::agent::completion::State::new(
                        message,
                        finish_reason,
                    ));
                }
                owner.clone().expect("actual Agent result")
            };
            let result = Box::pin(crate::runtime::agent::completion::agent(
                &hooks,
                saved,
                |saved| {
                    *completion.lock().expect("completion lock") = Some(saved);
                    Ok(())
                },
                &cancellation,
            ))
            .await?;
            if let (Some(evaluator), Some(input), Event::Complete { message, .. }) =
                (evaluator, input, &result)
            {
                let evaluation =
                    evaluator(input, Messages::new([message.clone()]), cancellation).await?;
                if !evaluation.passed {
                    let details = evaluation
                        .feedback
                        .and_then(|feedback| serde_json::to_string(&feedback).ok())
                        .unwrap_or_else(|| "final result was rejected".into());
                    return Err(
                        Error::new("EVALUATION_FAILED", "agent final result was rejected")
                            .with_details(details),
                    );
                }
            }
            Ok(result)
        })
    }

    fn capture_live(&self) {
        if !crate::runtime::agent::scope::active() {
            return;
        }
        crate::runtime::agent::scope::capture(self.live_snapshot());
    }
    fn live_snapshot(&self) -> Value {
        let steps = self.steps.lock().expect("Tool steps lock poisoned");
        let mut snapshot = if self.waits.is_empty() {
            checkpoint_state(
                &self.messages,
                self.run.usage(),
                (
                    self.input.as_ref(),
                    self.preparation.as_ref(),
                    self.completion.lock().expect("completion lock").as_ref(),
                ),
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
                (
                    self.input.as_ref(),
                    self.preparation.as_ref(),
                    self.completion.lock().expect("completion lock").as_ref(),
                ),
                &self.root,
                &steps,
                self.stream,
            )
        };
        snapshot["waits"] = serde_json::to_value(&self.waits).expect("waits serialization");
        snapshot
    }
    // Keep cancellation-only locals out of every normal nested stream poll.
    #[inline(never)]
    fn cancel_children(&mut self, cx: &mut Context<'_>) -> Result<Vec<Event>, Error> {
        let (completed, events) = crate::runtime::agent::subagent::cancel_ready(
            cx,
            &self.child_events,
            |cx| {
                let Some(tools) = self.pending_tools.as_mut() else {
                    return Poll::Ready(None);
                };
                match tools.as_mut().poll(cx) {
                    Poll::Ready(result) => {
                        self.pending_tools = None;
                        Poll::Ready(Some(result))
                    }
                    Poll::Pending => Poll::Pending,
                }
            },
            |observation| {
                self.root.observe(observation)?;
                self.run.settle_child(observation.usage.clone());
                Ok(())
            },
        )?;
        for result in completed {
            self.run.settle_child(match result {
                Ok((_, usage)) => usage,
                Err(failure) => failure.usage,
            });
        }
        self.root.cancel_pending();
        Ok(events)
    }

    // Finish state/history mutations before decorating the outgoing copy.
    fn poll_event(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<Event, Error>>> {
        let this = self;
        if this.finished {
            return Poll::Ready(None);
        }
        if !this.closing {
            if let Some(waiter) = &this.cancellation_waiter {
                waiter.update(cx.waker());
            } else {
                this.cancellation_waiter = Some(this.cancellation.register(cx.waker()));
            }
        }
        if this.cancellation.is_cancelled() && !this.closing {
            let events = match this.cancel_children(cx) {
                Ok(events) => events,
                Err(error) => {
                    this.close(Err(error));
                    return Pin::new(this).poll_next(cx);
                }
            };
            let result = this.cancelled();
            this.close(Ok(result));
            // Existing transient output queue; these are actual child events,
            // not synthesized results or another saved state.
            this.questions.extend(events);
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
                    this.model_span = Some(this.trace.child("agent.model"));
                    if let Some(request) = this
                        .request
                        .lock()
                        .expect("Model request lock poisoned")
                        .as_ref()
                        && let Some(model_span) = this.model_span.as_ref()
                    {
                        model_span.property_json("model.input", request.as_ref());
                    }
                    this.pending_provider = None;
                    return Pin::new(this).poll_next(cx);
                }
                Poll::Ready(Err(error)) => {
                    this.pending_provider = None;
                    if this.cancellation.is_cancelled() {
                        let result = this.cancelled();
                        this.close(Ok(result));
                    } else {
                        this.close(Err(error));
                    }
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
                        let feedback = matches!(
                            &change,
                            crate::runtime::agent::reply::Change::Feedback { .. }
                        );
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
                                    (
                                        this.input.as_ref(),
                                        this.preparation.as_ref(),
                                        this.completion.lock().expect("completion lock").as_ref(),
                                    ),
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
                            Ok(None) => {
                                this.capture_live();
                                if feedback && !this.waits.is_empty() {
                                    let mut snapshot = waiting_checkpoint_state(
                                        &this.messages,
                                        this.run.usage(),
                                        (
                                            this.input.as_ref(),
                                            this.preparation.as_ref(),
                                            this.completion
                                                .lock()
                                                .expect("completion lock")
                                                .as_ref(),
                                        ),
                                        &this.root,
                                        &this.steps.lock().expect("Tool steps lock poisoned"),
                                        this.stream,
                                    );
                                    snapshot["waits"] = serde_json::to_value(&this.waits)
                                        .expect("waits serialization");
                                    this.pending_checkpoint = Some(checkpoint_future(
                                        this.checkpoint.clone(),
                                        this.checkpoint_id,
                                        snapshot,
                                    ));
                                }
                            }
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
                                        step.status
                                            == crate::runtime::agent::node::status::step::Status::Running
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
                                .and_then(|_| {
                                    observation.commit_feedback(this.interaction.as_ref())
                                })
                            {
                                this.close(Err(error));
                                return Pin::new(this).poll_next(cx);
                            }
                            this.run.settle_child(observation.usage);
                            if question || observation.event.is_none() || observation.feedback {
                                let mut snapshot = waiting_checkpoint_state(
                                    &this.messages,
                                    this.run.usage(),
                                    (
                                        this.input.as_ref(),
                                        this.preparation.as_ref(),
                                        this.completion.lock().expect("completion lock").as_ref(),
                                    ),
                                    &this.root,
                                    &this.steps.lock().expect("Tool steps lock poisoned"),
                                    this.stream,
                                );
                                snapshot["waits"] =
                                    serde_json::to_value(&this.waits).expect("waits serialization");
                                if this.waits.is_empty() {
                                    snapshot = this.live_snapshot();
                                }
                                if let Some(event) = observation.event {
                                    this.questions.push_back(event);
                                }
                                this.pending_checkpoint = Some(checkpoint_future(
                                    this.checkpoint.clone(),
                                    this.checkpoint_id,
                                    snapshot,
                                ));
                                return Pin::new(this).poll_next(cx);
                            }
                            Poll::Ready(Some(Ok(observation.event.expect("ordinary event"))))
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
                            (
                                this.input.as_ref(),
                                this.preparation.as_ref(),
                                this.completion.lock().expect("completion lock").as_ref(),
                            ),
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
                    if this.cancellation.is_cancelled() {
                        let result = this.cancelled();
                        this.close(Ok(result));
                    } else {
                        this.close(Err(error.error));
                    }
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
                step::committed(&mut this.steps.lock().expect("Tool steps lock poisoned"));
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
                    this.steps.clone(),
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
                            if let Some(completion) =
                                this.completion.lock().expect("completion lock").as_mut()
                            {
                                completion.stage = "committed".into();
                            }
                            if calls.is_empty() {
                                this.close(Ok(event));
                                return Pin::new(this).poll_next(cx);
                            }
                            if this.dispatch.is_some() || !this.subagents.is_empty() {
                                let selected = step::selected(
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
                                            (
                                                this.input.as_ref(),
                                                this.preparation.as_ref(),
                                                this.completion
                                                    .lock()
                                                    .expect("completion lock")
                                                    .as_ref(),
                                            ),
                                            &this.root,
                                            &this.steps.lock().expect("Tool steps lock poisoned"),
                                            this.stream,
                                        ),
                                    ));
                                }
                                let dispatch = this
                                    .dispatch
                                    .clone()
                                    .unwrap_or_else(|| Arc::new(Dispatch::new()));
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
                                        dispatch,
                                        subagents: this.subagents.clone(),
                                        access,
                                        hooks,
                                        metadata,
                                        cancellation,
                                        max_concurrency,
                                        id_fn: this.id_fn,
                                        root: this.root.clone(),
                                        trace: this.trace.clone(),
                                    },
                                    calls,
                                )));
                                // The model Complete is emitted before its tool calls.
                                return Poll::Ready(Some(Ok(event)));
                            }
                            this.close(Err(Error::new(
                                "INVALID_STATE",
                                "model returned tool calls without configured tools",
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
                if let Some(model_span) = this.model_span.as_ref() {
                    model_span.property_json("model.output", &message);
                    model_span.property_json("model.usage", &usage);
                }
                this.model_span = None;
                crate::runtime::observation::clear_source(&mut message);
                let usage = usage.or_else(|| message.usage.clone());
                message.usage = usage.clone();
                this.run.settle(usage.clone());
                this.settled = true;
                this.replayed_completion = false;
                this.partial = message.clone();
                let saved = {
                    let mut steps = this.steps.lock().expect("Tool steps lock poisoned");
                    crate::runtime::agent::model_steps::returned(
                        &mut steps,
                        this.request
                            .lock()
                            .expect("Model request lock poisoned")
                            .take(),
                        message,
                        usage,
                        finish_reason,
                    );
                    steps.last().expect("actual model result").clone()
                };
                this.pending = Some(this.postprocess(Some(saved)));
                // Poll the owned hook future immediately; no extra Delta/Complete.
                Pin::new(this).poll_next(cx)
            }
            Poll::Ready(Some(Err(error))) => {
                this.model_span = None;
                this.close(Err(error));
                Pin::new(this).poll_next(cx)
            }
            Poll::Ready(None) => {
                this.model_span = None;
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

pub(crate) async fn prepare_model(
    hooks: &Hooks,
    steps: &Arc<Mutex<Vec<Step>>>,
    request: Request,
    cancellation: &Cancellation,
) -> Result<Request, Error> {
    let saved = crate::runtime::agent::model_steps::input(
        &mut steps.lock().expect("Model steps lock"),
        request,
    );
    Box::pin(crate::runtime::agent::lifecycle::model_prepare(
        hooks,
        saved,
        |saved| {
            let mut steps = steps.lock().expect("Model steps lock");
            let owner = steps
                .iter_mut()
                .find(|step| step.id == saved.id)
                .expect("actual Model input owner");
            *owner = saved;
            Ok(())
        },
        cancellation,
    ))
    .await
}

pub(crate) fn checkpoint_id(metadata: &Map<String, Value>) -> Result<i64, Error> {
    let id = metadata
        .get("chat_id")
        .and_then(Value::as_i64)
        .filter(|value| *value > 0)
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "checkpoint requires chat_id"))?;
    Ok(id)
}

#[doc(hidden)]
pub fn initial_messages(
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

pub(crate) fn estimate_output_tokens(message: &Message) -> u64 {
    crate::components::model::chat::estimate_tokens(
        message
            .content
            .iter()
            .map(crate::components::model::chat::part_bytes)
            .sum(),
    )
}
