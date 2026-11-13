//! Chat Agent stream execution and tool lifecycle.
//!
//! The parent module owns public construction and the Agent contract. This
//! module owns polling, model/provider transitions, tool batches, checkpoint
//! snapshots, and event closeout so those runtime concerns do not leak into
//! the public builder.

use super::*;

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

pub(crate) struct ToolBatchConfig {
    pub(super) interaction: Option<crate::runtime::agent::reply::Run>,
    pub(super) relay: Option<crate::runtime::agent::subagent::Relay>,
    pub(super) steps: Arc<Mutex<Vec<Step>>>,
    pub(super) dispatch: Arc<Dispatch>,
    pub(super) subagents: Vec<SubAgent>,
    pub(super) access: Access,
    pub(super) hooks: Hooks,
    pub(super) metadata: Map<String, Value>,
    pub(super) cancellation: Cancellation,
    pub(super) max_concurrency: usize,
    pub(super) root: Root,
    pub(super) id_fn: Option<fn() -> i64>,
    pub(super) trace: TraceRoot,
}

// Private per-call dependencies; not a public Tool context or saved state.
struct ToolExecution {
    dispatch: Arc<Dispatch>,
    access: Access,
    hooks: Hooks,
    metadata: Map<String, Value>,
    cancellation: Cancellation,
    steps: Arc<Mutex<Vec<Step>>>,
    interaction: Option<crate::runtime::agent::reply::Run>,
    id_fn: Option<fn() -> i64>,
    trace: TraceRoot,
}

pub(crate) struct EventsConfig<S> {
    pub(super) completion: Option<crate::runtime::agent::completion::State>,
    pub(super) interaction: Option<crate::runtime::agent::reply::Run>,
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
pub(crate) struct ToolBatchFailure {
    pub(super) error: Error,
    pub(super) usage: Option<crate::model::token::Usage>,
}

/// The concrete stream returned by this Chat Agent. Its items remain the
/// shared Event contract; this type only owns cancellation and partial state.
pub struct Events<S> {
    pub(super) completion: Arc<Mutex<Option<crate::runtime::agent::completion::State>>>,
    pub(super) preparation: Option<crate::runtime::agent::preparation::State>,
    pub(super) _interaction_owner: Option<crate::runtime::agent::reply::Owner>,
    pub(super) interaction: Option<crate::runtime::agent::reply::Run>,
    pub(super) waits: Vec<crate::runtime::agent::wait::Wait>,
    pub(super) questions: VecDeque<Event>,
    pub(super) stream: bool,
    pub(super) steps: Arc<Mutex<Vec<Step>>>,
    pub(super) root: Root,
    pub(super) provider: Option<Pin<Box<S>>>,
    pub(super) provider_factory: Option<ProviderFactory<S>>,
    pub(super) cancellation: Cancellation,
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
    pub(super) run: HistoryRun,
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
    #[cfg(test)]
    pub(super) fn new(
        provider: S,
        cancellation: Cancellation,
        input_tokens: u64,
        hooks: Hooks,
    ) -> Self {
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
            run: HistoryRun::default(),
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
            .map(crate::runtime::agent::reply::Run::owner);
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
            this.cancellation.register(cx.waker());
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

pub(crate) async fn execute_tool_batch(
    config: ToolBatchConfig,
    calls: Vec<Call>,
) -> Result<ToolBatch, Box<ToolBatchFailure>> {
    let ToolBatchConfig {
        interaction,
        relay,
        steps,
        dispatch,
        subagents,
        access,
        hooks,
        metadata,
        cancellation,
        max_concurrency,
        root,
        id_fn,
        trace,
    } = config;
    let task_count = calls.len().max(1);
    let slots = Arc::new(tokio::sync::Semaphore::new(max_concurrency.max(1)));
    let tasks = calls.into_iter().enumerate().map(|(index, call)| {
        let slots = slots.clone();
        let dispatch = Arc::clone(&dispatch);
        let steps = steps.clone();
        let interaction = interaction.clone();
        let subagents = subagents.clone();
        let access = access.clone();
        let hooks = hooks.clone();
        let metadata = metadata.clone();
        let cancellation = cancellation.clone();
        let root = root.clone();
        let relay = relay.clone();
        let trace = trace.clone();
        async move {
            let slot = crate::runtime::agent::reply::Slot::waiting(slots, Default::default());
            if let Err(error) = slot.acquire(&cancellation).await {
                return (
                    index,
                    call,
                    Err(Box::new(ToolBatchFailure { error, usage: None })),
                );
            }
            crate::runtime::agent::reply::feedback::with_slot(slot.clone(), async move {
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
                                let child =
                                    interaction.as_ref().map(|run| run.branch(slot.clone()));
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
                            dispatch,
                            access,
                            hooks,
                            metadata,
                            cancellation,
                            steps,
                            interaction,
                            id_fn,
                            trace,
                        },
                        call.clone(),
                    )
                    .await
                    .map(|messages| (messages, None))
                    .map_err(|error| Box::new(ToolBatchFailure { error, usage: None }))
                };
                (index, call, result)
            })
            .await
        }
    });
    let mut results = futures_util::stream::iter(tasks)
        .buffer_unordered(task_count)
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

pub(crate) async fn execute_subagent(
    subagents: Vec<SubAgent>,
    parent: Map<String, Value>,
    call: Call,
    restore: Option<Value>,
    cancellation: Cancellation,
    id_fn: Option<fn() -> i64>,
    observer: Option<crate::runtime::agent::subagent::Observer>,
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
    crate::runtime::agent::subagent::inherit_identity(&mut input, &parent)
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
    let error_value = json!({
        "code": error.code,
        "message": error.message,
        "details": error.details,
    });
    Message {
        role: Role::Tool,
        content: vec![Part {
            r#type: "text".into(),
            data: part_data(json!({"value": error_value.to_string()})),
        }],
        values: Map::from_iter([
            (String::from("error"), error_value),
            (String::from("call_id"), Value::String(call.call_id.clone())),
            (String::from("name"), Value::String(call.name.clone())),
        ]),
        metadata: Map::new(),
        usage: None,
    }
}

fn checkpoint_state(
    messages: &Messages,
    usage: Option<crate::model::token::Usage>,
    scene: (
        Option<&Input>,
        Option<&crate::runtime::agent::preparation::State>,
        Option<&crate::runtime::agent::completion::State>,
    ),
    result: &Result<Event, Error>,
    root: &Root,
    steps: &[Step],
    stream: bool,
) -> Value {
    let (input, preparation, completion) = scene;
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
    if let Some(preparation) = preparation {
        state["preparation"] =
            serde_json::to_value(preparation).expect("preparation serialization");
    }
    crate::runtime::agent::checkpoint::record_result(&mut state, result);
    if let Some(completion) = completion {
        state["completion"] = serde_json::to_value(completion).expect("completion serialization");
    }
    state
}

fn waiting_checkpoint_state(
    messages: &Messages,
    usage: Option<crate::model::token::Usage>,
    scene: (
        Option<&Input>,
        Option<&crate::runtime::agent::preparation::State>,
        Option<&crate::runtime::agent::completion::State>,
    ),
    root: &Root,
    steps: &[Step],
    stream: bool,
) -> Value {
    let mut state = checkpoint_state(
        messages,
        usage.clone(),
        scene,
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

pub(crate) fn checkpoint_pending_subagents(
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

pub(crate) fn checkpoint_messages(snapshot: &Value) -> Result<Messages, Error> {
    serde_json::from_value(snapshot["messages"].clone())
        .map_err(|_| Error::new("INVALID_STATE", "checkpoint has invalid Agent messages"))
}

pub(crate) fn checkpoint_steps(snapshot: &Value) -> Result<Vec<Step>, Error> {
    snapshot
        .get("steps")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| Error::new("INVALID_STATE", "checkpoint has invalid Tool steps"))
        .map(|steps| steps.unwrap_or_default())
}

pub(crate) fn checkpoint_query(snapshot: &Value) -> Result<Input, Error> {
    serde_json::from_value(snapshot["query"].clone()).map_err(|_| {
        Error::new(
            "INVALID_STATE",
            "checkpoint has invalid original Agent input",
        )
    })
}

pub(crate) fn is_waiting_snapshot(snapshot: &Value) -> bool {
    matches!(
        snapshot.get("status").and_then(Value::as_str),
        Some("running" | "waiting" | "paused")
    )
}

pub(crate) fn checkpoint_identity_matches(snapshot: &Value, metadata: &Map<String, Value>) -> bool {
    crate::runtime::agent::checkpoint::identity_matches(snapshot, metadata)
}

async fn execute_tool(execution: ToolExecution, call: Call) -> Result<Vec<Message>, Error> {
    let ToolExecution {
        dispatch,
        access,
        hooks,
        metadata,
        cancellation,
        steps,
        interaction,
        id_fn,
        trace,
    } = execution;
    let tool_span = trace.child("agent.tool");
    tool_span.property("tool", call.name.clone());
    tool_span.property_json("tool.call", &call);
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "tool call cancelled"));
    }
    let saved = {
        let mut steps = steps.lock().expect("Tool steps lock poisoned");
        if step::current(&mut steps, &call.call_id).is_none() {
            step::new_step(&mut steps, &call);
        }
        step::current(&mut steps, &call.call_id)
            .expect("selected Tool step")
            .clone()
    };
    if saved.stage.as_deref() == Some("complete") {
        return step_messages(&call, &saved);
    }
    let mut control_failed = false;
    let result = async {
        let effective = Box::pin(crate::runtime::agent::lifecycle::tool_prepare(
            &dispatch,
            &access,
            &hooks,
            &metadata,
            interaction.as_ref(),
            id_fn,
            &saved,
            &call,
            |effective| {
                let steps = &steps;
                let interaction = &interaction;
                let cancellation = &cancellation;
                async move {
                    retain_tool_stage(steps, interaction.as_ref(), &effective, None, cancellation)
                        .await
                }
            },
            |permission| {
                let steps = &steps;
                let interaction = &interaction;
                let cancellation = &cancellation;
                async move {
                    let call = permission.call()?;
                    if let Some(run) = interaction {
                        run.permission_stage(call, permission, cancellation).await
                    } else {
                        let mut steps = steps.lock().expect("Tool steps lock");
                        let owner = step::current(&mut steps, &call.call_id).ok_or_else(|| {
                            Error::new("INVALID_STATE", "permission owner missing")
                        })?;
                        owner.permission = Some(Box::new(permission));
                        Ok(())
                    }
                }
            },
            &cancellation,
        ))
        .await
        .map_err(|failure| match failure {
            crate::runtime::agent::lifecycle::Failure::Policy(error, _)
            | crate::runtime::agent::lifecycle::Failure::Tool(error) => error,
            crate::runtime::agent::lifecycle::Failure::Hook(error) => {
                control_failed = true;
                error
            }
        })?;
        if matches!(saved.stage.as_deref(), Some("before" | "call"))
            && let Some(run) = &interaction
        {
            run.continue_execution(&cancellation)
                .await
                .inspect_err(|_| {
                    control_failed = true;
                })?;
        }
        crate::runtime::agent::lifecycle::tool_call(
            &dispatch,
            &saved,
            effective,
            interaction.clone(),
            id_fn,
            |effective, business| {
                let steps = &steps;
                let interaction = &interaction;
                let cancellation = &cancellation;
                async move {
                    retain_tool_stage(
                        steps,
                        interaction.as_ref(),
                        &effective,
                        Some(&business),
                        cancellation,
                    )
                    .await
                }
            },
            &cancellation,
        )
        .await
    }
    .await;
    let (effective_call, business) = match result {
        Ok(result) => result,
        Err(error) => {
            tool_span.property_json("tool.error", &error);
            let mut steps = steps.lock().expect("Tool steps lock poisoned");
            step::stopped(&mut steps, &call.call_id, &error);
            if control_failed || matches!(error.code.as_str(), "CANCELLED" | "WAITING") {
                return Err(error);
            }
            let message = tool_error_message(&call, error);
            step::failed_message(&mut steps, &call.call_id, &message);
            return Ok(vec![message]);
        }
    };
    tool_span.property_json("tool.result", &business);
    step::called(
        &mut steps.lock().expect("Tool steps lock poisoned"),
        &effective_call,
        &business,
    );
    let evaluation = match dispatch
        .evaluate(&effective_call, &business, &cancellation)
        .await
    {
        Ok(evaluation) => evaluation,
        Err(error) => {
            step::stopped(
                &mut steps.lock().expect("Tool steps lock poisoned"),
                &call.call_id,
                &error,
            );
            return Err(error);
        }
    };
    let mut steps = steps.lock().expect("Tool steps lock poisoned");
    step::evaluated(&mut steps, &call.call_id, &evaluation);
    step_messages(
        &call,
        step::current(&mut steps, &call.call_id).expect("completed Tool step"),
    )
}

async fn retain_tool_stage(
    steps: &Arc<Mutex<Vec<Step>>>,
    interaction: Option<&crate::runtime::agent::reply::Run>,
    call: &Call,
    message: Option<&Message>,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    if let Some(run) = interaction {
        run.tool_stage(call.clone(), message.cloned(), cancellation)
            .await
    } else {
        let mut steps = steps.lock().expect("Tool steps lock poisoned");
        if let Some(message) = message {
            step::returned(&mut steps, call, message);
        } else {
            step::prepared(&mut steps, call);
        }
        Ok(())
    }
}

fn step_messages(call: &Call, step: &Step) -> Result<Vec<Message>, Error> {
    if step.status == "failed" {
        let error = step
            .error
            .clone()
            .ok_or_else(|| Error::new("INVALID_STATE", "failed Tool step has no error"))?;
        return Ok(vec![tool_error_message(call, error)]);
    }
    let business = step
        .message
        .clone()
        .ok_or_else(|| Error::new("INVALID_STATE", "Tool step lost its business result"))?;
    let mut messages = vec![tool_success_message(call, business)];
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

pub(crate) fn checkpoint_id(metadata: &Map<String, Value>) -> Result<i64, Error> {
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

pub(crate) fn estimate_output_tokens(message: &Message) -> u64 {
    crate::components::model::chat::estimate_tokens(
        message
            .content
            .iter()
            .map(crate::components::model::chat::part_bytes)
            .sum(),
    )
}
