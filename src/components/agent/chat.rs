use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use serde_json::Map;

use crate::components::agent::toolkit::Toolkit;
use crate::components::agent::{Agent as AgentContract, Error, Input};
use crate::components::event::Event;
use crate::components::message::{Message, Messages, Role};
use crate::components::model::chat::{Chat, Request};
use crate::components::model::content::Part;
use crate::harness::hooks::{Hooks, cancellable};
use crate::harness::memory::{History, Run};
use crate::runtime::cancellation::Cancellation;

pub struct Builder<M = ()> {
    model: Option<M>,
    system_prompt: Option<String>,
    toolkit: Option<Toolkit>,
    hooks: Hooks,
    history: History,
    max_concurrency: usize,
}

pub fn builder() -> Builder {
    Builder {
        model: None,
        system_prompt: None,
        toolkit: None,
        hooks: Hooks::default(),
        history: History::default(),
        max_concurrency: 100,
    }
}

impl<M> Builder<M> {
    pub fn model<N>(self, model: N) -> Builder<N> {
        Builder {
            model: Some(model),
            system_prompt: self.system_prompt,
            toolkit: self.toolkit,
            hooks: self.hooks,
            history: self.history,
            max_concurrency: self.max_concurrency,
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
        Ok(Agent {
            model,
            system_prompt: self.system_prompt,
            toolkit: self.toolkit,
            hooks: self.hooks,
            history: self.history,
            max_concurrency: self.max_concurrency,
        })
    }
}

pub struct Agent<M> {
    model: M,
    system_prompt: Option<String>,
    toolkit: Option<Toolkit>,
    hooks: Hooks,
    history: History,
    max_concurrency: usize,
}

type CompletionFuture = Pin<Box<dyn Future<Output = Result<Event, Error>>>>;

/// The concrete stream returned by this Chat Agent. Its items remain the
/// shared model Event; this type only owns cancellation and partial state.
pub struct Events<S> {
    provider: Pin<Box<S>>,
    cancellation: Cancellation,
    partial: Message,
    input_tokens: u64,
    finished: bool,
    hooks: Hooks,
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
        Self {
            provider: Box::pin(provider),
            cancellation,
            partial: Message::new(Role::Assistant),
            input_tokens,
            finished: false,
            hooks,
            pending: None,
            run: Run::default(),
            settled: false,
            closing: false,
        }
    }

    fn cancelled(&mut self) -> Event {
        let message = std::mem::replace(&mut self.partial, Message::new(Role::Assistant));
        let usage = {
            let output = estimate_output_tokens(&message);
            let mut usage = message.usage.clone().unwrap_or_default();
            usage.input.get_or_insert(self.input_tokens);
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
        if result.is_err() && !self.settled {
            // Keep already reported consumption, but never estimate an error
            // stream as though it had delivered a successful Complete.
            self.run.settle(self.partial.usage.clone());
            self.settled = true;
        }
        let run = self.run.clone();
        self.closing = true;
        self.pending = Some(Box::pin(async move { run.close(result).await }));
    }
}

impl<S> Stream for Events<S>
where
    S: Stream<Item = Result<Event, Error>>,
{
    type Item = Result<Event, Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
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
                    if !this.closing {
                        this.close(result);
                        return Pin::new(this).poll_next(cx);
                    }
                    this.finished = true;
                    this.pending = None;
                    Poll::Ready(Some(result))
                }
            };
        }
        match this.provider.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(Event::Delta { message }))) => {
                collect_delta(&mut this.partial, &message);
                Poll::Ready(Some(Ok(Event::Delta { message })))
            }
            Poll::Ready(Some(Ok(Event::Complete {
                mut message,
                usage,
                finish_reason,
            }))) => {
                let usage = usage.or_else(|| message.usage.clone());
                message.usage = usage.clone();
                this.run.settle(usage.clone());
                this.settled = true;
                this.partial = message.clone();
                let hooks = this.hooks.clone();
                let cancellation = this.cancellation.clone();
                this.pending = Some(Box::pin(async move {
                    let message = hooks.after_model(message, &cancellation).await?;
                    let message = hooks.after_execute(message, &cancellation).await?;
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
    M: Chat,
{
    type Stream = Events<M::Stream>;

    async fn run(&self, input: Input, cancellation: Cancellation) -> Result<Self::Stream, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "agent run cancelled"));
        }
        let original = input.clone();
        let input = self.hooks.before_execute(input, &cancellation).await?;
        self.history.check_identity(&original, &input)?;
        let prompt = self
            .hooks
            .prompt(self.system_prompt.clone(), &cancellation)
            .await?;
        let history = self
            .history
            .load(input.metadata.clone(), self.hooks.clone(), &cancellation)
            .await?;
        let run = self.history.begin(original, &cancellation).await?;
        let prepared = async {
            let mut request = Request {
                messages: initial_messages(input, prompt.as_deref(), history),
                stream: true,
                ..Request::default()
            };
            if let Some(toolkit) = &self.toolkit {
                request.tools = toolkit.prepared(&cancellation).await?.definitions();
            }
            let request = self.hooks.before_model(request, &cancellation).await?;
            let input_tokens = estimate_input_tokens(&request);
            let provider = cancellable(self.model.stream(request), &cancellation).await?;
            Ok::<_, Error>(Events::new(
                provider,
                cancellation,
                input_tokens,
                self.hooks.clone(),
            ))
        }
        .await;
        match prepared {
            Ok(mut events) => {
                events.run = run;
                Ok(events)
            }
            Err(error) => Err(run.fail(error).await),
        }
    }
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
            data: serde_json::json!({ "value": prompt }),
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
        .unwrap_or_else(|| part.data.to_string().len());
    part.r#type.len().saturating_add(body)
}

fn estimate_tokens(bytes: usize) -> u64 {
    u64::try_from(bytes.saturating_add(3) / 4).unwrap_or(u64::MAX)
}

pub(crate) fn user_message(input: Input) -> Message {
    let content = input
        .payload
        .into_iter()
        .map(|payload| {
            let mut data = match payload.data {
                serde_json::Value::Object(object) => object,
                value => Map::from_iter([(String::from("value"), value)]),
            };
            if let Some(mime_type) = payload.mime_type {
                data.insert("mime_type".into(), serde_json::Value::String(mime_type));
            }
            Part {
                r#type: payload.r#type,
                data: serde_json::Value::Object(data),
            }
        })
        .collect();
    Message {
        role: Role::User,
        content,
        values: Map::new(),
        metadata: input.metadata,
        usage: None,
    }
}

#[cfg(test)]
mod tests {
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
                data: json!({"value": "hel"}),
            });
            let mut complete = Message::new(Role::Assistant);
            complete.content.push(Part {
                r#type: "text".into(),
                data: json!({"value": "hello"}),
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
            metadata: Map::from_iter([(String::from("chat_id"), json!("chat_1"))]),
            payload: vec![crate::components::agent::Payload {
                r#type: "text".into(),
                data: json!("hello"),
                mime_type: Some("text/plain".into()),
            }],
        });
        assert_eq!(message.metadata["chat_id"], "chat_1");
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
        let mut stream =
            block_on(agent.run(Input::default(), Cancellation::new())).expect("provider stream");
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
        let error = block_on(agent.run(Input::default(), Cancellation::new())).unwrap_err();
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
        let error = block_on(agent.run(Input::default(), cancellation)).unwrap_err();
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
        let mut stream =
            block_on(agent.run(Input::default(), cancellation.clone())).expect("provider stream");
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
        let mut stream =
            block_on(agent.run(Input::default(), Cancellation::new())).expect("provider stream");
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
        let mut stream =
            block_on(agent.run(Input::default(), cancellation.clone())).expect("provider stream");
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
}
