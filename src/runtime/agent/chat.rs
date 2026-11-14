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

use crate::Cancellation;
use crate::cancellation::cancellable;
use crate::components::agent::{Error, Input};
use crate::components::agent_input::user_message;
use crate::components::evaluation::{Evaluation, Evaluator};
use crate::components::event::Event;
use crate::components::message::{Message, Messages, Role};
use crate::components::model::chat::{Chat, Request};
use crate::components::model::content::part::{Part, object as part_data};
use crate::components::tool::Call;
use crate::components::{mcp::Server, skill::Skill};
use crate::core::dispatch::{Dispatch, ToolInputs};
use crate::runtime::agent::access::Access;
use crate::runtime::agent::checkpoint::Backend;
use crate::runtime::agent::history::{Execute, History};
use crate::runtime::agent::hooks::Hooks;
use crate::runtime::agent::node::Step;
use crate::runtime::agent::step;
use crate::runtime::agent::subagent::SubAgent;
use crate::runtime::agent::subagent::{
    Failure as SubAgentFailure, Progress as SubAgentProgress, Root,
};
use crate::runtime::trace::fastrace::{Root as TraceRoot, SpanScope};

pub struct Builder<M = ()> {
    model: Option<M>,
    system_prompt: Option<String>,
    dispatch: Option<Dispatch>,
    subagents: Vec<SubAgent>,
    hooks: Hooks,
    access: Access,
    evaluator: Option<EvaluatorHandle>,
    checkpoint: Option<CheckpointSave>,
    history: History,
    #[doc(hidden)]
    pub max_concurrency: usize,
    id_fn: Option<fn() -> i64>,
}

pub fn builder() -> Builder {
    Builder {
        model: None,
        system_prompt: None,
        dispatch: None,
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
            dispatch: self.dispatch,
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

    /// Register application Tools directly on this Agent.
    pub fn tools<I>(mut self, tools: I) -> Self
    where
        I: ToolInputs,
    {
        let dispatch = self.dispatch.take().unwrap_or_default();
        self.dispatch = Some(dispatch.tools(tools));
        self
    }

    /// Register already connected and discovered MCP servers on this Agent.
    pub fn mcps<S>(mut self, servers: impl IntoIterator<Item = S>) -> Self
    where
        S: Server + 'static,
    {
        let dispatch = self.dispatch.take().unwrap_or_default();
        self.dispatch = Some(dispatch.mcps(servers));
        self
    }

    /// Register in-process Skills on this Agent.
    pub fn skills<S>(mut self, skills: impl IntoIterator<Item = S>) -> Result<Self, Error>
    where
        S: Skill + 'static,
    {
        let dispatch = self.dispatch.take().unwrap_or_default();
        self.dispatch = Some(dispatch.skills(skills)?);
        Ok(self)
    }

    /// Register the heterogeneous Agent set available to the built-in
    /// `agent_spawn` delegation path. SubAgents are not tool dispatch entries.
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
    pub fn memory<S, L>(mut self, config: crate::runtime::agent::memory::Config<S, L>) -> Self
    where
        S: crate::memory::short::Store + 'static,
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
            dispatch: self.dispatch,
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
    dispatch: Option<Dispatch>,
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

pub(crate) mod runner;
#[doc(hidden)]
pub use crate::components::model::chat::{estimate_input_tokens, fit_request};
pub use runner::Events;
pub(crate) use runner::{
    CheckpointSave, EvaluatorHandle, EventsConfig, ProviderFactory, ToolBatchConfig, checkpoint_id,
    checkpoint_identity_matches, checkpoint_messages, checkpoint_pending_subagents,
    checkpoint_query, checkpoint_steps, estimate_output_tokens, execute_tool_batch,
    initial_messages, is_waiting_snapshot, prepare_model,
};

impl<M> Agent<M> {
    pub fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }
}

impl<M> crate::components::agent::Agent for Agent<M>
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
        let trace = TraceRoot::start("agent.chat");
        // Record the public input at the Agent span boundary. Runtime State
        // remains the recovery truth; this is only the trace projection.
        trace.property_json("agent.input", &input);
        if let Some(chat_id) = input.metadata.get("chat_id").and_then(Value::as_i64) {
            trace.property("chat_id", chat_id.to_string());
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
        let mut preparation = None;
        let mut completion: Option<crate::runtime::agent::completion::State> = None;
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
            preparation = crate::runtime::agent::preparation::State::restore(&snapshot)?;
            completion = snapshot
                .get("completion")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|_| Error::new("INVALID_STATE", "invalid Agent completion"))?;
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
                && !crate::runtime::agent::reply::reply_duplicate(&snapshot, &input)?
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
            preparation = crate::runtime::agent::preparation::State::restore(&snapshot)?;
            completion = snapshot
                .get("completion")
                .filter(|value| !value.is_null())
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|_| Error::new("INVALID_STATE", "invalid Agent completion"))?;
            restored_steps = checkpoint_steps(&snapshot)?;
            restored_root = Root::restore(&snapshot)?;
        }
        if let Some(state) = &preparation {
            state.validate(
                self.hooks.len(),
                restored_query.as_ref().expect("restored original query"),
            )?;
        }
        crate::runtime::agent::model_steps::validate_owner(
            &restored_steps,
            completion.as_ref(),
            self.hooks.len(),
        )?;
        super::access::require_policy(&restored_steps, self.access.is_configured())?;
        let pending_model = crate::runtime::agent::model_steps::pending(&restored_steps).cloned();
        let processing_output = pending_model.is_some()
            || completion
                .as_ref()
                .is_some_and(crate::runtime::agent::completion::State::pending);
        let processing =
            processing_output || crate::runtime::agent::model_steps::continuing(&restored_steps);
        let restored_calls = if let Some(messages) = restored_messages.as_mut() {
            restored_root.resume_calls(&restored_steps, messages)?;
            step::resume_calls(&restored_steps, messages)?
        } else {
            Vec::new()
        };
        let original = restored_query.clone().unwrap_or_else(|| input.clone());
        let interaction = if embedded {
            Some(match crate::runtime::agent::scope::interaction() {
                Some(run) => run,
                None => self.replies.start(original.metadata.clone())?,
            })
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
        let discovered_dispatch = match &self.dispatch {
            Some(dispatch) => Some(dispatch.with_discovered_skills(&cancellation).await),
            None => None,
        };
        if !restoring {
            let system_prompt = discovered_dispatch
                .as_ref()
                .and_then(|result| result.as_ref().ok())
                .map_or_else(
                    || self.system_prompt.clone(),
                    |dispatch| dispatch.system_prompt(self.system_prompt.clone()),
                );
            preparation = Some(crate::runtime::agent::preparation::State::new(
                original.clone(),
                system_prompt,
            ));
        }
        let preparing = preparation.as_ref().is_some_and(|state| {
            state.stage != crate::runtime::agent::preparation::Stage::Complete
        });
        let mut run = Execute::default();
        // Preparation belongs to this run too. Retain its constructed input
        // before Dispatch preparation, model hooks or Provider dispatch stop it.
        let mut preparation_messages = restored_messages.clone().unwrap_or_default();
        let mut preparation_input_tokens = 0;
        let request_scene = Arc::new(Mutex::new(None));
        let model_steps = Arc::new(Mutex::new(restored_steps.clone()));
        let mut restore_validation_error = false;
        let prepared = async {
            if let Some(state) = preparation.as_mut() {
                Box::pin(super::preparation::advance(
                    state,
                    &original,
                    &mut preparation_messages,
                    &mut run,
                    &history,
                    &self.hooks,
                    &cancellation,
                ))
                .await?;
            } else {
                run = history.resume(original.clone(), &cancellation).await?;
            }
            let mut messages = preparation_messages.clone();
            if restored_waiting
                && !preparing
                && !processing
                && restored_pending_subagents.is_empty()
                && restored_calls.is_empty()
            {
                messages.push(user_message(input.clone()));
            }
            preparation_messages = messages.clone();
            let dispatch = match &discovered_dispatch {
                Some(Ok(dispatch)) => Some(dispatch.prepared(&cancellation).await?),
                Some(Err(error)) => return Err(error.clone()),
                None => None,
            };
            for step in restored_steps.iter().filter(|step| {
                step.kind == "tool"
                    && matches!(step.stage.as_deref(), Some("call" | "after" | "eval"))
            }) {
                let checked = dispatch
                    .as_ref()
                    .ok_or_else(|| {
                        Error::new("INVALID_STATE", "saved eval requires configured tools")
                    })
                    .and_then(|dispatch| {
                        dispatch.validate(step.call.as_ref().expect("validated saved Tool Call"))
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
            let tool_definitions = dispatch
                .as_ref()
                .map(Dispatch::tool_list)
                .unwrap_or_default();
            let mut tool_definitions = tool_definitions;
            if !self.subagents.is_empty() {
                tool_definitions.push(SubAgent::definition(&self.subagents));
            }
            let model = Arc::clone(&self.model);
            // A child-wait checkpoint already contains the exact child Call.
            // Do not invoke the parent model before restoring that hand-off.
            let (provider, input_tokens) = if restored_pending_subagents.is_empty()
                && restored_calls.is_empty()
                && !processing_output
            {
                let request = Request {
                    messages: messages.clone(),
                    stream,
                    tools: tool_definitions.clone(),
                    ..Request::default()
                };
                let request =
                    prepare_model(&self.hooks, &model_steps, request, &cancellation).await?;
                let request = fit_request(request, model.context_limit())?;
                let input_tokens = estimate_input_tokens(&request);
                preparation_input_tokens = input_tokens;
                *request_scene.lock().expect("Model request lock poisoned") =
                    Some(Box::new(request.clone()));
                crate::runtime::agent::model_steps::calling(
                    &mut model_steps.lock().expect("Model steps lock"),
                );
                let provider = match cancellable(model.stream(request), &cancellation).await {
                    Ok(provider) => provider,
                    Err(error) => {
                        trace.event_properties(
                            "model.error",
                            vec![
                                ("status".into(), "failed".into()),
                                ("error_code".into(), error.code.clone()),
                            ],
                        );
                        return Err(error);
                    }
                };
                (Some(provider), input_tokens)
            } else {
                (None, 0)
            };
            let hooks = self.hooks.clone();
            let factory_request = request_scene.clone();
            let factory: ProviderFactory<M::Stream> =
                Arc::new(move |messages, cancellation, input_tokens, steps| {
                    let model = Arc::clone(&model);
                    let hooks = hooks.clone();
                    let tools = tool_definitions.clone();
                    let request_scene = factory_request.clone();
                    Box::pin(async move {
                        let request = Request {
                            messages,
                            stream,
                            tools,
                            ..Request::default()
                        };
                        let request = prepare_model(&hooks, &steps, request, &cancellation).await?;
                        let request = fit_request(request, model.context_limit())?;
                        input_tokens.store(estimate_input_tokens(&request), Ordering::Relaxed);
                        *request_scene.lock().expect("Model request lock poisoned") =
                            Some(Box::new(request.clone()));
                        crate::runtime::agent::model_steps::calling(
                            &mut steps.lock().expect("Model steps lock"),
                        );
                        model.stream(request).await
                    })
                });
            let mut events = match provider {
                Some(provider) => Events::new_with_trace(
                    provider,
                    cancellation.clone(),
                    input_tokens,
                    self.hooks.clone(),
                    trace.clone(),
                ),
                None => Events::empty_with_trace(
                    cancellation.clone(),
                    self.hooks.clone(),
                    trace.clone(),
                ),
            }
            .configure_tools(EventsConfig {
                completion: completion.clone(),
                interaction: interaction.clone(),
                steps: model_steps.clone(),
                provider_factory: factory,
                dispatch,
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
                trace: trace.clone(),
            });
            events.stream = stream;
            events.request = request_scene.clone();
            if let Some(request) = events
                .request
                .lock()
                .expect("Model request lock poisoned")
                .as_ref()
                && let Some(model_span) = events.model_span.as_ref()
            {
                model_span.property_json("model.input", request.as_ref());
            }
            events.preparation = preparation.clone();
            events.waits = restored_waits.clone();
            if processing_output {
                events.partial = pending_model
                    .as_ref()
                    .and_then(|step| step.message.clone())
                    .or_else(|| completion.as_ref().map(|state| state.message.clone()))
                    .expect("validated output");
                events.settled = true;
                events.replayed_completion = true;
                events.input_tokens = restored_steps
                    .iter()
                    .rev()
                    .find(|step| step.kind == "model")
                    .and_then(|step| step.request.as_deref())
                    .map(estimate_input_tokens)
                    .unwrap_or(0);
                events.pending = Some(events.postprocess(pending_model.clone()));
            }
            if !restored_pending_subagents.is_empty() || !restored_calls.is_empty() {
                let calls = restored_pending_subagents
                    .values()
                    .map(|pending| (pending.index, pending.call.clone()))
                    .collect::<Vec<_>>();
                let mut calls = calls;
                calls.extend(restored_calls.clone());
                calls.sort_by_key(|(index, _)| *index);
                let calls = calls.into_iter().map(|(_, call)| call).collect::<Vec<_>>();
                let dispatch = events
                    .dispatch
                    .clone()
                    .unwrap_or_else(|| Arc::new(Dispatch::new()));
                events.provider = None;
                events.pending_tools = Some(Box::pin(execute_tool_batch(
                    ToolBatchConfig {
                        interaction: events.interaction.clone(),
                        relay: Some(events.child_events.clone()),
                        steps: events.steps.clone(),
                        dispatch,
                        subagents: events.subagents.clone(),
                        access: events.access.clone(),
                        hooks: events.hooks.clone(),
                        metadata: events.metadata.clone(),
                        cancellation: events.cancellation.clone(),
                        max_concurrency: events.max_concurrency,
                        id_fn,
                        root: events.root.clone(),
                        trace: events.trace.clone(),
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
                events.preparation = preparation;
                events.completion = Arc::new(Mutex::new(completion));
                events.checkpoint = checkpoint;
                events.checkpoint_id = checkpoint_id;
                events.input = Some(restored_query.unwrap_or(original));
                events.metadata = input.metadata;
                events.messages = preparation_messages;
                events.pending_subagents = restored_pending_subagents;
                events.root = restored_root;
                events.steps = model_steps;
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
                    // the same scene and terminal Chat.state as an in-stream error.
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
#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_runtime_agent_chat_tests.rs"]
mod tests;
