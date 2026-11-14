//! Runtime adapters from typed Agent/Node implementations to the workflow
//! executor's erased value boundary.

use super::*;

#[doc(hidden)]
pub type BoxFuture = Pin<Box<dyn Future<Output = Result<Signal<Value>, Error>> + Send + 'static>>;
#[doc(hidden)]
pub type RecoveryFuture<'a> = Pin<
    Box<
        dyn Future<
                Output = Result<
                    (Signal<Value>, Vec<Step>),
                    crate::runtime::workflow::tool::RecoveryError,
                >,
            > + Send
            + 'a,
    >,
>;
#[doc(hidden)]
pub type AgentFuture = Pin<Box<dyn Future<Output = AgentRun> + Send + 'static>>;

#[doc(hidden)]
pub struct AgentRun {
    pub result: Result<Messages, Error>,
    pub messages: Messages,
    pub steps: Vec<Step>,
    pub usage: Option<Usage>,
    pub scene: Option<Value>,
}

#[doc(hidden)]
pub trait ErasedAgent: Send + Sync + 'static {
    fn call(
        &self,
        input: agent::Input,
        cancellation: Cancellation,
        restore: Option<Value>,
        view: View,
    ) -> AgentFuture;
}

pub(crate) struct Agent<A> {
    pub(super) agent: Arc<A>,
}

impl<A: crate::components::agent::Agent + 'static> ErasedAgent for Agent<A> {
    fn call(
        &self,
        input: agent::Input,
        cancellation: Cancellation,
        restore: Option<Value>,
        view: View,
    ) -> AgentFuture {
        let agent = Arc::clone(&self.agent);
        Box::pin(async move {
            let interaction =
                crate::runtime::workflow::events::interaction().map(|run| run.child());
            if let Some(scene) = restore.as_ref() {
                let prepared = if input.metadata.contains_key("reply_id") {
                    crate::runtime::agent::reply::prepare_restore(scene, &input)
                } else {
                    crate::runtime::agent::reply::prepare_pending(scene)
                };
                let restored = match prepared {
                    Ok(Some(restored)) => Some(restored),
                    Ok(None) => None,
                    Err(error) => {
                        return AgentRun {
                            result: Err(error),
                            messages: Messages::default(),
                            steps: Vec::new(),
                            usage: None,
                            scene: None,
                        };
                    }
                };
                if let Some(restored) = restored {
                    interaction
                        .as_ref()
                        .expect("embedded Agent reply interaction")
                        .restore_approvals(restored);
                }
            }
            let scope = crate::runtime::agent::scope::Scope::new(restore)
                .with_feedback(crate::runtime::workflow::events::has_parent())
                .with_interaction(interaction);
            let observed = scope.clone();
            let mut run = crate::runtime::agent::scope::run(scope.clone(), async move {
                let mut messages = Messages::default();
                let mut steps = Vec::new();
                let mut usage = None;
                let stream = match agent
                    .run_with_view(input, true, cancellation.clone(), &view)
                    .await
                {
                    Ok(stream) => stream,
                    Err(error) => {
                        return AgentRun {
                            result: Err(error),
                            messages,
                            steps,
                            usage,
                            scene: None,
                        };
                    }
                };
                futures_util::pin_mut!(stream);
                let mut final_assistant = None;
                let mut cancelled_run = false;
                let mut published: Option<Arc<Value>> = None;
                enum Update {
                    Event(Box<Result<Event, Error>>),
                    Scene,
                }
                loop {
                    let update = std::future::poll_fn(|cx| {
                        match stream.as_mut().poll_next(cx) {
                            std::task::Poll::Ready(event) => std::task::Poll::Ready(
                                event.map(|event| Update::Event(Box::new(event))),
                            ),
                            std::task::Poll::Pending => {
                                if crate::runtime::workflow::events::has_parent()
                                    && let Some(scene) = observed.latest_shared()
                                    && published.as_ref().is_none_or(|old| !Arc::ptr_eq(old, &scene))
                                    && crate::runtime::agent::wait::has_input(&scene)
                                {
                                    published = Some(scene);
                                    std::task::Poll::Ready(Some(Update::Scene))
                                } else {
                                    std::task::Poll::Pending
                                }
                            }
                        }
                    }).await;
                    let Some(update) = update else { break };
                    let event = match update {
                        Update::Scene => {
                            if let Err(error) = crate::runtime::workflow::events::progress(
                                agent_attempt(&observed, &steps, &messages, usage.clone()), None,
                            ).await {
                                return AgentRun { result: Err(error), messages, steps, usage, scene: None };
                            }
                            continue;
                        }
                        Update::Event(event) => {
                            published = observed.latest_shared();
                            *event
                        }
                    };
                    match event {
                        Ok(Event::Delta { message }) => {
                            if let Err(error) = crate::runtime::workflow::events::progress(
                                crate::runtime::workflow::node::Attempt {
                                    idx: crate::runtime::workflow::events::attempt_index(),
                                    ..Default::default()
                                },
                                Some(Event::Delta { message }),
                            )
                            .await
                            {
                                return AgentRun {
                                    result: Err(error),
                                    messages,
                                    steps,
                                    usage,
                                    scene: None,
                                };
                            }
                        }
                        Ok(Event::Complete {
                            mut message,
                            usage: call_usage,
                            finish_reason,
                        }) => {
                            let call_usage = call_usage.or_else(|| {
                                if message.role == Role::Assistant {
                                    message.usage.clone()
                                } else {
                                    None
                                }
                            });
                            if call_usage.is_some() {
                                message.usage = call_usage.clone();
                            }
                            let notification = Event::Complete {
                                message: crate::runtime::workflow::events::project_message(&message),
                                usage: call_usage.clone(),
                                finish_reason: finish_reason.clone(),
                            };
                            if message.metadata.contains_key("reply_id") {
                                let attempt =
                                    agent_attempt(&observed, &steps, &messages, usage.clone());
                                let valid = message
                                    .metadata
                                    .get("reply_id")
                                    .and_then(Value::as_i64)
                                    .zip(attempt.agent.as_ref())
                                    .is_some_and(|(id, scene)| {
                                        crate::runtime::agent::wait::question(scene, id)
                                            .unwrap_or(false)
                                    });
                                if !valid {
                                    return AgentRun {
                                        result: Err(Error::new(
                                            "INVALID_STATE",
                                            "embedded question has no actual waiting Step",
                                        )),
                                        messages,
                                        steps,
                                        usage,
                                        scene: None,
                                    };
                                }
                                if !crate::runtime::workflow::events::has_parent() {
                                    return AgentRun {
                                        result: Err(Error::new(
                                            "WAITING",
                                            "embedded feedback requires its owning Workflow continuation",
                                        )),
                                        messages,
                                        steps,
                                        usage,
                                        scene: None,
                                    };
                                }
                                if let Err(error) =
                                    crate::runtime::workflow::events::progress(attempt, Some(notification)).await
                                {
                                    return AgentRun {
                                        result: Err(error),
                                        messages,
                                        steps,
                                        usage,
                                        scene: None,
                                    };
                                }
                                continue;
                            }
                            // The incoming Agent stream is already an observation
                            // projection. Keep its source on the forwarded copy,
                            // never in the Workflow's original message history.
                            crate::runtime::observation::clear_source(&mut message);
                            let cancelled = finish_reason.as_deref() == Some("cancelled");
                            let kind = match message.role {
                                Role::Assistant => "model",
                                Role::Tool => "tool",
                                Role::Function => "function",
                                _ => "agent",
                            };
                            let index = steps.len() + 1;
                            steps.push(Step {
                                id: format!("complete_{index}"),
                                kind: kind.into(),
                                status: if cancelled {
                                    super::super::node::status::step::Status::Cancelled
                                } else {
                                    super::super::node::status::step::Status::Succeeded
                                },
                                from: Vec::new(),
                                to: Vec::new(),
                                call_id: None,
                                attempts: vec![StepAttempt {
                                    status: if cancelled {
                                        super::super::node::status::attempt::Status::Cancelled
                                    } else {
                                        super::super::node::status::attempt::Status::Succeeded
                                    },
                                    created_time: None,
                                    finished_time: None,
                                    usage: call_usage.clone(),
                                    finish_reason,
                                    error: None,
                                }],
                                error: None,
                                ..Step::default()
                            });
                            usage = crate::components::model::token::merge(usage, call_usage);
                            if message.role == Role::Assistant {
                                final_assistant = Some(message.clone());
                            }
                            messages.push(message);
                            if let Err(error) = crate::runtime::workflow::events::progress(
                                agent_attempt(&observed, &steps, &messages, usage.clone()),
                                Some(notification),
                            )
                            .await
                            {
                                return AgentRun {
                                    result: Err(error),
                                    messages,
                                    steps,
                                    usage,
                                    scene: None,
                                };
                            }
                            if cancelled {
                                // A child invocation has its own Complete.
                                // Drain the owning Agent's closeout too, so
                                // its final scene and observations reach this
                                // Workflow before the invocation ends.
                                cancelled_run = true;
                                if observed.latest_shared().is_none() {
                                    // Opaque custom Agents expose no nested
                                    // native owner to finish. Their cancelled
                                    // Complete retains its terminal meaning.
                                    break;
                                }
                            }
                        }
                        Err(error) => {
                            return AgentRun {
                                result: Err(error),
                                messages,
                                steps,
                                usage,
                                scene: None,
                            };
                        }
                    }
                }
                AgentRun {
                    result: if cancelled_run {
                        Err(Error::new("CANCELLED", "agent run cancelled"))
                    } else {
                        final_assistant
                            .map(|message| Messages::new([message]))
                            .ok_or_else(|| {
                                Error::new(
                                    "INVALID_RESPONSE",
                                    "agent produced no final Assistant Complete",
                                )
                            })
                    },
                    messages,
                    steps,
                    usage,
                    scene: None,
                }
            })
            .await;
            run.scene = scope.latest();
            // A continuation only observes new calls; the captured scene owns
            // cumulative consumption for this same Agent execution. Do not
            // replace that total with the resumed segment's event usage.
            if let Some(value) = run.scene.as_ref().and_then(|scene| scene.get("usage")) {
                match serde_json::from_value(value.clone()) {
                    Ok(usage) => run.usage = usage,
                    Err(_) => {
                        run.result =
                            Err(Error::new("INVALID_STATE", "Agent scene has invalid usage"));
                    }
                }
            }
            run
        })
    }
}

fn agent_attempt(
    scope: &crate::runtime::agent::scope::Scope,
    steps: &[Step],
    messages: &Messages,
    usage: Option<Usage>,
) -> crate::runtime::workflow::node::Attempt {
    let scene = scope.latest();
    let usage = scene
        .as_ref()
        .and_then(|scene| scene.get("usage"))
        .and_then(|usage| serde_json::from_value(usage.clone()).ok())
        .unwrap_or(usage);
    crate::runtime::workflow::node::Attempt {
        idx: crate::runtime::workflow::events::attempt_index(),
        status: if scene
            .as_ref()
            .and_then(|scene| scene["waits"].as_array())
            .is_some_and(|waits| !waits.is_empty())
        {
            super::super::node::status::attempt::Status::Waiting
        } else {
            super::super::node::status::attempt::Status::Running
        },
        steps: steps.to_vec(),
        messages: messages.clone(),
        usage,
        agent: scene,
        ..Default::default()
    }
}

/// Internal heterogeneous execution boundary, not a second public Node API.
#[allow(dead_code)]
#[doc(hidden)]
pub trait ErasedNode: Send + Sync + 'static {
    fn call(&self, input: Value, cancellation: Cancellation) -> BoxFuture;

    fn call_with_view(&self, input: Value, cancellation: Cancellation, view: View) -> BoxFuture;

    fn validate_recovery(
        &self,
        _input: Value,
        _facts: &crate::runtime::workflow::node::Attempt,
    ) -> Result<(), Error> {
        Ok(())
    }

    /// Optional private recovery path for a node that can expose committed
    /// sub-call facts. Ordinary Nodes keep the normal single-call contract.
    fn call_with_recovery<'a>(
        &'a self,
        _input: Value,
        _cancellation: Cancellation,
        _committed: crate::runtime::workflow::node::Attempt,
    ) -> Option<RecoveryFuture<'a>> {
        None
    }
}

#[allow(dead_code)]
pub(crate) struct Node<N, I, O, E> {
    pub(super) node: Arc<N>,
    pub(super) input_schema: jsonschema::Validator,
    pub(super) output_schema: jsonschema::Validator,
    pub(super) marker: std::marker::PhantomData<fn(I, O, E)>,
}

impl<N, I, O, E> ErasedNode for Node<N, I, O, E>
where
    N: crate::components::node::Node<I, O, E> + Any,
    I: DeserializeOwned + Send + 'static,
    O: Serialize + Send + 'static,
    E: Into<Error> + Send + 'static,
{
    fn validate_recovery(
        &self,
        input: Value,
        facts: &crate::runtime::workflow::node::Attempt,
    ) -> Result<(), Error> {
        if let Some(tool) = (self.node.as_ref() as &dyn Any)
            .downcast_ref::<crate::runtime::workflow::tool::ToolNode>()
        {
            let input = serde_json::from_value::<Messages>(input)
                .map_err(|_| Error::new("INVALID_STATE", "invalid ToolNode recovery input"))?;
            tool.validate_recovery(&input, facts).map(|_| ())?;
        }
        Ok(())
    }

    fn call(&self, input: Value, cancellation: Cancellation) -> BoxFuture {
        self.call_with_view(input, cancellation, View::default())
    }

    fn call_with_view(&self, input: Value, cancellation: Cancellation, view: View) -> BoxFuture {
        // Validate before deserialization and before calling user code; Serde
        // alone does not enforce JSON Schema ranges, lengths or patterns.
        if let Err(error) = crate::components::node::schema::validate(
            &self.input_schema,
            &input,
            "INVALID_ARGUMENTS",
            "node input violates its declared schema",
        ) {
            return Box::pin(async move { Err(error) });
        }
        let node = Arc::clone(&self.node);
        let output_schema = self.output_schema.clone();
        Box::pin(async move {
            let input: I = serde_json::from_value(input)
                .map_err(|error| Error::new("INVALID_ARGUMENTS", error.to_string()))?;
            match node
                .call_with_view(input, &view, &cancellation)
                .await
                .map_err(Into::into)?
            {
                Signal::Complete(output) => {
                    let value = serde_json::to_value(output)
                        .map_err(|error| Error::new("INVALID_OUTPUT", error.to_string()))?;
                    crate::components::node::schema::validate(
                        &output_schema,
                        &value,
                        "INVALID_OUTPUT",
                        "node output violates its declared schema",
                    )?;
                    Ok(Signal::Complete(value))
                }
                Signal::Wait(wait) => Ok(Signal::Wait(wait)),
            }
        })
    }

    fn call_with_recovery<'a>(
        &'a self,
        input: Value,
        cancellation: Cancellation,
        committed: crate::runtime::workflow::node::Attempt,
    ) -> Option<RecoveryFuture<'a>> {
        let tool = (self.node.as_ref() as &dyn Any)
            .downcast_ref::<crate::runtime::workflow::tool::ToolNode>()?;
        let input = match serde_json::from_value::<Messages>(input) {
            Ok(input) => input,
            Err(error) => {
                return Some(Box::pin(async move {
                    Err(crate::runtime::workflow::tool::RecoveryError {
                        completed: Messages::default(),
                        steps: committed.steps,
                        error: Error::new("INVALID_ARGUMENTS", error.to_string()),
                    })
                }));
            }
        };
        let future = tool.call_with_recovery(input, cancellation, committed);
        Some(Box::pin(async move {
            match future.await {
                Ok((Signal::Complete(messages), steps)) => Ok((
                    Signal::Complete(serde_json::to_value(messages).map_err(|error| {
                        crate::runtime::workflow::tool::RecoveryError {
                            completed: Messages::default(),
                            steps: steps.clone(),
                            error: Error::new("INVALID_OUTPUT", error.to_string()),
                        }
                    })?),
                    steps,
                )),
                Ok((Signal::Wait(wait), steps)) => Ok((Signal::Wait(wait), steps)),
                Err(failure) => Err(failure),
            }
        }))
    }
}
