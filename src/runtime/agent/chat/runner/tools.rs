//! Chat Agent tool and SubAgent execution lifecycle.

use super::*;

#[doc(hidden)]
pub async fn execute_tool_batch(
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

#[doc(hidden)]
pub async fn execute_subagent(
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

pub(super) async fn execute_tool(
    execution: ToolExecution,
    call: Call,
) -> Result<Vec<Message>, Error> {
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

pub(super) async fn retain_tool_stage(
    steps: &Arc<Mutex<Vec<Step>>>,
    interaction: Option<&crate::runtime::agent::reply::Handle>,
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

pub(super) fn step_messages(call: &Call, step: &Step) -> Result<Vec<Message>, Error> {
    if step.status == crate::runtime::agent::node::status::step::Status::Failed {
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

pub(super) fn decode_call(part: &Part) -> Result<Call, Error> {
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

pub(super) fn tool_success_message(call: &Call, mut message: Message) -> Message {
    crate::runtime::observation::clear_source(&mut message);
    message
        .metadata
        .insert("call_id".into(), Value::String(call.call_id.clone()));
    message
        .metadata
        .insert("name".into(), Value::String(call.name.clone()));
    message
}

pub(super) fn tool_error_message(call: &Call, error: Error) -> Message {
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
