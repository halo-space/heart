//! SubAgent execution, usage settlement and result conversion.

use super::*;

#[derive(Debug)]
#[doc(hidden)]
pub struct Failure {
    pub error: Error,
    pub scene: Option<Value>,
    pub usage: Option<Usage>,
}

impl Failure {
    pub(crate) fn new(error: Error) -> Box<Self> {
        Box::new(Self {
            error,
            scene: None,
            usage: None,
        })
    }
}

/// Shared run/session ownership is not a model-selectable child argument.
pub(crate) fn inherit_identity(
    input: &mut Input,
    parent: &serde_json::Map<String, Value>,
) -> Result<(), Error> {
    for key in ["chat_id", "session_id", "user_id", "tenant_id"] {
        if let Some(value) = parent.get(key) {
            if input
                .metadata
                .get(key)
                .is_some_and(|supplied| supplied != value)
            {
                return Err(Error::new(
                    "CONFLICT",
                    "SubAgent input belongs to a different run or session",
                ));
            }
            input.metadata.insert(key.to_owned(), value.clone());
        }
    }
    if let Some(value) = parent.get("agent_id") {
        input
            .metadata
            .entry("agent_id")
            .or_insert_with(|| value.clone());
    }
    Ok(())
}

fn additional_usage(
    scene: Option<&Value>,
    previous: Option<&Value>,
    observed: Option<Usage>,
) -> Result<Option<Usage>, Error> {
    // A cached terminal child does not capture a new scene or make another
    // model call. Its replayed Complete is delivery, not new consumption.
    if scene.is_none()
        && previous.is_some_and(|previous| {
            matches!(
                previous["status"].as_str(),
                Some("succeeded" | "failed" | "cancelled")
            )
        })
    {
        return Ok(None);
    }
    let read = |scene: Option<&Value>| -> Result<Option<Usage>, Error> {
        scene
            .and_then(|scene| scene.get("usage"))
            .map(|usage| {
                serde_json::from_value(usage.clone())
                    .map_err(|_| Error::new("INVALID_STATE", "SubAgent scene has invalid usage"))
            })
            .transpose()
            .map(Option::flatten)
    };
    let Some(current) = read(scene)? else {
        return Ok(observed);
    };
    usage_since(Some(current), read(previous)?)
}

fn usage_since(current: Option<Usage>, previous: Option<Usage>) -> Result<Option<Usage>, Error> {
    if current == previous {
        return Ok(None);
    }
    let Some(current) = current else {
        return Ok(None);
    };
    let previous = previous.unwrap_or_default();
    let subtract = |current: Option<u64>, previous: Option<u64>| -> Result<Option<u64>, Error> {
        current
            .map(|current| {
                current.checked_sub(previous.unwrap_or(0)).ok_or_else(|| {
                    Error::new("INVALID_STATE", "SubAgent cumulative usage decreased")
                })
            })
            .transpose()
    };
    Ok(Some(Usage {
        input: subtract(current.input, previous.input)?,
        output: subtract(current.output, previous.output)?,
        total: subtract(current.total, previous.total)?,
        input_details: current
            .input_details
            .map(|v| {
                subtract(v.cached, previous.input_details.and_then(|v| v.cached))
                    .map(|cached| InputDetails { cached })
            })
            .transpose()?,
        output_details: current
            .output_details
            .map(|v| {
                subtract(v.think, previous.output_details.and_then(|v| v.think))
                    .map(|think| OutputDetails { think })
            })
            .transpose()?,
        details: Default::default(),
    }))
}

fn failure(
    error: Error,
    scope: &crate::runtime::agent::scope::Scope,
    previous: Option<&Value>,
    observed: Option<Usage>,
) -> Box<Failure> {
    let scene = scope.latest();
    match additional_usage(scene.as_ref(), previous, observed.clone()) {
        Ok(usage) => Box::new(Failure {
            error,
            scene,
            usage,
        }),
        Err(error) => Box::new(Failure {
            error,
            scene,
            usage: observed,
        }),
    }
}

pub(crate) async fn run(
    subagent: &SubAgent,
    input: Input,
    call: &Call,
    previous: Option<Value>,
    cancellation: Cancellation,
    id_fn: Option<fn() -> i64>,
    observer: Option<Observer>,
) -> Result<Progress, Box<Failure>> {
    // A descendant cannot enable progress handoff that its enclosing runtime
    // does not implement. Independent parents start the supported Agent path;
    // delegated Agent scopes inherit it, while Workflow scopes remain gated.
    let feedback = observer.is_some()
        && (!crate::runtime::agent::scope::active() || crate::runtime::agent::scope::feedback());
    let scope = crate::runtime::agent::scope::Scope::new(previous.clone())
        .with_id_fn(id_fn)
        .with_feedback(feedback)
        .with_interaction(
            observer
                .as_ref()
                .and_then(|observer| observer.interaction.clone()),
        );
    let view_input = serde_json::to_value(&input).map_err(|error| {
        failure(
            Error::new("SERIALIZATION", error.to_string()),
            &scope,
            previous.as_ref(),
            None,
        )
    })?;
    let view = View::agent(subagent.name.clone(), view_input, previous.clone());
    let mut stream = crate::runtime::agent::scope::run(
        scope.clone(),
        subagent.run_with_view(input, cancellation, view),
    )
    .await
    .map_err(|error| failure(error, &scope, previous.as_ref(), None))?;
    let mut final_message = None;
    let mut observed = None;
    let mut cancelled = false;
    let mut published = None;
    let mut published_scene: Option<Arc<Value>> = None;
    enum Update {
        Event(Box<Result<Event, Error>>),
        Scene(Arc<Value>),
    }
    loop {
        let item = crate::runtime::agent::scope::run(
            scope.clone(),
            futures_util::future::poll_fn(|cx| match stream.as_mut().poll_next(cx) {
                Poll::Ready(item) => Poll::Ready(item.map(|event| Update::Event(Box::new(event)))),
                Poll::Pending => {
                    let latest = scope.latest_shared();
                    if observer.is_some()
                        && let Some(scene) = latest
                        && published_scene
                            .as_ref()
                            .is_none_or(|old| !Arc::ptr_eq(old, &scene))
                        && crate::runtime::agent::wait::has_input(&scene)
                    {
                        Poll::Ready(Some(Update::Scene(scene)))
                    } else {
                        Poll::Pending
                    }
                }
            }),
        )
        .await;
        let Some(item) = item else { break };
        let item = match item {
            Update::Scene(scene) => {
                let current = additional_usage(Some(&scene), previous.as_ref(), observed.clone())
                    .map_err(Failure::new)?;
                let usage =
                    usage_since(current.clone(), published.clone()).map_err(Failure::new)?;
                published = current;
                let feedback = feedback_changed(&scene, published_scene.as_deref());
                published_scene = Some(scene.clone());
                observer
                    .as_ref()
                    .expect("progress has an observer")
                    .emit(None, Some(scene), usage, None, feedback)
                    .await;
                continue;
            }
            Update::Event(item) => *item,
        };
        match item {
            Ok(event) => {
                cancelled |= matches!(&event, Event::Complete { finish_reason: Some(reason), .. } if reason == "cancelled");
                if let Event::Complete { message, usage, .. } = &event
                    && message.role == Role::Assistant
                {
                    observed = crate::model::token::merge(
                        observed,
                        usage.clone().or_else(|| message.usage.clone()),
                    );
                    final_message = Some(message.clone());
                }
                if let Some(observer) = &observer {
                    let scene = scope.latest_shared();
                    let current =
                        additional_usage(scene.as_deref(), previous.as_ref(), observed.clone())
                            .map_err(Failure::new)?;
                    let usage =
                        usage_since(current.clone(), published.clone()).map_err(Failure::new)?;
                    published = current;
                    let changed = match (&scene, &published_scene) {
                        (Some(current), Some(previous)) => !Arc::ptr_eq(current, previous),
                        (Some(_), None) => true,
                        _ => false,
                    };
                    let message = if changed
                        && scene.as_ref().is_some_and(|scene| {
                            matches!(scene["status"].as_str(), Some("succeeded" | "cancelled"))
                        }) {
                        final_message
                            .clone()
                            .map(|message| result_message(message, call))
                    } else {
                        None
                    };
                    let feedback = changed
                        && scene.as_deref().is_some_and(|scene| {
                            feedback_changed(scene, published_scene.as_deref())
                        });
                    published_scene = scene.clone();
                    // Only real scene commits update the owner; text deltas
                    // do not copy the whole unchanged Agent State again.
                    observer
                        .emit(
                            Some(event),
                            scene.filter(|_| changed),
                            usage,
                            message,
                            feedback,
                        )
                        .await;
                }
            }
            Err(error) => {
                let mut failure = failure(error, &scope, previous.as_ref(), observed);
                failure.usage = usage_since(failure.usage, published).map_err(Failure::new)?;
                return Err(failure);
            }
        }
    }
    let mut message = match final_message {
        Some(message) => message,
        None => {
            let mut failure = failure(
                Error::new(
                    "INVALID_RESPONSE",
                    "SubAgent stream ended without Assistant Complete",
                ),
                &scope,
                previous.as_ref(),
                observed.clone(),
            );
            failure.usage = usage_since(failure.usage, published).map_err(Failure::new)?;
            return Err(failure);
        }
    };
    let scene = scope.latest();
    let usage = usage_since(
        additional_usage(scene.as_ref(), previous.as_ref(), observed).map_err(Failure::new)?,
        published,
    )
    .map_err(Failure::new)?;
    let final_message = message.clone();
    message = result_message(message, call);
    Ok(Progress {
        cancelled,
        message,
        final_message,
        usage,
        scene: scene.or(previous),
    })
}

fn feedback_changed(scene: &Value, previous: Option<&Value>) -> bool {
    let before = previous
        .map(crate::runtime::agent::wait::input_progress)
        .unwrap_or_default();
    crate::runtime::agent::wait::input_progress(scene)
        .iter()
        .any(|(id, count)| *count > before.get(id).copied().unwrap_or(0))
}

pub(crate) fn result_message(mut message: Message, call: &Call) -> Message {
    message.role = Role::Function;
    // A child Agent's private reasoning belongs to its own model transcript;
    // it is not part of the function result returned to the parent model.
    message.content.retain(|part| part.r#type != "think");
    crate::runtime::observation::clear_source(&mut message);
    message
        .metadata
        .insert("call_id".into(), Value::String(call.call_id.clone()));
    message
        .metadata
        .insert("name".into(), Value::String(call.name.clone()));
    message
}
