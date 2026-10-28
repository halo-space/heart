//! Agent-specific binding to the common Core phase driver. The passed Step is
//! a copy of the actual owner facts, not a new store or component Context.
use std::future::Future;

use crate::core::lifecycle::{Methods, Phase, advance};
use crate::runtime::agent::{node::Step, reply};
use crate::{Cancellation, Error, Message, tool::Call};

use super::{
    access::{self, Access},
    hooks::Hooks,
};
use crate::core::dispatch::Dispatch;
use serde_json::{Map, Value};

/// Close out observed unfinished Model output. Usage is accounted separately
/// from success: no Complete, Tool dispatch or prompt entry is synthesized.
pub(super) fn model_unfinished(
    steps: &mut [Step],
    partial: &Message,
) -> Option<crate::model::token::Usage> {
    let step = crate::runtime::agent::model_steps::pending_input(steps)?;
    if step.stage.as_deref() != Some("call")
        || step.attempts.is_empty()
        || (partial.content.is_empty() && partial.values.is_empty() && partial.usage.is_none())
    {
        return None;
    }
    let request = step.request.as_ref()?;
    let mut usage = partial.usage.clone().unwrap_or_default();
    usage
        .input
        .get_or_insert_with(|| super::chat::estimate_input_tokens(request));
    usage
        .output
        .get_or_insert_with(|| super::chat::estimate_output_tokens(partial));
    usage.total.get_or_insert_with(|| {
        usage
            .input
            .unwrap_or(0)
            .saturating_add(usage.output.unwrap_or(0))
    });
    crate::runtime::agent::model_steps::unfinished(steps, partial, &usage);
    Some(usage)
}

/// Bind actual Model input to the same method driver used for all Hooks.
/// Only acknowledged transformations enter the actual owning Step.
pub(super) async fn model_prepare(
    hooks: &Hooks,
    mut saved: Step,
    mut commit: impl FnMut(Step) -> Result<(), Error>,
    cancellation: &Cancellation,
) -> Result<crate::model::chat::Request, Error> {
    let request = saved
        .request
        .as_deref()
        .cloned()
        .ok_or_else(|| Error::new("INVALID_STATE", "missing Model input"))?;
    if saved.stage.as_deref() == Some("call") {
        return Ok(request);
    }
    let result = Box::pin(hooks.apply_from(
        super::hooks::Stage::Model,
        true,
        request,
        saved.idx.unwrap_or(0),
        |idx, request: crate::model::chat::Request| {
            request.validate()?;
            saved.idx = Some(idx);
            saved.request = Some(Box::new(request));
            commit(saved.clone())
        },
        cancellation,
    ))
    .await;
    match result {
        Ok(request) => {
            request.validate()?;
            saved.request = Some(Box::new(request.clone()));
            saved.stage = Some("call".into());
            saved.status = "running".into();
            saved.error = None;
            commit(saved)?;
            Ok(request)
        }
        Err(error) => {
            crate::runtime::agent::model_steps::stopped(std::slice::from_mut(&mut saved), &error);
            commit(saved)?;
            Err(error)
        }
    }
}

/// Keep component/business failures separate from permission and framework
/// control failures. Callers retain their own result presentation and State.
#[derive(Debug)]
pub(super) enum Failure {
    Policy(Error, Box<Call>),
    Hook(Error),
    Tool(Error),
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        Self::Hook(error)
    }
}

/// Shared by Chat/ReAct and their planned/delegated executions. Only native
/// successful facts are supplied/acknowledged; this binding owns no State,
/// storage, slots, retry loop or public Context.
#[allow(clippy::too_many_arguments)]
pub(super) async fn tool_prepare<K, KF, P, PF>(
    dispatch: &Dispatch,
    access: &Access,
    hooks: &Hooks,
    metadata: &Map<String, Value>,
    interaction: Option<&reply::Run>,
    id_fn: Option<fn() -> i64>,
    saved: &Step,
    call: &Call,
    mut commit: K,
    mut permission_commit: P,
    cancellation: &Cancellation,
) -> Result<Call, Failure>
where
    K: FnMut(Call) -> KF,
    KF: Future<Output = Result<(), Error>>,
    P: FnMut(crate::runtime::agent::permission::State) -> PF,
    PF: Future<Output = Result<(), Error>>,
{
    let phase = match saved.stage.as_deref() {
        Some("before") => Phase::Before,
        Some("call") => Phase::Call,
        Some("after" | "eval") => {
            return saved.call.clone().ok_or_else(|| {
                Failure::Hook(Error::new("INVALID_STATE", "Tool step has no actual Call"))
            });
        }
        _ => {
            return Err(Failure::Hook(Error::new(
                "INVALID_STATE",
                "invalid active Tool phase",
            )));
        }
    };
    let approval = reply::consume_restore(interaction, saved, cancellation)
        .await
        .map_err(|error| Failure::Policy(error, Box::new(call.clone())))?;
    dispatch.validate(call).map_err(Failure::Tool)?;
    let approval = approval.as_ref();
    authorize(
        access,
        hooks,
        metadata,
        interaction,
        id_fn,
        approval,
        call,
        saved.permission.as_deref().cloned(),
        &mut permission_commit,
        cancellation,
    )
    .await?;
    let effective = crate::core::lifecycle::prepare(
        phase,
        call.clone(),
        |call| async {
            dispatch
                .prepare_call(call, cancellation)
                .await
                .map_err(Failure::Tool)
        },
        |effective| {
            let acknowledged = commit(effective);
            async move { acknowledged.await.map_err(Failure::Hook) }
        },
        cancellation,
    )
    .await?;
    if effective.arguments != call.arguments {
        authorize(
            access,
            hooks,
            metadata,
            interaction,
            id_fn,
            approval,
            &effective,
            None,
            &mut permission_commit,
            cancellation,
        )
        .await?;
    }
    Ok(effective)
}

#[allow(clippy::too_many_arguments)]
async fn authorize<P, PF>(
    access: &Access,
    hooks: &Hooks,
    metadata: &Map<String, Value>,
    interaction: Option<&reply::Run>,
    id_fn: Option<fn() -> i64>,
    approval: Option<&Call>,
    call: &Call,
    saved: Option<crate::runtime::agent::permission::State>,
    commit: &mut P,
    cancellation: &Cancellation,
) -> Result<(), Failure>
where
    P: FnMut(crate::runtime::agent::permission::State) -> PF,
    PF: Future<Output = Result<(), Error>>,
{
    if !access.is_configured() {
        return Ok(());
    }
    Box::pin(access::check_tool(
        access,
        hooks,
        metadata.clone(),
        call,
        cancellation,
        interaction,
        id_fn,
        approval,
        saved,
        commit,
    ))
    .await
    .map_err(|failure| match failure {
        access::Failure::Policy(error) => Failure::Policy(error, Box::new(call.clone())),
        access::Failure::Hook(error) => Failure::Hook(error),
    })
}

pub(super) async fn tool_call<K, KF>(
    dispatch: &Dispatch,
    saved: &Step,
    input: Call,
    interaction: Option<reply::Run>,
    id_fn: Option<fn() -> i64>,
    mut commit: K,
    cancellation: &Cancellation,
) -> Result<(Call, Message), Error>
where
    K: FnMut(Call, Message) -> KF,
    KF: Future<Output = Result<(), Error>>,
{
    let phase = match saved.stage.as_deref() {
        // Before has been prepared and acknowledged by the caller before
        // admission/permission; this range starts at the actual invocation.
        Some("before" | "call") => Phase::Call,
        Some("after") => Phase::After,
        Some("eval") => Phase::Complete,
        _ => return Err(Error::new("INVALID_STATE", "invalid active Tool phase")),
    };
    let output = (phase >= Phase::After)
        .then(|| saved.message.clone())
        .flatten();
    let (call, output) = advance(
        phase,
        Phase::Complete,
        input,
        output,
        Methods {
            before: |input| async move { Ok(input) },
            call: |call: Call| {
                let mut step = saved.clone();
                step.call = Some(call.clone());
                step.stage = Some("call".into());
                let interaction = interaction.clone();
                async move {
                    reply::feedback::during(
                        interaction,
                        call.clone(),
                        step,
                        id_fn,
                        dispatch.invoke(call, cancellation),
                    )
                    .await
                    .map(|(_, message)| message)
                }
            },
            after: |call: Call, message: Message| {
                let mut step = saved.clone();
                step.call = Some(call.clone());
                step.message = Some(message.clone());
                step.stage = Some("after".into());
                let interaction = interaction.clone();
                async move {
                    reply::feedback::during(
                        interaction,
                        call.clone(),
                        step,
                        id_fn,
                        dispatch.finish(call, message, cancellation),
                    )
                    .await
                    .map(|(_, message)| message)
                }
            },
            commit: move |phase, call: &Call, message: Option<&Message>| {
                let pending = if phase == Phase::After {
                    Some(commit(
                        call.clone(),
                        message.expect("successful call has a result").clone(),
                    ))
                } else {
                    // The existing owner loop commits the completed after result
                    // before evaluation; it must not receive a second phase ack.
                    None
                };
                async move {
                    match pending {
                        Some(pending) => pending.await,
                        None => Ok(()),
                    }
                }
            },
        },
        cancellation,
    )
    .await?;
    Ok((call, output.expect("complete lifecycle has a result")))
}

#[cfg(test)]
#[path = "../../../../tests/unit/agent_tool_prepare.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../tests/unit/agent_model_partial.rs"]
mod model_partial_tests;
