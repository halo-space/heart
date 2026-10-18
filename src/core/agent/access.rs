//! Concrete permission-policy integration used by the Core Agent implementations.
use std::{future::Future, pin::Pin, sync::Arc};

use crate::{
    Cancellation, Error,
    permission::{Behavior, Decision, Permission, Request},
};

type CheckFuture = Pin<Box<dyn Future<Output = Result<Decision, Error>> + Send + 'static>>;

trait Entry: Send + Sync {
    fn check(&self, request: Request, cancellation: Cancellation) -> CheckFuture;
}

struct Typed<P>(Arc<P>);
impl<P: Permission + 'static> Entry for Typed<P> {
    fn check(&self, request: Request, cancellation: Cancellation) -> CheckFuture {
        let policy = self.0.clone();
        Box::pin(async move { policy.check(&request, &cancellation).await })
    }
}

#[derive(Clone, Default)]
pub(crate) struct Access(Option<Arc<dyn Entry>>);

impl Access {
    pub(crate) fn set<P: Permission + 'static>(&mut self, policy: P) {
        self.0 = Some(Arc::new(Typed(Arc::new(policy))));
    }

    pub(crate) fn is_configured(&self) -> bool {
        self.0.is_some()
    }

    pub(crate) async fn check(
        &self,
        request: crate::permission::Request,
        cancellation: &Cancellation,
    ) -> Result<Decision, Error> {
        if let Some(policy) = &self.0 {
            super::hooks::cancellable(policy.check(request, cancellation.clone()), cancellation)
                .await
        } else {
            Err(Error::new(
                "INVALID_STATE",
                "permission policy is not configured",
            ))
        }
    }
}

pub(crate) enum Failure {
    Policy(Error),
    Hook(Error),
}

/// Interpret a completed decision in the Agent's existing lifecycle. This
/// does not create a Message, consume a reply or grant/copy authorization.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn check_permission<K, KF>(
    access: &Access,
    hooks: &super::hooks::Hooks,
    mut metadata: serde_json::Map<String, serde_json::Value>,
    call: &crate::tool::Call,
    saved: Option<crate::runtime::agent::permission::State>,
    mut commit: K,
    cancellation: &Cancellation,
) -> Result<Decision, Failure>
where
    K: FnMut(crate::runtime::agent::permission::State) -> KF,
    KF: Future<Output = Result<(), Error>>,
{
    metadata.insert(
        "call".into(),
        serde_json::to_value(call).map_err(|_| {
            Failure::Hook(Error::new(
                "INVALID_ARGUMENTS",
                "cannot encode permission call",
            ))
        })?,
    );
    let request = Request {
        action: "call".into(),
        resource: call.name.clone(),
        metadata,
    };
    use crate::runtime::agent::permission::{State, identity};
    let mut state = match saved {
        Some(state) if state.stage != "complete" => {
            state.validate(call, hooks.len()).map_err(Failure::Hook)?;
            state
        }
        _ => State::new(request.clone()),
    };
    commit(state.clone()).await.map_err(Failure::Hook)?;
    if state.stage == "before" {
        let input = Box::pin(hooks.apply_from_async(
            super::hooks::Stage::Permission,
            true,
            state.input.clone(),
            state.idx,
            |idx, input: Request| {
                let pending = identity(&request, &input).map(|()| {
                    state.input = input;
                    state.idx = idx;
                    commit(state.clone())
                });
                async move { pending?.await }
            },
            cancellation,
        ))
        .await
        .map_err(Failure::Hook)?;
        state.input = input;
        state.stage = "check".into();
        state.idx = 0;
        commit(state.clone()).await.map_err(Failure::Hook)?;
    }
    // A saved decision is audit data, not a grant. Restoration always checks
    // current policy against the acknowledged before input, never after edits.
    let decision = access
        .check(state.input.clone(), cancellation)
        .await
        .map_err(Failure::Policy)?;
    if state.stage != "after" {
        state.stage = "after".into();
        state.idx = 0;
        state.output = Some(state.input.clone());
    }
    state.decision = Some(decision.clone());
    commit(state.clone()).await.map_err(Failure::Hook)?;
    let output = Box::pin(hooks.apply_from_async(
        super::hooks::Stage::Permission,
        false,
        state.output.clone().expect("after Request"),
        state.idx,
        |idx, output: Request| {
            let pending = identity(&state.input, &output).map(|()| {
                state.output = Some(output);
                state.idx = idx;
                commit(state.clone())
            });
            async move { pending?.await }
        },
        cancellation,
    ))
    .await
    .map_err(Failure::Hook)?;
    state.output = Some(output);
    state.stage = "complete".into();
    commit(state).await.map_err(Failure::Hook)?;
    Ok(decision)
}

/// Apply permission policy to an actual tool call and, when configured, route
/// an Ask decision through the Agent's existing reply interaction.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn check_tool<K, KF>(
    access: &Access,
    hooks: &super::hooks::Hooks,
    metadata: serde_json::Map<String, serde_json::Value>,
    call: &crate::tool::Call,
    cancellation: &Cancellation,
    interaction: Option<&crate::runtime::agent::reply::Run>,
    id_fn: Option<fn() -> i64>,
    approval: Option<&crate::tool::Call>,
    saved: Option<crate::runtime::agent::permission::State>,
    commit: K,
) -> Result<(), Failure>
where
    K: FnMut(crate::runtime::agent::permission::State) -> KF,
    KF: Future<Output = Result<(), Error>>,
{
    let decision = Box::pin(check_permission(
        access,
        hooks,
        metadata,
        call,
        saved,
        commit,
        cancellation,
    ))
    .await?;
    match decision.behavior {
        Behavior::Allow => Ok(()),
        Behavior::Deny => Err(Failure::Policy(Error::new(
            "PERMISSION_DENIED",
            decision
                .reason
                .unwrap_or_else(|| "operation denied by permission policy".into()),
        ))),
        Behavior::Ask => {
            let reason = decision
                .reason
                .unwrap_or_else(|| "operation requires user confirmation".into());
            if approval == Some(call) {
                return Ok(());
            }
            if let (Some(run), Some(id_fn)) = (interaction, id_fn) {
                run.ask(call, &reason, id_fn, cancellation)
                    .await
                    .map_err(Failure::Policy)
            } else {
                Err(Failure::Policy(Error::new("WAITING", reason)))
            }
        }
    }
}

pub(crate) fn require_policy(
    steps: &[crate::runtime::agent::node::Step],
    configured: bool,
) -> Result<(), Error> {
    if !configured
        && steps.iter().any(|step| {
            step.permission
                .as_ref()
                .is_some_and(|state| state.stage != "complete")
        })
    {
        Err(Error::new(
            "INVALID_STATE",
            "unfinished permission requires its configured policy",
        ))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_scene(scene: &serde_json::Value, configured: bool) -> Result<(), Error> {
    if configured {
        return Ok(());
    }
    let steps: Vec<crate::runtime::agent::node::Step> = scene
        .get("steps")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|_| Error::new("INVALID_STATE", "invalid permission owner"))?
        .unwrap_or_default();
    require_policy(&steps, configured)?;
    if let Some(nodes) = scene.get("nodes").and_then(serde_json::Value::as_object) {
        for node in nodes.values() {
            if let Some(attempt) = node["execution"]["attempts"]
                .as_array()
                .and_then(|attempts| attempts.last())
            {
                validate_scene(attempt, configured)?;
            }
        }
    }
    Ok(())
}
