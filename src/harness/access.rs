//! Harness integration of an explicitly configured Permission component.
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
pub(crate) async fn check_tool(
    access: &Access,
    hooks: &super::hooks::Hooks,
    mut metadata: serde_json::Map<String, serde_json::Value>,
    call: &crate::tool::Call,
    cancellation: &Cancellation,
) -> Result<(), Failure> {
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
    let request = hooks
        .before_permission(request, cancellation)
        .await
        .map_err(Failure::Hook)?;
    let decision = access
        .check(request.clone(), cancellation)
        .await
        .map_err(Failure::Policy)?;
    // Deny and ask are successful checks too, so normal after hooks unwind.
    hooks
        .after_permission(request, cancellation)
        .await
        .map_err(Failure::Hook)?;
    match decision.behavior {
        Behavior::Allow => Ok(()),
        Behavior::Deny => Err(Failure::Policy(Error::new(
            "PERMISSION_DENIED",
            decision
                .reason
                .unwrap_or_else(|| "operation denied by permission policy".into()),
        ))),
        Behavior::Ask => Err(Failure::Policy(Error::new(
            "WAITING",
            decision
                .reason
                .unwrap_or_else(|| "operation requires user confirmation".into()),
        ))),
    }
}
