use std::future::Future;

use crate::{
    Cancellation, Error,
    permission::{Permission as Contract, Request},
};

/// Caller-defined permission policy; no implicit global allow, model call or
/// approval workflow. A rejected decision is a normal shared Error.
pub struct Permission<F> {
    check: F,
}

impl<F> Permission<F> {
    pub fn new(check: F) -> Self {
        Self { check }
    }
}

impl<F, Fut> Contract for Permission<F>
where
    F: Fn(Request, Cancellation) -> Fut + Send + Sync,
    Fut: Future<Output = Result<(), Error>>,
{
    async fn check(&self, request: &Request, cancellation: &Cancellation) -> Result<(), Error> {
        super::operation::check(cancellation)?;
        if request.action.trim().is_empty() || request.resource.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "permission action and resource are required",
            ));
        }
        super::operation::cancellable(
            (self.check)(request.clone(), cancellation.clone()),
            cancellation,
        )
        .await
    }
}
