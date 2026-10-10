//! Harness integration of an explicitly configured Permission component.
use std::{future::Future, pin::Pin, sync::Arc};

use crate::{
    Cancellation, Error,
    permission::{Permission, Request},
};

type CheckFuture = Pin<Box<dyn Future<Output = Result<(), Error>>>>;

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
    ) -> Result<(), Error> {
        if let Some(policy) = &self.0 {
            super::hooks::cancellable(policy.check(request, cancellation.clone()), cancellation)
                .await
        } else {
            Ok(())
        }
    }
}
