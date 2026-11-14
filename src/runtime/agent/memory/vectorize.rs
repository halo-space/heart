//! Private vector preparation for automatic long-memory writeback.
//!
//! The configured Embed component lives in the Agent/Harness composition, not
//! in a Memory backend. The backend still receives only the prepared vector
//! through `memory::Request`.

use std::{future::Future, pin::Pin, sync::Arc};

use crate::{
    Cancellation, Error, Message,
    cancellation::cancellable,
    model::embed::{Embed, Request},
};

type FutureResult = Pin<Box<dyn Future<Output = Result<Message, Error>> + Send + 'static>>;

trait Embedder: Send + Sync {
    fn embed(&self, request: Request) -> FutureResult;
}

mod adapter {
    use std::sync::Arc;

    pub(super) struct Embedder<E>(pub(super) Arc<E>);
}

impl<E: Embed + 'static> Embedder for adapter::Embedder<E> {
    fn embed(&self, request: Request) -> FutureResult {
        let component = self.0.clone();
        Box::pin(async move { component.embed(request).await })
    }
}

#[derive(Clone)]
pub(crate) struct Configured(Arc<dyn Embedder>);

impl Configured {
    pub(crate) fn new<E: Embed + 'static>(component: E) -> Self {
        Self(Arc::new(adapter::Embedder(Arc::new(component))))
    }

    pub(crate) async fn embed(
        &self,
        input: String,
        cancellation: &Cancellation,
    ) -> Result<Message, Error> {
        let request = Request {
            input,
            dimensions: None,
            options: Default::default(),
        };
        request.validate()?;
        cancellable(self.0.embed(request), cancellation).await
    }
}
