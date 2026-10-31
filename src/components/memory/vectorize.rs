//! Private vector preparation for automatic long-memory writeback.
//!
//! The configured Embed component lives in the Agent/Harness composition, not
//! in a Memory backend. The backend still receives only the prepared vector
//! through `memory::Request`.

use std::{future::Future, pin::Pin, sync::Arc};

use crate::{
    Cancellation, Error, Message,
    model::embed::{Embed, Request},
};

type FutureResult = Pin<Box<dyn Future<Output = Result<Message, Error>> + Send + 'static>>;

trait Entry: Send + Sync {
    fn embed(&self, request: Request) -> FutureResult;
}

struct Typed<E>(Arc<E>);

impl<E: Embed + 'static> Entry for Typed<E> {
    fn embed(&self, request: Request) -> FutureResult {
        let component = self.0.clone();
        Box::pin(async move { component.embed(request).await })
    }
}

#[derive(Clone)]
pub(crate) struct Configured(Arc<dyn Entry>);

impl Configured {
    pub(crate) fn new<E: Embed + 'static>(component: E) -> Self {
        Self(Arc::new(Typed(Arc::new(component))))
    }

    pub(crate) async fn embed(
        &self,
        input: String,
        cancellation: &Cancellation,
    ) -> Result<Message, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "embedding cancelled"));
        }
        let request = Request {
            input,
            dimensions: None,
            options: Default::default(),
        };
        request.validate()?;
        let mut future = std::pin::pin!(self.0.embed(request));
        std::future::poll_fn(|context| {
            cancellation.register(context.waker());
            if cancellation.is_cancelled() {
                return std::task::Poll::Ready(Err(Error::new("CANCELLED", "embedding cancelled")));
            }
            let result = future.as_mut().poll(context);
            if cancellation.is_cancelled() {
                return std::task::Poll::Ready(Err(Error::new("CANCELLED", "embedding cancelled")));
            }
            result
        })
        .await
    }
}
