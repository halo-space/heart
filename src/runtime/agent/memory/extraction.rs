//! Runtime wrapper for a configured long-memory extractor.

use crate::components::memory::Item;
use crate::components::memory::extraction::Extractor;
use crate::{Cancellation, Error, Messages};
use std::{future::Future, pin::Pin, sync::Arc};
type ExtractFuture = Pin<Box<dyn Future<Output = Result<Messages, Error>> + Send + 'static>>;
type MergeFuture = Pin<Box<dyn Future<Output = Result<String, Error>> + Send + 'static>>;
trait Extraction: Send + Sync {
    fn extract(&self, input: Messages, cancellation: Cancellation) -> ExtractFuture;
    fn merge(&self, current: Item, content: String, cancellation: Cancellation) -> MergeFuture;
}
mod adapter {
    use std::sync::Arc;

    pub(super) struct Extraction<E>(pub(super) Arc<E>);
}

impl<E: Extractor + 'static> Extraction for adapter::Extraction<E> {
    fn extract(&self, input: Messages, cancellation: Cancellation) -> ExtractFuture {
        let component = self.0.clone();
        Box::pin(async move { component.extract(&input, &cancellation).await })
    }
    fn merge(&self, current: Item, content: String, cancellation: Cancellation) -> MergeFuture {
        let component = self.0.clone();
        Box::pin(async move { component.merge(&current, &content, &cancellation).await })
    }
}
#[derive(Clone)]
pub(crate) struct Configured {
    component: Arc<dyn Extraction>,
}
impl Configured {
    pub(crate) fn new<E>(component: E) -> Self
    where
        E: Extractor + 'static,
    {
        Self {
            component: Arc::new(adapter::Extraction(Arc::new(component))),
        }
    }
    pub(crate) async fn extract(
        &self,
        input: Messages,
        cancellation: &Cancellation,
    ) -> Result<Messages, Error> {
        self.component.extract(input, cancellation.clone()).await
    }
    pub(crate) async fn merge(
        &self,
        current: Item,
        content: String,
        cancellation: &Cancellation,
    ) -> Result<String, Error> {
        self.component
            .merge(current, content, cancellation.clone())
            .await
    }
}
