//! Long-memory transformation, separate from Session compression and storage.
use crate::{Cancellation, Error, Messages, memory::Item};
use serde_json::{Map, Value};
use std::{future::Future, pin::Pin, sync::Arc};

/// Return one assistant Message with values shaped as
/// {"user":{"fact":"..."},"agent":{"preference":"..."}}.
/// Only summary/fact/preference are accepted; omitted types have no new content.
/// The caller supplies identity, source ranges, IDs and persistence.
#[allow(async_fn_in_trait)]
pub trait Extractor: Send + Sync {
    async fn extract(
        &self,
        input: &Messages,
        cancellation: &Cancellation,
    ) -> Result<Messages, Error>;
    /// Complete replacement text, recomputed against fresh state on conflict.
    async fn merge(
        &self,
        current: &Item,
        content: &str,
        cancellation: &Cancellation,
    ) -> Result<String, Error>;
}

type ExtractFuture = Pin<Box<dyn Future<Output = Result<Messages, Error>>>>;
type MergeFuture = Pin<Box<dyn Future<Output = Result<String, Error>>>>;
trait Entry: Send + Sync {
    fn extract(&self, input: Messages, cancellation: Cancellation) -> ExtractFuture;
    fn merge(&self, current: Item, content: String, cancellation: Cancellation) -> MergeFuture;
}
struct Typed<E>(Arc<E>);
impl<E: Extractor + 'static> Entry for Typed<E> {
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
    component: Arc<dyn Entry>,
}
impl Configured {
    pub(crate) fn new<E>(component: E) -> Self
    where
        E: Extractor + 'static,
    {
        Self {
            component: Arc::new(Typed(Arc::new(component))),
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

pub(crate) fn validate(values: &Map<String, Value>) -> Result<(), Error> {
    let invalid = || Error::new("INVALID_RESPONSE", "invalid long memory ownership or text");
    for (level, types) in values {
        if !matches!(level.as_str(), "user" | "agent") {
            return Err(invalid());
        }
        let types = types.as_object().ok_or_else(invalid)?;
        for (kind, content) in types {
            if !matches!(kind.as_str(), "summary" | "fact" | "preference")
                || content.as_str().is_none_or(|text| text.trim().is_empty())
            {
                return Err(invalid());
            }
        }
    }
    Ok(())
}
