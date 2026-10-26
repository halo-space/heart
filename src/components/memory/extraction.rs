//! Long-memory transformation, separate from Session compression and storage.
use crate::{Cancellation, Error, Messages, memory::Item};
use serde_json::{Map, Value};
use std::{future::Future, pin::Pin, sync::Arc};

/// Return one assistant Message with values shaped as
/// {"user":{"fact":"..."},"agent":{"preference":"..."}}.
/// Only summary/fact/preference are accepted; omitted types have no new content.
/// The caller supplies identity, source ranges, IDs and persistence.
pub trait Extractor: Send + Sync {
    fn extract(
        &self,
        input: &Messages,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Messages, Error>>;
    /// Complete replacement text, recomputed against fresh state on conflict.
    fn merge(
        &self,
        current: &Item,
        content: &str,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<String, Error>>;
}

type ExtractFuture = Pin<Box<dyn Future<Output = Result<Messages, Error>> + 'static>>;
type MergeFuture = Pin<Box<dyn Future<Output = Result<String, Error>> + 'static>>;
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

#[cfg(test)]
mod tests {
    use super::validate;
    use serde_json::json;

    #[test]
    fn accepts_only_nonempty_durable_text_by_owner_and_type() {
        for valid in [
            json!({}),
            json!({"user":{"summary":"User context"}}),
            json!({"agent":{"fact":"Agent-specific fact","preference":"Concise"}}),
        ] {
            let values = valid.as_object().unwrap().clone();
            assert!(validate(&values).is_ok(), "{valid}");
        }

        for invalid in [
            json!({"session":{"summary":"not a durable owner"}}),
            json!({"user":["not an object"]}),
            json!({"user":{"id":"not a memory type"}}),
            json!({"agent":{"fact":"  "}}),
        ] {
            let values = invalid.as_object().unwrap().clone();
            assert_eq!(
                validate(&values).unwrap_err().code,
                "INVALID_RESPONSE",
                "{invalid}"
            );
        }
    }
}
