//! Long-memory transformation, separate from Session compression and storage.
use crate::components::memory::Item;
use crate::{Cancellation, Error, Messages};
use serde_json::{Map, Value};

/// Return one assistant Message with values shaped as
/// {"user":{"fact":"..."},"agent":{"preference":"..."}}.
/// Only summary/fact/preference are accepted; omitted types have no new content.
/// The caller supplies identity, source ranges, IDs and persistence.
pub trait Extractor: Send + Sync {
    fn extract(
        &self,
        input: &Messages,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Messages, Error>> + Send;
    /// Complete replacement text, recomputed against fresh state on conflict.
    fn merge(
        &self,
        current: &Item,
        content: &str,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<String, Error>> + Send;
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
#[path = "../../../tests/unit/embedded/src_components_memory_extraction_tests.rs"]
mod tests;
