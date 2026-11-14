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
