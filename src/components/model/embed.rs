use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::message::{Message, Role};
use crate::components::model::Error;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Config {
    pub model_name: String,
    #[serde(default)]
    pub options: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Request {
    pub input: String,
    pub dimensions: Option<u32>,
    #[serde(default)]
    pub options: Map<String, Value>,
}

impl Request {
    pub fn validate(&self) -> Result<(), Error> {
        if self.input.trim().is_empty() {
            return Err(Error::new("INVALID_ARGUMENTS", "embed input is empty"));
        }
        if self.dimensions == Some(0) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "dimensions must be positive",
            ));
        }
        Ok(())
    }
}

/// Validate and normalize one Provider embedding response into the shared
/// Message shape. This helper never generates vectors or guesses dimensions.
pub fn message(
    vector: Vec<f32>,
    provider_name: &str,
    model_name: &str,
    requested_dimensions: Option<u32>,
) -> Result<Message, Error> {
    if vector.is_empty() || vector.iter().any(|component| !component.is_finite()) {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "embedding vector is invalid",
        ));
    }
    let dimension = u32::try_from(vector.len())
        .map_err(|_| Error::new("INVALID_RESPONSE", "embedding dimension exceeds u32"))?;
    if requested_dimensions.is_some_and(|requested| requested != dimension) {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "embedding dimension does not match request",
        ));
    }
    let mut output = Message::new(Role::Assistant);
    output.values.insert(
        "vector".into(),
        serde_json::to_value(vector)
            .map_err(|error| Error::new("INVALID_RESPONSE", error.to_string()))?,
    );
    output
        .metadata
        .insert("provider_name".into(), Value::String(provider_name.into()));
    output
        .metadata
        .insert("model_name".into(), Value::String(model_name.into()));
    output
        .metadata
        .insert("dimension".into(), Value::from(dimension));
    Ok(output)
}

pub trait Embed: Send + Sync {
    fn embed(
        &self,
        request: Request,
    ) -> impl std::future::Future<Output = Result<Message, Error>> + Send;
}

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_components_model_embed_tests.rs"]
mod tests;
