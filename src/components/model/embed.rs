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

#[allow(async_fn_in_trait)]
pub trait Embed: Send + Sync {
    async fn embed(&self, request: Request) -> Result<Message, Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_single_input_and_dimension() {
        let request = Request {
            input: " \n".into(),
            dimensions: None,
            options: Map::new(),
        };
        assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");
        let request = Request {
            input: "北京".into(),
            dimensions: Some(0),
            options: Map::new(),
        };
        assert_eq!(request.validate().unwrap_err().code, "INVALID_ARGUMENTS");
    }

    #[test]
    fn vector_uses_values_and_source_uses_metadata() {
        let output = message(vec![0.1, -0.3], "example", "embed-v1", Some(2)).unwrap();
        assert_eq!(output.role, Role::Assistant);
        assert_eq!(output.values["vector"].as_array().unwrap().len(), 2);
        assert_eq!(output.metadata["dimension"], 2);
        assert!(output.values.get("dimension").is_none());
        assert_eq!(
            message(vec![0.1], "example", "embed-v1", Some(2))
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
    }
}
