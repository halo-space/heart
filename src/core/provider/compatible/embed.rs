//! OpenAI-compatible embedding request encoding and response decoding.

use serde_json::{Map, Value};

use super::common::{self, Connection, Kind};
use crate::message::Message;
use crate::model;

type Error = model::Error;

#[derive(Clone)]
pub struct Embed {
    pub(super) connection: Connection,
    pub(super) config: model::embed::Config,
}

impl model::embed::Embed for Embed {
    async fn embed(&self, request: model::embed::Request) -> Result<Message, Error> {
        let dimensions = request.dimensions;
        let body = encode_embed_request(self.connection.kind, &self.config, &request)?;
        let value: Value = self
            .connection
            .post("embeddings", Value::Object(body))
            .await?
            .json()
            .await
            .map_err(common::transport_error)?;
        let data = value
            .get("data")
            .and_then(Value::as_array)
            .filter(|data| data.len() == 1)
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "expected one embedding"))?;
        if data[0].get("index").and_then(Value::as_u64) != Some(0) {
            return Err(Error::new("INVALID_RESPONSE", "embedding index is invalid"));
        }
        let vector = data[0]
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "embedding vector is missing"))?
            .iter()
            .map(|value| value.as_f64().map(|value| value as f32))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "embedding components are invalid"))?;
        let mut message = model::embed::message(
            vector,
            self.connection.kind.name(),
            &self.config.model_name,
            dimensions,
        )?;
        message
            .metadata
            .insert("provider_id".into(), self.connection.provider_id.into());
        message.usage = common::parse_usage(value.get("usage"))?;
        Ok(message)
    }
}

pub(super) fn encode_embed_request(
    kind: Kind,
    config: &model::embed::Config,
    request: &model::embed::Request,
) -> Result<Map<String, Value>, Error> {
    request.validate()?;
    if kind == Kind::Zhipu {
        super::super::zhipu::validate_dimensions(request.dimensions)?;
    }
    let mut options = config.options.clone();
    options.extend(request.options.clone());
    common::validate_options(
        &options,
        &["model", "input", "encoding_format", "dimensions"],
    )?;
    let mut body = options;
    body.insert("model".into(), config.model_name.clone().into());
    body.insert("input".into(), request.input.clone().into());
    body.insert("encoding_format".into(), "float".into());
    if let Some(dimensions) = request.dimensions {
        body.insert("dimensions".into(), dimensions.into());
    }
    Ok(body)
}
