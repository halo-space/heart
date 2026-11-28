use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::event::Event;
use crate::components::message::Messages;
use crate::components::model::Error;

/// Provider-specific ASR options.
pub type Options = Map<String, Value>;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Config {
    pub model_name: String,
    #[serde(default)]
    pub options: Options,
}

impl Config {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            options: Options::new(),
        }
    }

    pub fn options(mut self, options: Options) -> Self {
        self.options = options;
        self
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Request {
    /// Audio input is carried by the shared Message/Part contract.
    pub messages: Messages,
    pub stream: bool,
    #[serde(default)]
    pub options: Options,
}

impl Request {
    pub fn validate(&self) -> Result<(), Error> {
        if self.messages.is_empty() {
            return Err(Error::new("INVALID_ARGUMENTS", "asr messages are empty"));
        }
        let mut has_audio = false;
        for message in self.messages.as_slice() {
            for part in &message.content {
                if part.r#type == "audio" {
                    let source = part
                        .data
                        .get("source")
                        .and_then(Value::as_object)
                        .ok_or_else(|| {
                            Error::new("INVALID_ARGUMENTS", "audio source is required")
                        })?;
                    let kind = source.get("kind").and_then(Value::as_str).unwrap_or("");
                    let value = source.get("value").and_then(Value::as_str).unwrap_or("");
                    if kind.trim().is_empty() || value.trim().is_empty() {
                        return Err(Error::new(
                            "INVALID_ARGUMENTS",
                            "audio source kind and value are required",
                        ));
                    }
                    has_audio = true;
                }
            }
        }
        if !has_audio {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "asr request needs an audio content part",
            ));
        }
        Ok(())
    }
}

/// Speech-to-text capability. Providers return transcript text through the
/// shared Event and Message contracts instead of introducing an ASR response.
pub trait Asr: Send + Sync {
    type Stream: Stream<Item = Result<Event, Error>> + Send;

    fn stream(
        &self,
        request: Request,
    ) -> impl std::future::Future<Output = Result<Self::Stream, Error>> + Send;
}

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_components_model_asr_tests.rs"]
mod tests;
