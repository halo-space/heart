use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::event::Event;
use crate::components::message::{Messages, Role};
use crate::components::model::Error;
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
    pub messages: Messages,
    pub voice: Option<String>,
    pub format: Option<String>,
    pub speed: Option<f32>,
    pub stream: bool,
    #[serde(default)]
    pub options: Options,
}

impl Request {
    pub fn validate(&self) -> Result<(), Error> {
        if self
            .messages
            .as_slice()
            .iter()
            .all(|message| !message.content.iter().any(|part| part.r#type == "text"))
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "tts request needs a text content part",
            ));
        }
        for value in [&self.voice, &self.format] {
            if value
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
            {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "tts request text options must be nonempty",
                ));
            }
        }
        if self
            .speed
            .is_some_and(|speed| !speed.is_finite() || speed <= 0.0)
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "tts speed must be a positive finite number",
            ));
        }
        for message in self.messages.as_slice() {
            if message.role != Role::User && message.role != Role::Assistant {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "tts messages must use user or assistant roles",
                ));
            }
            if message.content.iter().any(|part| part.r#type != "text") {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "tts messages may contain text parts only",
                ));
            }
        }
        Ok(())
    }
}

pub trait Tts: Send + Sync {
    type Stream: Stream<Item = Result<Event, Error>> + Send;

    fn stream(
        &self,
        request: Request,
    ) -> impl std::future::Future<Output = Result<Self::Stream, Error>> + Send;
}

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_components_model_tts_tests.rs"]
mod tests;
