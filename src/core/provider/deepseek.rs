use std::fmt;

use serde::Deserialize;

use super::compatible::{Connection, Kind};
use crate::model;

#[derive(Clone, Deserialize)]
pub struct Config {
    pub provider_id: i64,
    pub provider_name: String,
    pub base_url: String,
    pub api_key: String,
    /// Total HTTP request timeout, in seconds.
    pub timeout: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider_id: 0,
            provider_name: "deepseek".into(),
            base_url: "https://api.deepseek.com".into(),
            api_key: String::new(),
            timeout: 30,
        }
    }
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("provider_id", &self.provider_id)
            .field("provider_name", &self.provider_name)
            .field("api_key", &"[redacted]")
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct DeepSeek {
    connection: Connection,
}

impl DeepSeek {
    pub fn new(config: Config) -> Result<Self, model::Error> {
        Ok(Self {
            connection: Connection::new(
                Kind::DeepSeek,
                config.provider_id,
                &config.provider_name,
                &config.base_url,
                &config.api_key,
                config.timeout,
            )?,
        })
    }

    pub fn chat(&self, config: model::chat::Config) -> Result<chat::Chat, model::Error> {
        self.connection.chat(config)
    }

    pub fn embed(&self, _config: model::embed::Config) -> Result<embed::Embed, model::Error> {
        Err(model::Error::new(
            "UNSUPPORTED",
            "DeepSeek does not expose a supported embedding endpoint",
        ))
    }
}

pub mod chat {
    pub use super::super::compatible::{Chat, Events};
}

pub mod embed {
    pub use super::super::compatible::Embed;
}
