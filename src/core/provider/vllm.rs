use std::fmt;

use serde::Deserialize;

use super::compatible::{Connection, Kind};
use crate::model;

/// Configuration for a VLLM OpenAI-compatible server.
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
            provider_name: "vllm".into(),
            base_url: "http://127.0.0.1:8000/v1".into(),
            api_key: String::new(),
            timeout: 120,
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
pub struct Vllm {
    connection: Connection,
}

impl Vllm {
    /// Start configuring a VLLM provider connection.
    pub fn builder() -> Builder {
        Builder {
            config: Config::default(),
        }
    }

    pub fn new(config: Config) -> Result<Self, model::Error> {
        Ok(Self {
            connection: Connection::new(
                Kind::Vllm,
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

    pub fn embed(&self, config: model::embed::Config) -> Result<embed::Embed, model::Error> {
        self.connection.embed(config)
    }
}

/// Fluent configuration for a VLLM provider connection.
pub struct Builder {
    config: Config,
}

impl Builder {
    pub fn provider_id(mut self, provider_id: i64) -> Self {
        self.config.provider_id = provider_id;
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.provider_name = name.into();
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.config.base_url = base_url.into();
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.config.api_key = api_key.into();
        self
    }

    pub fn timeout(mut self, timeout: u64) -> Self {
        self.config.timeout = timeout;
        self
    }

    pub fn build(self) -> Result<Vllm, model::Error> {
        Vllm::new(self.config)
    }
}

pub mod chat {
    pub use super::super::compatible::{Chat, Events};
}

pub mod embed {
    pub use super::super::compatible::Embed;
}
