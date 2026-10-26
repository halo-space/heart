use std::fmt;

use serde::Deserialize;

use super::compatible::{Connection, Kind};
use crate::model;

/// Configuration for the Zhipu AI API.
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
            provider_name: "zhipu".into(),
            base_url: "https://open.bigmodel.cn/api/paas/v4".into(),
            api_key: String::new(),
            timeout: 60,
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
pub struct Zhipu {
    connection: Connection,
}

impl Zhipu {
    pub fn builder() -> Builder {
        Builder {
            config: Config::default(),
        }
    }

    pub fn new(config: Config) -> Result<Self, model::Error> {
        Ok(Self {
            connection: Connection::new(
                Kind::Zhipu,
                config.provider_id,
                &config.provider_name,
                &config.base_url,
                &config.api_key,
                config.timeout,
            )?,
        })
    }

    /// Build an Embed component for a Zhipu embedding model.
    pub fn embed(&self, config: model::embed::Config) -> Result<embed::Embed, model::Error> {
        self.connection.embed(config)
    }
}

pub(super) fn validate_dimensions(dimensions: Option<u32>) -> Result<(), model::Error> {
    if dimensions.is_some_and(|value| !(256..=2048).contains(&value)) {
        return Err(model::Error::new(
            "INVALID_ARGUMENTS",
            "Zhipu Embedding-3 dimensions must be between 256 and 2048",
        ));
    }
    Ok(())
}

pub struct Builder {
    config: Config,
}

impl Builder {
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

    pub fn build(self) -> Result<Zhipu, model::Error> {
        Zhipu::new(self.config)
    }
}

pub mod embed {
    pub use super::super::compatible::Embed;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_hides_api_key_and_uses_zhipu_v4_endpoint() {
        let config = Config {
            api_key: "do-not-log-this".into(),
            ..Config::default()
        };
        assert_eq!(config.provider_name, "zhipu");
        assert_eq!(config.base_url, "https://open.bigmodel.cn/api/paas/v4");
        assert!(!format!("{config:?}").contains("do-not-log-this"));
    }

    #[test]
    fn embedding_dimensions_are_checked_against_supported_values() {
        for dimensions in [256, 512, 768, 1024, 2048] {
            assert!(validate_dimensions(Some(dimensions)).is_ok());
        }

        assert_eq!(
            validate_dimensions(Some(255)).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
        assert_eq!(
            validate_dimensions(Some(2049)).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
        assert!(validate_dimensions(None).is_ok());
    }
}
