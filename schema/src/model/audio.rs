use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Error;

pub mod speech {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Options {
        pub stream: bool,
        pub voice: Option<String>,
        pub instructions: Option<String>,
        pub response_format: Option<String>,
        pub speed: Option<f32>,
        pub stream_format: Option<String>,
        pub extra_headers: Option<BTreeMap<String, String>>,
        pub extra_query: Option<Value>,
        pub extra_body: Option<Value>,
        pub timeout: Option<u64>,
    }

    impl Default for Options {
        fn default() -> Self {
            Self {
                stream: false,
                voice: None,
                instructions: None,
                response_format: None,
                speed: None,
                stream_format: None,
                extra_headers: None,
                extra_query: None,
                extra_body: None,
                timeout: None,
            }
        }
    }

    impl Options {
        pub fn validate(&self) -> Result<(), Error> {
            if let Some(speed) = self.speed
                && (!speed.is_finite() || !(0.25..=4.0).contains(&speed))
            {
                return Err(Error::invalid_input(
                    "speech speed must be between 0.25 and 4.0",
                ));
            }

            Ok(())
        }
    }
}

pub mod transcription {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Options {
        pub stream: bool,
        pub chunking_strategy: Option<Value>,
        pub include: Vec<String>,
        pub keywords: Vec<String>,
        pub language: Option<String>,
        pub languages: Vec<String>,
        pub prompt: Option<String>,
        pub response_format: Option<String>,
        pub temperature: Option<f32>,
        pub timestamp_granularities: Vec<String>,
        pub extra_headers: Option<BTreeMap<String, String>>,
        pub extra_query: Option<Value>,
        pub extra_body: Option<Value>,
        pub timeout: Option<u64>,
    }

    impl Default for Options {
        fn default() -> Self {
            Self {
                stream: false,
                chunking_strategy: None,
                include: Vec::new(),
                keywords: Vec::new(),
                language: None,
                languages: Vec::new(),
                prompt: None,
                response_format: None,
                temperature: None,
                timestamp_granularities: Vec::new(),
                extra_headers: None,
                extra_query: None,
                extra_body: None,
                timeout: None,
            }
        }
    }

    impl Options {
        pub fn validate(&self) -> Result<(), Error> {
            if let Some(temperature) = self.temperature
                && (!temperature.is_finite() || !(0.0..=1.0).contains(&temperature))
            {
                return Err(Error::invalid_input(
                    "transcription temperature must be between 0.0 and 1.0",
                ));
            }

            Ok(())
        }
    }
}
