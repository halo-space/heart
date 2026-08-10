use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool;

use super::Error;

pub mod audio {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Options {
        pub format: Option<String>,
        pub voice: Option<String>,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Options {
    pub stream: bool,
    pub tools: Vec<tool::Definition>,
    #[serde(default = "default_feats")]
    pub feats: Vec<Feat>,
    pub audio: Option<audio::Options>,
    #[serde(default = "default_complete_audio")]
    pub complete_audio: bool,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub extra_headers: Option<BTreeMap<String, String>>,
    pub extra_query: Option<Value>,
    pub extra_body: Option<Value>,
    pub timeout: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Feat {
    Text,
    Audio,
}

fn default_feats() -> Vec<Feat> {
    vec![Feat::Text]
}

fn default_complete_audio() -> bool {
    true
}

impl Default for Options {
    fn default() -> Self {
        Self {
            stream: false,
            tools: Vec::new(),
            feats: default_feats(),
            audio: None,
            complete_audio: true,
            max_tokens: None,
            temperature: None,
            top_p: None,
            extra_headers: None,
            extra_query: None,
            extra_body: None,
            timeout: None,
        }
    }
}

impl Options {
    pub fn validate(&self) -> Result<(), Error> {
        if self.feats.is_empty() {
            return Err(Error::invalid_input("feats must not be empty"));
        }

        for (index, feat) in self.feats.iter().enumerate() {
            if self.feats[index + 1..].contains(feat) {
                return Err(Error::invalid_input("feats must not contain duplicates"));
            }
        }

        if !self.feats.contains(&Feat::Audio) && self.audio.is_some() {
            return Err(Error::invalid_input("audio options require the audio feat"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_default_to_text() {
        assert_eq!(Options::default().feats, vec![Feat::Text]);
    }

    #[test]
    fn options_reject_audio_without_audio_feat() {
        let options = Options {
            audio: Some(audio::Options {
                format: Some("wav".into()),
                voice: Some("alloy".into()),
            }),
            ..Options::default()
        };

        assert_eq!(
            options.validate().unwrap_err().kind,
            super::super::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn options_reject_duplicate_feats() {
        let options = Options {
            feats: vec![Feat::Text, Feat::Text],
            ..Options::default()
        };

        assert_eq!(
            options.validate().unwrap_err().kind,
            super::super::ErrorKind::InvalidInput
        );
    }
}
