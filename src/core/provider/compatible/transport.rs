use std::time::Duration;

use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde_json::{Map, Value};

use super::{Chat, Embed};
use crate::model;
use crate::model::token::{InputDetails, OutputDetails, Usage};

type Error = model::Error;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    OpenAI,
    DeepSeek,
    Vllm,
    Zhipu,
}

impl Kind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::OpenAI => "openai",
            Self::DeepSeek => "deepseek",
            Self::Vllm => "vllm",
            Self::Zhipu => "zhipu",
        }
    }
}

#[derive(Clone)]
pub(crate) struct Connection {
    pub(crate) client: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) provider_id: i64,
    pub(crate) kind: Kind,
}

impl Connection {
    pub(crate) fn new(
        kind: Kind,
        provider_id: i64,
        name: &str,
        base_url: &str,
        api_key: &str,
        timeout: u64,
    ) -> Result<Self, Error> {
        if name != kind.name() || timeout == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "provider name or timeout is invalid",
            ));
        }
        let url = reqwest::Url::parse(base_url)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid provider base URL"))?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "provider URL must be HTTP(S) without credentials, query or fragment",
            ));
        }
        if api_key.trim().is_empty() {
            return Err(Error::new("AUTHENTICATION", "provider API key is required"));
        }
        let mut authorization = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|_| Error::new("AUTHENTICATION", "provider API key is not a valid header"))?;
        authorization.set_sensitive(true);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, authorization);
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(timeout))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| Error::new("TRANSPORT", "provider HTTP client initialization failed"))?;
        Ok(Self {
            client,
            base_url: url.as_str().trim_end_matches('/').into(),
            provider_id,
            kind,
        })
    }

    pub(crate) fn chat(&self, config: model::chat::Config) -> Result<Chat, Error> {
        validate_model(&config.model_name)?;
        Ok(Chat {
            connection: self.clone(),
            config,
        })
    }

    pub(crate) fn embed(&self, config: model::embed::Config) -> Result<Embed, Error> {
        validate_model(&config.model_name)?;
        Ok(Embed {
            connection: self.clone(),
            config,
        })
    }

    pub(crate) async fn post(
        &self,
        endpoint: &str,
        body: Value,
    ) -> Result<reqwest::Response, Error> {
        let response = self
            .client
            .post(format!("{}/{endpoint}", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(transport_error)?;
        if response.status().is_success() {
            return Ok(response);
        }
        let status = response.status().as_u16();
        let body = response.json::<Value>().await.unwrap_or(Value::Null);
        Err(api_error(status, &body))
    }

    pub(crate) fn metadata(&self, model_name: &str) -> Map<String, Value> {
        Map::from_iter([
            ("provider_id".into(), self.provider_id.into()),
            ("provider_name".into(), self.kind.name().into()),
            ("model_name".into(), model_name.into()),
        ])
    }
}

pub(crate) fn validate_model(name: &str) -> Result<(), Error> {
    if name.trim().is_empty() {
        Err(Error::new("INVALID_ARGUMENTS", "model name is empty"))
    } else {
        Ok(())
    }
}

pub(crate) fn transport_error(error: reqwest::Error) -> Error {
    if error.is_timeout() {
        Error::new("TIMEOUT", "provider request timed out")
    } else if error.is_decode() {
        Error::new("INVALID_RESPONSE", "provider response cannot be decoded")
    } else {
        Error::new("TRANSPORT", "provider HTTP transport failed")
    }
}

pub(crate) fn api_error(status: u16, body: &Value) -> Error {
    let raw_code = body
        .pointer("/error/code")
        .and_then(Value::as_str)
        .unwrap_or("");
    let code = match raw_code {
        "context_length_exceeded" => "CONTEXT_LENGTH_EXCEEDED",
        "content_filter" | "content_policy_violation" => "CONTENT_FILTERED",
        "unsupported_parameter" | "unsupported_value" => "UNSUPPORTED",
        _ => match status {
            400 | 422 => "INVALID_ARGUMENTS",
            401 => "AUTHENTICATION",
            403 => "PERMISSION_DENIED",
            408 => "TIMEOUT",
            429 => "RATE_LIMITED",
            500..=599 => "UNAVAILABLE",
            _ => "PROVIDER",
        },
    };
    // Do not copy raw Provider messages/body: they can echo credentials or prompts.
    Error::new(code, "provider request failed").with_details(format!("http_status={status}"))
}

fn count(value: Option<&Value>) -> Result<Option<u64>, Error> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "provider usage count is invalid")),
    }
}

pub(crate) fn validate_options(
    options: &Map<String, Value>,
    reserved: &[&str],
) -> Result<(), Error> {
    if let Some(key) = options.keys().find(|key| reserved.contains(&key.as_str())) {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "provider options contain a reserved request field",
        )
        .with_details(format!("field={key}")));
    }
    Ok(())
}

pub(crate) fn parse_usage(value: Option<&Value>) -> Result<Option<Usage>, Error> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let object = value
        .as_object()
        .ok_or_else(|| Error::new("INVALID_RESPONSE", "provider usage must be an object"))?;
    let input = count(object.get("prompt_tokens"))?;
    let output = count(object.get("completion_tokens"))?;
    let total = count(object.get("total_tokens"))?.or_else(|| {
        input
            .zip(output)
            .and_then(|(input, output)| input.checked_add(output))
    });
    let cached = count(
        value
            .pointer("/prompt_tokens_details/cached_tokens")
            .or_else(|| object.get("prompt_cache_hit_tokens")),
    )?;
    let think = count(value.pointer("/completion_tokens_details/reasoning_tokens"))?;
    let details = object
        .iter()
        .filter(|(key, value)| {
            !matches!(
                key.as_str(),
                "prompt_tokens" | "completion_tokens" | "total_tokens"
            ) && value.is_number()
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    Ok(Some(Usage {
        input,
        output,
        total,
        input_details: cached.map(|cached| InputDetails {
            cached: Some(cached),
        }),
        output_details: think.map(|think| OutputDetails { think: Some(think) }),
        details,
    }))
}
