//! Minimal Webhook delivery for already projected A2A events.

use crate::Error;
use crate::components::protocol::a2a::StreamResponse;

#[derive(Clone)]
pub struct Webhook {
    http: reqwest::Client,
    endpoint: String,
}

impl Webhook {
    pub fn new(endpoint: impl Into<String>) -> Result<Self, Error> {
        let endpoint = endpoint.into();
        if endpoint.trim().is_empty() {
            return Err(Error::new("INVALID_ARGUMENTS", "Webhook endpoint is empty"));
        }
        Ok(Self {
            http: reqwest::Client::new(),
            endpoint,
        })
    }

    pub fn with_http(endpoint: impl Into<String>, http: reqwest::Client) -> Result<Self, Error> {
        let endpoint = endpoint.into();
        if endpoint.trim().is_empty() {
            return Err(Error::new("INVALID_ARGUMENTS", "Webhook endpoint is empty"));
        }
        Ok(Self { http, endpoint })
    }

    pub async fn send(&self, response: &StreamResponse) -> Result<(), Error> {
        let result = self
            .http
            .post(&self.endpoint)
            .json(response)
            .send()
            .await
            .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
        if !result.status().is_success() {
            return Err(Error::new(
                "TRANSPORT",
                format!("Webhook HTTP status {}", result.status().as_u16()),
            ));
        }
        Ok(())
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

#[cfg(test)]
mod tests {
    use super::Webhook;

    #[test]
    fn validates_endpoint() {
        assert!(Webhook::new(" ").is_err());
        assert_eq!(
            Webhook::new("http://127.0.0.1/hook").unwrap().endpoint(),
            "http://127.0.0.1/hook"
        );
    }
}
