//! Minimal A2A HTTP client.
//!
//! The client keeps transport concerns outside Runtime. It sends the public
//! JSON-RPC envelope and returns the public A2A response without introducing
//! a second task or message model.

use crate::Error;
use crate::components::protocol::a2a::StreamResponse;
use crate::server::axum::{Request, Response};

#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    endpoint: String,
}

impl Client {
    pub fn new(endpoint: impl Into<String>) -> Result<Self, Error> {
        let endpoint = endpoint.into();
        if endpoint.trim().is_empty() {
            return Err(Error::new("INVALID_ARGUMENTS", "A2A endpoint is empty"));
        }
        Ok(Self {
            http: reqwest::Client::new(),
            endpoint,
        })
    }

    pub fn with_http(endpoint: impl Into<String>, http: reqwest::Client) -> Result<Self, Error> {
        let endpoint = endpoint.into();
        if endpoint.trim().is_empty() {
            return Err(Error::new("INVALID_ARGUMENTS", "A2A endpoint is empty"));
        }
        Ok(Self { http, endpoint })
    }

    pub async fn call(&self, request: Request) -> Result<Response, Error> {
        request.validate()?;
        let response = self
            .http
            .post(&self.endpoint)
            .json(&request)
            .send()
            .await
            .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
        let status = response.status();
        let body = response
            .json::<Response>()
            .await
            .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
        if !status.is_success() {
            return Err(Error::new(
                "TRANSPORT",
                format!("A2A HTTP status {}", status.as_u16()),
            ));
        }
        if let Some(error) = body.error.as_ref() {
            return Err(Error::new("A2A_ERROR", error.message.clone()));
        }
        Ok(body)
    }

    /// Request one SSE response. The returned body is intentionally left as
    /// the transport response so applications can choose their SSE parser.
    pub async fn stream(&self, request: Request) -> Result<reqwest::Response, Error> {
        request.validate()?;
        self.http
            .post(&self.endpoint)
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .json(&request)
            .send()
            .await
            .map_err(|error| Error::new("TRANSPORT", error.to_string()))
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn decode_event(bytes: &[u8]) -> Result<StreamResponse, Error> {
        serde_json::from_slice(bytes)
            .map_err(|error| Error::new("SERIALIZATION", error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::Client;
    use crate::components::protocol::a2a::{StreamResponse, Task, TaskState, TaskStatus};

    #[test]
    fn validates_endpoint_and_decodes_events() {
        assert!(Client::new(" ").is_err());
        let response = StreamResponse::Task(Task {
            id: "task-1".into(),
            context_id: "session-1".into(),
            status: TaskStatus {
                state: TaskState::Working,
                message: None,
            },
            artifacts: Vec::new(),
            history: Vec::new(),
        });
        let bytes = serde_json::to_vec(&response).unwrap();
        assert_eq!(
            serde_json::to_value(Client::decode_event(&bytes).unwrap()).unwrap()["task"]["id"],
            json!("task-1")
        );
    }
}
