//! Real HTTP acceptance for the Axum A2A binding.
//!
//! These tests intentionally use a live DeepSeek provider and a real local
//! TCP listener. They are ignored by the default suite because they consume
//! provider quota; run them with the project's env loaded.

use std::time::Duration;

use halo::agent::{
    self, Agent, Error,
    model::chat,
    provider::deepseek::DeepSeek,
    server::{a2a::Service, axum::router},
};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;

fn live_agent() -> Result<impl Agent, Error> {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
            Error::new(
                "INVALID_ARGUMENTS",
                "set DEEPSEEK_API_KEY for the A2A HTTP acceptance",
            )
        })?)
        .timeout(90)
        .build()?;
    let model = provider.chat(chat::Config::new(
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
    ))?;
    agent::chat()
        .model(model)
        .system_prompt("只回复 A2A_HTTP_OK，不要解释。")
        .build()
}

fn request(id: i64, method: &str, params: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params
    })
}

fn message(session_id: i64, chat_id: i64, message_id: &str) -> Value {
    json!({
        "message": {
            "role": "ROLE_USER",
            "parts": [{"text": "只回复 A2A_HTTP_OK"}],
            "messageId": message_id,
            "contextId": session_id.to_string()
        },
        "metadata": {
            "session_id": session_id,
            "chat_id": chat_id
        }
    })
}

async fn start() -> Result<(String, tokio::task::JoinHandle<()>), Error> {
    let service = std::sync::Arc::new(Service::new(live_agent()?));
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let address = listener
        .local_addr()
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let task = tokio::spawn(async move {
        axum::serve(listener, router(service))
            .await
            .expect("A2A test server");
    });
    Ok((format!("http://{address}/a2a"), task))
}

async fn wait_for_completion(
    client: &Client,
    endpoint: &str,
    task_id: &str,
) -> Result<Value, Error> {
    for _ in 0..90 {
        let response = client
            .post(endpoint)
            .json(&request(2, "tasks/get", json!({"id": task_id})))
            .send()
            .await
            .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
        let body: Value = response
            .json()
            .await
            .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
        let state = body["result"]["status"]["state"]
            .as_str()
            .unwrap_or_default();
        if state == "TASK_STATE_COMPLETED" {
            return Ok(body);
        }
        if matches!(
            state,
            "TASK_STATE_FAILED" | "TASK_STATE_CANCELED" | "TASK_STATE_REJECTED"
        ) {
            return Err(Error::new(
                "A2A_ERROR",
                format!("live A2A task ended in {state}"),
            ));
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    Err(Error::new(
        "TIMEOUT",
        "live A2A task did not complete within 90 seconds",
    ))
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and a live DeepSeek request"]
async fn real_http_send_and_task_get_complete() -> Result<(), Error> {
    let (endpoint, server) = start().await?;
    let client = Client::new();
    let response = client
        .post(&endpoint)
        .json(&request(
            1,
            "message/send",
            message(31_001, 32_001, "message-send-1"),
        ))
        .send()
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let body: Value = response
        .json()
        .await
        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
    let task_id = body["result"]["id"]
        .as_str()
        .ok_or_else(|| Error::new("INVALID_RESPONSE", "message/send returned no task id"))?;
    let completed = wait_for_completion(&client, &endpoint, task_id).await?;
    assert_eq!(
        completed["result"]["status"]["state"],
        "TASK_STATE_COMPLETED"
    );
    assert!(
        completed["result"]["artifacts"]
            .as_array()
            .is_some_and(|artifacts| !artifacts.is_empty())
    );
    server.abort();
    Ok(())
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and a live DeepSeek request"]
async fn real_http_message_stream_returns_sse_events() -> Result<(), Error> {
    let (endpoint, server) = start().await?;
    let client = Client::new();
    let response = client
        .post(&endpoint)
        .header("accept", "text/event-stream")
        .json(&request(
            3,
            "message/stream",
            message(31_002, 32_002, "message-stream-1"),
        ))
        .send()
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    assert!(response.status().is_success());
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream"))
    );
    let body = response
        .text()
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    assert!(body.contains("statusUpdate"));
    assert!(body.contains("TASK_STATE_COMPLETED"));
    server.abort();
    Ok(())
}
