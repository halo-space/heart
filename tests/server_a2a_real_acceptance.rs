//! Real HTTP acceptance for the Axum A2A binding.
//!
//! These tests intentionally use a live DeepSeek provider and a real local
//! TCP listener. They are ignored by the default suite because they consume
//! provider quota; run them with the project's env loaded.

use std::time::Duration;

use halo::agent::{
    self, Agent, Cancellation, Error, Message,
    message::Role,
    middleware::Middleware,
    model::chat,
    provider::deepseek::DeepSeek,
    server::{a2a::Service, axum::router},
    tool::{Call, Tool, ToolDefinition},
};
use reqwest::Client;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicI64, AtomicUsize, Ordering},
};
use tokio::net::TcpListener;

fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(9_410_001);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

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

/// Hold the second real model response so two concurrent HTTP replies can
/// observe the same waiting task. The model still makes both calls against
/// DeepSeek; this only creates a deterministic cancellation/idempotency window
/// around the real continuation.
#[derive(Clone)]
struct HoldSecondModel {
    calls: Arc<AtomicUsize>,
}

impl Middleware for HoldSecondModel {
    async fn after_model(
        &self,
        _output: &mut Value,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 1 {
            loop {
                if cancellation.is_cancelled() {
                    return Err(Error::new(
                        "CANCELLED",
                        "A2A test model continuation cancelled",
                    ));
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct HoldModel;

impl Middleware for HoldModel {
    async fn after_model(
        &self,
        _output: &mut Value,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        loop {
            if cancellation.is_cancelled() {
                return Err(Error::new("CANCELLED", "A2A test model call cancelled"));
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

struct AskForApproval;

impl Tool for AskForApproval {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: std::sync::LazyLock<ToolDefinition> =
            std::sync::LazyLock::new(|| ToolDefinition {
                name: "ask_for_approval".into(),
                description: "Ask the user for one approval reply".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "count": {"type": "integer", "minimum": 1}
                    },
                    "required": ["question", "count"],
                    "additionalProperties": false
                })
                .as_object()
                .expect("approval schema")
                .clone(),
            });
        &DEFINITION
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        let question = call.arguments["question"]
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "question is required"))?;
        let count = call.arguments["count"]
            .as_u64()
            .filter(|value| *value == 1)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "count must be 1"))?;
        let mut prompt = Message::new(Role::Assistant);
        prompt
            .content
            .push(halo::agent::model::content::part::Part {
                r#type: "text".into(),
                data: json!({"value": question}).as_object().unwrap().clone(),
            });
        let feedback = halo::agent::wait::ask(prompt, count as usize, cancellation).await?;
        Ok(Message::function(json!({"user_feedback": feedback.0})))
    }
}

fn waiting_live_agent() -> Result<impl Agent, Error> {
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
        .build()?;
    let model = provider.chat(chat::Config::new(
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
    ))?;
    agent::chat()
        .model(model)
        .system_prompt(
            "必须先调用 ask_for_approval 工具询问用户是否批准，count 必须为 1；收到用户回复后只回复 A2A_REPLY_OK。",
        )
        .tools([AskForApproval])
        .middleware(HoldSecondModel {
            calls: Arc::new(AtomicUsize::new(0)),
        })
        .with_id_fn(next_id)
        .build()
}

fn cancellable_live_agent() -> Result<impl Agent, Error> {
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
        .build()?;
    let model = provider.chat(chat::Config::new(
        std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
    ))?;
    agent::chat()
        .model(model)
        .system_prompt("只回复 A2A_CANCEL_OK，不要解释。")
        .middleware(HoldModel)
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

fn reply_message(session_id: i64, task_id: &str, message_id: &str) -> Value {
    json!({
        "message": {
            "role": "ROLE_USER",
            "parts": [{"text": "批准，继续执行"}],
            "messageId": message_id,
            "contextId": session_id.to_string(),
            "taskId": task_id
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

async fn wait_for_state(
    client: &Client,
    endpoint: &str,
    task_id: &str,
    expected: &str,
) -> Result<Value, Error> {
    let mut last_state = String::new();
    for _ in 0..180 {
        let response = client
            .post(endpoint)
            .json(&request(20, "tasks/get", json!({"id": task_id})))
            .send()
            .await
            .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
        let body: Value = response
            .json()
            .await
            .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
        last_state = body["result"]["status"]["state"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        if last_state == "TASK_STATE_FAILED" {
            return Err(Error::new(
                "A2A_ERROR",
                format!("A2A task failed: {}", body["result"]["status"]["message"]),
            ));
        }
        if last_state == expected {
            return Ok(body);
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Err(Error::new(
        "TIMEOUT",
        format!("A2A task did not reach {expected} within 90 seconds (last state: {last_state})"),
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
    let reply = client
        .post(&endpoint)
        .json(&request(
            4,
            "message/send",
            reply_message(31_001, task_id, "reply-after-complete"),
        ))
        .send()
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let reply_body: Value = reply
        .json()
        .await
        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
    assert_eq!(reply_body["error"]["data"]["code"], "CONFLICT");
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

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and a live DeepSeek request"]
async fn real_http_cancel_and_resubscribe_preserve_terminal_task() -> Result<(), Error> {
    let service = std::sync::Arc::new(Service::new(cancellable_live_agent()?));
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let address = listener
        .local_addr()
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let server = tokio::spawn(async move {
        axum::serve(listener, router(service))
            .await
            .expect("A2A cancellation test server");
    });
    let endpoint = format!("http://{address}/a2a");
    let client = Client::new();
    let response = client
        .post(&endpoint)
        .json(&request(
            11,
            "message/send",
            message(31_003, 32_003, "message-cancel-1"),
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
        .ok_or_else(|| Error::new("INVALID_RESPONSE", "message/send returned no task id"))?
        .to_owned();
    client
        .post(&endpoint)
        .json(&request(12, "tasks/cancel", json!({"id": task_id})))
        .send()
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let canceled = wait_for_state(&client, &endpoint, &task_id, "TASK_STATE_CANCELED").await?;
    assert_eq!(canceled["result"]["id"], task_id);

    let response = client
        .post(&endpoint)
        .json(&request(13, "tasks/resubscribe", json!({"id": task_id})))
        .send()
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    assert!(response.status().is_success());
    let body = response
        .text()
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    assert!(body.contains("TASK_STATE_CANCELED"));
    server.abort();
    Ok(())
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and a live DeepSeek request"]
async fn real_http_duplicate_reply_is_idempotent_while_waiting() -> Result<(), Error> {
    let service = std::sync::Arc::new(Service::new(waiting_live_agent()?));
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let address = listener
        .local_addr()
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let server = tokio::spawn(async move {
        axum::serve(listener, router(service))
            .await
            .expect("A2A idempotency test server");
    });
    let endpoint = format!("http://{address}/a2a");
    let client = Client::new();
    let response = client
        .post(&endpoint)
        .json(&request(
            21,
            "message/send",
            message(31_004, 32_004, "message-wait-1"),
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
        .ok_or_else(|| Error::new("INVALID_RESPONSE", "message/send returned no task id"))?
        .to_owned();
    wait_for_state(&client, &endpoint, &task_id, "TASK_STATE_INPUT_REQUIRED").await?;

    let first = client.clone();
    let first_endpoint = endpoint.clone();
    let first_task_id = task_id.clone();
    let first_reply = tokio::spawn(async move {
        first
            .post(&first_endpoint)
            .json(&request(
                22,
                "message/send",
                reply_message(31_004, &first_task_id, "reply-1"),
            ))
            .send()
            .await
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    let duplicate = client
        .post(&endpoint)
        .json(&request(
            23,
            "message/send",
            reply_message(31_004, &task_id, "reply-1"),
        ))
        .send()
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let duplicate_body: Value = duplicate
        .json()
        .await
        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
    assert_eq!(duplicate_body["result"]["id"], task_id);
    assert!(duplicate_body["error"].is_null());

    let mut conflict = reply_message(31_004, &task_id, "reply-1");
    conflict["message"]["parts"][0]["text"] = json!("不同的批准内容");
    let conflict = client
        .post(&endpoint)
        .json(&request(24, "message/send", conflict))
        .send()
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let conflict_body: Value = conflict
        .json()
        .await
        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))?;
    assert_eq!(conflict_body["error"]["data"]["code"], "CONFLICT");

    client
        .post(&endpoint)
        .json(&request(25, "tasks/cancel", json!({"id": task_id})))
        .send()
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    let first_response = first_reply
        .await
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?
        .map_err(|error| Error::new("TRANSPORT", error.to_string()))?;
    assert!(first_response.status().is_success());
    wait_for_state(&client, &endpoint, &task_id, "TASK_STATE_CANCELED").await?;
    server.abort();
    Ok(())
}
