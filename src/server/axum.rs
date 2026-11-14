//! Axum bindings for the outer Server layer.
//!
//! These types only bind HTTP concerns to the existing A2A objects. Runtime
//! execution, task projection and `run/reply` selection stay in Server code.

use std::sync::Arc;

use axum::Router;
use axum::extract::{Json, State};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::post;
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Error;
use crate::components::agent::Agent;
use crate::components::protocol::a2a::{SendMessageRequest, StreamResponse};
use crate::server::a2a::Service;
use crate::server::task::Store;

/// JSON-RPC 2.0 request envelope used by the A2A HTTP binding.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Request {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

impl Request {
    pub fn validate(&self) -> Result<(), Error> {
        if self.jsonrpc != "2.0" {
            return Err(Error::new("INVALID_ARGUMENTS", "jsonrpc must be 2.0"));
        }
        if self.method.trim().is_empty() {
            return Err(Error::new("INVALID_ARGUMENTS", "method must not be empty"));
        }
        Ok(())
    }

    pub fn method(&self) -> Result<Method, Error> {
        self.validate()?;
        match self.method.as_str() {
            "message/send" => Ok(Method::Send),
            "message/stream" => Ok(Method::Stream),
            "tasks/get" => Ok(Method::Get),
            "tasks/cancel" => Ok(Method::Cancel),
            "tasks/resubscribe" => Ok(Method::Resubscribe),
            _ => Err(Error::new(
                "METHOD_NOT_FOUND",
                format!("unsupported A2A method: {}", self.method),
            )),
        }
    }
}

/// A2A method names supported by the first Server binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    Send,
    Stream,
    Get,
    Cancel,
    Resubscribe,
}

/// JSON-RPC 2.0 response envelope.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Response {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorObject>,
}

impl Response {
    pub fn result(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Value, error: ErrorObject) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(error),
        }
    }

    pub fn from_error(id: Value, error: &Error) -> Self {
        Self::error(id, ErrorObject::from_error(error))
    }
}

/// JSON-RPC error object. The public Heart error remains the source of the
/// code and message; JSON-RPC only supplies its transport shape.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ErrorObject {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ErrorObject {
    pub fn from_error(error: &Error) -> Self {
        let code = match error.code.as_str() {
            "INVALID_ARGUMENTS" => -32602,
            "NOT_FOUND" => -32004,
            "CONFLICT" => -32009,
            "CANCELLED" => -32800,
            "METHOD_NOT_FOUND" => -32601,
            _ => -32000,
        };
        Self {
            code,
            message: error.message.clone(),
            data: Some(serde_json::json!({"code": error.code, "details": error.details})),
        }
    }
}

/// Convert one protocol response to an Axum SSE event.
pub fn event(response: &StreamResponse) -> Result<Event, Error> {
    Event::default()
        .json_data(response)
        .map_err(|error| Error::new("SERIALIZATION", error.to_string()))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TaskRequest {
    pub id: String,
}

/// Build the first Axum A2A endpoint set for one Agent.
///
/// `message/send` starts the task projection and drives the Agent in a
/// detached task on the caller's Tokio executor. `message/stream` keeps the
/// projection stream attached to the HTTP response, so a waiting Agent can
/// receive `reply`.
pub fn router<A, S>(service: Arc<Service<A, S>>) -> Router
where
    A: Agent + Send + Sync + 'static,
    S: Store,
{
    Router::new()
        .route("/a2a", post(handle::<A, S>))
        .with_state(service)
}

async fn handle<A, S>(
    State(service): State<Arc<Service<A, S>>>,
    Json(request): Json<Request>,
) -> HttpResponse
where
    A: Agent + Send + Sync + 'static,
    S: Store,
{
    let id = request.id.clone();
    let method = match request.method() {
        Ok(method) => method,
        Err(error) => return Json(Response::from_error(id, &error)).into_response(),
    };

    match method {
        Method::Send => {
            let params: SendMessageRequest = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(error) => {
                    return Json(Response::from_error(
                        id,
                        &Error::new("INVALID_ARGUMENTS", error.to_string()),
                    ))
                    .into_response();
                }
            };
            if params.message.task_id.is_some() {
                let result = service.reply(params).await;
                return json_result(id, result);
            }
            let (context, input) = match service.begin(params).await {
                Ok(value) => value,
                Err(error) => return Json(Response::from_error(id, &error)).into_response(),
            };
            let task_id = context.task_id.clone();
            let background = service.clone();
            tokio::spawn(async move {
                let _ = background.drive(context, input).await;
            });
            match service.task(&task_id).await {
                Ok(task) => json_result(id, Ok(task)),
                Err(error) => Json(Response::from_error(id, &error)).into_response(),
            }
        }
        Method::Stream => {
            let params: SendMessageRequest = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(error) => {
                    return Json(Response::from_error(
                        id,
                        &Error::new("INVALID_ARGUMENTS", error.to_string()),
                    ))
                    .into_response();
                }
            };
            if let Some(task_id) = params.message.task_id.clone() {
                let reply = service.reply(params).await;
                if let Err(error) = reply {
                    return Json(Response::from_error(id, &error)).into_response();
                }
                return match service.resubscribe(&task_id).await {
                    Ok(events) => Sse::new(events.map(|item| {
                        Ok::<_, std::convert::Infallible>(match item {
                            Ok(response) => sse_event(&response),
                            Err(error) => Event::default().event("error").data(error.message),
                        })
                    }))
                    .into_response(),
                    Err(error) => Json(Response::from_error(id, &error)).into_response(),
                };
            }
            let (context, input) = match service.begin(params).await {
                Ok(value) => value,
                Err(error) => return Json(Response::from_error(id, &error)).into_response(),
            };
            let (sender, receiver) = tokio::sync::mpsc::unbounded_channel::<StreamResponse>();
            let background = service.clone();
            let error_context = context.clone();
            tokio::spawn(async move {
                match background.stream_from(context, input).await {
                    Ok(events) => {
                        futures_util::pin_mut!(events);
                        while let Some(item) = events.next().await {
                            match item {
                                Ok(response) => {
                                    if sender.send(response).is_err() {
                                        break;
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                    }
                    Err(error) => {
                        let _ = background.record_error(&error_context, &error).await;
                        let _ =
                            sender.send(crate::server::a2a::error_status(&error_context, &error));
                    }
                }
            });
            let output = stream::unfold(receiver, |mut receiver| async move {
                receiver.recv().await.map(|response| {
                    (
                        Ok::<_, std::convert::Infallible>(sse_event(&response)),
                        receiver,
                    )
                })
            });
            Sse::new(output).into_response()
        }
        Method::Get => {
            let params: TaskRequest = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(error) => {
                    return Json(Response::from_error(
                        id,
                        &Error::new("INVALID_ARGUMENTS", error.to_string()),
                    ))
                    .into_response();
                }
            };
            json_result(id, service.task(&params.id).await)
        }
        Method::Cancel => {
            let params: TaskRequest = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(error) => {
                    return Json(Response::from_error(
                        id,
                        &Error::new("INVALID_ARGUMENTS", error.to_string()),
                    ))
                    .into_response();
                }
            };
            json_result(id, service.cancel(&params.id).await)
        }
        Method::Resubscribe => {
            let params: TaskRequest = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(error) => {
                    return Json(Response::from_error(
                        id,
                        &Error::new("INVALID_ARGUMENTS", error.to_string()),
                    ))
                    .into_response();
                }
            };
            match service.resubscribe(&params.id).await {
                Ok(events) => Sse::new(events.map(|item| {
                    Ok::<_, std::convert::Infallible>(match item {
                        Ok(response) => sse_event(&response),
                        Err(error) => Event::default().event("error").data(error.message),
                    })
                }))
                .into_response(),
                Err(error) => Json(Response::from_error(id, &error)).into_response(),
            }
        }
    }
}

fn sse_event(response: &StreamResponse) -> Event {
    event(response).unwrap_or_else(|error| Event::default().event("error").data(error.message))
}

fn json_result<T: Serialize>(id: Value, result: Result<T, Error>) -> HttpResponse {
    match result {
        Ok(value) => match serde_json::to_value(value) {
            Ok(value) => Json(Response::result(id, value)).into_response(),
            Err(error) => Json(Response::from_error(
                id,
                &Error::new("SERIALIZATION", error.to_string()),
            ))
            .into_response(),
        },
        Err(error) => Json(Response::from_error(id, &error)).into_response(),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/embedded/src_server_axum_tests.rs"]
mod tests;
