//! Database Workspace backed by an application-initialized Elasticsearch client.
//! Only four ID-based CRUD Tools are installed. Client configuration, index
//! provisioning and business-specific queries remain application-owned.
//! Cancellation or a lost response after dispatch does not prove a mutation
//! was rolled back. No automatic retries or compensating writes are performed.

use std::sync::Arc;

use elasticsearch::indices::IndicesExistsParts;
use elasticsearch::{CreateParts, DeleteParts, Elasticsearch, GetParts, UpdateParts};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::tools::{definition, tool_message};
use crate::core::{agent::toolkit::Toolkit, operation::cancellable};
use crate::tool::{Call, Tool, ToolDefinition};
use crate::workspace::{Definition, Mode, Workspace as Contract};
use crate::{Cancellation, Error, Message};

/// Binds an existing client to one concrete index. Performs no network I/O,
/// connection setup, index creation or mapping modification.
pub fn new(client: Elasticsearch, index: impl Into<String>) -> Result<Workspace, Error> {
    let index = index.into();
    if index.is_empty()
        || index.len() > 255
        || matches!(index.as_str(), "." | "..")
        || index.starts_with(['-', '_', '+'])
        || index.chars().any(|c| {
            c.is_whitespace() || c.is_control() || c.is_uppercase() || "\\/*?\"<>|,#:%".contains(c)
        })
    {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "workspace requires one concrete Elasticsearch index",
        ));
    }
    Ok(Workspace {
        backend: Arc::new(Backend { client, index }),
        definition: Definition {
            mode: Mode::Server,
            path: None,
            metadata: Map::new(),
        },
        toolkit: Toolkit::new(),
    })
}

pub struct Workspace {
    backend: Arc<Backend>,
    definition: Definition,
    toolkit: Toolkit,
}

impl Workspace {
    /// Stores the application's existing capabilities. Tools are bound when
    /// assembling the Toolkit, without discovery or database access.
    pub fn toolkit(mut self, toolkit: Toolkit) -> Self {
        self.toolkit = toolkit;
        self
    }
}

impl Contract for Workspace {
    fn definition(&self) -> &Definition {
        &self.definition
    }
}

impl From<Workspace> for Toolkit {
    fn from(workspace: Workspace) -> Self {
        let mut toolkit = workspace.toolkit;
        for kind in [Kind::Write, Kind::Read, Kind::Update, Kind::Delete] {
            toolkit = toolkit.tool(Operation {
                backend: Arc::clone(&workspace.backend),
                kind,
                definition: kind.definition(),
            });
        }
        toolkit
    }
}

struct Backend {
    client: Elasticsearch,
    index: String,
}

impl Backend {
    async fn ensure_index(&self) -> Result<(), Error> {
        // Both create and update can auto-create an absent ES index. In
        // particular, doc_as_upsert=false prevents document insertion but
        // does NOT prevent index creation. Do not cache this check.
        // Applications must still prohibit auto-creation in cluster policy /
        // credentials to cover concurrent index deletion after this request.
        let response = self
            .client
            .indices()
            .exists(IndicesExistsParts::Index(&[&self.index]))
            .send()
            .await
            .map_err(transport_error)?;
        success(response.status_code().as_u16())
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Write,
    Read,
    Update,
    Delete,
}

impl Kind {
    fn definition(self) -> ToolDefinition {
        let id = json!({"type":"integer", "minimum":i64::MIN, "maximum":i64::MAX});
        let data = json!({"type":"object"});
        let (name, description, properties, required) = match self {
            Self::Write => (
                "workspace_write",
                "Insert one database record. data must contain an application-provided i64 id; an existing id is a conflict.",
                json!({"data":{"type":"object","properties":{"id":id},"required":["id"]}}),
                vec!["data"],
            ),
            Self::Read => (
                "workspace_read",
                "Read one database record by i64 id; values.data is an empty or one-record array.",
                json!({"id":id}),
                vec!["id"],
            ),
            Self::Update => (
                "workspace_update",
                "Patch the supplied fields of one database record by i64 id. Does not insert missing records or change the record id.",
                json!({"id":id,"data":data}),
                vec!["id", "data"],
            ),
            Self::Delete => (
                "workspace_delete",
                "Physically delete one database record by i64 id.",
                json!({"id":id}),
                vec!["id"],
            ),
        };
        definition(
            name,
            description,
            json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),
        )
    }
}

struct Operation {
    backend: Arc<Backend>,
    kind: Kind,
    definition: ToolDefinition,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Id {
    id: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Write {
    data: Map<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Update {
    id: i64,
    data: Map<String, Value>,
}

impl Tool for Operation {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        // The existing Tool chain owns Schema/identity/Permission and retries.
        // Cancellation also covers a pending response body, not only headers.
        cancellable(self.perform(call), cancellation).await
    }
}

impl Operation {
    async fn perform(&self, call: &Call) -> Result<Message, Error> {
        let client = &self.backend.client;
        let index = self.backend.index.as_str();
        let values = match self.kind {
            Kind::Write => {
                let Write { data } = arguments(call)?;
                let id = data.get("id").and_then(Value::as_i64).ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", "write data.id must be an i64")
                })?;
                let id = id.to_string();
                // Do not provision an application index as a side effect of a
                // normal Tool call. ES cluster privileges/settings must also
                // forbid auto-creation if an index is deleted concurrently.
                self.backend.ensure_index().await?;
                let response = client
                    .create(CreateParts::IndexId(index, &id))
                    .body(&data)
                    .send()
                    .await
                    .map_err(transport_error)?;
                let (status, body) = decode(response).await?;
                success(status)?;
                if body.get("result").and_then(Value::as_str) != Some("created") {
                    return Err(invalid_response());
                }
                // A default ingest pipeline may transform the submitted data.
                // Realtime GET returns stored _source, not a made-up copy of
                // the request or a driver metadata envelope. No refresh needed.
                let response = client
                    .get(GetParts::IndexId(index, &id))
                    .send()
                    .await
                    .map_err(transport_error)?;
                let (status, body) = decode(response).await?;
                success(status)?;
                json!({"data":source(&body)?})
            }
            Kind::Read => {
                let Id { id } = arguments(call)?;
                let id = id.to_string();
                let response = client
                    .get(GetParts::IndexId(index, &id))
                    .send()
                    .await
                    .map_err(transport_error)?;
                let (status, body) = decode(response).await?;
                if status == 404
                    && body.get("found") == Some(&Value::Bool(false))
                    && body.get("error").is_none()
                {
                    json!({"data":[]})
                } else {
                    success(status)?;
                    json!({"data":[source(&body)?]})
                }
            }
            Kind::Update => {
                let Update { id, data } = arguments(call)?;
                if data
                    .get("id")
                    .is_some_and(|value| value.as_i64() != Some(id))
                {
                    return Err(Error::new(
                        "INVALID_ARGUMENTS",
                        "update cannot change the record id",
                    ));
                }
                let id = id.to_string();
                self.backend.ensure_index().await?;
                let response = client
                    .update(UpdateParts::IndexId(index, &id))
                    .body(json!({"doc":data,"doc_as_upsert":false}))
                    .send()
                    .await
                    .map_err(transport_error)?;
                let (status, body) = decode(response).await?;
                if status == 404
                    && body.pointer("/error/type").and_then(Value::as_str)
                        == Some("document_missing_exception")
                {
                    json!({"count":0})
                } else {
                    success(status)?;
                    match body.get("result").and_then(Value::as_str) {
                        Some("updated") => json!({"count":1}),
                        Some("noop") => json!({"count":0}),
                        _ => return Err(invalid_response()),
                    }
                }
            }
            Kind::Delete => {
                let Id { id } = arguments(call)?;
                let id = id.to_string();
                let response = client
                    .delete(DeleteParts::IndexId(index, &id))
                    .send()
                    .await
                    .map_err(transport_error)?;
                let (status, body) = decode(response).await?;
                if status == 404
                    && body.get("result").and_then(Value::as_str) == Some("not_found")
                    && body.get("error").is_none()
                {
                    json!({"count":0})
                } else {
                    success(status)?;
                    if body.get("result").and_then(Value::as_str) != Some("deleted") {
                        return Err(invalid_response());
                    }
                    json!({"count":1})
                }
            }
        };
        Ok(tool_message(values))
    }
}

fn arguments<T: serde::de::DeserializeOwned>(call: &Call) -> Result<T, Error> {
    serde_json::from_value(Value::Object(call.arguments.clone()))
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid workspace CRUD arguments"))
}

async fn decode(response: elasticsearch::http::response::Response) -> Result<(u16, Value), Error> {
    let status = response.status_code().as_u16();
    // Do not copy raw driver errors, URLs, headers or response bodies into
    // model-visible errors: upstream messages may echo credentials or data.
    // For errors without a JSON body retain the useful HTTP error category.
    let body = response.json::<Value>().await.map_err(|error| {
        if !(200..300).contains(&status) {
            status_error(status)
        } else if error.is_timeout() {
            Error::new("TIMEOUT", "workspace database response timed out")
        } else {
            invalid_response()
        }
    })?;
    Ok((status, body))
}

fn success(status: u16) -> Result<(), Error> {
    if (200..300).contains(&status) {
        Ok(())
    } else {
        Err(status_error(status))
    }
}

fn status_error(status: u16) -> Error {
    let code = match status {
        400 | 422 => "INVALID_ARGUMENTS",
        401 => "AUTHENTICATION",
        403 => "PERMISSION_DENIED",
        404 => "NOT_FOUND",
        408 | 504 => "TIMEOUT",
        409 => "CONFLICT",
        429 => "RATE_LIMITED",
        500..=599 => "UNAVAILABLE",
        _ => "INVALID_RESPONSE",
    };
    Error::new(code, "workspace database request failed")
        .with_details(format!("http_status={status}"))
}

fn invalid_response() -> Error {
    Error::new("INVALID_RESPONSE", "invalid workspace database response")
}

fn source(body: &Value) -> Result<&Value, Error> {
    if body.get("found") != Some(&Value::Bool(true)) {
        return Err(invalid_response());
    }
    body.get("_source")
        .filter(|value| value.is_object())
        .ok_or_else(invalid_response)
}

fn transport_error(error: elasticsearch::Error) -> Error {
    if error.is_timeout() {
        Error::new("TIMEOUT", "workspace database request timed out")
    } else {
        Error::new("TRANSPORT", "workspace database transport failed")
    }
}
