//! Official MCP SDK-backed transport; tool results and errors use the existing
//! direct Message/Error contracts. No Node or Runtime identities are created.

use std::{collections::BTreeSet, sync::Arc, time::Duration};

use rmcp::{
    ClientLifecycleMode, ClientServiceExt, RoleClient,
    model::{
        ClientCapabilities, ClientConfig, ClientNotification, ClientRequest, Implementation,
        ProtocolVersion, ServerResult,
    },
    service::{PeerRequestOptions, RunningService, ServiceError},
    transport::{
        StreamableHttpClientTransport, TokioChildProcess, common::client_side_sse::NeverRetry,
        streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde_json::{Map, Value, json};

use crate::{
    Cancellation, Error, Message,
    mcp::{Definition, Server as Contract},
    message::Role,
    model::content::Part,
    tool::{Call, ToolDefinition},
};

use super::operation::{cancellable, check};

const TIMEOUT: Duration = Duration::from_secs(30);

pub struct Server {
    definition: Definition,
    client: RunningService<RoleClient, ClientConfig>,
}

fn client() -> ClientConfig {
    ClientConfig::new(
        ClientCapabilities::default(),
        Implementation::new("halo-agents", env!("CARGO_PKG_VERSION")),
    )
    .with_protocol_version(ProtocolVersion::V_2025_11_25)
}

impl Server {
    /// Connects to an explicit endpoint. Headers are transport-only secrets,
    /// never copied to Definition, Messages, error text or Debug output.
    pub async fn http(
        definition: Definition,
        endpoint: &str,
        headers: Map<String, Value>,
        cancellation: &Cancellation,
    ) -> Result<Self, Error> {
        validate_definition(&definition)?;
        check(cancellation)?;
        check_runtime()?;
        let url = reqwest_mcp::Url::parse(endpoint)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid MCP endpoint"))?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "MCP endpoint requires an HTTP URL without embedded credentials",
            ));
        }
        let mut custom = std::collections::HashMap::new();
        for (key, value) in headers {
            if matches!(
                key.to_ascii_lowercase().as_str(),
                "accept"
                    | "content-type"
                    | "mcp-session-id"
                    | "mcp-protocol-version"
                    | "host"
                    | "content-length"
            ) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "MCP protocol headers cannot be overridden",
                ));
            }
            let name = reqwest_mcp::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid MCP header"))?;
            let mut value =
                reqwest_mcp::header::HeaderValue::from_str(value.as_str().ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", "MCP header value must be text")
                })?)
                .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid MCP header value"))?;
            value.set_sensitive(true);
            custom.insert(name, value);
        }
        let http = reqwest_mcp::Client::builder()
            .redirect(reqwest_mcp::redirect::Policy::none())
            .timeout(TIMEOUT)
            .build()
            .map_err(|_| Error::new("TRANSPORT", "cannot create MCP HTTP client"))?;
        let mut config = StreamableHttpClientTransportConfig::with_uri(endpoint.to_owned())
            .custom_headers(custom)
            .reinit_on_expired_session(false);
        config.retry_config = Arc::new(NeverRetry::default());
        let transport = StreamableHttpClientTransport::with_client(http, config);
        let client = cancellable(
            async {
                tokio::time::timeout(
                    TIMEOUT,
                    client().serve_with_lifecycle(transport, ClientLifecycleMode::Initialize),
                )
                .await
                .map_err(|_| Error::new("TIMEOUT", "MCP initialization timed out"))?
                .map_err(|_| Error::new("TRANSPORT", "MCP initialization failed"))
            },
            cancellation,
        )
        .await?;
        Ok(Self { definition, client })
    }

    /// Starts only the application-supplied command, without a shell wrapper.
    /// The application owns program/args/env and sandbox/permission decisions.
    pub async fn stdio(
        definition: Definition,
        command: tokio::process::Command,
        cancellation: &Cancellation,
    ) -> Result<Self, Error> {
        validate_definition(&definition)?;
        check(cancellation)?;
        check_runtime()?;
        let transport = TokioChildProcess::new(command)
            .map_err(|_| Error::new("TRANSPORT", "cannot start MCP server"))?;
        let client = cancellable(
            async {
                tokio::time::timeout(
                    TIMEOUT,
                    client().serve_with_lifecycle(transport, ClientLifecycleMode::Initialize),
                )
                .await
                .map_err(|_| Error::new("TIMEOUT", "MCP initialization timed out"))?
                .map_err(|_| Error::new("TRANSPORT", "MCP initialization failed"))
            },
            cancellation,
        )
        .await?;
        Ok(Self { definition, client })
    }

    pub async fn close(mut self) -> Result<(), Error> {
        self.client
            .close_with_timeout(Duration::from_secs(5))
            .await
            .map_err(|_| Error::new("TRANSPORT", "MCP shutdown failed"))?
            .ok_or_else(|| Error::new("TIMEOUT", "MCP shutdown timed out"))?;
        Ok(())
    }

    async fn request(
        &self,
        request: Value,
        cancellation: &Cancellation,
    ) -> Result<ServerResult, Error> {
        let request: ClientRequest = serde_json::from_value(request)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid MCP request"))?;
        let handle = cancellable(
            async {
                self.client
                    .send_cancellable_request(request, PeerRequestOptions::with_timeout(TIMEOUT))
                    .await
                    .map_err(service_error)
            },
            cancellation,
        )
        .await?;
        let id = handle.id.clone();
        let result = cancellable(
            async { handle.await_response().await.map_err(service_error) },
            cancellation,
        )
        .await;
        if result
            .as_ref()
            .is_err_and(|error| error.code == "CANCELLED")
        {
            // Cancellation is explicit on the wire, not just disconnection.
            // A remote side effect already committed cannot be rolled back.
            let notification: ClientNotification = serde_json::from_value(json!({"method":"notifications/cancelled","params":{"requestId":id,"reason":"caller cancelled"}})).map_err(|_| Error::new("INTERNAL", "cannot encode MCP cancellation"))?;
            let _ = tokio::time::timeout(
                Duration::from_secs(1),
                self.client.send_notification(notification),
            )
            .await;
        }
        result
    }
}

impl Contract for Server {
    fn definition(&self) -> &Definition {
        &self.definition
    }

    async fn list_tools(&self, cancellation: &Cancellation) -> Result<Vec<ToolDefinition>, Error> {
        check(cancellation)?;
        let mut cursor = None;
        let mut cursors = BTreeSet::new();
        let mut names = BTreeSet::new();
        let mut tools = Vec::new();
        let mut pages = 0;
        loop {
            pages += 1;
            if pages > 1024 {
                return Err(Error::new(
                    "INVALID_RESPONSE",
                    "MCP discovery exceeded page limit",
                ));
            }
            let params = match cursor.take() {
                None => json!({}),
                Some(cursor) => json!({"cursor":cursor}),
            };
            let ServerResult::ListToolsResult(page) = self
                .request(json!({"method":"tools/list","params":params}), cancellation)
                .await?
            else {
                return Err(Error::new(
                    "INVALID_RESPONSE",
                    "invalid MCP discovery result",
                ));
            };
            for tool in page.tools {
                let name = tool.name.into_owned();
                if !names.insert(name.clone()) {
                    return Err(Error::new("INVALID_RESPONSE", "duplicate MCP tool name"));
                }
                let tool = ToolDefinition {
                    description: tool
                        .description
                        .map(|description| description.into_owned())
                        .filter(|text| !text.trim().is_empty())
                        .unwrap_or_else(|| name.clone()),
                    name,
                    parameters: (*tool.input_schema).clone(),
                };
                tool.validate()
                    .map_err(|_| Error::new("INVALID_RESPONSE", "invalid MCP tool schema"))?;
                tools.push(tool);
            }
            match page.next_cursor {
                None => break,
                Some(next) if next.is_empty() || !cursors.insert(next.clone()) => {
                    return Err(Error::new(
                        "INVALID_RESPONSE",
                        "invalid MCP pagination cursor",
                    ));
                }
                Some(next) => cursor = Some(next),
            }
        }
        check(cancellation)?;
        Ok(tools)
    }

    async fn call(&self, call: Call, cancellation: &Cancellation) -> Result<Message, Error> {
        check(cancellation)?;
        if call.call_id.trim().is_empty() || call.name.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "MCP call id and name are required",
            ));
        }
        let response = self.request(json!({"method":"tools/call","params":{"name":call.name,"arguments":call.arguments}}), cancellation).await?;
        let ServerResult::CallToolResult(result) = response else {
            return Err(Error::new(
                "UNSUPPORTED",
                "MCP deferred input/task results are not supported by this single-call contract",
            ));
        };
        if result.is_error == Some(true) {
            // Return the Tool's error itself, never a tool_result envelope.
            if let Some(value) = &result.structured_content
                && let Ok(error) = serde_json::from_value::<Error>(value.clone())
            {
                return Err(error);
            }
            let text = result
                .content
                .iter()
                .filter_map(|part| part.as_text().map(|text| text.text.as_str()))
                .collect::<Vec<_>>()
                .join("\n");
            return Err(Error::new(
                "TOOL_ERROR",
                if text.trim().is_empty() {
                    "MCP tool execution failed".to_owned()
                } else {
                    text
                },
            ));
        }
        let mut message = Message::new(Role::Tool);
        message
            .metadata
            .insert("call_id".into(), json!(call.call_id));
        message.metadata.insert("name".into(), json!(call.name));
        if let Some(value) = result.structured_content {
            message.values = match value {
                Value::Object(values) => values,
                value => Map::from_iter([("value".into(), value)]),
            };
        }
        for part in result.content {
            if let Some(text) = part.as_text() {
                message.content.push(Part {
                    r#type: "text".into(),
                    data: json!({"value":text.text}),
                });
            } else {
                // Full media contracts are V3+, not invented by this adapter.
                return Err(Error::new(
                    "UNSUPPORTED",
                    "MCP non-text tool content requires the V3 media contract",
                ));
            }
        }
        check(cancellation)?;
        Ok(message)
    }
}

fn validate_definition(definition: &Definition) -> Result<(), Error> {
    if definition.name.trim().is_empty() {
        Err(Error::new(
            "INVALID_ARGUMENTS",
            "MCP server name is required",
        ))
    } else {
        Ok(())
    }
}

fn check_runtime() -> Result<(), Error> {
    tokio::runtime::Handle::try_current()
        .map(|_| ())
        .map_err(|_| Error::new("UNAVAILABLE", "MCP transport requires a Tokio runtime"))
}

fn service_error(error: ServiceError) -> Error {
    match error {
        ServiceError::Timeout { .. } => Error::new("TIMEOUT", "MCP request timed out"),
        ServiceError::Cancelled { .. } => Error::new("CANCELLED", "MCP request cancelled"),
        ServiceError::McpError(_) => Error::new("MCP_ERROR", "MCP server rejected request"),
        ServiceError::TransportSend(_) | ServiceError::TransportClosed => {
            Error::new("TRANSPORT", "MCP transport failed")
        }
        _ => Error::new("INVALID_RESPONSE", "invalid MCP response"),
    }
}
