//! Adapter for the official Model Context Protocol Rust SDK.
//!
//! Only this module knows the SDK types. The rest of Heart consumes the small
//! transport-neutral values below and maps them to the public MCP contract.

use std::{collections::BTreeSet, sync::Arc, time::Duration};

use rmcp::{
    ClientLifecycleMode, ClientServiceExt, RoleClient,
    model::{
        ClientCapabilities, ClientConfig, ClientNotification, ClientRequest, Implementation,
        PaginatedRequestParams, ProtocolVersion, ServerResult,
    },
    service::{PeerRequestOptions, RunningService, ServiceError},
    transport::{
        StreamableHttpClientTransport, TokioChildProcess, common::client_side_sse::NeverRetry,
        streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde_json::{Map, Value};

use crate::{Cancellation, Error};

const TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug)]
pub(crate) struct Tool {
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) input_schema: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub(crate) enum Content {
    Text(String),
    Other,
}

#[derive(Clone, Debug)]
pub(crate) struct ToolResult {
    pub(crate) structured_content: Option<Value>,
    pub(crate) is_error: bool,
    pub(crate) content: Vec<Content>,
}

pub(crate) struct Client {
    inner: RunningService<RoleClient, ClientConfig>,
}

fn config() -> ClientConfig {
    ClientConfig::new(
        ClientCapabilities::default(),
        Implementation::new("halo-agents", env!("CARGO_PKG_VERSION")),
    )
    .with_protocol_version(ProtocolVersion::V_2025_11_25)
}

impl Client {
    pub(crate) async fn http(
        endpoint: &str,
        headers: Map<String, Value>,
        cancellation: &Cancellation,
    ) -> Result<Self, Error> {
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
        let custom = headers
            .into_iter()
            .map(|(key, value)| {
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
                    reqwest_mcp::header::HeaderValue::from_str(value.as_str().ok_or_else(
                        || Error::new("INVALID_ARGUMENTS", "MCP header value must be text"),
                    )?)
                    .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid MCP header value"))?;
                value.set_sensitive(true);
                Ok((name, value))
            })
            .collect::<Result<std::collections::HashMap<_, _>, Error>>()?;
        let http = reqwest_mcp::Client::builder()
            .redirect(reqwest_mcp::redirect::Policy::none())
            .timeout(TIMEOUT)
            .build()
            .map_err(|_| Error::new("TRANSPORT", "cannot create MCP HTTP client"))?;
        let mut transport = StreamableHttpClientTransportConfig::with_uri(endpoint.to_owned())
            .custom_headers(custom)
            .reinit_on_expired_session(false);
        transport.retry_config = Arc::new(NeverRetry::default());
        let client = cancellable(
            async {
                tokio::time::timeout(
                    TIMEOUT,
                    config().serve_with_lifecycle(
                        StreamableHttpClientTransport::with_client(http, transport),
                        ClientLifecycleMode::Initialize,
                    ),
                )
                .await
                .map_err(|_| Error::new("TIMEOUT", "MCP initialization timed out"))?
                .map_err(|_| Error::new("TRANSPORT", "MCP initialization failed"))
            },
            cancellation,
        )
        .await?;
        Ok(Self { inner: client })
    }

    pub(crate) async fn stdio(
        command: tokio::process::Command,
        cancellation: &Cancellation,
    ) -> Result<Self, Error> {
        check(cancellation)?;
        check_runtime()?;
        let transport = TokioChildProcess::new(command)
            .map_err(|_| Error::new("TRANSPORT", "cannot start MCP server"))?;
        let client = cancellable(
            async {
                tokio::time::timeout(
                    TIMEOUT,
                    config().serve_with_lifecycle(transport, ClientLifecycleMode::Initialize),
                )
                .await
                .map_err(|_| Error::new("TIMEOUT", "MCP initialization timed out"))?
                .map_err(|_| Error::new("TRANSPORT", "MCP initialization failed"))
            },
            cancellation,
        )
        .await?;
        Ok(Self { inner: client })
    }

    pub(crate) async fn list_tools(&self, cancellation: &Cancellation) -> Result<Vec<Tool>, Error> {
        check(cancellation)?;
        let mut names = BTreeSet::new();
        let mut tools = Vec::new();
        let mut cursor = None;
        let mut cursors = BTreeSet::new();
        let mut pages = 0;
        loop {
            pages += 1;
            if pages > 1024 {
                return Err(Error::new(
                    "INVALID_RESPONSE",
                    "MCP discovery exceeded page limit",
                ));
            }
            let params = Some(PaginatedRequestParams::default().with_cursor(cursor.take()));
            let page = cancellable(
                async { self.inner.list_tools(params).await.map_err(service_error) },
                cancellation,
            )
            .await?;
            for tool in page.tools {
                let name = tool.name.into_owned();
                if !names.insert(name.clone()) {
                    return Err(Error::new("INVALID_RESPONSE", "duplicate MCP tool name"));
                }
                tools.push(Tool {
                    name,
                    description: tool.description.map(|value| value.into_owned()),
                    input_schema: (*tool.input_schema).clone(),
                });
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

    pub(crate) async fn call(
        &self,
        name: &str,
        arguments: Map<String, Value>,
        cancellation: &Cancellation,
    ) -> Result<ToolResult, Error> {
        check(cancellation)?;
        let response = self
            .request(
                ClientRequest::CallToolRequest(rmcp::model::CallToolRequest::new(
                    rmcp::model::CallToolRequestParams::new(name.to_owned())
                        .with_arguments(arguments),
                )),
                cancellation,
            )
            .await?;
        let ServerResult::CallToolResult(result) = response else {
            return Err(Error::new(
                "UNSUPPORTED",
                "MCP deferred input/task results are not supported by this single-call contract",
            ));
        };
        Ok(ToolResult {
            structured_content: result.structured_content,
            is_error: result.is_error == Some(true),
            content: result
                .content
                .into_iter()
                .map(|part| {
                    part.as_text()
                        .map_or(Content::Other, |text| Content::Text(text.text.clone()))
                })
                .collect(),
        })
    }

    pub(crate) async fn close(mut self) -> Result<(), Error> {
        self.inner
            .close_with_timeout(Duration::from_secs(5))
            .await
            .map_err(|_| Error::new("TRANSPORT", "MCP shutdown failed"))?
            .ok_or_else(|| Error::new("TIMEOUT", "MCP shutdown timed out"))?;
        Ok(())
    }

    async fn request(
        &self,
        request: ClientRequest,
        cancellation: &Cancellation,
    ) -> Result<ServerResult, Error> {
        let handle = cancellable(
            async {
                self.inner
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
            let notification =
                ClientNotification::CancelledNotification(rmcp::model::CancelledNotification::new(
                    rmcp::model::CancelledNotificationParam::new(
                        Some(id),
                        Some("caller cancelled".into()),
                    ),
                ));
            let _ = tokio::time::timeout(
                Duration::from_secs(1),
                self.inner.send_notification(notification),
            )
            .await;
        }
        result
    }
}

fn check(cancellation: &Cancellation) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        Err(Error::new("CANCELLED", "operation cancelled"))
    } else {
        Ok(())
    }
}

async fn cancellable<T>(
    future: impl std::future::Future<Output = Result<T, Error>>,
    cancellation: &Cancellation,
) -> Result<T, Error> {
    let mut future = std::pin::pin!(future);
    futures_util::future::poll_fn(|cx| {
        cancellation.register(cx.waker());
        if let Err(error) = check(cancellation) {
            return std::task::Poll::Ready(Err(error));
        }
        let result = future.as_mut().poll(cx);
        if let Err(error) = check(cancellation) {
            return std::task::Poll::Ready(Err(error));
        }
        result
    })
    .await
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
