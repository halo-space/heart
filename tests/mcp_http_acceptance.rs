//! End-to-end HTTP MCP acceptance against a local, protocol-speaking service.
//! The tool returns the host's current clock rather than a canned response.

use std::{
    io,
    time::{SystemTime, UNIX_EPOCH},
};

use halo::agent::{
    Cancellation,
    mcp::{Definition, Server as _},
    tool::Call,
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::watch,
    task::JoinHandle,
};

struct LocalMcpService {
    endpoint: String,
    shutdown: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl LocalMcpService {
    async fn start() -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let endpoint = format!("http://{}/mcp", listener.local_addr()?);
        let (shutdown, mut stopped) = watch::channel(false);
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    changed = stopped.changed() => {
                        if changed.is_err() || *stopped.borrow() {
                            break;
                        }
                    }
                    accepted = listener.accept() => {
                        let Ok((stream, _)) = accepted else { break };
                        tokio::spawn(async move {
                            let _ = handle_connection(stream).await;
                        });
                    }
                }
            }
        });
        Ok(Self {
            endpoint,
            shutdown,
            task,
        })
    }

    async fn stop(self) {
        let _ = self.shutdown.send(true);
        self.task.await.expect("local MCP service task");
    }
}

async fn handle_connection(mut stream: TcpStream) -> io::Result<()> {
    let request = read_http_request(&mut stream).await?;
    if !request.starts_with("POST ") {
        return write_http_response(&mut stream, 405, "application/json", b"{}").await;
    }
    let body = read_http_body(&mut stream, &request).await?;
    let rpc: Value = serde_json::from_slice(&body)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let Some(id) = rpc.get("id").filter(|value| !value.is_null()).cloned() else {
        // MCP notifications such as notifications/initialized have no reply.
        return write_http_response(&mut stream, 202, "application/json", b"").await;
    };
    let method = rpc["method"].as_str().unwrap_or_default();
    let result = match method {
        "initialize" => json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "heart-local-http-acceptance", "version": "0.1.0"}
        }),
        "tools/list" => json!({
            "tools": [{
                "name": "current_time",
                "description": "Return the current Unix timestamp from the host clock",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }]
        }),
        "tools/call" if rpc["params"]["name"] == "current_time" => {
            let unix_seconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(io::Error::other)?
                .as_secs();
            json!({
                "content": [{"type": "text", "text": format!("Unix time: {unix_seconds}")}],
                "structuredContent": {"unix_seconds": unix_seconds},
                "isError": false
            })
        }
        "ping" => json!({}),
        _ => {
            let response = json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": "method not found"}
            });
            return write_json_response(&mut stream, &response).await;
        }
    };
    write_json_response(
        &mut stream,
        &json!({"jsonrpc": "2.0", "id": id, "result": result}),
    )
    .await
}

async fn read_http_request(stream: &mut TcpStream) -> io::Result<String> {
    let mut headers = Vec::new();
    let mut window = [0; 4];
    while window != *b"\r\n\r\n" {
        let mut byte = [0];
        if stream.read_exact(&mut byte).await.is_err() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "HTTP request ended before headers completed",
            ));
        }
        headers.push(byte[0]);
        window.rotate_left(1);
        window[3] = byte[0];
        if headers.len() > 64 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HTTP headers exceeded the acceptance service limit",
            ));
        }
    }
    String::from_utf8(headers).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

async fn read_http_body(stream: &mut TcpStream, headers: &str) -> io::Result<Vec<u8>> {
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or_default();
    if content_length > 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "HTTP body exceeded the acceptance service limit",
        ));
    }
    let mut body = vec![0; content_length];
    stream.read_exact(&mut body).await?;
    Ok(body)
}

async fn write_json_response(stream: &mut TcpStream, value: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(value).map_err(io::Error::other)?;
    write_http_response(stream, 200, "application/json", &body).await
}

async fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        202 => "Accepted",
        405 => "Method Not Allowed",
        _ => "Error",
    };
    let headers = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.shutdown().await
}

#[tokio::test]
async fn heart_http_client_discovers_and_calls_a_real_local_mcp_service() {
    let service = LocalMcpService::start().await.unwrap();
    let server = halo::agent::core::mcp::Server::http(
        Definition {
            name: "local-http-clock".into(),
            ..Definition::default()
        },
        &service.endpoint,
        Default::default(),
        &Cancellation::new(),
    )
    .await
    .expect("initialize the real local Streamable HTTP MCP service");

    assert_eq!(server.tools().len(), 1);
    assert_eq!(server.tools()[0].name, "current_time");

    let before = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let message = server
        .call(
            Call {
                call_id: "http-live-clock-call".into(),
                name: "current_time".into(),
                arguments: Default::default(),
            },
            &Cancellation::new(),
        )
        .await
        .expect("call the MCP service's live clock tool");
    let after = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let returned = message.values["unix_seconds"]
        .as_u64()
        .expect("MCP result should contain structured timestamp data");
    assert!((before..=after).contains(&returned));
    assert_eq!(message.metadata["call_id"], "http-live-clock-call");
    server.close().await.expect("close the MCP client session");
    service.stop().await;
    println!("LIVE_HTTP_MCP_CURRENT_TIME={returned}");
}
