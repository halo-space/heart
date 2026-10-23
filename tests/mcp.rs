use halo::{
    Cancellation, core,
    mcp::{Definition, Server as _},
    message::Role,
    tool::Call,
};
use serde_json::{Map, Value, json};
use std::{sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::{Mutex, Notify},
    task::{JoinHandle, JoinSet},
};

#[derive(Clone, Copy)]
enum Mode {
    Json,
    Sse,
    ToolError,
    StructuredError,
    RpcError,
    Cycle,
    Media,
    Pending,
}
struct Fixture {
    url: String,
    requests: Arc<Mutex<Vec<(String, Value)>>>,
    called: Arc<Notify>,
    task: JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Fixture {
    async fn new(mode: Mode) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let records = requests.clone();
        let called = Arc::new(Notify::new());
        let notified = called.clone();
        let task = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            loop {
                tokio::select! {
                    accepted=listener.accept() => {
                        let Ok((mut socket,_))=accepted else { break; };
                        let records=records.clone();let notified=notified.clone();
                        connections.spawn(async move {
                            let mut bytes=Vec::new();let mut buffer=[0;2048];
                            let (end,length)=loop {
                                let count=socket.read(&mut buffer).await.unwrap();if count==0 { return; }
                                bytes.extend_from_slice(&buffer[..count]);
                                if let Some(end)=bytes.windows(4).position(|part|part==b"\r\n\r\n") {
                                    let headers=String::from_utf8_lossy(&bytes[..end]);
                                    let length=headers.lines().find_map(|line| {
                                        let (key,value)=line.split_once(':')?;
                                        key.eq_ignore_ascii_case("content-length").then(||value.trim().parse::<usize>().unwrap())
                                    }).unwrap_or(0);
                                    break (end+4,length);
                                }
                            };
                            while bytes.len()<end+length { let count=socket.read(&mut buffer).await.unwrap();if count==0 { return; }bytes.extend_from_slice(&buffer[..count]); }
                            let headers=String::from_utf8(bytes[..end].to_vec()).unwrap();
                            let request:Value=if length>0 { serde_json::from_slice(&bytes[end..end+length]).unwrap() } else { Value::Null };
                            records.lock().await.push((headers.clone(),request.clone()));
                            let method=request["method"].as_str().unwrap_or("");let id=request["id"].clone();
                            let result=match method {
                                "initialize" => json!({"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"1"}}),
                                "tools/list" => {
                                    let next=request["params"]["cursor"].is_string();let name=if next { "forecast" } else { "weather" };
                                    let mut result=json!({"tools":[{"name":name,"description":"Weather lookup","inputSchema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]});
                                    if !next || matches!(mode,Mode::Cycle) { result["nextCursor"]=json!("page2"); }result
                                }
                                "tools/call" => {
                                    notified.notify_one();if matches!(mode,Mode::Pending) { std::future::pending::<()>().await; }
                                    match mode {
                                        Mode::ToolError => json!({"isError":true,"content":[{"type":"text","text":"city missing"}]}),
                                        Mode::StructuredError => json!({"isError":true,"content":[],"structuredContent":{"code":"INVALID_ARGUMENTS","message":"city missing","details":"city"}}),
                                        Mode::Media => json!({"content":[{"type":"image","data":"AA==","mimeType":"image/png"}]}),
                                        _ => json!({"content":[{"type":"text","text":"北京晴天"}],"structuredContent":{"temperature":22}}),
                                    }
                                },
                                _ => Value::Null,
                            };
                            let (status,content_type,body)=if headers.starts_with("GET ") { (405,"application/json",String::new()) }
                                else if headers.starts_with("DELETE ") || id.is_null() { (202,"application/json",String::new()) }
                                else {
                                    let rpc=if matches!(mode,Mode::RpcError) && method=="tools/call" { json!({"jsonrpc":"2.0","id":id,"error":{"code":-32602,"message":"fixture-secret"}}) }
                                    else { json!({"jsonrpc":"2.0","id":id,"result":result}) };
                                    if matches!(mode,Mode::Sse) && method=="tools/call" {
                                        (200,"text/event-stream",format!("event: message\ndata: {{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{{\"level\":\"info\",\"data\":\"progress\"}}}}\n\nevent: message\ndata: {rpc}\n\n"))
                                    } else { (200,"application/json",rpc.to_string()) }
                                };
                            let session=if method=="initialize" { "Mcp-Session-Id: fixture-session\r\n" } else { "" };
                            let response=format!("HTTP/1.1 {status} Fixture\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{session}Connection: close\r\n\r\n",body.len());
                            if socket.write_all(response.as_bytes()).await.is_err() { return; }
                            for fragment in body.as_bytes().chunks(7) { if socket.write_all(fragment).await.is_err() { return; }tokio::task::yield_now().await; }
                        });
                    }
                    _=connections.join_next(), if !connections.is_empty() => {},
                }
            }
        });
        Self {
            url,
            requests,
            called,
            task,
        }
    }
}
fn definition() -> Definition {
    Definition {
        name: "weather".into(),
        ..Definition::default()
    }
}
fn call() -> Call {
    Call {
        call_id: "call_001".into(),
        name: "weather".into(),
        arguments: json!({"city":"北京"}).as_object().unwrap().clone(),
    }
}

#[tokio::test]
async fn mcp_http_initializes_discovers_pages_and_returns_direct_message() {
    let fixture = Fixture::new(Mode::Json).await;
    let cancel = Cancellation::new();
    let server = core::mcp::Server::http(
        definition(),
        &fixture.url,
        json!({"Authorization":"Bearer fixture-secret"})
            .as_object()
            .unwrap()
            .clone(),
        &cancel,
    )
    .await
    .unwrap();
    let tools = server.tools();
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["weather", "forecast"]
    );
    let message = server.call(call(), &cancel).await.unwrap();
    assert_eq!(message.role, Role::Tool);
    assert_eq!(message.values["temperature"], json!(22));
    assert_eq!(message.content[0].data["value"], json!("北京晴天"));
    assert_eq!(message.metadata["call_id"], json!("call_001"));
    assert!(
        message
            .content
            .iter()
            .all(|part| part.r#type != "tool_result")
    );
    assert!(
        !serde_json::to_string(server.definition())
            .unwrap()
            .contains("fixture-secret")
    );
    server.close().await.unwrap();
    let records = fixture.requests.lock().await;
    let requests = records
        .iter()
        .filter(|(_, body)| body["method"] == "tools/list" || body["method"] == "tools/call")
        .collect::<Vec<_>>();
    assert_eq!(requests.len(), 3);
    for (headers, _) in requests {
        let headers = headers.to_ascii_lowercase();
        assert!(headers.contains("mcp-session-id: fixture-session"));
        assert!(headers.contains("mcp-protocol-version: 2025-11-25"));
        assert!(headers.contains("authorization: bearer fixture-secret"));
    }
    assert!(
        records
            .iter()
            .any(|(_, body)| body["method"] == "notifications/initialized")
    );
}
#[tokio::test]
async fn mcp_sse_ignores_notifications_and_reassembles_fragmented_utf8() {
    let fixture = Fixture::new(Mode::Sse).await;
    let cancel = Cancellation::new();
    let server = core::mcp::Server::http(definition(), &fixture.url, Map::new(), &cancel)
        .await
        .unwrap();
    assert_eq!(
        server.call(call(), &cancel).await.unwrap().content[0].data["value"],
        json!("北京晴天")
    );
    server.close().await.unwrap();
}
#[tokio::test]
async fn mcp_tool_failure_is_direct_error_not_success_message() {
    for mode in [Mode::ToolError, Mode::StructuredError] {
        let fixture = Fixture::new(mode).await;
        let cancel = Cancellation::new();
        let server = core::mcp::Server::http(definition(), &fixture.url, Map::new(), &cancel)
            .await
            .unwrap();
        let error = server.call(call(), &cancel).await.unwrap_err();
        assert_eq!(error.message, "city missing");
        if matches!(mode, Mode::StructuredError) {
            assert_eq!(error.code, "INVALID_ARGUMENTS");
            assert_eq!(error.details.as_deref(), Some("city"));
        } else {
            assert_eq!(error.code, "TOOL_ERROR");
        }
        server.close().await.unwrap();
    }
}
#[tokio::test]
async fn mcp_rpc_error_does_not_leak_remote_protocol_body() {
    let fixture = Fixture::new(Mode::RpcError).await;
    let cancel = Cancellation::new();
    let server = core::mcp::Server::http(definition(), &fixture.url, Map::new(), &cancel)
        .await
        .unwrap();
    let error = server.call(call(), &cancel).await.unwrap_err();
    assert_eq!(error.code, "MCP_ERROR");
    assert!(!format!("{error:?}").contains("fixture-secret"));
    server.close().await.unwrap();
}
#[tokio::test]
async fn mcp_repeated_cursor_fails_without_partial_discovery() {
    let fixture = Fixture::new(Mode::Cycle).await;
    let cancel = Cancellation::new();
    let error = match core::mcp::Server::http(definition(), &fixture.url, Map::new(), &cancel).await
    {
        Ok(_) => panic!("cyclic discovery must prevent server construction"),
        Err(error) => error,
    };
    assert_eq!(error.code, "INVALID_RESPONSE");
}
#[tokio::test]
async fn mcp_media_requires_v3_contract_instead_of_silent_drop() {
    let fixture = Fixture::new(Mode::Media).await;
    let cancel = Cancellation::new();
    let server = core::mcp::Server::http(definition(), &fixture.url, Map::new(), &cancel)
        .await
        .unwrap();
    assert_eq!(
        server.call(call(), &cancel).await.unwrap_err().code,
        "UNSUPPORTED"
    );
    server.close().await.unwrap();
}
#[tokio::test]
async fn mcp_cancellation_wakes_pending_request_and_notifies_server() {
    let fixture = Fixture::new(Mode::Pending).await;
    let cancel = Cancellation::new();
    let server = core::mcp::Server::http(definition(), &fixture.url, Map::new(), &cancel)
        .await
        .unwrap();
    let token = cancel.clone();
    let called = fixture.called.clone();
    let task = tokio::spawn(async move {
        called.notified().await;
        token.cancel();
    });
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(3), server.call(call(), &cancel))
            .await
            .unwrap()
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    task.await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if fixture
                .requests
                .lock()
                .await
                .iter()
                .any(|(_, body)| body["method"] == "notifications/cancelled")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    server.close().await.unwrap();
}
#[tokio::test]
async fn mcp_rejects_bad_endpoint_headers_and_pre_cancelled_initialization() {
    let cancel = Cancellation::new();
    for endpoint in [
        "file:///tmp/x",
        "http://user:pass@localhost/mcp",
        "not a URL",
    ] {
        assert_eq!(
            core::mcp::Server::http(definition(), endpoint, Map::new(), &cancel)
                .await
                .err()
                .unwrap()
                .code,
            "INVALID_ARGUMENTS"
        );
    }
    for headers in [
        json!({"Mcp-Session-Id":"bad"}),
        json!({"Authorization":100}),
        json!({"Host":"override"}),
    ] {
        assert_eq!(
            core::mcp::Server::http(
                definition(),
                "http://127.0.0.1:1/mcp",
                headers.as_object().unwrap().clone(),
                &cancel
            )
            .await
            .err()
            .unwrap()
            .code,
            "INVALID_ARGUMENTS"
        );
    }
    cancel.cancel();
    assert_eq!(
        core::mcp::Server::http(definition(), "http://127.0.0.1:1/mcp", Map::new(), &cancel)
            .await
            .err()
            .unwrap()
            .code,
        "CANCELLED"
    );
}
#[cfg(unix)]
#[tokio::test]
async fn mcp_stdio_starts_explicit_command_and_calls_tool() {
    // Static POSIX fixture only, not a Python bridge or production shell wrapper.
    let script = r#"while IFS= read -r line; do
id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
case "$line" in
*'"method":"initialize"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"1"}}}\n' "$id";;
*'"method":"tools/list"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"weather","description":"Weather lookup","inputSchema":{"type":"object"}}]}}\n' "$id";;
*'"method":"tools/call"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"sunny"}],"structuredContent":{"temperature":22}}}\n' "$id";;
esac
done"#;
    let mut command = tokio::process::Command::new("/bin/sh");
    command.arg("-c").arg(script);
    let cancel = Cancellation::new();
    let server = core::mcp::Server::stdio(definition(), command, &cancel)
        .await
        .unwrap();
    assert_eq!(server.tools().len(), 1);
    assert_eq!(
        server.call(call(), &cancel).await.unwrap().values["temperature"],
        json!(22)
    );
    server.close().await.unwrap();
}
