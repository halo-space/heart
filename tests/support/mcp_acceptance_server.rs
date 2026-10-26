//! A small, functional MCP stdio service for end-to-end client acceptance.
//! Its current_time tool reads the actual host clock; it does not replay data.

use std::io::{self, BufRead, Write};

use serde_json::{Value, json};

fn result(method: &str, params: &Value) -> Value {
    match method {
        "initialize" => json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "heart-local-acceptance", "version": "0.1.0"}
        }),
        "tools/list" => json!({
            "tools": [{
                "name": "current_time",
                "description": "Read the current Unix timestamp from this host",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }]
        }),
        "tools/call" if params["name"] == "current_time" => {
            let unix_seconds = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or_default();
            json!({
                "content": [{"type": "text", "text": format!("Unix time: {unix_seconds}")}],
                "structuredContent": {"unix_seconds": unix_seconds}
            })
        }
        _ => json!({
            "isError": true,
            "content": [{"type": "text", "text": format!("Unsupported MCP method or tool: {method}")}]
        }),
    }
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = serde_json::from_str(&line)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let Some(id) = request.get("id").filter(|id| !id.is_null()).cloned() else {
            // MCP notifications, including notifications/initialized, do not
            // receive JSON-RPC responses.
            continue;
        };
        let method = request["method"].as_str().unwrap_or_default();
        let response = json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result(method, &request["params"])
        });
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}
