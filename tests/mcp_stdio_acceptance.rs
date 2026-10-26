//! End-to-end MCP stdio acceptance against a real child service process.

use std::time::{SystemTime, UNIX_EPOCH};

use halo::agent::{
    Cancellation,
    mcp::{Definition, Server as _},
    tool::Call,
};
use serde_json::Map;

#[tokio::test]
async fn local_mcp_service_discovers_tools_and_returns_live_host_data() {
    let server = halo::agent::core::mcp::Server::stdio(
        Definition {
            name: "local-clock".into(),
            metadata: Map::new(),
            ..Definition::default()
        },
        tokio::process::Command::new(env!("CARGO_BIN_EXE_heart_mcp_acceptance_server")),
        &Cancellation::new(),
    )
    .await
    .expect("initialize the local MCP service and discover its tools");

    let discovered = server.tools();
    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].name, "current_time");

    let before = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let message = server
        .call(
            Call {
                call_id: "live-clock-call".into(),
                name: "current_time".into(),
                arguments: Map::new(),
            },
            &Cancellation::new(),
        )
        .await
        .expect("execute the real local MCP tool");
    let after = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let returned = message.values["unix_seconds"]
        .as_u64()
        .expect("MCP tool should return structured timestamp data");
    assert!((before..=after).contains(&returned));
    assert_eq!(message.metadata["call_id"], "live-clock-call");
    println!("LIVE_MCP_CURRENT_TIME={returned}");

    server.close().await.expect("close local MCP service");
}
