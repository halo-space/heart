//! Workspace MCP acceptance against a separate, functional stdio service.
//! The service reads the host clock instead of replaying a fixture response.

use std::time::{SystemTime, UNIX_EPOCH};

use halo::agent::{
    Cancellation, core::workspace::Local, mcp::Definition, tool::Call, workspace::Workspace as _,
};
use tokio::process::Command;

async fn live_workspace() -> Local {
    let server = halo::agent::core::mcp::Server::stdio(
        Definition {
            name: "local-clock".into(),
            ..Definition::default()
        },
        Command::new(env!("CARGO_BIN_EXE_heart_mcp_acceptance_server")),
        &Cancellation::new(),
    )
    .await
    .expect("connect to local MCP service and discover its tools");
    Local::new(std::env::temp_dir())
        .expect("open local Workspace")
        .mcps([server])
}

#[tokio::test]
async fn workspace_discovers_and_executes_a_real_mcp_tool() {
    let workspace = live_workspace().await;
    let tools = workspace.tool();
    assert!(tools.iter().any(|tool| tool.name == "current_time"));

    let before = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_secs();
    let (message, evaluation) = workspace
        .execute(
            Call {
                call_id: "workspace-live-clock".into(),
                name: "current_time".into(),
                arguments: Default::default(),
            },
            &Cancellation::new(),
        )
        .await
        .expect("execute current_time over MCP");
    let after = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_secs();

    let returned = message.values["unix_seconds"]
        .as_u64()
        .expect("MCP response should contain the measured timestamp");
    assert!((before..=after).contains(&returned));
    assert_eq!(message.metadata["call_id"], "workspace-live-clock");
    assert!(evaluation.passed);
    println!("LIVE_WORKSPACE_MCP_TIMESTAMP={returned}");
}

#[tokio::test]
async fn workspace_rejects_unknown_mcp_tools_before_remote_dispatch() {
    let workspace = live_workspace().await;
    let error = workspace
        .execute(
            Call {
                call_id: "unknown-tool".into(),
                name: "unknown_tool".into(),
                arguments: Default::default(),
            },
            &Cancellation::new(),
        )
        .await
        .expect_err("unknown tool should not be dispatched to MCP");
    assert_eq!(error.code, "NOT_FOUND");
}
