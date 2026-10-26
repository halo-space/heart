//! Elasticsearch Workspace contract checks that do not require a local
//! protocol fixture. Cluster behavior is covered by `workspace_es_real`.

use std::time::Duration;

use elasticsearch::{
    Elasticsearch,
    http::transport::{SingleNodeConnectionPool, TransportBuilder},
};
use halo::agent::{Cancellation, tool::Call, workspace::Workspace as _};
use halo::core;
use serde_json::json;

fn unavailable_client() -> Elasticsearch {
    let pool = SingleNodeConnectionPool::new("http://127.0.0.1:1".parse().unwrap());
    Elasticsearch::new(
        TransportBuilder::new(pool)
            .disable_proxy()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap(),
    )
}

fn call(name: &str, arguments: serde_json::Value) -> Call {
    Call {
        call_id: format!("test-{name}"),
        name: format!("workspace_{name}"),
        arguments: arguments.as_object().unwrap().clone(),
    }
}

#[test]
fn es_workspace_assembly_is_lazy_and_exposes_only_database_operations() {
    let client = unavailable_client();
    let workspace = core::workspace::es::new(client.clone(), "records").unwrap();
    assert!(workspace.definition().path.is_none());
    let tools = workspace.tool();
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        [
            "workspace_write",
            "workspace_read",
            "workspace_update",
            "workspace_delete"
        ]
    );
    for tool in tools {
        tool.validate().unwrap();
        let properties = tool.parameters["properties"].as_object().unwrap();
        for removed in [
            "filter", "offset", "limit", "sort", "path", "index", "config", "schema",
        ] {
            assert!(!properties.contains_key(removed));
        }
    }
    for index in [
        "", "*", "a,b", "a/b", "a\\b", "_all", "a?", "a#x", "a:x", "a%2fb", ".", "..", "Bad", "a b",
    ] {
        assert_eq!(
            core::workspace::es::new(client.clone(), index)
                .err()
                .unwrap()
                .code,
            "INVALID_ARGUMENTS"
        );
    }
}

#[tokio::test]
async fn malformed_operations_are_rejected_before_any_network_request() {
    let workspace = core::workspace::es::new(unavailable_client(), "records").unwrap();
    let invalid = [
        ("write", json!({"data":{}})),
        ("write", json!({"data":{"id":"1"}})),
        ("read", json!({"id":1,"index":"other"})),
        ("read", json!({"id":1.5})),
        ("delete", json!({"id":"1"})),
        ("update", json!({"id":1,"data":[]})),
        ("update", json!({"id":1,"data":{"id":2}})),
    ];
    for (name, arguments) in invalid {
        let error = workspace
            .execute(call(name, arguments), &Cancellation::new())
            .await
            .expect_err("invalid input must fail before network access");
        assert_eq!(error.code, "INVALID_ARGUMENTS", "{name}: {error:?}");
    }
}

#[tokio::test]
async fn a_real_connection_refusal_is_returned_without_retry_or_secret_leakage() {
    let workspace = core::workspace::es::new(unavailable_client(), "records").unwrap();
    let error = workspace
        .execute(call("read", json!({"id":1})), &Cancellation::new())
        .await
        .expect_err("port 1 has no Elasticsearch service");
    assert_eq!(error.code, "TRANSPORT");
    let serialized = serde_json::to_string(&error).unwrap_or_else(|_| format!("{error:?}"));
    assert!(!serialized.contains("127.0.0.1:1"));
}
