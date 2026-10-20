//! ES protocol regressions use a local HTTP fixture and the official client.
//! They are not claims of running against an actual Elasticsearch cluster.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

use elasticsearch::{
    Elasticsearch,
    auth::Credentials,
    http::transport::{SingleNodeConnectionPool, TransportBuilder},
};
use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message,
    agent::{Agent, toolkit::Toolkit},
    core,
    event::Event,
    harness,
    message::Role,
    model::chat::{Chat, Request},
    tool::{Call, Tool, ToolDefinition},
    workspace::{Mode, Workspace},
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    task::JoinHandle,
};

#[derive(Debug)]
struct Recorded {
    method: String,
    target: String,
    body: Value,
}

struct Fixture {
    client: Elasticsearch,
    requests: Arc<Mutex<Vec<Recorded>>>,
    task: JoinHandle<()>,
}

impl Fixture {
    async fn new(replies: Vec<(u16, Value)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client = client(&format!("http://{}", listener.local_addr().unwrap()));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let task = tokio::spawn(async move {
            for (status, body) in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let request = receive(&mut socket).await;
                let is_head = request.method == "HEAD";
                captured.lock().unwrap().push(request);
                let body = if is_head {
                    String::new()
                } else {
                    body.to_string()
                };
                let headers = format!(
                    "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nX-Elastic-Product: Elasticsearch\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len(),
                );
                socket.write_all(headers.as_bytes()).await.unwrap();
                socket.write_all(body.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            }
        });
        Self {
            client,
            requests,
            task,
        }
    }

    fn workspace(&self) -> core::workspace::es::Workspace {
        core::workspace::es::new(self.client.clone(), "records").unwrap()
    }

    async fn finish(&mut self) {
        tokio::time::timeout(Duration::from_secs(5), &mut self.task)
            .await
            .expect("fixture did not receive the expected requests")
            .expect("fixture failed");
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn client(url: &str) -> Elasticsearch {
    let pool = SingleNodeConnectionPool::new(url.parse().unwrap());
    Elasticsearch::new(
        TransportBuilder::new(pool)
            .disable_proxy()
            .auth(Credentials::Basic(
                "fixture-user".into(),
                "fixture-secret".into(),
            ))
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap(),
    )
}

async fn receive(socket: &mut TcpStream) -> Recorded {
    let mut bytes = Vec::new();
    let mut buffer = [0; 2048];
    let (end, length) = loop {
        let count = socket.read(&mut buffer).await.unwrap();
        assert!(count > 0, "unexpected end of HTTP request");
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]);
            let length = headers
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            break (end + 4, length);
        }
    };
    while bytes.len() < end + length {
        let count = socket.read(&mut buffer).await.unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&buffer[..count]);
    }
    let headers = String::from_utf8_lossy(&bytes[..end]);
    let mut first = headers.lines().next().unwrap().split_whitespace();
    Recorded {
        method: first.next().unwrap().into(),
        target: first.next().unwrap().into(),
        body: if length == 0 {
            Value::Null
        } else {
            serde_json::from_slice(&bytes[end..end + length]).unwrap()
        },
    }
}

fn call(name: &str, arguments: Value) -> Call {
    Call {
        call_id: format!("call_{name}"),
        name: format!("workspace_{name}"),
        arguments: arguments.as_object().unwrap().clone(),
    }
}

async fn run(toolkit: &Toolkit, name: &str, args: Value) -> Result<Message, Error> {
    let (message, evaluation) = toolkit
        .execute(call(name, args), &Cancellation::new())
        .await?;
    assert!(evaluation.passed);
    assert_eq!(message.role, Role::Tool);
    assert_eq!(message.content.len(), 1);
    assert_eq!(message.content[0].r#type, "text");
    assert_eq!(
        serde_json::from_str::<Value>(message.content[0].data["value"].as_str().unwrap()).unwrap(),
        Value::Object(message.values.clone())
    );
    Ok(message)
}

fn document(data: Value) -> Value {
    json!({"found":true,"_source":data})
}

#[test]
fn es_assembly_has_no_file_methods_config_layer_or_eager_network_access() {
    let client = client("http://127.0.0.1:1");
    let workspace = core::workspace::es::new(client.clone(), "records").unwrap();
    assert_eq!(workspace.definition().mode, Mode::Server);
    assert!(workspace.definition().path.is_none());
    assert!(workspace.definition().metadata.is_empty());
    let toolkit = Toolkit::from(workspace);
    let definitions = toolkit.definitions();
    assert_eq!(
        definitions
            .iter()
            .map(|v| v.name.as_str())
            .collect::<Vec<_>>(),
        [
            "workspace_write",
            "workspace_read",
            "workspace_update",
            "workspace_delete"
        ]
    );
    for definition in definitions {
        definition.validate().unwrap();
        let properties = definition.parameters["properties"].as_object().unwrap();
        for removed in [
            "filter", "offset", "limit", "sort", "path", "index", "config", "schema",
        ] {
            assert!(!properties.contains_key(removed));
        }
        assert!(
            !serde_json::to_string(&definition)
                .unwrap()
                .contains("fixture-secret")
        );
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
async fn es_crud_uses_atomic_create_partial_update_physical_delete_and_stored_source() {
    let stored = json!({"id":3001,"content":"pipeline normalized", "kept":true});
    let updated = json!({"id":3001,"content":"new", "kept":true});
    let mut fixture = Fixture::new(vec![
        (200, Value::Null),
        (201, json!({"result":"created"})),
        (200, document(stored.clone())),
        (200, document(stored.clone())),
        (200, Value::Null),
        (200, json!({"result":"updated"})),
        (200, document(updated.clone())),
        (200, json!({"result":"deleted"})),
        (404, json!({"found":false})),
    ])
    .await;
    let toolkit = Toolkit::from(fixture.workspace().toolkit(Toolkit::new()));
    assert_eq!(
        run(
            &toolkit,
            "write",
            json!({"data":{"id":3001,"content":"raw","kept":true}})
        )
        .await
        .unwrap()
        .values,
        json!({"data":stored}).as_object().unwrap().clone()
    );
    assert_eq!(
        run(&toolkit, "read", json!({"id":3001}))
            .await
            .unwrap()
            .values["data"],
        json!([stored])
    );
    assert_eq!(
        run(
            &toolkit,
            "update",
            json!({"id":3001,"data":{"content":"new"}})
        )
        .await
        .unwrap()
        .values["count"],
        1
    );
    assert_eq!(
        run(&toolkit, "read", json!({"id":3001}))
            .await
            .unwrap()
            .values["data"],
        json!([updated])
    );
    assert_eq!(
        run(&toolkit, "delete", json!({"id":3001}))
            .await
            .unwrap()
            .values["count"],
        1
    );
    assert_eq!(
        run(&toolkit, "read", json!({"id":3001}))
            .await
            .unwrap()
            .values["data"],
        json!([])
    );
    fixture.finish().await;
    let requests = fixture.requests.lock().unwrap();
    assert_eq!(
        requests
            .iter()
            .map(|v| (v.method.as_str(), v.target.as_str()))
            .collect::<Vec<_>>(),
        [
            ("HEAD", "/records"),
            ("POST", "/records/_create/3001"),
            ("GET", "/records/_doc/3001"),
            ("GET", "/records/_doc/3001"),
            ("HEAD", "/records"),
            ("POST", "/records/_update/3001"),
            ("GET", "/records/_doc/3001"),
            ("DELETE", "/records/_doc/3001"),
            ("GET", "/records/_doc/3001"),
        ]
    );
    assert_eq!(
        requests[1].body,
        json!({"id":3001,"content":"raw","kept":true})
    );
    assert_eq!(
        requests[5].body,
        json!({"doc":{"content":"new"},"doc_as_upsert":false})
    );
}

#[tokio::test]
async fn es_noop_and_missing_records_return_zero_without_upsert_or_soft_delete() {
    let mut fixture = Fixture::new(vec![
        (200, Value::Null),
        (200, json!({"result":"noop"})),
        (200, Value::Null),
        (404, json!({"error":{"type":"document_missing_exception"}})),
        (404, json!({"result":"not_found"})),
        (404, json!({"found":false})),
    ])
    .await;
    let toolkit = Toolkit::from(fixture.workspace());
    for _ in 0..2 {
        assert_eq!(
            run(&toolkit, "update", json!({"id":1,"data":{"content":"new"}}))
                .await
                .unwrap()
                .values["count"],
            0
        );
    }
    assert_eq!(
        run(&toolkit, "delete", json!({"id":1}))
            .await
            .unwrap()
            .values["count"],
        0
    );
    assert_eq!(
        run(&toolkit, "read", json!({"id":1})).await.unwrap().values["data"],
        json!([])
    );
    fixture.finish().await;
}

#[tokio::test]
async fn es_create_conflict_is_direct_error_and_never_overwrites() {
    let mut fixture = Fixture::new(vec![
        (200, Value::Null),
        (409, json!({"error":{"reason":"fixture-secret"}})),
    ])
    .await;
    let error = run(
        &Toolkit::from(fixture.workspace()),
        "write",
        json!({"data":{"id":1}}),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code, "CONFLICT");
    assert!(
        !serde_json::to_string(&error)
            .unwrap()
            .contains("fixture-secret")
    );
    fixture.finish().await;
    assert_eq!(
        fixture.requests.lock().unwrap().len(),
        2,
        "no retry or readback after failed insert"
    );
}

#[tokio::test]
async fn es_missing_index_is_error_not_empty_success_or_automatic_provisioning() {
    for (name, args) in [
        ("write", json!({"data":{"id":1}})),
        ("read", json!({"id":1})),
        ("update", json!({"id":1,"data":{"content":"x"}})),
        ("delete", json!({"id":1})),
    ] {
        let mut fixture = Fixture::new(vec![(
            404,
            json!({"error":{"type":"index_not_found_exception"}}),
        )])
        .await;
        assert_eq!(
            run(&Toolkit::from(fixture.workspace()), name, args)
                .await
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
        fixture.finish().await;
        assert_eq!(fixture.requests.lock().unwrap().len(), 1);
        if matches!(name, "write" | "update") {
            assert_eq!(fixture.requests.lock().unwrap()[0].method, "HEAD");
        }
    }
}

#[tokio::test]
async fn es_schema_and_i64_identity_validation_precede_network_access() {
    let toolkit =
        Toolkit::from(core::workspace::es::new(client("http://127.0.0.1:1"), "records").unwrap());
    for (name, args) in [
        ("write", json!({"data":{}})),
        ("write", json!({"data":{"id":"1"}})),
        ("write", json!({"data":{"id":18446744073709551615u64}})),
        ("read", json!({"id":1,"filter":{}})),
        ("read", json!({"filter":{"id":1}})),
        ("read", json!({"id":1,"index":"other"})),
        ("read", json!({"id":1.5})),
        ("delete", json!({"id":"1"})),
        ("update", json!({"id":1,"data":[]})),
        ("update", json!({"id":1,"data":{"id":2}})),
        ("update", json!({"id":1,"data":{"id":null}})),
    ] {
        assert_eq!(
            run(&toolkit, name, args).await.unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
    }
}

#[tokio::test]
async fn es_ids_do_not_round_through_floats_or_get_positive_only_restrictions() {
    let mut fixture = Fixture::new(vec![(404, json!({"found":false})); 3]).await;
    let toolkit = Toolkit::from(fixture.workspace());
    for id in [i64::MIN, 0, i64::MAX] {
        run(&toolkit, "read", json!({"id":id})).await.unwrap();
    }
    fixture.finish().await;
    let requests = fixture.requests.lock().unwrap();
    assert_eq!(
        requests
            .iter()
            .map(|v| v.target.as_str())
            .collect::<Vec<_>>(),
        [
            "/records/_doc/-9223372036854775808",
            "/records/_doc/0",
            "/records/_doc/9223372036854775807",
        ]
    );
}

#[tokio::test]
async fn es_http_errors_are_categorized_sanitized_and_not_retried() {
    for (status, code) in [
        (400, "INVALID_ARGUMENTS"),
        (401, "AUTHENTICATION"),
        (403, "PERMISSION_DENIED"),
        (408, "TIMEOUT"),
        (429, "RATE_LIMITED"),
        (500, "UNAVAILABLE"),
        (503, "UNAVAILABLE"),
        (504, "TIMEOUT"),
        (404, "NOT_FOUND"),
    ] {
        let mut fixture = Fixture::new(vec![(
            status,
            json!({"error":{"type":"secret","reason":"fixture-secret"},"found":false}),
        )])
        .await;
        let error = run(&Toolkit::from(fixture.workspace()), "read", json!({"id":1}))
            .await
            .unwrap_err();
        assert_eq!(error.code, code);
        assert_eq!(error.details, Some(format!("http_status={status}")));
        assert!(
            !serde_json::to_string(&error)
                .unwrap()
                .contains("fixture-secret")
        );
        fixture.finish().await;
        assert_eq!(fixture.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn es_invalid_success_envelopes_are_not_reported_as_empty_data() {
    for body in [
        json!({}),
        json!({"found":true}),
        json!({"found":true,"_source":[]}),
        json!({"found":false,"_source":{}}),
    ] {
        let mut fixture = Fixture::new(vec![(200, body)]).await;
        assert_eq!(
            run(&Toolkit::from(fixture.workspace()), "read", json!({"id":1}))
                .await
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
        fixture.finish().await;
    }
    let mut fixture =
        Fixture::new(vec![(200, Value::Null), (200, json!({"result":"created"}))]).await;
    assert_eq!(
        run(
            &Toolkit::from(fixture.workspace()),
            "update",
            json!({"id":1,"data":{}})
        )
        .await
        .unwrap_err()
        .code,
        "INVALID_RESPONSE"
    );
    fixture.finish().await;
}

#[tokio::test]
async fn es_update_preflight_failure_never_sends_mutation() {
    for (status, code) in [
        (404, "NOT_FOUND"),
        (403, "PERMISSION_DENIED"),
        (503, "UNAVAILABLE"),
    ] {
        let mut fixture = Fixture::new(vec![(status, Value::Null)]).await;
        let error = run(
            &Toolkit::from(fixture.workspace()),
            "update",
            json!({"id":1,"data":{"content":"must not dispatch"}}),
        )
        .await
        .unwrap_err();
        assert_eq!(error.code, code);
        fixture.finish().await;
        let requests = fixture.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].method, "HEAD");
        assert_eq!(requests[0].target, "/records");
    }
}

#[tokio::test]
async fn es_cancellation_before_send_and_during_headers_or_body_does_not_hang() {
    let toolkit =
        Toolkit::from(core::workspace::es::new(client("http://127.0.0.1:1"), "records").unwrap());
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        toolkit
            .execute(call("read", json!({"id":1})), &cancellation)
            .await
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    for body_started in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let toolkit = Toolkit::from(
            core::workspace::es::new(
                client(&format!("http://{}", listener.local_addr().unwrap())),
                "records",
            )
            .unwrap(),
        );
        let (ready, received) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            receive(&mut socket).await;
            if body_started {
                socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\nContent-Type: application/json\r\nX-Elastic-Product: Elasticsearch\r\n\r\n{\"found\":").await.unwrap();
            }
            ready.send(()).unwrap();
            let mut buffer = [0; 32];
            let _ = socket.read(&mut buffer).await;
        });
        let cancellation = Cancellation::new();
        let signal = cancellation.clone();
        let operation = tokio::spawn(async move {
            toolkit
                .execute(call("read", json!({"id":1})), &signal)
                .await
        });
        tokio::time::timeout(Duration::from_secs(5), received)
            .await
            .unwrap()
            .unwrap();
        cancellation.cancel();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), operation)
                .await
                .unwrap()
                .unwrap()
                .unwrap_err()
                .code,
            "CANCELLED"
        );
        server.abort();
    }
}

#[tokio::test]
async fn es_transport_failure_has_no_credentials_url_or_raw_driver_details() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let client = client(&format!("http://{}", listener.local_addr().unwrap()));
    drop(listener);
    let error = run(
        &Toolkit::from(core::workspace::es::new(client, "records").unwrap()),
        "read",
        json!({"id":1}),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code, "TRANSPORT");
    assert_eq!(error.details, None);
    assert!(!error.message.contains("http"));
}

#[tokio::test]
async fn es_non_json_responses_preserve_http_errors_without_echoing_body() {
    for (status, code) in [(200, "INVALID_RESPONSE"), (403, "PERMISSION_DENIED")] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let toolkit = Toolkit::from(
            core::workspace::es::new(
                client(&format!("http://{}", listener.local_addr().unwrap())),
                "records",
            )
            .unwrap(),
        );
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            receive(&mut socket).await;
            let body = "<html>fixture-secret</html>";
            socket.write_all(format!(
                "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len(),
            ).as_bytes()).await.unwrap();
        });
        let error = run(&toolkit, "read", json!({"id":1})).await.unwrap_err();
        assert_eq!(error.code, code);
        assert!(
            !serde_json::to_string(&error)
                .unwrap()
                .contains("fixture-secret")
        );
        server.await.unwrap();
    }
}

#[tokio::test]
async fn es_write_readback_failure_is_direct_error_not_fabricated_inserted_data() {
    let mut fixture = Fixture::new(vec![
        (200, Value::Null),
        (201, json!({"result":"created"})),
        (503, json!({"error":{"reason":"unavailable"}})),
    ])
    .await;
    assert_eq!(
        run(
            &Toolkit::from(fixture.workspace()),
            "write",
            json!({"data":{"id":1}})
        )
        .await
        .unwrap_err()
        .code,
        "UNAVAILABLE"
    );
    fixture.finish().await;
    let requests = fixture.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests[2].method, "GET",
        "no rollback/delete or retry after an acknowledged write"
    );
}

/// Opt-in only. The application must prepare an isolated existing index and
/// grant document CRUD, index metadata reads and _source reads. This test does
/// not create/change indices or mappings; it deletes only its generated ID.
#[tokio::test]
#[ignore = "requires HEART_ES_URL and HEART_ES_TEST_INDEX for an isolated real ES cluster"]
async fn es_real_cluster_crud_smoke() {
    let url = std::env::var("HEART_ES_URL").expect("set HEART_ES_URL");
    let index = std::env::var("HEART_ES_TEST_INDEX")
        .expect("set HEART_ES_TEST_INDEX to an isolated pre-created index");
    let mut transport = TransportBuilder::new(SingleNodeConnectionPool::new(url.parse().unwrap()))
        .disable_proxy()
        .timeout(Duration::from_secs(15));
    if let Ok(key) = std::env::var("HEART_ES_API_KEY") {
        transport = transport.auth(Credentials::EncodedApiKey(key));
    }
    let client = Elasticsearch::new(transport.build().unwrap());
    let toolkit = Toolkit::from(core::workspace::es::new(client, index).unwrap());
    // Test/application supplies identity; the framework does not allocate it.
    let id = rand::random::<i64>();
    let data = json!({"id":id,"content":"heart workspace smoke","metadata":{"kept":true}});
    run(&toolkit, "write", json!({"data":data})).await.unwrap();
    let conflict = run(&toolkit, "write", json!({"data":{"id":id}})).await;
    let before = run(&toolkit, "read", json!({"id":id})).await;
    let changed = run(
        &toolkit,
        "update",
        json!({"id":id,"data":{"content":"updated"}}),
    )
    .await;
    let after = run(&toolkit, "read", json!({"id":id})).await;
    // Perform the scoped cleanup before assertions on the preceding results.
    let deleted = run(&toolkit, "delete", json!({"id":id})).await.unwrap();
    let missing = run(&toolkit, "read", json!({"id":id})).await.unwrap();
    let no_upsert = run(
        &toolkit,
        "update",
        json!({"id":id,"data":{"content":"absent"}}),
    )
    .await
    .unwrap();
    assert_eq!(conflict.unwrap_err().code, "CONFLICT");
    assert_eq!(before.unwrap().values["data"], json!([data]));
    assert_eq!(changed.unwrap().values["count"], 1);
    let after = after.unwrap();
    assert_eq!(after.values["data"][0]["metadata"], json!({"kept":true}));
    assert_eq!(after.values["data"][0]["content"], "updated");
    assert_eq!(deleted.values["count"], 1);
    assert_eq!(missing.values["data"], json!([]));
    assert_eq!(no_upsert.values["count"], 0);
}

struct Model {
    output: Mutex<VecDeque<Message>>,
    requests: Arc<Mutex<Vec<Request>>>,
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.requests.lock().unwrap().push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: self
                .output
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected model call"),
            usage: None,
            finish_reason: None,
        })]))
    }
}

fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message
        .content
        .push(halo_agents::model::content::part::Part {
            r#type: "text".into(),
            data: json!({"value":value}).as_object().unwrap().clone(),
        });
    message
}

struct Custom(ToolDefinition);
impl Tool for Custom {
    fn definition(&self) -> &ToolDefinition {
        &self.0
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        Ok(text(Role::Tool, "custom query result"))
    }
}

#[tokio::test]
async fn es_harness_uses_existing_model_loop_permission_and_custom_toolkit() {
    for allowed in [true, false] {
        let mut fixture = Fixture::new(if allowed {
            vec![(200, document(json!({"id":1,"content":"saved"})))]
        } else {
            vec![]
        })
        .await;
        let requests = Arc::new(Mutex::new(Vec::new()));
        let mut selection = Message::new(Role::Assistant);
        selection
            .content
            .push(halo_agents::model::content::part::Part {
                r#type: "tool_call".into(),
                data: serde_json::to_value(call("read", json!({"id":1})))
                    .unwrap()
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        let model = Model {
            output: Mutex::new(vec![selection, text(Role::Assistant, "done")].into()),
            requests: Arc::clone(&requests),
        };
        let custom = Custom(ToolDefinition {
            name: "custom_query".into(),
            description: "Application-specific query".into(),
            parameters: json!({"type":"object"}).as_object().unwrap().clone(),
        });
        let permission_calls = Arc::new(Mutex::new(Vec::new()));
        let checked = Arc::clone(&permission_calls);
        let permission = core::permission::Permission::new(
            move |request: halo_agents::permission::Request, _: Cancellation| {
                checked.lock().unwrap().push(request);
                async move {
                    Ok(halo_agents::permission::Decision {
                        behavior: if allowed {
                            halo_agents::permission::Behavior::Allow
                        } else {
                            halo_agents::permission::Behavior::Deny
                        },
                        reason: None,
                    })
                }
            },
        );
        let agent = harness::builder()
            .model(model)
            .workspace(fixture.workspace().toolkit(Toolkit::new().tools(custom)))
            .permission(permission)
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(
                halo_agents::agent::Input {
                    metadata: json!({"chat_id":101,"agent_id":201,"session_id":301})
                        .as_object()
                        .unwrap()
                        .clone(),
                    payload: vec![halo_agents::agent::Payload {
                        r#type: "text".into(),
                        data: json!({"value":"read record 1"}),
                        mime_type: None,
                    }],
                },
                false,
                Cancellation::new(),
            )
            .await
            .unwrap()
            .collect()
            .await;
        assert!(events.iter().all(Result::is_ok));
        fixture.finish().await;
        assert_eq!(fixture.requests.lock().unwrap().len(), usize::from(allowed));
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let names: Vec<_> = requests[0].tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names.len(),
            6,
            "four CRUD tools, the custom Tool and built-in plan"
        );
        for name in [
            "custom_query",
            "workspace_read",
            "workspace_write",
            "workspace_update",
            "workspace_delete",
            "plan",
        ] {
            assert!(names.contains(&name));
        }
        assert!(!names.contains(&"workspace_glob"));
        assert!(!permission_calls.lock().unwrap().is_empty());
        if allowed {
            assert!(events.iter().any(|e|matches!(e,Ok(Event::Complete {message,..}) if message.values.get("data")==Some(&json!([{"id":1,"content":"saved"}])))));
        }
    }
}
