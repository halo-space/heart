//! Opt-in actual Elasticsearch acceptance. The caller prepares the cluster,
//! isolated base index and optional pipeline index; Workspace never provisions
//! them. No tests use a production index or provision a Docker service.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

use elasticsearch::{
    Elasticsearch,
    auth::Credentials,
    http::transport::{SingleNodeConnectionPool, TransportBuilder},
    indices::{IndicesDeleteParts, IndicesExistsParts},
};
use futures_util::{StreamExt, future::join_all, stream};
use halo::{
    Cancellation, Error, Message,
    agent::{Agent, Input, Payload, toolkit::Toolkit},
    core,
    event::Event,
    harness,
    message::Role,
    model::chat::{Chat, Request},
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Value, json};

fn environment() -> (Elasticsearch, String) {
    let url = std::env::var("HEART_ES_URL").expect("set HEART_ES_URL");
    let index = std::env::var("HEART_ES_TEST_INDEX").expect("provide an isolated existing index");
    let mut transport = TransportBuilder::new(SingleNodeConnectionPool::new(url.parse().unwrap()))
        .disable_proxy()
        .timeout(Duration::from_secs(15));
    if let Ok(key) = std::env::var("HEART_ES_API_KEY") {
        transport = transport.auth(Credentials::EncodedApiKey(key));
    }
    (Elasticsearch::new(transport.build().unwrap()), index)
}

fn call(name: &str, args: Value) -> Call {
    Call {
        call_id: format!("es_acceptance_{name}"),
        name: format!("workspace_{name}"),
        arguments: args.as_object().unwrap().clone(),
    }
}

async fn run(capabilities: &Toolkit, name: &str, args: Value) -> Result<Message, Error> {
    let (message, evaluation) = capabilities
        .execute(call(name, args), &Cancellation::new())
        .await?;
    assert!(evaluation.passed);
    assert_eq!(message.role, Role::Tool);
    assert_eq!(
        serde_json::from_str::<Value>(message.content[0].data["value"].as_str().unwrap()).unwrap(),
        Value::Object(message.values.clone())
    );
    Ok(message)
}

#[tokio::test]
#[ignore = "requires application-prepared HEART_ES_URL and isolated HEART_ES_TEST_INDEX"]
async fn real_i64_extremes_noop_and_physical_delete() {
    let (client, index) = environment();
    let capabilities = Toolkit::from(core::workspace::es::new(client, index).unwrap());
    for id in [i64::MIN, i64::MAX, -1, 0] {
        let data = json!({"id":id,"content":"边界测试","metadata":{"kept":true}});
        // A conflicting pre-existing record is NOT owned by this test. Stop
        // before cleanup instead of deleting somebody else's record.
        let written = run(&capabilities, "write", json!({"data":data}))
            .await
            .unwrap();
        let read = run(&capabilities, "read", json!({"id":id})).await;
        let noop = run(
            &capabilities,
            "update",
            json!({"id":id,"data":{"content":"边界测试"}}),
        )
        .await;
        let changed_id = run(
            &capabilities,
            "update",
            json!({"id":id,"data":{"id":if id == 0 {1} else {0}}}),
        )
        .await;
        let patched = run(
            &capabilities,
            "update",
            json!({"id":id,"data":{"content":"changed"}}),
        )
        .await;
        let after = run(&capabilities, "read", json!({"id":id})).await;
        // Perform cleanup before checking intermediate results.
        let deleted = run(&capabilities, "delete", json!({"id":id}))
            .await
            .unwrap();
        let absent = run(&capabilities, "read", json!({"id":id})).await.unwrap();
        let repeated_delete = run(&capabilities, "delete", json!({"id":id}))
            .await
            .unwrap();
        let no_upsert = run(
            &capabilities,
            "update",
            json!({"id":id,"data":{"content":"must not insert"}}),
        )
        .await
        .unwrap();
        assert_eq!(written.values["data"], data);
        assert_eq!(read.unwrap().values["data"], json!([data]));
        assert_eq!(noop.unwrap().values["count"], 0);
        assert_eq!(changed_id.unwrap_err().code, "INVALID_ARGUMENTS");
        assert_eq!(patched.unwrap().values["count"], 1);
        let after = after.unwrap();
        assert_eq!(after.values["data"][0]["id"].as_i64(), Some(id));
        assert_eq!(after.values["data"][0]["metadata"]["kept"], true);
        assert_eq!(after.values["data"][0]["content"], "changed");
        assert_eq!(deleted.values["count"], 1);
        assert_eq!(absent.values["data"], json!([]));
        assert_eq!(repeated_delete.values["count"], 0);
        assert_eq!(no_upsert.values["count"], 0);
        assert_eq!(
            run(&capabilities, "read", json!({"id":id}))
                .await
                .unwrap()
                .values["data"],
            json!([])
        );
    }
}

#[tokio::test]
#[ignore = "requires application-prepared HEART_ES_URL and isolated HEART_ES_TEST_INDEX"]
async fn real_concurrent_create_has_one_winner() {
    let (client, index) = environment();
    let capabilities = Toolkit::from(core::workspace::es::new(client, index).unwrap());
    let id = rand::random::<i64>();
    let results = join_all((0..8).map(|writer| {
        run(
            &capabilities,
            "write",
            json!({"data":{"id":id,"content":format!("writer {writer}")}}),
        )
    }))
    .await;
    let winners: Vec<_> = results
        .iter()
        .filter_map(|result| result.as_ref().ok())
        .collect();
    // No successful create => the test cannot claim ownership of this ID.
    assert!(!winners.is_empty(), "no owned record; not deleting ID {id}");
    let saved = run(&capabilities, "read", json!({"id":id})).await;
    let cleanup = run(&capabilities, "delete", json!({"id":id})).await;
    assert_eq!(winners.len(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| result.as_ref().is_err_and(|error| error.code == "CONFLICT"))
            .count(),
        7
    );
    assert_eq!(
        saved.unwrap().values["data"],
        json!([winners[0].values["data"]])
    );
    assert_eq!(cleanup.unwrap().values["count"], 1);
}

#[tokio::test]
#[ignore = "requires application-prepared HEART_ES_URL; uses only a new random absent-index name"]
async fn real_missing_index_is_error_without_provisioning() {
    let (client, _) = environment();
    let index = format!("heart-workspace-absent-{:016x}", rand::random::<u64>());
    assert_eq!(
        client
            .indices()
            .exists(IndicesExistsParts::Index(&[&index]))
            .send()
            .await
            .unwrap()
            .status_code()
            .as_u16(),
        404
    );
    let capabilities = Toolkit::from(core::workspace::es::new(client.clone(), &index).unwrap());
    for (name, args) in [
        ("read", json!({"id":1})),
        ("write", json!({"data":{"id":1}})),
        ("update", json!({"id":1,"data":{"content":"absent"}})),
        ("delete", json!({"id":1})),
    ] {
        let result = run(&capabilities, name, args).await;
        let status = client
            .indices()
            .exists(IndicesExistsParts::Index(&[&index]))
            .send()
            .await
            .unwrap()
            .status_code()
            .as_u16();
        // If this detects a regression, remove only this newly chosen name,
        // whose nonexistence the test verified before making any request.
        if status == 200 {
            let response = client
                .indices()
                .delete(IndicesDeleteParts::Index(&[&index]))
                .send()
                .await
                .unwrap();
            assert!(response.status_code().is_success());
        }
        assert_eq!(status, 404, "{name} must not auto-create the missing index");
        assert_eq!(result.unwrap_err().code, "NOT_FOUND", "{name}");
    }
}

#[tokio::test]
#[ignore = "requires HEART_ES_URL and HEART_ES_PIPELINE_TEST_INDEX with the documented default ingest pipeline"]
async fn real_write_returns_pipeline_transformed_source() {
    let (client, _) = environment();
    let index =
        std::env::var("HEART_ES_PIPELINE_TEST_INDEX").expect("provide a dedicated pipeline index");
    let capabilities = Toolkit::from(core::workspace::es::new(client, index).unwrap());
    let id = rand::random::<i64>();
    let data = json!({"id":id,"content":"MiXeD Text","metadata":{"kept":true}});
    let written = run(&capabilities, "write", json!({"data":data}))
        .await
        .unwrap();
    let read = run(&capabilities, "read", json!({"id":id})).await;
    let cleanup = run(&capabilities, "delete", json!({"id":id})).await;
    assert_eq!(
        written.values["data"],
        json!({"id":id,"content":"mixed text","metadata":{"kept":true,"ingested":true}})
    );
    assert_eq!(
        read.unwrap().values["data"],
        json!([written.values["data"]])
    );
    assert_eq!(cleanup.unwrap().values["count"], 1);
}

struct Model {
    outputs: Mutex<VecDeque<Message>>,
    requests: Arc<Mutex<Vec<Request>>>,
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.requests.lock().unwrap().push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: self
                .outputs
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected model replay"),
            usage: None,
            finish_reason: None,
        })]))
    }
}

fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message
        .content
        .push(halo::model::content::part::Part {
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
        Ok(text(Role::Tool, "custom result"))
    }
}

#[tokio::test]
#[ignore = "requires application-prepared HEART_ES_URL and isolated HEART_ES_TEST_INDEX; model is scripted, DB is real"]
async fn real_harness_crud_and_permission_use_the_existing_tool_chain() {
    let (client, index) = environment();
    for allowed in [true, false] {
        let capabilities = Toolkit::from(core::workspace::es::new(client.clone(), &index).unwrap());
        let id = rand::random::<i64>();
        let data = json!({"id":id,"content":"original","metadata":{"kept":true}});
        let operations = if allowed {
            vec![
                ("write", json!({"data":data})),
                ("read", json!({"id":id})),
                ("update", json!({"id":id,"data":{"content":"changed"}})),
                ("read", json!({"id":id})),
                ("delete", json!({"id":id})),
            ]
        } else {
            vec![("write", json!({"data":data}))]
        };
        let mut outputs: Vec<Message> = operations
            .iter()
            .enumerate()
            .map(|(idx, (name, args))| {
                let mut actual_call = call(name, args.clone());
                // Each model-selected call has its own identity, including repeated read.
                actual_call.call_id = format!("es_real_{idx}");
                let mut message = Message::new(Role::Assistant);
                message
                    .content
                    .push(halo::model::content::part::Part {
                        r#type: "tool_call".into(),
                        data: serde_json::to_value(actual_call)
                            .unwrap()
                            .as_object()
                            .unwrap()
                            .clone(),
                    });
                message
            })
            .collect();
        outputs.push(text(Role::Assistant, "done"));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let checks = Arc::new(Mutex::new(Vec::new()));
        let checked = Arc::clone(&checks);
        let agent = harness::builder()
            .model(Model {
                outputs: Mutex::new(outputs.into()),
                requests: Arc::clone(&requests),
            })
            .workspace(
                core::workspace::es::new(client.clone(), &index)
                    .unwrap()
                    .toolkit(Toolkit::new().tools(Custom(ToolDefinition {
                        name: "custom_query".into(),
                        description: "application query".into(),
                        parameters: json!({"type":"object"}).as_object().unwrap().clone(),
                    }))),
            )
            .permission(core::permission::Permission::new(
                move |request: halo::permission::Request, _: Cancellation| {
                    checked.lock().unwrap().push(request.resource);
                    async move {
                        Ok(halo::permission::Decision {
                            behavior: if allowed {
                                halo::permission::Behavior::Allow
                            } else {
                                halo::permission::Behavior::Deny
                            },
                            reason: None,
                        })
                    }
                },
            ))
            .build()
            .unwrap();
        let input = Input {
            metadata: json!({"chat_id":11,"agent_id":22,"session_id":33})
                .as_object()
                .unwrap()
                .clone(),
            payload: vec![Payload {
                r#type: "text".into(),
                data: json!({"value":"test real database operations"}),
                mime_type: None,
            }],
        };
        let result = tokio::time::timeout(Duration::from_secs(30), async {
            agent
                .run(input, false, Cancellation::new())
                .await?
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, Error>>()
        })
        .await;
        // Capture any leftover record, then clean before asserting the run.
        let leftover = run(&capabilities, "read", json!({"id":id})).await;
        if allowed {
            run(&capabilities, "delete", json!({"id":id}))
                .await
                .unwrap();
        }
        let events = result.unwrap().unwrap();
        assert_eq!(events.len(), operations.len() * 2 + 1);
        assert_eq!(leftover.unwrap().values["data"], json!([]));
        assert_eq!(checks.lock().unwrap().len(), operations.len());
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), operations.len() + 1);
        assert!(
            requests[0]
                .tools
                .iter()
                .any(|tool| tool.name == "custom_query")
        );
        assert!(
            !requests[0]
                .tools
                .iter()
                .any(|tool| tool.name == "workspace_glob")
        );
        let messages: Vec<_> = requests
            .last()
            .unwrap()
            .messages
            .as_slice()
            .iter()
            .filter(|message| message.role == Role::Tool)
            .collect();
        if allowed {
            assert_eq!(messages[0].values["data"], data);
            assert_eq!(messages[1].values["data"], json!([data]));
            assert_eq!(messages[2].values["count"], 1);
            assert_eq!(messages[3].values["data"][0]["content"], "changed");
            assert_eq!(messages[4].values["count"], 1);
        }
    }
}
