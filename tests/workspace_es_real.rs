//! Opt-in actual Elasticsearch acceptance. The caller prepares the cluster,
//! isolated base index and optional pipeline index; Workspace never provisions
//! them. No tests use a production index or provision a Docker service.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use elasticsearch::{
    Elasticsearch,
    auth::Credentials,
    http::transport::{SingleNodeConnectionPool, TransportBuilder},
    indices::{IndicesDeleteParts, IndicesExistsParts},
};
use futures_util::{StreamExt, future::join_all};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::core,
    agent::message::Role,
    agent::model::chat,
    agent::provider::deepseek::DeepSeek,
    agent::tool::Call,
    agent::workspace::Workspace,
    agent::{Agent, Input, Payload},
    harness,
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

async fn run(workspace: &impl Workspace, name: &str, args: Value) -> Result<Message, Error> {
    let (message, evaluation) = workspace
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
    let capabilities = core::workspace::es::new(client, index).unwrap();
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
    let capabilities = core::workspace::es::new(client, index).unwrap();
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
    let capabilities = core::workspace::es::new(client.clone(), &index).unwrap();
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
    let capabilities = core::workspace::es::new(client, index).unwrap();
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

#[tokio::test]
#[ignore = "requires real DeepSeek credentials and an application-prepared isolated Elasticsearch index"]
async fn real_harness_crud_and_permission_use_the_existing_tool_chain() {
    let (client, index) = environment();
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct real DeepSeek provider");
    let model = provider
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct real DeepSeek Chat");
    for allowed in [true, false] {
        let capabilities = core::workspace::es::new(client.clone(), &index).unwrap();
        // Keep the ID exactly representable by JSON-number handling in all
        // model/provider layers while remaining unique in the test index.
        let id = i64::from(rand::random::<u32>() & 0x7fff_ffff).max(1);
        let data = json!({"id":id,"content":"original","metadata":{"kept":true}});
        let instruction = if allowed {
            format!(
                "请严格按顺序真实执行以下数据库操作，每一步都必须调用对应的 workspace 工具，不要跳过或编造结果：\n1. workspace_write，参数 {{\"data\":{data}}}\n2. workspace_read，参数 {{\"id\":{id}}}\n3. workspace_update，参数 {{\"id\":{id},\"data\":{{\"content\":\"changed\"}}}}\n4. workspace_read，参数 {{\"id\":{id}}}\n5. workspace_delete，参数 {{\"id\":{id}}}\n全部完成后只回复 DONE。"
            )
        } else {
            format!(
                "请调用 workspace_write，参数 {{\"data\":{data}}}。即使权限被拒绝，也只报告实际工具结果，不要声称写入成功。"
            )
        };
        let checks = Arc::new(Mutex::new(Vec::new()));
        let checked = Arc::clone(&checks);
        let agent = harness::builder()
            .model(model.clone())
            .system_prompt("你是数据库操作 Agent。只根据工具的真实结果继续操作，绝不编造成功状态。")
            .workspace(core::workspace::es::new(client.clone(), &index).unwrap())
            .permission(core::permission::Permission::new(
                move |request: halo::agent::permission::Request, _: Cancellation| {
                    checked.lock().unwrap().push(request.resource);
                    async move {
                        Ok(halo::agent::permission::Decision {
                            behavior: if allowed {
                                halo::agent::permission::Behavior::Allow
                            } else {
                                halo::agent::permission::Behavior::Deny
                            },
                            reason: None,
                        })
                    }
                },
            ))
            .with_max_iters(12)
            .build()
            .unwrap();
        let input = Input {
            metadata:
                json!({"chat_id":rand::random::<u32>() as i64 + 1,"agent_id":22,"session_id":33})
                    .as_object()
                    .unwrap()
                    .clone(),
            payload: vec![Payload {
                r#type: "text".into(),
                data: json!({"value":instruction}),
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
        assert!(!events.is_empty());
        assert_eq!(leftover.unwrap().values["data"], json!([]));
        let checks = checks.lock().unwrap();
        assert!(checks.iter().any(|resource| resource == "workspace_write"));
        if allowed {
            for operation in ["workspace_read", "workspace_update", "workspace_delete"] {
                assert!(
                    checks.iter().any(|resource| resource == operation),
                    "missing authorized {operation}: {checks:?}"
                );
            }
        }
    }
}
