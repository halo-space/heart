use std::{
    future::pending,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures_executor::block_on;
use halo::agent::{Cancellation, Error};
use halo::{
    core,
    memory::short::{
        Type,
        chat::{self, Store as _},
    },
    middleware::{Retry, RetryPolicy},
    permission::{Behavior, Decision, Permission as _, Request},
    workspace::{Mode, Workspace as _},
};
use serde_json::{Map, Value, json};

#[test]
fn retry_default_calls_once_and_preserves_error() {
    let count = AtomicUsize::new(0);
    let error = Error::new("TIMEOUT", "original").with_details("detail");
    let result = block_on(Retry::default().call(
        || {
            count.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Err::<(), _>(error.clone()))
        },
        &Cancellation::new(),
    ));
    assert_eq!(result.unwrap_err(), error);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn retry_applies_budget_and_never_retries_permanent_errors() {
    let retry = Retry::new(RetryPolicy {
        max_retries: 2,
        min_delay_ms: 0,
        max_delay_ms: 0,
    })
    .unwrap();
    for (code, expected) in [
        ("TIMEOUT", 3),
        ("INVALID_ARGUMENTS", 1),
        ("CANCELLED", 1),
        ("PERMISSION_DENIED", 1),
    ] {
        let count = AtomicUsize::new(0);
        let result = block_on(retry.call(
            || {
                count.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Err::<(), _>(Error::new(code, "failure")))
            },
            &Cancellation::new(),
        ));
        assert_eq!(result.unwrap_err().code, code);
        assert_eq!(count.load(Ordering::SeqCst), expected);
    }
}

#[test]
fn retry_returns_first_success_and_validates_policy() {
    assert!(
        Retry::new(RetryPolicy {
            max_retries: 1,
            min_delay_ms: 2,
            max_delay_ms: 1
        })
        .is_err()
    );
    let retry = Retry::new(RetryPolicy {
        max_retries: 4,
        min_delay_ms: 0,
        max_delay_ms: 0,
    })
    .unwrap();
    let mut count = 0;
    let result = block_on(retry.call(
        || {
            count += 1;
            std::future::ready(if count < 3 {
                Err(Error::new("TRANSPORT", "network"))
            } else {
                Ok(42)
            })
        },
        &Cancellation::new(),
    ))
    .unwrap();
    assert_eq!(result, 42);
    assert_eq!(count, 3);
}

#[tokio::test]
async fn retry_cancellation_wakes_pending_operation() {
    let cancellation = Cancellation::new();
    let cancel = cancellation.clone();
    let work = Retry::default();
    let cancel_task = tokio::spawn(async move {
        tokio::task::yield_now().await;
        cancel.cancel();
    });
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        work.call(pending::<Result<(), Error>>, &cancellation),
    )
    .await
    .unwrap();
    assert_eq!(result.unwrap_err().code, "CANCELLED");
    cancel_task.await.unwrap();
}

#[tokio::test]
async fn retry_backoff_is_cancellable_without_next_invocation() {
    let count = Arc::new(AtomicUsize::new(0));
    let seen = count.clone();
    let cancellation = Cancellation::new();
    let cancel = cancellation.clone();
    let task = tokio::spawn(async move {
        while seen.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        cancel.cancel();
    });
    let retry = Retry::new(RetryPolicy {
        max_retries: 3,
        min_delay_ms: 30_000,
        max_delay_ms: 30_000,
    })
    .unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        retry.call(
            || {
                count.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Err::<(), _>(Error::new("TIMEOUT", "network")))
            },
            &cancellation,
        ),
    )
    .await
    .unwrap();
    assert_eq!(result.unwrap_err().code, "CANCELLED");
    assert_eq!(count.load(Ordering::SeqCst), 1);
    task.await.unwrap();
}

#[test]
fn cancelled_retry_does_not_enter_user_operation() {
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let error = block_on(Retry::default().call(
        || async {
            panic!("must not call");
            #[allow(unreachable_code)]
            Ok::<(), Error>(())
        },
        &cancellation,
    ))
    .unwrap_err();
    assert_eq!(error.code, "CANCELLED");
}

#[test]
fn permission_executes_user_policy_and_returns_normal_decisions() {
    let permission =
        core::permission::Permission::new(|request: Request, _: Cancellation| async move {
            if request.action == "read" && request.metadata.get("allowed") == Some(&json!(true)) {
                Ok(Decision {
                    behavior: Behavior::Allow,
                    reason: None,
                })
            } else {
                Ok(Decision {
                    behavior: Behavior::Deny,
                    reason: Some("policy denied".into()),
                })
            }
        });
    let request = Request {
        action: "read".into(),
        resource: "notes".into(),
        metadata: json!({"allowed":true}).as_object().unwrap().clone(),
    };
    assert_eq!(
        block_on(permission.check(&request, &Cancellation::new()))
            .unwrap()
            .behavior,
        Behavior::Allow
    );
    let denied = Request {
        action: "delete".into(),
        ..request
    };
    assert_eq!(
        block_on(permission.check(&denied, &Cancellation::new())).unwrap(),
        Decision {
            behavior: Behavior::Deny,
            reason: Some("policy denied".into())
        }
    );
}

#[test]
fn permission_decision_has_only_behavior_and_optional_reason_and_keeps_errors_separate() {
    for behavior in [Behavior::Allow, Behavior::Deny, Behavior::Ask] {
        let decision = Decision {
            behavior,
            reason: Some("policy explanation".into()),
        };
        let value = serde_json::to_value(&decision).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 2);
        assert!(value.get("behavior").is_some() && value.get("reason").is_some());
        assert_eq!(serde_json::from_value::<Decision>(value).unwrap(), decision);
    }
    assert!(
        serde_json::from_value::<Decision>(json!({"behavior":"passthrough","reason":null}))
            .is_err()
    );
    let ask = core::permission::Permission::new(|_: Request, _: Cancellation| async {
        Ok(Decision {
            behavior: Behavior::Ask,
            reason: None,
        })
    });
    let request = Request {
        action: "call".into(),
        resource: "tool".into(),
        metadata: Map::new(),
    };
    assert_eq!(
        block_on(ask.check(&request, &Cancellation::new()))
            .unwrap()
            .behavior,
        Behavior::Ask
    );
    let error =
        Error::new("UNAVAILABLE", "policy unavailable").with_details("unchanged backend details");
    let expected = error.clone();
    let policy = core::permission::Permission::new(move |_: Request, _: Cancellation| {
        let error = error.clone();
        async move { Err(error) }
    });
    assert_eq!(
        block_on(policy.check(&request, &Cancellation::new())).unwrap_err(),
        expected
    );
}

#[test]
fn permission_rejects_empty_requests_and_pre_cancelled_calls() {
    let policy = core::permission::Permission::new(|_: Request, _: Cancellation| async {
        panic!("must not invoke")
    });
    assert_eq!(
        block_on(policy.check(&Request::default(), &Cancellation::new()))
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    let cancel = Cancellation::new();
    cancel.cancel();
    assert_eq!(
        block_on(policy.check(&Request::default(), &cancel))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
}

#[tokio::test]
async fn permission_pending_policy_can_be_cancelled() {
    let policy = core::permission::Permission::new(|_: Request, _: Cancellation| {
        pending::<Result<Decision, Error>>()
    });
    let cancel = Cancellation::new();
    let token = cancel.clone();
    let task = tokio::spawn(async move {
        tokio::task::yield_now().await;
        token.cancel();
    });
    let request = Request {
        action: "read".into(),
        resource: "notes".into(),
        metadata: Map::new(),
    };
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), policy.check(&request, &cancel))
            .await
            .unwrap()
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    task.await.unwrap();
}

struct Directory(std::path::PathBuf);
impl Directory {
    fn new() -> Self {
        static IDX: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "halo-workspace-test-{}-{}",
            std::process::id(),
            IDX.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn local_workspace_reads_overwrites_and_physically_deletes() {
    let dir = Directory::new();
    let workspace = core::workspace::Local::new(&dir.0).unwrap();
    let cancel = Cancellation::new();
    assert_eq!(workspace.definition().mode, Mode::Local);
    assert!(workspace.definition().path.is_some());
    workspace
        .write("mcp/config.json", b"first".to_vec(), &cancel)
        .await
        .unwrap();
    workspace
        .write("mcp/config.json", b"second".to_vec(), &cancel)
        .await
        .unwrap();
    assert_eq!(
        workspace.read("mcp/config.json", &cancel).await.unwrap(),
        b"second"
    );
    workspace.delete("mcp/config.json", &cancel).await.unwrap();
    assert_eq!(
        workspace
            .read("mcp/config.json", &cancel)
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert_eq!(std::fs::read_dir(dir.0.join("mcp")).unwrap().count(), 0);
}

#[tokio::test]
async fn local_workspace_rejects_traversal_absolute_and_directory_delete() {
    let dir = Directory::new();
    let workspace = core::workspace::Local::new(&dir.0).unwrap();
    let cancel = Cancellation::new();
    for path in [
        "",
        "../secret",
        "/tmp/file",
        "a/../b",
        "./x",
        "a//b",
        "a\\b",
        "a/",
    ] {
        assert_eq!(
            workspace
                .write(path, vec![], &cancel)
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
    }
    std::fs::create_dir(dir.0.join("folder")).unwrap();
    assert!(workspace.delete("folder", &cancel).await.is_err());
    assert!(dir.0.join("folder").exists());
}

#[cfg(unix)]
#[tokio::test]
async fn local_workspace_rejects_symlink_resource_and_parent() {
    let root = Directory::new();
    let outside = Directory::new();
    std::fs::write(outside.0.join("secret"), b"keep").unwrap();
    std::os::unix::fs::symlink(&outside.0, root.0.join("link")).unwrap();
    std::os::unix::fs::symlink(outside.0.join("secret"), root.0.join("file")).unwrap();
    let workspace = core::workspace::Local::new(&root.0).unwrap();
    let cancel = Cancellation::new();
    for path in ["link/secret", "file"] {
        assert_eq!(
            workspace.read(path, &cancel).await.unwrap_err().code,
            "PERMISSION_DENIED"
        );
        assert_eq!(
            workspace
                .write(path, b"change".to_vec(), &cancel)
                .await
                .unwrap_err()
                .code,
            "PERMISSION_DENIED"
        );
        assert_eq!(
            workspace.delete(path, &cancel).await.unwrap_err().code,
            "PERMISSION_DENIED"
        );
    }
    assert_eq!(std::fs::read(outside.0.join("secret")).unwrap(), b"keep");
}

#[tokio::test]
async fn local_workspace_cancelled_write_preserves_original() {
    let dir = Directory::new();
    let workspace = core::workspace::Local::new(&dir.0).unwrap();
    let cancel = Cancellation::new();
    workspace
        .write("original", b"keep".to_vec(), &cancel)
        .await
        .unwrap();
    cancel.cancel();
    assert_eq!(
        workspace
            .write("original", b"overwrite".to_vec(), &cancel)
            .await
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(std::fs::read(dir.0.join("original")).unwrap(), b"keep");
    assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
}

#[tokio::test]
async fn local_workspace_returns_owned_data_and_physically_deletes_records() {
    let directory = Directory::new();
    let workspace = core::workspace::Local::new(&directory.0).unwrap();
    let cancel = Cancellation::new();
    assert!(workspace.definition().path.is_some());
    workspace
        .write("notes", vec![1, 2, 3], &cancel)
        .await
        .unwrap();
    let mut snapshot = workspace.read("notes", &cancel).await.unwrap();
    snapshot.clear();
    assert_eq!(
        workspace.read("notes", &cancel).await.unwrap(),
        vec![1, 2, 3]
    );
    workspace.delete("notes", &cancel).await.unwrap();
    assert_eq!(
        workspace.delete("notes", &cancel).await.unwrap_err().code,
        "NOT_FOUND"
    );
}

fn chat(id: i64, session_id: i64) -> chat::Chat {
    chat::Chat {
        id,
        tenant_id: 1,
        user_id: 2,
        agent_id: 3,
        session_id,
        r#type: Type::Agent,
        status: chat::Status::Completed,
        input: None,
        message: None,
        usage: None,
        graph: None,
        metadata: Map::new(),
        started_time: None,
        completed_time: None,
        created_time: 0,
        updated_time: 0,
    }
}
fn options() -> Map<String, Value> {
    json!({"filter":{"tenant_id":1,"user_id":2,"agent_id":3,"session_id":10,"type":"agent"}})
        .as_object()
        .unwrap()
        .clone()
}

#[test]
fn history_requires_explicit_ownership_and_validates_options() {
    let store = core::memory::short::chat::InMemory::new();
    let cancel = Cancellation::new();
    for field in ["tenant_id", "user_id", "agent_id", "session_id", "type"] {
        let mut option = options();
        option
            .get_mut("filter")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert_eq!(
            block_on(store.list(option, &cancel)).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
    }
    for invalid in [
        json!({"pagination":{"limit":0}}),
        json!({"pagination":{"offset":"wrong"}}),
        json!({"sort":{"field":"idx","order":"asc"}}),
        json!({"other":true}),
    ] {
        let mut option = options();
        option.extend(invalid.as_object().unwrap().clone());
        assert_eq!(
            block_on(store.list(option, &cancel)).unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
    }
}

#[test]
fn history_is_filtered_ordered_paginated_and_time_bounded() {
    let store = core::memory::short::chat::InMemory::new();
    let cancel = Cancellation::new();
    for id in [4, 1, 3, 2] {
        block_on(store.create(chat(id, 10), &cancel)).unwrap();
    }
    block_on(store.create(chat(5, 11), &cancel)).unwrap();
    let mut other = chat(6, 10);
    other.r#type = Type::Workflow;
    block_on(store.create(other, &cancel)).unwrap();
    let result = block_on(store.list(options(), &cancel)).unwrap();
    assert_eq!(result.len(), 4);
    assert!(
        result
            .windows(2)
            .all(|pair| (pair[0].created_time, pair[0].id) < (pair[1].created_time, pair[1].id))
    );
    let mut option = options();
    option.insert(
        "sort".into(),
        json!({"field":"created_time","order":"desc"}),
    );
    option.insert("pagination".into(), json!({"offset":1,"limit":2}));
    let paged = block_on(store.list(option, &cancel)).unwrap();
    assert_eq!(
        paged.iter().map(|chat| chat.id).collect::<Vec<_>>(),
        result
            .iter()
            .rev()
            .skip(1)
            .take(2)
            .map(|chat| chat.id)
            .collect::<Vec<_>>()
    );
    let mut option = options();
    option.get_mut("filter").unwrap()["created_time"] =
        json!({"from":result[0].created_time,"to":result.last().unwrap().created_time+1});
    assert_eq!(
        block_on(store.list(option.clone(), &cancel)).unwrap(),
        result
    );
    option.get_mut("filter").unwrap()["created_time"] =
        json!({"from":0,"to":result[0].created_time});
    assert!(block_on(store.list(option, &cancel)).unwrap().is_empty());
    cancel.cancel();
    assert_eq!(
        block_on(store.list(options(), &cancel)).unwrap_err().code,
        "CANCELLED"
    );
}
