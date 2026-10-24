//! Built-in operations use only isolated temporary files and static commands.
#![cfg(unix)]

use std::{path::PathBuf, sync::Arc, time::Duration};

use halo::agent::{Cancellation, Error, Message};
use halo::{
    core,
    workspace::Workspace,
    message::Role,
    sandbox::{Local as Sandbox, local::Config},
    tool::Call,
};
use serde_json::{Value, json};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "heart-workspace-operations-{}-{}",
            std::process::id(),
            rand::random::<u64>(),
        ));
        std::fs::create_dir(&directory).unwrap();
        Self(directory)
    }
    fn workspace(&self) -> core::workspace::Local {
        core::workspace::Local::new(&self.0).unwrap()
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn call(name: &str, arguments: Value) -> Call {
    Call {
        call_id: format!("call_{name}"),
        name: format!("workspace_{name}"),
        arguments: arguments.as_object().unwrap().clone(),
    }
}

async fn run(workspace: &impl Workspace, name: &str, arguments: Value) -> Result<Message, Error> {
    let (message, evaluation) = workspace
        .execute(call(name, arguments), &Cancellation::new())
        .await?;
    assert!(evaluation.passed);
    assert_eq!(message.role, Role::Tool);
    assert_eq!(message.content[0].r#type, "text");
    assert_eq!(
        serde_json::from_str::<Value>(message.content[0].data["value"].as_str().unwrap()).unwrap(),
        Value::Object(message.values.clone())
    );
    Ok(message)
}

#[tokio::test]
async fn local_defaults_include_delete_but_not_host_commands() {
    let directory = Directory::new();
    let capabilities = directory.workspace();
    assert_eq!(
        halo::workspace::Workspace::tool(&capabilities)
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        [
            "workspace_read",
            "workspace_write",
            "workspace_list",
            "workspace_edit",
            "workspace_glob",
            "workspace_grep",
            "workspace_delete",
        ]
    );
    for definition in halo::workspace::Workspace::tool(&capabilities) {
        definition.validate().unwrap();
    }
    assert_eq!(
        run(
            &capabilities,
            "exec_shell",
            json!({"args":["/bin/sh","-c","printf wrong > unexpected"]})
        )
        .await
        .unwrap_err()
        .code,
        "NOT_FOUND"
    );
    assert!(!directory.0.join("unexpected").exists());
}

#[tokio::test]
async fn deletion_is_relative_single_file_physical_and_cancel_safe() {
    let directory = Directory::new();
    let capabilities = directory.workspace();
    run(
        &capabilities,
        "write",
        json!({"path":"notes/draft.txt","content":"keep"}),
    )
    .await
    .unwrap();
    for path in [
        "",
        "../outside",
        "/outside",
        "notes/../draft.txt",
        ".",
        "notes/",
    ] {
        assert_eq!(
            run(&capabilities, "delete", json!({"path":path}))
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
    }
    assert!(
        run(&capabilities, "delete", json!({"path":"notes"}))
            .await
            .is_err()
    );
    assert_eq!(
        std::fs::read_to_string(directory.0.join("notes/draft.txt")).unwrap(),
        "keep"
    );
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        capabilities
            .execute(
                call("delete", json!({"path":"notes/draft.txt"})),
                &cancellation
            )
            .await
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    let message = run(&capabilities, "delete", json!({"path":"notes/draft.txt"}))
        .await
        .unwrap();
    assert_eq!(
        message.values,
        json!({"path":"notes/draft.txt"})
            .as_object()
            .unwrap()
            .clone()
    );
    assert!(!directory.0.join("notes/draft.txt").exists());
    assert!(directory.0.join("notes").is_dir());
    assert_eq!(
        run(&capabilities, "delete", json!({"path":"notes/draft.txt"}))
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
}

#[tokio::test]
async fn deletion_does_not_follow_symlink_resources_or_parents() {
    let inside = Directory::new();
    let outside = Directory::new();
    std::fs::write(outside.0.join("keep.txt"), "keep").unwrap();
    std::os::unix::fs::symlink(outside.0.join("keep.txt"), inside.0.join("file-link")).unwrap();
    std::os::unix::fs::symlink(&outside.0, inside.0.join("directory-link")).unwrap();
    let capabilities = inside.workspace();
    for path in ["file-link", "directory-link/keep.txt"] {
        assert_eq!(
            run(&capabilities, "delete", json!({"path":path}))
                .await
                .unwrap_err()
                .code,
            "PERMISSION_DENIED"
        );
    }
    assert_eq!(
        std::fs::read_to_string(outside.0.join("keep.txt")).unwrap(),
        "keep"
    );
}

#[tokio::test]
async fn configured_sandbox_automatically_adds_command_and_uses_the_same_files() {
    let directory = Directory::new();
    let original_cwd = std::env::current_dir().unwrap();
    let workspace = directory.workspace().sandbox(Sandbox::default());
    assert_eq!(
        std::fs::read_dir(&directory.0).unwrap().count(),
        0,
        "configuration must not execute"
    );
    let capabilities = workspace;
    assert_eq!(halo::workspace::Workspace::tool(&capabilities).len(), 8);
    run(
        &capabilities,
        "write",
        json!({"path":"draft.txt","content":"draft"}),
    )
    .await
    .unwrap();
    let read = run(
        &capabilities,
        "exec_shell",
        json!({"args":["/bin/cat","draft.txt"]}),
    )
    .await
    .unwrap();
    assert_eq!(
        read.values,
        json!({"stdout":"draft","stderr":"","exit_code":0})
            .as_object()
            .unwrap()
            .clone()
    );
    run(
        &capabilities,
        "exec_shell",
        json!({"args":["/bin/sh","-c","printf 'new data' > draft.txt"]}),
    )
    .await
    .unwrap();
    assert_eq!(
        run(&capabilities, "read", json!({"path":"draft.txt"}))
            .await
            .unwrap()
            .values["content"],
        "new data"
    );
    run(&capabilities, "delete", json!({"path":"draft.txt"}))
        .await
        .unwrap();
    run(
        &capabilities,
        "exec_shell",
        json!({"args":["/bin/sh","-c","test ! -e draft.txt"]}),
    )
    .await
    .unwrap();
    assert_eq!(std::env::current_dir().unwrap(), original_cwd);
}

#[tokio::test]
async fn command_argv_is_literal_and_call_cannot_override_environment_configuration() {
    let directory = Directory::new();
    let capabilities = directory.workspace().sandbox(Sandbox::default());
    let message = run(
        &capabilities,
        "exec_shell",
        json!({"args":["/usr/bin/printf","%s","$(exit 7); * $HOME"]}),
    )
    .await
    .unwrap();
    assert_eq!(message.values["stdout"], "$(exit 7); * $HOME");
    for arguments in [
        json!({"args":[]}),
        json!({"args":[""]}),
        json!({"args":[" "]}),
        json!({"args":["x\u{0}"]}),
        json!({"args":[1]}),
        json!({"command":"pwd"}),
        json!({"args":["/bin/pwd"],"cwd":"/"}),
        json!({"args":["/bin/pwd"],"env":{}}),
        json!({"args":["/bin/pwd"],"timeout":100}),
    ] {
        assert_eq!(
            run(&capabilities, "exec_shell", arguments)
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
    }
    assert_eq!(
        run(
            &capabilities,
            "exec_shell",
            json!({"args":["/nonexistent-heart-workspace-command"]})
        )
        .await
        .unwrap_err()
        .code,
        "NOT_FOUND"
    );
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        capabilities
            .execute(
                call(
                    "exec_shell",
                    json!({"args":["/bin/sh","-c","printf wrong > unexpected"]})
                ),
                &cancellation
            )
            .await
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 0);
}

#[tokio::test]
async fn command_failure_is_direct_not_retried_and_does_not_undo_side_effects() {
    let directory = Directory::new();
    let capabilities = directory.workspace().sandbox(Sandbox::default());
    let error = run(
        &capabilities,
        "exec_shell",
        json!({"args":["/bin/sh","-c","printf x >> attempts; printf out; printf err >&2; exit 7"]}),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code, "EXECUTION_FAILED");
    assert_eq!(
        serde_json::from_str::<Value>(error.details.as_ref().unwrap()).unwrap(),
        json!({"stdout":"out","stderr":"err","exit_code":7})
    );
    assert_eq!(
        std::fs::read_to_string(directory.0.join("attempts")).unwrap(),
        "x"
    );
}

#[tokio::test]
async fn command_reuses_configured_output_limit_timeout_and_utf8_errors() {
    let directory = Directory::new();
    let capabilities =
        directory.workspace().sandbox(
            Sandbox::new(Config {
                max_output: 3,
                ..Default::default()
            })
            .unwrap(),
        );
    assert_eq!(
        run(
            &capabilities,
            "exec_shell",
            json!({"args":["/usr/bin/printf","abcd"]})
        )
        .await
        .unwrap_err()
        .code,
        "OUTPUT_LIMIT"
    );
    assert_eq!(
        run(
            &capabilities,
            "exec_shell",
            json!({"args":["/bin/sh","-c","printf '\\377'"]})
        )
        .await
        .unwrap_err()
        .code,
        "INVALID_RESPONSE"
    );
    let capabilities =
        directory.workspace().sandbox(
            Sandbox::new(Config {
                timeout: Duration::from_millis(50),
                ..Default::default()
            })
            .unwrap(),
        );
    assert_eq!(
        tokio::time::timeout(
            Duration::from_secs(5),
            run(
                &capabilities,
                "exec_shell",
                json!({"args":["/bin/sleep","20"]})
            )
        )
        .await
        .unwrap()
        .unwrap_err()
        .code,
        "TIMEOUT"
    );
    // The writer lock is released after the failed/timed-out process.
    run(
        &capabilities,
        "write",
        json!({"path":"after.txt","content":"ok"}),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn independent_workspaces_do_not_share_cwd_or_execution_configuration() {
    let first = Directory::new();
    let second = Directory::new();
    std::fs::write(first.0.join("value.txt"), "first").unwrap();
    std::fs::write(second.0.join("value.txt"), "second").unwrap();
    let first_mounted_capabilities = first.workspace().sandbox(Sandbox::default());
    let second_mounted_capabilities = second.workspace().sandbox(Sandbox::default());
    let original_cwd = std::env::current_dir().unwrap();
    let (a, b) = tokio::join!(
        run(
            &first_mounted_capabilities,
            "exec_shell",
            json!({"args":["/bin/cat","value.txt"]})
        ),
        run(
            &second_mounted_capabilities,
            "exec_shell",
            json!({"args":["/bin/cat","value.txt"]})
        ),
    );
    assert_eq!(a.unwrap().values["stdout"], "first");
    assert_eq!(b.unwrap().values["stdout"], "second");
    assert_eq!(std::env::current_dir().unwrap(), original_cwd);
}

#[tokio::test]
async fn cancellation_while_waiting_for_a_command_lock_never_runs_the_file_operation() {
    let directory = Directory::new();
    std::fs::write(directory.0.join("draft.txt"), "keep").unwrap();
    let capabilities = Arc::new(
        directory.workspace().sandbox(Sandbox::default()),
    );
    let cancellation = Cancellation::new();
    let signal = cancellation.clone();
    let running = Arc::clone(&capabilities);
    let process = tokio::spawn(async move {
        running
            .execute(
                call(
                    "exec_shell",
                    json!({"args":["/bin/sh","-c","printf started > started; exec /bin/sleep 20"]}),
                ),
                &signal,
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while tokio::fs::metadata(directory.0.join("started"))
            .await
            .is_err()
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    for (name, args) in [
        ("read", json!({"path":"draft.txt"})),
        ("delete", json!({"path":"draft.txt"})),
        ("write", json!({"path":"draft.txt","content":"wrong"})),
        (
            "edit",
            json!({"path":"draft.txt","old_string":"keep","new_string":"wrong"}),
        ),
        ("list", json!({})),
    ] {
        let waiting = Cancellation::new();
        let mut operation = std::pin::pin!(capabilities.execute(call(name, args), &waiting));
        assert!(futures_util::poll!(&mut operation).is_pending());
        waiting.cancel();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), operation)
                .await
                .unwrap()
                .unwrap_err()
                .code,
            "CANCELLED"
        );
    }
    cancellation.cancel();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), process)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(
        run(&capabilities, "read", json!({"path":"draft.txt"}))
            .await
            .unwrap()
            .values["content"],
        "keep"
    );
    run(&capabilities, "delete", json!({"path":"draft.txt"}))
        .await
        .unwrap();
}
