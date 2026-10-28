//! Process-local fake Docker protocol regressions. The opt-in final test is
//! separate and is the only test exercising an actual Docker daemon/container.
#![cfg(unix)]

use std::{path::PathBuf, process::Command, time::Duration};

use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::core::{
        sandbox,
        workspace::{Docker, docker::Config},
    },
    agent::tool::Call,
    agent::workspace::Mode,
    agent::workspace::Workspace,
};
use serde_json::{Value, json};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        use std::os::unix::fs::PermissionsExt;
        let path = std::env::temp_dir().join(format!(
            "heart-workspace-docker-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::create_dir(path.join("bin")).unwrap();
        std::fs::create_dir(path.join("state")).unwrap();
        std::fs::write(
            path.join("bin/docker"),
            include_bytes!("fixtures/workspace_docker.sh"),
        )
        .unwrap();
        std::fs::set_permissions(
            path.join("bin/docker"),
            std::fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn protocol(case: &str) {
    // Set PATH only in a new process. Never mutate the test runner's global
    // environment, expose a public test-only CLI setting, or invoke real Docker.
    let fixture = Fixture::new();
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "run_protocol_case", "--nocapture"])
        .env("HEART_WORKSPACE_CASE", case)
        .env("HEART_WORKSPACE_FIXTURE", &fixture.0)
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", fixture.0.join("bin").display()),
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{case}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

macro_rules! protocol_test {
    ($test:ident, $case:ident) => {
        #[test]
        fn $test() {
            protocol(stringify!($case));
        }
    };
}
protocol_test!(protocol_shared_files, shared_files_and_default_tools);
protocol_test!(protocol_literal_paths, literal_paths_and_binary_data);
protocol_test!(
    protocol_concurrent_initialization,
    concurrent_init_and_isolated_workspaces
);
protocol_test!(protocol_failures, failures_do_not_recreate_environment);
protocol_test!(
    protocol_timeout_and_cancellation,
    timeout_and_cancellation_close_owned_environment
);
protocol_test!(
    protocol_lock_wait,
    cancelled_lock_wait_does_not_close_environment
);
protocol_test!(protocol_dropped_future, drop_in_flight_future_cleans_up);
protocol_test!(
    protocol_dropped_configured_tools,
    drop_configured_tools_releases_environment
);
protocol_test!(protocol_foreign_resource, foreign_resource_is_never_removed);
protocol_test!(protocol_mount_and_env, configured_mount_and_env);
// The model-to-permission path is exercised with a real Provider by the
// opt-in live acceptance below, not by a scripted Chat fixture.
protocol_test!(
    protocol_validation,
    pre_cancel_and_invalid_arguments_are_no_io
);

protocol_test!(
    protocol_missing_container_cleanup,
    missing_container_cleanup
);

protocol_test!(
    protocol_limits_and_startup_failure,
    limits_and_startup_failure
);

#[tokio::test]
#[ignore = "subprocess dispatcher; default protocol_* tests invoke this automatically"]
async fn run_protocol_case() {
    match std::env::var("HEART_WORKSPACE_CASE").unwrap().as_str() {
        "shared_files_and_default_tools" => shared_files_and_default_tools().await,
        "literal_paths_and_binary_data" => literal_paths_and_binary_data().await,
        "concurrent_init_and_isolated_workspaces" => {
            concurrent_init_and_isolated_workspaces().await
        }
        "failures_do_not_recreate_environment" => failures_do_not_recreate_environment().await,
        "timeout_and_cancellation_close_owned_environment" => {
            timeout_and_cancellation_close_owned_environment().await
        }
        "cancelled_lock_wait_does_not_close_environment" => {
            cancelled_lock_wait_does_not_close_environment().await
        }
        "drop_in_flight_future_cleans_up" => drop_in_flight_future_cleans_up().await,
        "drop_configured_tools_releases_environment" => {
            drop_configured_tools_releases_environment().await
        }
        "foreign_resource_is_never_removed" => foreign_resource_is_never_removed().await,
        "configured_mount_and_env" => configured_mount_and_env().await,
        "real_model_workspace_permission_denial" => real_model_workspace_permission_denial().await,
        "pre_cancel_and_invalid_arguments_are_no_io" => {
            pre_cancel_and_invalid_arguments_are_no_io().await
        }
        "missing_container_cleanup" => missing_container_cleanup().await,
        "limits_and_startup_failure" => limits_and_startup_failure().await,
        other => panic!("unknown fixture case {other}"),
    }
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY; subprocesses a Docker-call probe"]
async fn live_model_workspace_permission_denial_prevents_docker_dispatch() {
    protocol("real_model_workspace_permission_denial");
}

fn config() -> Config {
    Config {
        sandbox: sandbox::docker::Config {
            image: "prepared:test".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
fn state() -> PathBuf {
    PathBuf::from(
        std::env::var_os("HEART_WORKSPACE_FIXTURE").expect("run through protocol_regressions"),
    )
    .join("state")
}
fn operations() -> Vec<String> {
    std::fs::read_to_string(state().join("operations"))
        .unwrap_or_default()
        .lines()
        .map(str::to_owned)
        .collect()
}
fn count(operation: &str) -> usize {
    operations()
        .iter()
        .filter(|item| item.as_str() == operation)
        .count()
}
fn call(name: &str, args: Value) -> Call {
    Call {
        call_id: format!("call_{name}"),
        name: format!("workspace_{name}"),
        arguments: args.as_object().unwrap().clone(),
    }
}
async fn run(workspace: &impl Workspace, name: &str, args: Value) -> Result<Message, Error> {
    workspace
        .execute(call(name, args), &Cancellation::new())
        .await
        .map(|(message, _)| message)
}
async fn wait_for_file(path: PathBuf) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !path.exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("fixture stage was not reached");
}
async fn wait_for_removal() {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if std::fs::read_dir(state()).unwrap().any(|entry| {
                entry
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|value| value == "removed")
            }) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("owned environment was not removed");
}

async fn shared_files_and_default_tools() {
    let workspace = Docker::new(config()).unwrap();
    assert_eq!(workspace.definition().mode, Mode::Local);
    assert_eq!(workspace.definition().path.as_deref(), Some("/workspace"));
    let capabilities = workspace;
    assert_eq!(
        halo::agent::workspace::Workspace::tool(&capabilities).len(),
        8
    );
    for tool in halo::agent::workspace::Workspace::tool(&capabilities) {
        tool.validate().unwrap();
    }
    assert!(operations().is_empty());
    run(
        &capabilities,
        "write",
        json!({"path":"notes/a.txt","content":"alpha\nbeta\n"}),
    )
    .await
    .unwrap();
    let output = run(
        &capabilities,
        "exec_shell",
        json!({"args":["/bin/sh","-c","cat notes/a.txt > result.txt; printf gamma >> result.txt"]}),
    )
    .await
    .unwrap();
    assert_eq!(output.values["exit_code"], 0);
    assert_eq!(
        run(&capabilities, "read", json!({"path":"result.txt"}))
            .await
            .unwrap()
            .values["content"],
        "alpha\nbeta\ngamma"
    );
    run(
        &capabilities,
        "edit",
        json!({"path":"result.txt","old_string":"gamma","new_string":"delta"}),
    )
    .await
    .unwrap();
    assert_eq!(
        run(&capabilities, "glob", json!({"pattern":"**/*.txt"}))
            .await
            .unwrap()
            .values["files"],
        json!(["notes/a.txt", "result.txt"])
    );
    let grep = run(&capabilities, "grep", json!({"pattern":"delta"}))
        .await
        .unwrap();
    assert!(
        serde_json::to_string(&grep.values)
            .unwrap()
            .contains("result.txt")
    );
    let listed = run(&capabilities, "list", json!({"recursive":true}))
        .await
        .unwrap();
    assert_eq!(
        listed.values["entries"],
        json!(["notes/a.txt", "result.txt"])
    );
    run(&capabilities, "delete", json!({"path":"result.txt"}))
        .await
        .unwrap();
    assert_eq!(
        run(&capabilities, "read", json!({"path":"result.txt"}))
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert_eq!(count("create"), 1);
    assert_eq!(count("start"), 1);
    assert_eq!(count("rm"), 0);
    let args = std::fs::read_to_string(state().join("arguments")).unwrap();
    for flag in [
        "ARG:--pull\nARG:never",
        "ARG:--read-only",
        "ARG:--user\nARG:65534:65534",
        "ARG:--network\nARG:none",
    ] {
        assert!(args.contains(flag));
    }
    assert!(!args.contains("ARG:--mount"));
    drop(capabilities);
    wait_for_removal().await;
}

async fn literal_paths_and_binary_data() {
    let workspace = Docker::new(config()).unwrap();
    let cancellation = Cancellation::new();
    let path = "notes/$(touch nope) 'quoted'\n.txt";
    workspace
        .write(path, vec![0, 255, 1, 2], &cancellation)
        .await
        .unwrap();
    assert_eq!(
        workspace.read(path, &cancellation).await.unwrap(),
        [0, 255, 1, 2]
    );
    assert_eq!(
        workspace.list(".", true, &cancellation).await.unwrap(),
        [path]
    );
    let capabilities = workspace;
    assert_eq!(
        run(&capabilities, "read", json!({"path":path}))
            .await
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    run(
        &capabilities,
        "exec_shell",
        json!({"args":["ln","-s","/tmp","link"]}),
    )
    .await
    .unwrap();
    for name in ["read", "write", "delete"] {
        let mut args = json!({"path":"link/escape"});
        if name == "write" {
            args["content"] = json!("no");
        }
        assert_eq!(
            run(&capabilities, name, args).await.unwrap_err().code,
            "PERMISSION_DENIED"
        );
    }
    run(&capabilities, "delete", json!({"path":path}))
        .await
        .unwrap();
    assert!(
        run(&capabilities, "delete", json!({"path":"notes"}))
            .await
            .is_err()
    );
    assert_eq!(
        run(
            &capabilities,
            "exec_shell",
            json!({"args":["/bin/sh","-c","test ! -e nope"]})
        )
        .await
        .unwrap()
        .values["exit_code"],
        0
    );
    drop(capabilities);
    wait_for_removal().await;
}

async fn concurrent_init_and_isolated_workspaces() {
    let first = Docker::new(config()).unwrap();
    let second = Docker::new(config()).unwrap();
    let cancellation = Cancellation::new();
    futures_util::try_join!(
        first.write("a", b"one".to_vec(), &cancellation),
        first.write("b", b"two".to_vec(), &cancellation),
        second.write("a", b"other".to_vec(), &cancellation)
    )
    .unwrap();
    assert_eq!(count("create"), 2);
    assert_eq!(first.read("a", &cancellation).await.unwrap(), b"one");
    assert_eq!(second.read("a", &cancellation).await.unwrap(), b"other");
    first.close().await.unwrap();
    second.close().await.unwrap();
    first.close().await.unwrap();
    assert_eq!(count("rm"), 2);
    assert_eq!(
        first.read("a", &cancellation).await.unwrap_err().code,
        "UNAVAILABLE"
    );
    assert_eq!(count("create"), 2);
}

async fn failures_do_not_recreate_environment() {
    let capabilities = Docker::new(config()).unwrap();
    assert_eq!(
        run(&capabilities, "read", json!({"path":"missing"}))
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    let error = run(
        &capabilities,
        "exec_shell",
        json!({"args":["/bin/sh","-c","printf kept > kept; printf failed >&2; exit 7"]}),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code, "EXECUTION_FAILED");
    assert!(error.details.unwrap().contains("failed"));
    assert_eq!(
        run(&capabilities, "read", json!({"path":"kept"}))
            .await
            .unwrap()
            .values["content"],
        "kept"
    );
    assert_eq!(count("create"), 1);
    std::fs::write(state().join("oom"), "").unwrap();
    assert_eq!(
        run(&capabilities, "read", json!({"path":"kept"}))
            .await
            .unwrap_err()
            .code,
        "RESOURCE_EXHAUSTED"
    );
    assert_eq!(count("rm"), 1);
    assert_eq!(
        run(&capabilities, "read", json!({"path":"kept"}))
            .await
            .unwrap_err()
            .code,
        "UNAVAILABLE"
    );
    assert_eq!(count("create"), 1);
    std::fs::remove_file(state().join("oom")).unwrap();
    std::fs::write(state().join("create_fail"), "").unwrap();
    let workspace = Docker::new(config()).unwrap();
    let error = workspace.read("x", &Cancellation::new()).await.unwrap_err();
    assert_eq!(error.code, "UNAVAILABLE");
    assert!(!format!("{error:?}").contains("ENV_SECRET"));
    assert_eq!(count("rm"), 2);
}

async fn timeout_and_cancellation_close_owned_environment() {
    let mut short = config();
    short.sandbox.timeout = Duration::from_secs(1);
    let workspace = Docker::new(short).unwrap();
    workspace
        .write("a", b"data".to_vec(), &Cancellation::new())
        .await
        .unwrap();
    std::fs::write(state().join("hang"), "").unwrap();
    assert_eq!(
        workspace
            .read("a", &Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "TIMEOUT"
    );
    assert_eq!(count("rm"), 1);
    std::fs::remove_file(state().join("hang")).unwrap();
    let workspace = Docker::new(config()).unwrap();
    workspace
        .write("a", b"data".to_vec(), &Cancellation::new())
        .await
        .unwrap();
    std::fs::write(state().join("hang"), "").unwrap();
    std::fs::remove_file(state().join("exec_started")).unwrap();
    let cancellation = Cancellation::new();
    let pending = workspace.read("a", &cancellation);
    let cancel = async {
        wait_for_file(state().join("exec_started")).await;
        cancellation.cancel();
    };
    let (result, ()) = tokio::join!(pending, cancel);
    assert_eq!(result.unwrap_err().code, "CANCELLED");
    assert_eq!(count("rm"), 2);
    assert_eq!(
        workspace
            .read("a", &Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "UNAVAILABLE"
    );
}

async fn cancelled_lock_wait_does_not_close_environment() {
    let workspace = Docker::new(config()).unwrap();
    workspace
        .write("x", b"value".to_vec(), &Cancellation::new())
        .await
        .unwrap();
    std::fs::remove_file(state().join("exec_started")).unwrap();
    std::fs::write(state().join("wait"), "").unwrap();
    let active = Cancellation::new();
    let waiter = Cancellation::new();
    let first = workspace.read("x", &active);
    let second = async {
        wait_for_file(state().join("exec_started")).await;
        let waiting = workspace.read("x", &waiter);
        let cancel = async {
            tokio::task::yield_now().await;
            waiter.cancel();
        };
        let (result, ()) = tokio::join!(waiting, cancel);
        assert_eq!(result.unwrap_err().code, "CANCELLED");
        assert_eq!(count("rm"), 0);
        std::fs::remove_file(state().join("wait")).unwrap();
    };
    let (first, ()) = tokio::join!(first, second);
    assert_eq!(first.unwrap(), b"value");
    assert_eq!(count("rm"), 0);
    assert_eq!(count("create"), 1);
    workspace.close().await.unwrap();
}

async fn drop_in_flight_future_cleans_up() {
    let workspace = Docker::new(config()).unwrap();
    std::fs::write(state().join("hang"), "").unwrap();
    let cancellation = Cancellation::new();
    let mut future = Box::pin(workspace.read("x", &cancellation));
    tokio::select! {
        _=wait_for_file(state().join("exec_started"))=>{},
        result=&mut future=>panic!("unexpected early result {result:?}"),
    }
    drop(future);
    wait_for_removal().await;
    assert_eq!(
        workspace.read("x", &cancellation).await.unwrap_err().code,
        "UNAVAILABLE"
    );
    assert_eq!(count("create"), 1);
}

async fn drop_configured_tools_releases_environment() {
    let capabilities = Docker::new(config()).unwrap();
    run(&capabilities, "write", json!({"path":"x","content":"kept"}))
        .await
        .unwrap();
    drop(capabilities);
    wait_for_removal().await;
    assert_eq!(count("create"), 1);
    assert_eq!(count("rm"), 1);
}

async fn foreign_resource_is_never_removed() {
    let workspace = Docker::new(config()).unwrap();
    workspace
        .write("x", vec![], &Cancellation::new())
        .await
        .unwrap();
    std::fs::write(state().join("foreign"), "").unwrap();
    assert_eq!(
        workspace.close().await.unwrap_err().code,
        "PERMISSION_DENIED"
    );
    assert_eq!(count("rm"), 0);
    // Restore ownership for the guard's bounded Drop retry before fixture cleanup.
    std::fs::remove_file(state().join("foreign")).unwrap();
    wait_for_removal().await;
}

async fn configured_mount_and_env() {
    let host = state().join("host");
    std::fs::create_dir(&host).unwrap();
    std::fs::write(host.join("keep"), "application data").unwrap();
    let mut config = config();
    config.workdir = "/project".into();
    config.host_workdir = Some(host.clone());
    config
        .env
        .insert("MY_SETTING".into(), "secret-not-in-definition".into());
    let workspace = Docker::new(config).unwrap();
    assert!(!format!("{:?}", workspace.definition()).contains("secret"));
    assert!(operations().is_empty());
    workspace
        .write("x", vec![], &Cancellation::new())
        .await
        .unwrap();
    workspace.close().await.unwrap();
    assert_eq!(
        std::fs::read_to_string(host.join("keep")).unwrap(),
        "application data"
    );
    let args = std::fs::read_to_string(state().join("arguments")).unwrap();
    assert!(args.contains("ARG:--mount\nARG:type=bind,source="));
    assert!(args.contains(",target=/project"));
    assert!(args.contains("ARG:--env\nARG:MY_SETTING=secret-not-in-definition"));
}

async fn real_model_workspace_permission_denial() {
    use futures_util::StreamExt;
    use halo::{
        agent::model::chat,
        agent::permission::{Behavior, Decision},
        agent::provider::deepseek::DeepSeek,
        agent::{Agent, Input},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct WriteOnly(Docker);
    impl Workspace for WriteOnly {
        fn definition(&self) -> &halo::agent::workspace::Definition {
            self.0.definition()
        }

        fn tool(&self) -> Vec<halo::agent::tool::ToolDefinition> {
            self.0
                .tool()
                .into_iter()
                .filter(|tool| tool.name == "workspace_write")
                .collect()
        }

        fn mcp(&self) -> Vec<halo::agent::mcp::Definition> {
            Vec::new()
        }

        fn skill(&self) -> Vec<halo::agent::skill::Definition> {
            Vec::new()
        }

        fn execute<'a>(
            &'a self,
            call: Call,
            cancellation: &'a Cancellation,
        ) -> halo::agent::workspace::ToolFuture<'a> {
            Box::pin(async move { self.0.execute(call, cancellation).await })
        }
    }

    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct real DeepSeek Provider");
    let model = provider
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct real Chat");
    let checks = Arc::new(AtomicUsize::new(0));
    let workspace = WriteOnly(Docker::new(config()).unwrap());
    let agent = halo::harness::builder()
        .model(model)
        .workspace(workspace)
        .system_prompt("你必须调用 workspace_write，将内容写入路径 denied.txt；不要改用其他操作。")
        .permission(halo::agent::core::permission::Permission::new({
            let checks = Arc::clone(&checks);
            move |request: halo::agent::permission::Request, _: Cancellation| {
                let checks = Arc::clone(&checks);
                async move {
                    assert_eq!(request.resource, "workspace_write");
                    checks.fetch_add(1, Ordering::SeqCst);
                    Ok(Decision {
                        behavior: Behavior::Deny,
                        reason: Some("live acceptance denies file writes".into()),
                    })
                }
            }
        }))
        .with_max_iters(2)
        .build()
        .unwrap();
    let input = Input {
        metadata: json!({"chat_id":1,"agent_id":2,"session_id":3})
            .as_object()
            .unwrap()
            .clone(),
        payload: vec![halo::agent::Payload {
            r#type: "text".into(),
            data: json!({"value":"请创建文件 denied.txt，内容为 no。"}),
            mime_type: None,
        }],
    };
    let events: Vec<_> = agent
        .run(input, false, Cancellation::new())
        .await
        .unwrap()
        .collect()
        .await;
    assert!(!events.is_empty());
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    assert!(
        checks.load(Ordering::SeqCst) > 0,
        "the real model must select the protected operation"
    );
    assert!(
        operations().is_empty(),
        "Permission denial must prevent even container startup"
    );
}

async fn pre_cancel_and_invalid_arguments_are_no_io() {
    let mut invalid = config();
    invalid.workdir = "/".into();
    assert!(Docker::new(invalid).is_err());
    let mut invalid = config();
    invalid.env.insert("BAD=KEY".into(), "secret".into());
    assert!(Docker::new(invalid).is_err());
    let workspace = Docker::new(config()).unwrap();
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        workspace.read("x", &cancellation).await.unwrap_err().code,
        "CANCELLED"
    );
    for path in ["../x", "/tmp/x", "a/../x", "a//x", "a/./x", ""] {
        assert_eq!(
            workspace
                .read(path, &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
    }
    let capabilities = workspace;
    assert_eq!(
        run(
            &capabilities,
            "exec_shell",
            json!({"args":["echo"],"env":{}})
        )
        .await
        .unwrap_err()
        .code,
        "INVALID_ARGUMENTS"
    );
    assert!(operations().is_empty());
}

async fn missing_container_cleanup() {
    let workspace = Docker::new(config()).unwrap();
    workspace
        .write("x", vec![], &Cancellation::new())
        .await
        .unwrap();
    std::fs::write(state().join("absent"), "").unwrap();
    // Docker 29 emits lowercase "error: no such object". Missing is still
    // idempotent; authentication/daemon/foreign-resource errors must not be.
    workspace.close().await.unwrap();
    assert_eq!(count("rm"), 0);
}

async fn limits_and_startup_failure() {
    std::fs::write(state().join("start_fail"), "").unwrap();
    let workspace = Docker::new(config()).unwrap();
    assert_eq!(
        workspace
            .read("x", &Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "UNAVAILABLE"
    );
    assert_eq!(count("rm"), 1);
    std::fs::remove_file(state().join("start_fail")).unwrap();
    let mut config = config();
    config.sandbox.max_output = 4;
    let capabilities = Docker::new(config).unwrap();
    assert_eq!(
        run(
            &capabilities,
            "exec_shell",
            json!({"args":["/bin/sh","-c","printf 123456789"]})
        )
        .await
        .unwrap_err()
        .code,
        "OUTPUT_LIMIT"
    );
    assert_eq!(count("rm"), 2);
    assert_eq!(
        run(&capabilities, "read", json!({"path":"x"}))
            .await
            .unwrap_err()
            .code,
        "UNAVAILABLE"
    );
    assert_eq!(count("create"), 2);
}

#[tokio::test]
#[ignore = "requires running Docker and HEART_WORKSPACE_TEST_IMAGE already present; never pulls"]
async fn docker_workspace_real_container_smoke() {
    let image = std::env::var("HEART_WORKSPACE_TEST_IMAGE")
        .expect("provide a prepared image with sh/coreutils/find");
    let workspace = Docker::new(Config {
        sandbox: sandbox::docker::Config {
            image,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();
    let capabilities = workspace;
    let mut id = None;
    let result = async {
        let hostname=run(&capabilities,"exec_shell",json!({"args":["/bin/cat","/etc/hostname"]})).await?;
        let hostname=hostname.values["stdout"].as_str().unwrap().trim().to_owned();
        assert_eq!(hostname.len(),12); assert!(hostname.bytes().all(|byte|byte.is_ascii_hexdigit()));
        id=Some(hostname);
        run(&capabilities,"write",json!({"path":"input.txt","content":"shared"})).await?;
        run(&capabilities,"exec_shell",json!({"args":["/bin/sh","-c","test $(id -u) != 0 && cat input.txt > output.txt; printf ' environment' >> output.txt"]})).await?;
        assert_eq!(run(&capabilities,"read",json!({"path":"output.txt"})).await?.values["content"],"shared environment");
        run(&capabilities,"edit",json!({"path":"output.txt","old_string":"environment","new_string":"container"})).await?;
        assert_eq!(run(&capabilities,"grep",json!({"pattern":"container"})).await?.values["files"],json!(["output.txt"]));
        run(&capabilities,"delete",json!({"path":"output.txt"})).await?;
        Ok::<_, Error>(())
    }
    .await;
    drop(capabilities);
    if let Some(id) = id {
        real_removal(&id).await;
    }
    result.unwrap();
}

async fn real_removal(id: &str) {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let output = tokio::process::Command::new("docker")
                .args(["inspect", id])
                .output()
                .await
                .unwrap();
            if !output.status.success() {
                assert!(
                    String::from_utf8_lossy(&output.stderr)
                        .to_ascii_lowercase()
                        .contains("no such"),
                    "cleanup lookup failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("test-owned container was not released");
}

#[tokio::test]
#[ignore = "requires running Docker and HEART_WORKSPACE_TEST_IMAGE already present; never pulls"]
async fn docker_workspace_real_bind_mount_smoke() {
    use std::os::unix::fs::PermissionsExt;
    let fixture = Fixture::new();
    let host = fixture.0.join("host");
    std::fs::create_dir(&host).unwrap();
    std::fs::set_permissions(&host, std::fs::Permissions::from_mode(0o777)).unwrap();
    std::fs::write(host.join("source.txt"), "host source").unwrap();
    let workspace = Docker::new(Config {
        sandbox: sandbox::docker::Config {
            image: std::env::var("HEART_WORKSPACE_TEST_IMAGE").unwrap(),
            ..Default::default()
        },
        host_workdir: Some(host.clone()),
        env: std::collections::BTreeMap::from([("WORKSPACE_SETTING".into(), "configured".into())]),
        ..Default::default()
    })
    .unwrap();
    let capabilities = workspace;
    let mut id = None;
    let result=async {
        let hostname=run(&capabilities,"exec_shell",json!({"args":["/bin/cat","/etc/hostname"]})).await?;
        id=Some(hostname.values["stdout"].as_str().unwrap().trim().to_owned());
        assert_eq!(run(&capabilities,"read",json!({"path":"source.txt"})).await?.values["content"],"host source");
        run(&capabilities,"write",json!({"path":"from_tool.txt","content":"tool"})).await?;
        run(&capabilities,"exec_shell",json!({"args":["/bin/sh","-c","cat from_tool.txt > from_command.txt; printf ' %s' \"$WORKSPACE_SETTING\" >> from_command.txt"]})).await?;
        assert_eq!(run(&capabilities,"read",json!({"path":"from_command.txt"})).await?.values["content"],"tool configured");
        Ok::<_,Error>(())
    }.await;
    drop(capabilities);
    if let Some(id) = id {
        real_removal(&id).await;
    }
    result.unwrap();
    // Container/owned-volume cleanup must never delete the application bind.
    assert_eq!(
        std::fs::read_to_string(host.join("from_command.txt")).unwrap(),
        "tool configured"
    );
    assert_eq!(
        std::fs::read_to_string(host.join("source.txt")).unwrap(),
        "host source"
    );
}

#[tokio::test]
#[ignore = "requires running Docker and HEART_WORKSPACE_TEST_IMAGE already present; never pulls"]
async fn docker_workspace_real_cancellation_smoke() {
    use std::os::unix::fs::PermissionsExt;
    let fixture = Fixture::new();
    let host = fixture.0.join("host");
    std::fs::create_dir(&host).unwrap();
    std::fs::set_permissions(&host, std::fs::Permissions::from_mode(0o777)).unwrap();
    let workspace = Docker::new(Config {
        sandbox: sandbox::docker::Config {
            image: std::env::var("HEART_WORKSPACE_TEST_IMAGE").unwrap(),
            ..Default::default()
        },
        host_workdir: Some(host.clone()),
        ..Default::default()
    })
    .unwrap();
    let capabilities = workspace;
    let hostname = run(
        &capabilities,
        "exec_shell",
        json!({"args":["/bin/cat","/etc/hostname"]}),
    )
    .await
    .unwrap();
    let id = hostname.values["stdout"]
        .as_str()
        .unwrap()
        .trim()
        .to_owned();
    let cancellation = Cancellation::new();
    let command = capabilities.execute(
        call(
            "exec_shell",
            json!({"args":["/bin/sh","-c","printf started > started; exec sleep 60"]}),
        ),
        &cancellation,
    );
    let cancel = async {
        wait_for_file(host.join("started")).await;
        cancellation.cancel();
    };
    let (result, ()) = tokio::join!(command, cancel);
    real_removal(&id).await;
    assert_eq!(result.unwrap_err().code, "CANCELLED");
    assert_eq!(
        run(&capabilities, "read", json!({"path":"started"}))
            .await
            .unwrap_err()
            .code,
        "UNAVAILABLE"
    );
    assert_eq!(
        std::fs::read_to_string(host.join("started")).unwrap(),
        "started"
    );
}
