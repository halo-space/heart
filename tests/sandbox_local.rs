//! Host execution regressions use only static trusted test commands.
#![cfg(unix)]

use halo::agent::Cancellation;
use halo::sandbox::{Local, Request, Sandbox, local::Config};
use serde_json::json;
use std::time::Duration;

fn request(args: &[&str]) -> Request {
    Request {
        operation: "command".into(),
        input: json!({"args": args}),
        metadata: Default::default(),
    }
}

#[tokio::test]
async fn argv_is_literal_and_metadata_is_preserved() {
    let mut request = request(&["/usr/bin/printf", "%s", "$(exit 7); * $HOME"]);
    request.metadata.insert("request".into(), json!(42));
    let result = Local::default()
        .execute(request, &Cancellation::new())
        .await
        .unwrap();
    assert_eq!(
        result.data,
        json!({"stdout":"$(exit 7); * $HOME", "stderr":"", "exit_code":0})
    );
    assert_eq!(result.metadata["request"], 42);
}

#[tokio::test]
async fn nonzero_exit_returns_direct_error_with_actual_output() {
    let error = Local::default()
        .execute(
            request(&["/bin/sh", "-c", "printf out; printf err >&2; exit 7"]),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, "EXECUTION_FAILED");
    let data: serde_json::Value = serde_json::from_str(error.details.as_ref().unwrap()).unwrap();
    assert_eq!(data, json!({"stdout":"out", "stderr":"err", "exit_code":7}));
}

#[tokio::test]
async fn output_limit_and_invalid_utf8_return_errors() {
    let local = Local::new(Config {
        max_output: 3,
        ..Default::default()
    })
    .unwrap();
    for (args, code) in [
        (vec!["/usr/bin/printf", "abcd"], "OUTPUT_LIMIT"),
        (vec!["/bin/sh", "-c", "printf '\\377'"], "INVALID_RESPONSE"),
        (vec!["/bin/sh", "-c", "printf abcd >&2"], "OUTPUT_LIMIT"),
    ] {
        assert_eq!(
            local
                .execute(request(&args), &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            code
        );
    }
}

#[tokio::test]
async fn timeout_stops_waiting_for_direct_child() {
    let local = Local::new(Config {
        timeout: Duration::from_millis(50),
        ..Default::default()
    })
    .unwrap();
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        local.execute(request(&["/bin/sleep", "20"]), &Cancellation::new()),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert_eq!(error.code, "TIMEOUT");
}

#[tokio::test]
async fn cancellation_wakes_pending_command() {
    let cancellation = Cancellation::new();
    let local = Local::default();
    let operation = local.execute(request(&["/bin/sleep", "20"]), &cancellation);
    let cancel = async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancellation.cancel();
    };
    let (result, ()) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(operation, cancel)
    })
    .await
    .unwrap();
    assert_eq!(result.unwrap_err().code, "CANCELLED");
}

#[tokio::test]
async fn invalid_request_and_precancel_do_not_execute() {
    let local = Local::default();
    for input in [
        json!({"args": []}),
        json!({"args": [" "]}),
        json!({"args": ["x\u{0}"]}),
        json!({"args": ["x"], "extra":1}),
    ] {
        let mut r = request(&["x"]);
        r.input = input;
        assert_eq!(
            local
                .execute(r, &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
    }
    let mut r = request(&["x"]);
    r.operation = "unknown".into();
    assert_eq!(
        local
            .execute(r, &Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "UNSUPPORTED"
    );
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        local
            .execute(request(&["missing-program"]), &cancellation)
            .await
            .unwrap_err()
            .code,
        "CANCELLED"
    );
}

#[tokio::test]
async fn missing_executable_is_not_found_without_fallback() {
    assert_eq!(
        Local::default()
            .execute(
                request(&["/halo-static-missing-executable"]),
                &Cancellation::new()
            )
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
}

#[test]
fn config_and_runtime_requirements_are_explicit() {
    for config in [
        Config {
            timeout: Duration::ZERO,
            ..Default::default()
        },
        Config {
            max_output: 0,
            ..Default::default()
        },
    ] {
        assert!(Local::new(config).is_err());
    }
    let result = futures_executor::block_on(
        Local::default().execute(request(&["/usr/bin/printf", "x"]), &Cancellation::new()),
    );
    assert_eq!(result.unwrap_err().code, "UNAVAILABLE");
}

#[tokio::test]
async fn dropping_future_terminates_the_direct_child() {
    struct Marker(std::path::PathBuf);
    impl Drop for Marker {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let marker = Marker(
        std::env::temp_dir().join(format!("halo-local-child-{:032x}", rand::random::<u128>())),
    );
    let request = Request {
        operation: "command".into(),
        input: json!({"args":["/bin/sh", "-c", "printf '%s' \"$$\" > \"$1\"; exec /bin/sleep 20", "test", marker.0]}),
        metadata: Default::default(),
    };
    let local = Local::default();
    let cancellation = Cancellation::new();
    let mut operation = Box::pin(local.execute(request, &cancellation));
    let pid = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            tokio::select! {
                result = &mut operation => panic!("command ended before marker: {result:?}"),
                _ = tokio::time::sleep(Duration::from_millis(10)) => {},
            }
            if let Ok(pid) = std::fs::read_to_string(&marker.0)
                && let Ok(pid) = pid.parse::<u32>()
            {
                break pid;
            }
        }
    })
    .await
    .unwrap();
    drop(operation);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let status = tokio::process::Command::new("/bin/kill")
                .args(["-0", &pid.to_string()])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await
                .unwrap();
            if !status.success() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("direct child should terminate after the future is dropped");
}
