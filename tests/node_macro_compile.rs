//! Definition-site compiler checks, including a renamed library dependency.
//! A separate target prevents taking the parent cargo test build lock.
use std::{path::PathBuf, process::Command};

#[test]
fn node_declarations_compile_or_fail_at_the_definition_site() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cases = [
        ("pass_renamed", None),
        ("pass_cfg", None),
        ("keyword_alias", None),
        ("sync_function", Some("must be async")),
        ("wrong_arity", Some("one owned business input")),
        ("borrowed_input", Some("concrete owned node I/O")),
        ("borrowed_output", Some("concrete owned node I/O")),
        ("borrowed_cancellation", Some("Cancellation by value")),
        (
            "unknown_option",
            Some("expected call = method or signal = method"),
        ),
        ("conflicting_options", Some("exactly one")),
        ("loose_method", Some("one owned business input")),
        ("missing_method", Some("defined in this inherent impl")),
        ("mutable_receiver", Some("immutable &self")),
        ("method_owned_cancellation", Some("must take &Cancellation")),
        ("call_returns_signal", Some("Signal results require")),
        (
            "signal_returns_plain",
            Some("explicitly return Result<node::Signal<O>, E>"),
        ),
        ("generic_function", Some("generic node signatures")),
        ("trait_impl", Some("concrete inherent impl")),
        ("missing_schema", Some("JsonSchema")),
        ("missing_serialize", Some("Serialize")),
        ("missing_error_conversion", Some("OtherError")),
        (
            "non_send_function",
            Some("cannot be sent between threads safely"),
        ),
        (
            "non_send_method",
            Some("cannot be sent between threads safely"),
        ),
    ];
    for (case, expected_error) in cases {
        let manifest = if case == "keyword_alias" {
            "tests/fixtures/node_macro_keyword/Cargo.toml"
        } else {
            "tests/fixtures/node_macro/Cargo.toml"
        };
        let output = Command::new(env!("CARGO"))
            .args(["check", "--offline", "--quiet", "--manifest-path"])
            .arg(root.join(manifest))
            .args(["--bin", case])
            .arg("--target-dir")
            .arg(root.join("target/node-macro-ui"))
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .expect("run the local Rust compiler");
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        match expected_error {
            None => assert!(output.status.success(), "{case} must compile: {diagnostic}"),
            Some(expected) => {
                assert!(!output.status.success(), "{case} unexpectedly compiled");
                assert!(
                    diagnostic.contains(expected),
                    "{case}: expected {expected:?}, got {diagnostic}"
                );
            }
        }
    }
}
