//! Opt-in real-container smoke test. No image pull or Docker daemon startup.
use halo::agent::Cancellation;
use halo::{
    agent::core::sandbox::docker::{Config, Docker},
    agent::sandbox::{Request, Sandbox},
};
use serde_json::json;

#[tokio::test]
#[ignore = "requires a running Docker daemon and HALO_SANDBOX_TEST_IMAGE already present"]
async fn real_container_uses_nonroot_user_and_writable_scratch() {
    let image =
        std::env::var("HALO_SANDBOX_TEST_IMAGE").expect("select an existing image with /bin/sh");
    let sandbox = Docker::new(Config {
        image,
        ..Default::default()
    })
    .unwrap();
    let result = sandbox
        .execute(
            Request {
                operation: "command".into(),
                input: json!({"args":["/bin/sh", "-c",
            "test $(id -u) != 0 && printf isolated >/workspace/probe && cat /workspace/probe"]}),
                metadata: Default::default(),
            },
            &Cancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(result.data["stdout"], "isolated");
}
