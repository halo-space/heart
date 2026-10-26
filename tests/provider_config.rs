use halo::agent::provider::openai::{self, OpenAI};

#[test]
fn provider_configuration_rejects_unsafe_urls_and_redacts_credentials() {
    let config = openai::Config {
        api_key: "redaction-test-key".into(),
        base_url: "https://example.com/private-path".into(),
        ..Default::default()
    };
    let debug = format!("{config:?}");
    assert!(!debug.contains("redaction-test-key"));
    assert!(!debug.contains("private-path"));

    for url in [
        "file:///tmp/x",
        "https://name:secret@example.com",
        "https://example.com?api_key=secret",
        "https://example.com#secret",
    ] {
        assert_eq!(
            OpenAI::new(openai::Config {
                base_url: url.into(),
                api_key: "redaction-test-key".into(),
                ..Default::default()
            })
            .err()
            .unwrap()
            .code,
            "INVALID_ARGUMENTS"
        );
    }

    assert_eq!(
        OpenAI::new(openai::Config::default()).err().unwrap().code,
        "AUTHENTICATION"
    );
}
