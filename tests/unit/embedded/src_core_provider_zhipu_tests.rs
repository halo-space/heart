use super::*;

#[test]
fn config_hides_api_key_and_uses_zhipu_v4_endpoint() {
    let config = Config {
        api_key: "do-not-log-this".into(),
        ..Config::default()
    };
    assert_eq!(config.provider_name, "zhipu");
    assert_eq!(config.base_url, "https://open.bigmodel.cn/api/paas/v4");
    assert!(!format!("{config:?}").contains("do-not-log-this"));
}

#[test]
fn embedding_dimensions_are_checked_against_supported_values() {
    for dimensions in [256, 512, 768, 1024, 2048] {
        assert!(validate_dimensions(Some(dimensions)).is_ok());
    }

    assert_eq!(
        validate_dimensions(Some(255)).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(
        validate_dimensions(Some(2049)).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
    assert!(validate_dimensions(None).is_ok());
}
