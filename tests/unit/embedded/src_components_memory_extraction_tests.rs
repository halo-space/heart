use super::validate;
use serde_json::json;

#[test]
fn accepts_only_nonempty_durable_text_by_owner_and_type() {
    for valid in [
        json!({}),
        json!({"user":{"summary":"User context"}}),
        json!({"agent":{"fact":"Agent-specific fact","preference":"Concise"}}),
    ] {
        let values = valid.as_object().unwrap().clone();
        assert!(validate(&values).is_ok(), "{valid}");
    }

    for invalid in [
        json!({"session":{"summary":"not a durable owner"}}),
        json!({"user":["not an object"]}),
        json!({"user":{"id":"not a memory type"}}),
        json!({"agent":{"fact":"  "}}),
    ] {
        let values = invalid.as_object().unwrap().clone();
        assert_eq!(
            validate(&values).unwrap_err().code,
            "INVALID_RESPONSE",
            "{invalid}"
        );
    }
}
