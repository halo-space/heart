use super::retry::Policy;
use crate::Error;

#[test]
fn retry_policy_only_retries_transient_errors_and_caps_delay() {
    let policy = Policy {
        max_retries: 2,
        min_delay_ms: 10,
        max_delay_ms: 15,
    };
    assert!(policy.validate().is_ok());
    assert!(policy.should_retry(0, &Error::new("TIMEOUT", "temporary")));
    assert!(!policy.should_retry(2, &Error::new("TIMEOUT", "temporary")));
    assert!(!policy.should_retry(0, &Error::new("INVALID_ARGUMENTS", "permanent")));
    assert_eq!(policy.delay(0).as_millis(), 10);
    assert_eq!(policy.delay(2).as_millis(), 15);
}
