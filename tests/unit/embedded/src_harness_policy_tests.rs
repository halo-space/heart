use super::Policy;

#[test]
fn default_policy_is_valid() {
    assert!(Policy::default().validate().is_ok());
    assert!(
        Policy {
            max_iters: 0,
            ..Policy::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        Policy {
            max_concurrency: 0,
            ..Policy::default()
        }
        .validate()
        .is_err()
    );
}
