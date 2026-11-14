use super::Webhook;

#[test]
fn validates_endpoint() {
    assert!(Webhook::new(" ").is_err());
    assert_eq!(
        Webhook::new("http://127.0.0.1/hook").unwrap().endpoint(),
        "http://127.0.0.1/hook"
    );
}
