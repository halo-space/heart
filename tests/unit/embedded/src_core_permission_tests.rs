use futures_executor::block_on;

use crate::components::permission::Permission as _;
use crate::{Cancellation, permission::Behavior};

fn request(action: &str) -> crate::permission::Request {
    crate::permission::Request {
        action: action.into(),
        resource: "resource".into(),
        metadata: Default::default(),
    }
}

#[test]
fn built_in_policies_keep_allow_deny_ask_distinct() {
    let cancellation = Cancellation::new();
    let cases = [
        (super::default::new(), "read", Behavior::Allow),
        (super::default::new(), "write", Behavior::Ask),
        (super::explore::new(), "write", Behavior::Deny),
        (super::edit::new(), "write", Behavior::Allow),
        (super::edit::new(), "delete", Behavior::Ask),
        (super::bypass::new(), "anything", Behavior::Allow),
    ];

    for (policy, action, expected) in cases {
        let decision = block_on(policy.check(&request(action), &cancellation)).unwrap();
        assert_eq!(decision.behavior, expected, "action={action}");
    }
}

#[test]
fn built_in_policy_rejects_cancelled_checks() {
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let result = block_on(super::default::new().check(&request("read"), &cancellation));
    assert_eq!(result.unwrap_err().code, "CANCELLED");
}
