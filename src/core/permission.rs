use std::future::Future;

use crate::{
    Cancellation, Error,
    permission::{Decision, Permission as Contract, Request},
};

/// Built-in permission policies. The policy is intentionally small: it only
/// maps the request's action to the existing `Allow`/`Deny`/`Ask` contract.
/// Waiting, user replies and checkpoint state remain Runtime concerns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Default,
    Explore,
    Edit,
    Bypass,
}

/// A built-in permission implementation selected through one of the
/// `default`, `explore`, `edit` or `bypass` modules below.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Policy {
    mode: Mode,
}

impl Policy {
    const fn new(mode: Mode) -> Self {
        Self { mode }
    }
}

pub mod bypass;
pub mod default;
pub mod edit;
pub mod explore;

fn action(request: &Request) -> String {
    request.action.trim().to_ascii_lowercase()
}

fn is_read(action: &str) -> bool {
    matches!(
        action,
        "read" | "list" | "search" | "inspect" | "glob" | "grep" | "stat"
    )
}

fn is_edit(action: &str) -> bool {
    matches!(
        action,
        "write" | "create" | "update" | "edit" | "append" | "mkdir" | "copy" | "move"
    )
}

fn is_destructive(action: &str) -> bool {
    matches!(
        action,
        "delete" | "remove" | "execute" | "exec" | "run" | "call" | "shell"
    )
}

fn decision(mode: Mode, request: &Request) -> Decision {
    let action = action(request);
    let (behavior, reason) = match mode {
        Mode::Bypass => (crate::permission::Behavior::Allow, "bypass policy"),
        Mode::Explore if is_read(&action) => {
            (crate::permission::Behavior::Allow, "read-only action")
        }
        Mode::Explore => (
            crate::permission::Behavior::Deny,
            "explore policy is read-only",
        ),
        Mode::Edit if is_read(&action) || is_edit(&action) => {
            (crate::permission::Behavior::Allow, "read or edit action")
        }
        Mode::Edit if is_destructive(&action) => (
            crate::permission::Behavior::Ask,
            "destructive action requires confirmation",
        ),
        Mode::Edit => (
            crate::permission::Behavior::Ask,
            "action requires confirmation",
        ),
        Mode::Default if is_read(&action) => {
            (crate::permission::Behavior::Allow, "read-only action")
        }
        Mode::Default => (
            crate::permission::Behavior::Ask,
            "action requires confirmation",
        ),
    };
    Decision {
        behavior,
        reason: Some(reason.into()),
    }
}

impl Contract for Policy {
    async fn check(
        &self,
        request: &Request,
        cancellation: &Cancellation,
    ) -> Result<Decision, Error> {
        super::operation::check(cancellation)?;
        if request.action.trim().is_empty() || request.resource.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "permission action and resource are required",
            ));
        }
        Ok(decision(self.mode, request))
    }
}

/// Caller-defined permission policy; no implicit global allow, model call or
/// approval workflow. Allow/deny/ask are decisions; only check failures are errors.
pub struct Permission<F> {
    check: F,
}

impl<F> Permission<F> {
    pub fn new(check: F) -> Self {
        Self { check }
    }
}

impl<F, Fut> Contract for Permission<F>
where
    F: Fn(Request, Cancellation) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Decision, Error>> + Send,
{
    async fn check(
        &self,
        request: &Request,
        cancellation: &Cancellation,
    ) -> Result<Decision, Error> {
        super::operation::check(cancellation)?;
        if request.action.trim().is_empty() || request.resource.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "permission action and resource are required",
            ));
        }
        super::operation::cancellable(
            (self.check)(request.clone(), cancellation.clone()),
            cancellation,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use futures_executor::block_on;

    use super::Contract;
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
}
