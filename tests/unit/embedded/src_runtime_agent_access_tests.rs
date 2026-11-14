use super::{Access, Failure, check_permission, check_tool};
use crate::{
    Cancellation,
    permission::{Behavior, Decision, Permission, Request},
    tool::Call,
};
use serde_json::{Map, Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone)]
struct Policy {
    decision: Decision,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl Permission for Policy {
    async fn check(&self, request: &Request, _: &Cancellation) -> Result<Decision, crate::Error> {
        self.requests.lock().unwrap().push(request.clone());
        Ok(self.decision.clone())
    }
}

#[derive(Clone)]
struct Hook {
    name: &'static str,
    calls: Arc<Mutex<Vec<&'static str>>>,
    change_resource: bool,
    change_after: bool,
    fail: Option<&'static str>,
}

impl crate::middleware::Middleware for Hook {
    async fn before_permission(
        &self,
        request: &mut Value,
        _: &Cancellation,
    ) -> Result<(), crate::Error> {
        self.calls.lock().unwrap().push(self.name);
        if self.fail == Some("before") {
            return Err(crate::Error::new("HOOK_FAILED", "before hook failed"));
        }
        if self.change_resource {
            request["resource"] = json!("different-operation");
        }
        Ok(())
    }

    async fn after_permission(
        &self,
        request: &mut Value,
        _: &Cancellation,
    ) -> Result<(), crate::Error> {
        self.calls.lock().unwrap().push("after");
        if self.fail == Some("after") {
            return Err(crate::Error::new("HOOK_FAILED", "after hook failed"));
        }
        if self.change_after {
            request["resource"] = json!("different-operation");
        }
        Ok(())
    }
}

fn call() -> Call {
    Call {
        call_id: "call-live-fact".into(),
        name: "lookup".into(),
        arguments: serde_json::from_value(json!({"city":"Beijing"})).unwrap(),
    }
}

fn policy(behavior: Behavior) -> (Access, Arc<Mutex<Vec<Request>>>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut access = Access::default();
    access.set(Policy {
        decision: Decision {
            behavior,
            reason: Some("policy reason".into()),
        },
        requests: requests.clone(),
    });
    (access, requests)
}

#[tokio::test]
async fn permission_checks_the_actual_call_between_ordered_hooks() {
    let (access, requests) = policy(Behavior::Allow);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut hooks = crate::runtime::agent::hooks::Hooks::default();
    hooks.add(Hook {
        name: "before",
        calls: calls.clone(),
        change_resource: false,
        change_after: false,
        fail: None,
    });
    let mut metadata = Map::new();
    metadata.insert("chat_id".into(), json!(101));

    let decision = check_permission(
        &access,
        &hooks,
        metadata,
        &call(),
        None,
        |_| std::future::ready(Ok(())),
        &Cancellation::new(),
    )
    .await
    .unwrap_or_else(|_| panic!("permission check should complete"));

    assert_eq!(decision.behavior, Behavior::Allow);
    assert_eq!(*calls.lock().unwrap(), ["before", "after"]);
    assert_eq!(
        requests.lock().unwrap().as_slice(),
        &[Request {
            action: "call".into(),
            resource: "lookup".into(),
            metadata: serde_json::from_value(json!({
                "chat_id":101,
                "call":{"call_id":"call-live-fact","name":"lookup","arguments":{"city":"Beijing"}},
            }))
            .unwrap(),
        }]
    );
}

#[tokio::test]
async fn deny_is_a_normal_decision_and_policy_failure_stays_technical() {
    let (access, _) = policy(Behavior::Deny);
    let decision = check_permission(
        &access,
        &crate::runtime::agent::hooks::Hooks::default(),
        Map::new(),
        &call(),
        None,
        |_| std::future::ready(Ok(())),
        &Cancellation::new(),
    )
    .await
    .unwrap_or_else(|_| panic!("deny is a normal policy decision"));
    assert_eq!(decision.behavior, Behavior::Deny);

    struct Broken;
    impl Permission for Broken {
        async fn check(&self, _: &Request, _: &Cancellation) -> Result<Decision, crate::Error> {
            Err(crate::Error::new(
                "UNAVAILABLE",
                "policy backend unavailable",
            ))
        }
    }
    let mut access = Access::default();
    access.set(Broken);
    let error = check_permission(
        &access,
        &crate::runtime::agent::hooks::Hooks::default(),
        Map::new(),
        &call(),
        None,
        |_| std::future::ready(Ok(())),
        &Cancellation::new(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Failure::Policy(error) if error.code == "UNAVAILABLE"));
}

#[tokio::test]
async fn permission_hook_cannot_redirect_the_authorized_operation() {
    let (access, requests) = policy(Behavior::Allow);
    let mut hooks = crate::runtime::agent::hooks::Hooks::default();
    hooks.add(Hook {
        name: "redirect",
        calls: Arc::default(),
        change_resource: true,
        change_after: false,
        fail: None,
    });
    let error = check_permission(
        &access,
        &hooks,
        Map::new(),
        &call(),
        None,
        |_| std::future::ready(Ok(())),
        &Cancellation::new(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Failure::Hook(error) if error.code == "INVALID_ARGUMENTS"));
    assert!(requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn tool_gate_distinguishes_allow_deny_ask_and_matching_approval() {
    let call = call();
    for (behavior, approval, expected) in [
        (Behavior::Allow, None, None),
        (Behavior::Deny, None, Some("PERMISSION_DENIED")),
        (Behavior::Ask, None, Some("WAITING")),
        (Behavior::Ask, Some(call.clone()), None),
    ] {
        let (access, requests) = policy(behavior);
        let result = check_tool(
            &access,
            &crate::runtime::agent::hooks::Hooks::default(),
            Map::new(),
            &call,
            &Cancellation::new(),
            None,
            None,
            approval.as_ref(),
            None,
            |_| std::future::ready(Ok(())),
        )
        .await;
        match expected {
            Some(code) => {
                assert!(matches!(result, Err(Failure::Policy(error)) if error.code == code))
            }
            None => assert!(result.is_ok()),
        }
        assert_eq!(requests.lock().unwrap().len(), 1);
        assert_eq!(
            requests.lock().unwrap()[0].metadata["call"]["call_id"],
            call.call_id
        );
    }
}

#[tokio::test]
async fn after_hook_cannot_rewrite_a_decision_into_authorizing_another_operation() {
    let (access, requests) = policy(Behavior::Allow);
    let mut hooks = crate::runtime::agent::hooks::Hooks::default();
    hooks.add(Hook {
        name: "redirect-after",
        calls: Arc::default(),
        change_resource: false,
        change_after: true,
        fail: None,
    });
    let result = check_tool(
        &access,
        &hooks,
        Map::new(),
        &call(),
        &Cancellation::new(),
        None,
        None,
        None,
        None,
        |_| std::future::ready(Ok(())),
    )
    .await;
    assert!(matches!(result, Err(Failure::Hook(error)) if error.code == "INVALID_ARGUMENTS"));
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn permission_hook_failures_are_control_errors_at_their_actual_stage() {
    for stage in ["before", "after"] {
        let (access, requests) = policy(Behavior::Allow);
        let mut hooks = crate::runtime::agent::hooks::Hooks::default();
        hooks.add(Hook {
            name: "failure",
            calls: Arc::default(),
            change_resource: false,
            change_after: false,
            fail: Some(stage),
        });
        let error = check_permission(
            &access,
            &hooks,
            Map::new(),
            &call(),
            None,
            |_| std::future::ready(Ok(())),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Failure::Hook(error) if error.code == "HOOK_FAILED"));
        assert_eq!(
            requests.lock().unwrap().len(),
            usize::from(stage == "after")
        );
    }
}

#[tokio::test]
async fn pending_permission_check_is_cancellable_without_retrying_policy() {
    struct Pending {
        entered: Arc<AtomicBool>,
    }
    impl Permission for Pending {
        async fn check(&self, _: &Request, _: &Cancellation) -> Result<Decision, crate::Error> {
            self.entered.store(true, Ordering::SeqCst);
            std::future::pending().await
        }
    }

    let entered = Arc::new(AtomicBool::new(false));
    let mut access = Access::default();
    access.set(Pending {
        entered: entered.clone(),
    });
    let cancellation = Cancellation::new();
    let cancel = cancellation.clone();
    let cancel_task = tokio::spawn(async move {
        while !entered.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
        cancel.cancel();
    });
    let error = check_tool(
        &access,
        &crate::runtime::agent::hooks::Hooks::default(),
        Map::new(),
        &call(),
        &cancellation,
        None,
        None,
        None,
        None,
        |_| std::future::ready(Ok(())),
    )
    .await
    .unwrap_err();
    cancel_task.await.expect("cancel task");
    assert!(matches!(error, Failure::Policy(error) if error.code == "CANCELLED"));
}

#[tokio::test]
async fn permission_resume_reuses_acknowledged_hooks_and_rechecks_current_policy() {
    struct WaitThenAllow {
        requests: Arc<Mutex<Vec<Request>>>,
    }
    impl Permission for WaitThenAllow {
        async fn check(
            &self,
            request: &Request,
            _: &Cancellation,
        ) -> Result<Decision, crate::Error> {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request.clone());
            if requests.len() == 1 {
                Err(crate::Error::new("WAITING", "policy pending"))
            } else {
                Ok(Decision {
                    behavior: Behavior::Allow,
                    reason: None,
                })
            }
        }
    }

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut access = Access::default();
    access.set(WaitThenAllow {
        requests: requests.clone(),
    });
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut hooks = crate::runtime::agent::hooks::Hooks::default();
    hooks.add(Hook {
        name: "acknowledged",
        calls: calls.clone(),
        change_resource: false,
        change_after: false,
        fail: None,
    });

    let mut saved = None;
    let first = check_permission(
        &access,
        &hooks,
        Map::new(),
        &call(),
        None,
        |state| {
            saved = Some(state);
            std::future::ready(Ok(()))
        },
        &Cancellation::new(),
    )
    .await
    .unwrap_err();
    assert!(matches!(first, Failure::Policy(error) if error.code == "WAITING"));
    let saved = saved.expect("the current permission state must be checkpointable");
    assert_eq!(saved.stage, "check");
    assert!(saved.output.is_none());
    assert_eq!(*calls.lock().unwrap(), ["acknowledged"]);

    let mut restored = None;
    let decision = check_permission(
        &access,
        &hooks,
        Map::new(),
        &call(),
        Some(saved),
        |state| {
            restored = Some(state);
            std::future::ready(Ok(()))
        },
        &Cancellation::new(),
    )
    .await
    .unwrap_or_else(|_| panic!("restored permission should recheck and allow"));
    assert_eq!(decision.behavior, Behavior::Allow);
    assert_eq!(*calls.lock().unwrap(), ["acknowledged", "after"]);
    assert_eq!(requests.lock().unwrap().len(), 2);
    assert_eq!(restored.expect("final state").stage, "complete");
}

#[tokio::test]
async fn permission_is_rechecked_per_call_and_run_metadata_is_not_cached() {
    let (access, requests) = policy(Behavior::Allow);
    let call = call();
    for chat_id in [201, 202] {
        let metadata = Map::from_iter([("chat_id".into(), json!(chat_id))]);
        check_tool(
            &access,
            &crate::runtime::agent::hooks::Hooks::default(),
            metadata,
            &call,
            &Cancellation::new(),
            None,
            None,
            None,
            None,
            |_| std::future::ready(Ok(())),
        )
        .await
        .unwrap_or_else(|_| panic!("each permission request should be allowed"));
    }
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].metadata["chat_id"], 201);
    assert_eq!(requests[1].metadata["chat_id"], 202);
    assert_eq!(requests[0].metadata["call"], requests[1].metadata["call"]);
}

#[test]
fn restored_unfinished_permission_requires_policy_at_its_nested_owner() {
    use crate::runtime::agent::permission::State;

    let mut metadata = Map::new();
    metadata.insert("call".into(), serde_json::to_value(call()).unwrap());
    let pending = State::new(Request {
        action: "call".into(),
        resource: "lookup".into(),
        metadata,
    });
    let agent_scene = agent_scene(pending);
    let nested_scene = json!({
        "nodes": {
            "agent-node": {
                "execution": {
                    "attempts": [agent_scene],
                },
            },
        },
    });

    for scene in [&agent_scene, &nested_scene] {
        let error = super::validate_scene(scene, false).unwrap_err();
        assert_eq!(error.code, "INVALID_STATE");
        assert!(super::validate_scene(scene, true).is_ok());
    }
}

#[test]
fn completed_permission_history_does_not_require_a_policy_on_restore() {
    use crate::runtime::agent::permission::State;

    let mut metadata = Map::new();
    metadata.insert("call".into(), serde_json::to_value(call()).unwrap());
    let mut completed = State::new(Request {
        action: "call".into(),
        resource: "lookup".into(),
        metadata,
    });
    completed.stage = "complete".into();
    let scene = agent_scene(completed);

    assert!(super::validate_scene(&scene, false).is_ok());
}

fn agent_scene(permission: crate::runtime::agent::permission::State) -> Value {
    let step = crate::runtime::agent::node::Step {
        id: "tool-step".into(),
        kind: "tool".into(),
        status: crate::runtime::agent::node::status::step::Status::Running,
        permission: Some(Box::new(permission)),
        ..Default::default()
    };
    json!({"steps":[step]})
}
