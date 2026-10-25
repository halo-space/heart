use super::*;
use crate::{
    middleware::Middleware,
    permission::{Decision, Permission, Request},
    tool::{Tool, ToolDefinition},
};
use futures_executor::block_on;
use serde_json::json;
use std::sync::{Arc, Mutex};

type Log = Arc<Mutex<Vec<String>>>;

struct Lookup {
    definition: ToolDefinition,
    log: Log,
    fail: bool,
}

impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn before(&self, call: &mut Call, _: &Cancellation) -> Result<(), Error> {
        self.log.lock().unwrap().push("before".into());
        call.arguments.insert("city".into(), json!("Paris"));
        if self.fail {
            Err(Error::new("INVALID_ARGUMENTS", "invalid business input"))
        } else {
            Ok(())
        }
    }

    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        panic!("preparation must not execute the Tool")
    }
}

struct Policy(Log);
impl Permission for Policy {
    async fn check(&self, request: &Request, _: &Cancellation) -> Result<Decision, Error> {
        self.0.lock().unwrap().push(format!(
            "authorize:{}",
            request.metadata["call"]["arguments"]["city"]
                .as_str()
                .unwrap()
        ));
        Ok(Decision {
            behavior: crate::permission::Behavior::Allow,
            reason: None,
        })
    }
}

struct BrokenPermissionHook;
impl Middleware for BrokenPermissionHook {
    async fn before_permission(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        Err(Error::new("BACKEND", "permission hook failed"))
    }
}

fn fixture(fail: bool) -> (Dispatch, Access, Log, Step, Call) {
    let log: Log = Default::default();
    let dispatch = Dispatch::new().tools(Lookup {
        definition: ToolDefinition {
            name: "lookup".into(),
            description: "find weather".into(),
            parameters: serde_json::from_value(json!({
                "type":"object", "properties":{"city":{"type":"string"}},
                "required":["city"], "additionalProperties":false
            }))
            .unwrap(),
        },
        log: log.clone(),
        fail,
    });
    let mut access = Access::default();
    access.set(Policy(log.clone()));
    let call = Call {
        call_id: "call_1".into(),
        name: "lookup".into(),
        arguments: serde_json::from_value(json!({"city":"Beijing"})).unwrap(),
    };
    let saved = Step {
        id: "tool_0".into(),
        kind: "tool".into(),
        stage: Some("before".into()),
        call: Some(call.clone()),
        ..Default::default()
    };
    (dispatch, access, log, saved, call)
}

#[test]
fn both_strategies_share_reauthorization_and_native_before_acknowledgement() {
    block_on(async {
        let (dispatch, access, log, mut saved, call) = fixture(false);
        let hooks = Hooks::default();
        let metadata = Map::new();
        let cancellation = Cancellation::new();
        let starting = saved.clone();
        let effective = tool_prepare(
            &dispatch,
            &access,
            &hooks,
            &metadata,
            None,
            None,
            &starting,
            &call,
            |effective| {
                saved.call = Some(effective);
                saved.stage = Some("call".into());
                log.lock().unwrap().push("commit".into());
                async { Ok(()) }
            },
            |_| async { Ok(()) },
            &cancellation,
        )
        .await
        .unwrap();
        assert_eq!(
            *log.lock().unwrap(),
            ["authorize:Beijing", "before", "commit", "authorize:Paris"]
        );
        assert_eq!(saved.call.as_ref(), Some(&effective));
        log.lock().unwrap().clear();

        let restored = tool_prepare(
            &dispatch,
            &access,
            &hooks,
            &metadata,
            None,
            None,
            &saved,
            &effective,
            |_| async { panic!("successful before must not be committed twice") },
            |_| async { Ok(()) },
            &cancellation,
        )
        .await
        .unwrap();
        assert_eq!(restored, effective);
        assert_eq!(*log.lock().unwrap(), ["authorize:Paris"]);

        for stage in ["after", "eval"] {
            log.lock().unwrap().clear();
            saved.stage = Some(stage.into());
            let restored = tool_prepare(
                &dispatch,
                &access,
                &hooks,
                &metadata,
                None,
                None,
                &saved,
                &effective,
                |_| async { panic!("post-call restore must not rerun before") },
                |_| async { Ok(()) },
                &cancellation,
            )
            .await
            .unwrap();
            assert_eq!(restored, effective);
            assert!(log.lock().unwrap().is_empty());
        }
    });
}

#[test]
fn business_before_error_keeps_original_native_facts() {
    block_on(async {
        let (dispatch, access, log, saved, call) = fixture(true);
        let original = saved.clone();
        let result = tool_prepare(
            &dispatch,
            &access,
            &Hooks::default(),
            &Map::new(),
            None,
            None,
            &saved,
            &call,
            |_| async { panic!("failed method has no successful acknowledgement") },
            |_| async { Ok(()) },
            &Cancellation::new(),
        )
        .await;
        assert!(matches!(result, Err(Failure::Tool(error)) if error.code == "INVALID_ARGUMENTS"));
        assert_eq!(saved, original);
        assert_eq!(*log.lock().unwrap(), ["authorize:Beijing", "before"]);
    });
}

#[test]
fn permission_middleware_error_is_control_failure_not_tool_feedback() {
    block_on(async {
        let (dispatch, access, log, saved, call) = fixture(false);
        let mut hooks = Hooks::default();
        hooks.add(BrokenPermissionHook);
        let result = tool_prepare(
            &dispatch,
            &access,
            &hooks,
            &Map::new(),
            None,
            None,
            &saved,
            &call,
            |_| async { panic!("permission middleware failed before preparation") },
            |_| async { Ok(()) },
            &Cancellation::new(),
        )
        .await;
        assert!(matches!(result, Err(Failure::Hook(error)) if error.code == "BACKEND"));
        assert!(log.lock().unwrap().is_empty());
    });
}

#[test]
fn failed_native_handoff_is_control_failure_without_second_authorization() {
    block_on(async {
        let (dispatch, access, log, saved, call) = fixture(false);
        let result = tool_prepare(
            &dispatch,
            &access,
            &Hooks::default(),
            &Map::new(),
            None,
            None,
            &saved,
            &call,
            |_| async { Err(Error::new("BACKEND", "native owner handoff failed")) },
            |_| async { Ok(()) },
            &Cancellation::new(),
        )
        .await;
        assert!(matches!(result, Err(Failure::Hook(error)) if error.code == "BACKEND"));
        assert_eq!(*log.lock().unwrap(), ["authorize:Beijing", "before"]);
        assert_eq!(saved.stage.as_deref(), Some("before"));
        assert_eq!(saved.call, Some(call));
    });
}
