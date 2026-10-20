use std::{
    future::pending,
    sync::{Arc, Mutex},
    time::Duration,
};

use futures_executor::block_on;
use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, toolkit::Toolkit},
    event::Event,
    mcp::{Definition as McpDefinition, Server},
    message::Role,
    middleware::Middleware,
    model::{
        chat::{Chat, Request},
        content::part::Part,
    },
    permission::{Behavior, Decision, Permission, Request as PermissionRequest},
    tool::{Call, Evaluation, Tool, ToolDefinition},
};
use serde_json::{Value, json};

type Log = Arc<Mutex<Vec<String>>>;

fn definition() -> ToolDefinition {
    ToolDefinition {name:"lookup".into(), description:"weather".into(), parameters:serde_json::from_value(json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false})).unwrap()}
}

struct Lookup {
    definition: ToolDefinition,
    log: Log,
}
impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn before(&self, _: &mut Call, _: &Cancellation) -> Result<(), Error> {
        self.log.lock().unwrap().push("tool:before".into());
        Ok(())
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.log.lock().unwrap().push("tool:call".into());
        Ok(Message::function(json!({"temperature":22})))
    }
    async fn after(&self, _: &Call, _: &mut Message, _: &Cancellation) -> Result<(), Error> {
        self.log.lock().unwrap().push("tool:after".into());
        Ok(())
    }
    async fn eval(&self, _: &Call, _: &Message, _: &Cancellation) -> Result<Evaluation, Error> {
        self.log.lock().unwrap().push("tool:eval".into());
        Ok(Evaluation::passed())
    }
}

#[derive(Clone)]
struct Model {
    requests: Arc<Mutex<Vec<Request>>>,
    name: String,
    arguments: Value,
    multiple: bool,
}
impl Model {
    fn new() -> Self {
        Self {
            requests: Default::default(),
            name: "lookup".into(),
            arguments: json!({"city":"Beijing"}),
            multiple: false,
        }
    }
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let first = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request);
            requests.len() % 2 == 1
        };
        let mut message = Message::new(Role::Assistant);
        if first {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"c1","name":self.name,"arguments":self.arguments})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
            if self.multiple {
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: json!({"call_id":"c2","name":self.name,"arguments":{"city":"Paris"}})
                        .as_object()
                        .unwrap()
                        .clone(),
                });
            }
        } else {
            message.values.insert("answer".into(), json!("done"));
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Allow,
    Deny,
    Ask,
    AskChanged,
    Pending,
    Cancel,
    Fault,
    DenyChanged,
    CancelSecond,
    PendingSecond,
}
#[derive(Clone)]
struct Policy {
    requests: Arc<Mutex<Vec<PermissionRequest>>>,
    log: Log,
    mode: Mode,
}
impl Policy {
    fn new(log: &Log, mode: Mode) -> Self {
        Self {
            requests: Default::default(),
            log: log.clone(),
            mode,
        }
    }
}
fn denied() -> Error {
    Error::new("PERMISSION_DENIED", "no access").with_details("original reason")
}

fn allow() -> Decision {
    Decision {
        behavior: Behavior::Allow,
        reason: None,
    }
}

fn deny() -> Decision {
    Decision {
        behavior: Behavior::Deny,
        reason: Some("no access".into()),
    }
}
impl Permission for Policy {
    async fn check(
        &self,
        request: &PermissionRequest,
        cancellation: &Cancellation,
    ) -> Result<Decision, Error> {
        let count = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request.clone());
            requests.len()
        };
        self.log.lock().unwrap().push("policy".into());
        match self.mode {
            Mode::Allow => Ok(allow()),
            Mode::Deny => Ok(deny()),
            Mode::Ask => Ok(Decision {
                behavior: Behavior::Ask,
                reason: Some("approval required".into()),
            }),
            Mode::AskChanged => {
                if request.metadata["call"]["arguments"]["city"] == "Beijing" {
                    Ok(allow())
                } else {
                    Ok(Decision {
                        behavior: Behavior::Ask,
                        reason: Some("changed operation needs approval".into()),
                    })
                }
            }
            Mode::Pending => pending().await,
            Mode::Cancel => {
                cancellation.cancel();
                Ok(allow())
            }
            Mode::Fault => Err(Error::new("UNAVAILABLE", "policy backend unavailable")
                .with_details("original backend error")),
            Mode::DenyChanged => {
                if request.metadata["call"]["arguments"]["city"] == "Beijing" {
                    Ok(allow())
                } else {
                    Ok(deny())
                }
            }
            Mode::CancelSecond if count > 1 => {
                cancellation.cancel();
                Ok(allow())
            }
            Mode::PendingSecond if count > 1 => pending().await,
            Mode::CancelSecond | Mode::PendingSecond => Ok(allow()),
        }
    }
}

#[derive(Clone)]
struct Hook {
    name: &'static str,
    log: Log,
    fail: Option<&'static str>,
    wait: Option<&'static str>,
    change: Option<&'static str>,
    change_at: Option<&'static str>,
}
impl Hook {
    fn new(name: &'static str, log: &Log) -> Self {
        Self {
            name,
            log: log.clone(),
            fail: None,
            wait: None,
            change: None,
            change_at: None,
        }
    }
}
impl Hook {
    async fn record(&self, stage: &'static str, request: &mut Value) -> Result<(), Error> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:{stage}", self.name));
        if self.fail == Some(stage) {
            return Err(denied());
        }
        if self.wait == Some(stage) {
            pending::<()>().await;
        }
        request["metadata"][format!("hook_{}", self.name)] = json!(true);
        if self.change_at.is_some_and(|target| target != stage) {
            return Ok(());
        }
        match self.change {
            Some("action") => request["action"] = json!("other"),
            Some("resource") => request["resource"] = json!("other"),
            Some("call") => request["metadata"]["call"]["arguments"]["city"] = json!("other"),
            Some("shape") => *request = json!([]),
            _ => {}
        }
        Ok(())
    }
}
impl Middleware for Hook {
    async fn before_permission(&self, request: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.record("before", request).await
    }
    async fn after_permission(&self, request: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.record("after", request).await
    }
}

fn input(chat_id: i64) -> Input {
    Input {
        metadata: serde_json::from_value(
            json!({"chat_id":chat_id,"user_id":7,"session_id":8,"agent_id":9,"call":"not actual"}),
        )
        .unwrap(),
        payload: vec![],
    }
}
fn toolkit(log: &Log) -> Toolkit {
    Toolkit::new().tools(Lookup {
        definition: definition(),
        log: log.clone(),
    })
}
async fn collect<A: Agent>(
    agent: &A,
    chat_id: i64,
    cancellation: Cancellation,
) -> Vec<Result<Event, Error>> {
    match agent.run(input(chat_id), true, cancellation).await {
        Ok(events) => events.collect().await,
        Err(error) => vec![Err(error)],
    }
}

#[test]
fn allowed_call_checks_actual_request_and_onion_hooks_before_tool() {
    let log = Log::default();
    let model = Model::new();
    let policy = Policy::new(&log, Mode::Allow);
    let agent = agent::react()
        .permission(policy.clone())
        .middleware(Hook::new("A", &log))
        .model(model)
        .toolkit(toolkit(&log))
        .middleware(Hook::new("B", &log))
        .build()
        .unwrap();
    let events = block_on(collect(&agent, 10, Cancellation::new()));
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(
        *log.lock().unwrap(),
        [
            "A:before",
            "B:before",
            "policy",
            "B:after",
            "A:after",
            "tool:before",
            "tool:call",
            "tool:after",
            "tool:eval"
        ]
    );
    let requests = policy.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].action, "call");
    assert_eq!(requests[0].resource, "lookup");
    assert_eq!(requests[0].metadata["chat_id"], 10);
    assert_eq!(requests[0].metadata["user_id"], 7);
    assert_eq!(requests[0].metadata["session_id"], 8);
    assert_eq!(requests[0].metadata["agent_id"], 9);
    assert_eq!(
        requests[0].metadata["call"],
        json!({"call_id":"c1","name":"lookup","arguments":{"city":"Beijing"}})
    );
    assert_eq!(requests[0].metadata["hook_A"], true);
    assert_eq!(requests[0].metadata["hook_B"], true);
}

#[test]
fn denial_is_a_normal_decision_and_skips_all_tool_hooks() {
    let log = Log::default();
    let model = Model::new();
    let policy = Policy::new(&log, Mode::Deny);
    let agent = agent::react()
        .model(model.clone())
        .toolkit(toolkit(&log))
        .permission(policy.clone())
        .middleware(Hook::new("A", &log))
        .build()
        .unwrap();
    let events = block_on(collect(&agent, 10, Cancellation::new()));
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(events.len(), 3);
    assert_eq!(*log.lock().unwrap(), ["A:before", "policy", "A:after"]);
    assert_eq!(policy.requests.lock().unwrap().len(), 1);
    let requests = model.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let error = &requests[1].messages.0.last().unwrap().values["error"];
    assert_eq!(
        *error,
        json!({"code":"PERMISSION_DENIED","message":"no access","details":null})
    );
    assert!(
        !requests[1]
            .messages
            .0
            .last()
            .unwrap()
            .values
            .contains_key("tool_result")
    );
}

#[test]
fn absent_policy_does_not_activate_permission_hooks() {
    let log = Log::default();
    let mut hook = Hook::new("A", &log);
    hook.fail = Some("before");
    let agent = agent::react()
        .model(Model::new())
        .toolkit(toolkit(&log))
        .middleware(hook)
        .build()
        .unwrap();
    assert!(
        block_on(collect(&agent, 1, Cancellation::new()))
            .iter()
            .all(Result::is_ok)
    );
    assert_eq!(
        *log.lock().unwrap(),
        ["tool:before", "tool:call", "tool:after", "tool:eval"]
    );
}

#[test]
fn invalid_call_never_checks_policy_or_enters_hooks() {
    for unknown in [false, true] {
        let log = Log::default();
        let policy = Policy::new(&log, Mode::Allow);
        let mut model = Model::new();
        if unknown {
            model.name = "unknown".into();
        } else {
            model.arguments = json!({"city":123});
        }
        let agent = agent::react()
            .model(model.clone())
            .toolkit(toolkit(&log))
            .permission(policy.clone())
            .middleware(Hook::new("A", &log))
            .build()
            .unwrap();
        assert!(
            block_on(collect(&agent, 1, Cancellation::new()))
                .iter()
                .all(Result::is_ok)
        );
        assert!(log.lock().unwrap().is_empty());
        assert!(policy.requests.lock().unwrap().is_empty());
        assert_eq!(
            model.requests.lock().unwrap()[1]
                .messages
                .0
                .last()
                .unwrap()
                .values["error"]["code"],
            if unknown {
                "NOT_FOUND"
            } else {
                "INVALID_ARGUMENTS"
            }
        );
    }
}

#[test]
fn hook_error_terminates_agent_instead_of_becoming_tool_error_context() {
    for stage in ["before", "after"] {
        let log = Log::default();
        let policy = Policy::new(&log, Mode::Allow);
        let model = Model::new();
        let mut hook = Hook::new("A", &log);
        hook.fail = Some(stage);
        let agent = agent::react()
            .model(model.clone())
            .toolkit(toolkit(&log))
            .permission(policy.clone())
            .middleware(hook)
            .build()
            .unwrap();
        let events = block_on(collect(&agent, 1, Cancellation::new()));
        assert_eq!(events.last().unwrap().as_ref().unwrap_err(), &denied());
        assert_eq!(model.requests.lock().unwrap().len(), 1);
        assert_eq!(
            policy.requests.lock().unwrap().len(),
            usize::from(stage == "after")
        );
        assert!(!log.lock().unwrap().iter().any(|s| s.starts_with("tool:")));
    }
}

#[test]
fn hooks_cannot_authorize_a_different_resource_or_argument_set() {
    for change in ["action", "resource", "call", "shape"] {
        let log = Log::default();
        let policy = Policy::new(&log, Mode::Allow);
        let model = Model::new();
        let mut hook = Hook::new("A", &log);
        hook.change = Some(change);
        let agent = agent::react()
            .model(model.clone())
            .toolkit(toolkit(&log))
            .permission(policy.clone())
            .middleware(hook)
            .build()
            .unwrap();
        assert_eq!(
            block_on(collect(&agent, 1, Cancellation::new()))
                .last()
                .unwrap()
                .as_ref()
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        assert!(policy.requests.lock().unwrap().is_empty());
        assert_eq!(model.requests.lock().unwrap().len(), 1);
    }
}

#[test]
fn reused_agent_checks_each_run_without_cached_decisions() {
    let log = Log::default();
    let policy = Policy::new(&log, Mode::Allow);
    let agent = agent::react()
        .model(Model::new())
        .toolkit(toolkit(&log))
        .permission(policy.clone())
        .build()
        .unwrap();
    for id in [10, 11] {
        assert!(
            block_on(collect(&agent, id, Cancellation::new()))
                .iter()
                .all(Result::is_ok)
        );
    }
    let requests = policy.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].metadata["chat_id"], 10);
    assert_eq!(requests[1].metadata["chat_id"], 11);
}

struct Remote {
    definition: McpDefinition,
    log: Log,
}
impl Server for Remote {
    fn definition(&self) -> &McpDefinition {
        &self.definition
    }
    fn tools(&self) -> &[ToolDefinition] {
        static TOOLS: std::sync::OnceLock<Vec<ToolDefinition>> = std::sync::OnceLock::new();
        TOOLS.get_or_init(|| vec![definition()])
    }
    async fn call(&self, _: Call, _: &Cancellation) -> Result<Message, Error> {
        self.log.lock().unwrap().push("mcp:call".into());
        Ok(Message::function(json!({"temperature":22})))
    }
}
#[test]
fn discovered_mcp_tools_use_the_same_permission_gate() {
    for mode in [Mode::Allow, Mode::Deny] {
        let log = Log::default();
        let policy = Policy::new(&log, mode);
        let toolkit = Toolkit::new().mcps(vec![Remote {
            definition: McpDefinition {
                name: "server-not-tool-name".into(),
                ..Default::default()
            },
            log: log.clone(),
        }]);
        let agent = agent::react()
            .model(Model::new())
            .toolkit(toolkit)
            .permission(policy.clone())
            .build()
            .unwrap();
        assert!(
            block_on(collect(&agent, 1, Cancellation::new()))
                .iter()
                .all(Result::is_ok)
        );
        assert_eq!(policy.requests.lock().unwrap()[0].resource, "lookup");
        assert_eq!(policy.requests.lock().unwrap().len(), 1);
        assert_eq!(
            log.lock()
                .unwrap()
                .iter()
                .filter(|s| *s == "mcp:call")
                .count(),
            usize::from(matches!(mode, Mode::Allow))
        );
    }
}

#[test]
fn synchronous_policy_cancellation_never_enters_after_or_tool() {
    let log = Log::default();
    let policy = Policy::new(&log, Mode::Cancel);
    let agent = agent::react()
        .model(Model::new())
        .toolkit(toolkit(&log))
        .permission(policy)
        .middleware(Hook::new("A", &log))
        .build()
        .unwrap();
    let events = block_on(collect(&agent, 1, Cancellation::new()));
    assert!(
        matches!(events.last().unwrap(),Ok(Event::Complete{finish_reason:Some(reason),..}) if reason=="cancelled")
    );
    assert_eq!(*log.lock().unwrap(), ["A:before", "policy"]);
}

#[test]
fn multiple_calls_are_authorized_individually_with_their_actual_arguments() {
    let log = Log::default();
    let policy = Policy::new(&log, Mode::Allow);
    let mut model = Model::new();
    model.multiple = true;
    let agent = agent::react()
        .model(model)
        .toolkit(toolkit(&log))
        .permission(policy.clone())
        .build()
        .unwrap();
    assert!(
        block_on(collect(&agent, 1, Cancellation::new()))
            .iter()
            .all(Result::is_ok)
    );
    let requests = policy.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].metadata["call"]["call_id"], "c1");
    assert_eq!(requests[0].metadata["call"]["arguments"]["city"], "Beijing");
    assert_eq!(requests[1].metadata["call"]["call_id"], "c2");
    assert_eq!(requests[1].metadata["call"]["arguments"]["city"], "Paris");
    assert_eq!(
        log.lock()
            .unwrap()
            .iter()
            .filter(|s| *s == "tool:call")
            .count(),
        2
    );
}

#[test]
fn policy_backend_errors_are_direct_context_without_internal_retry() {
    let log = Log::default();
    let policy = Policy::new(&log, Mode::Fault);
    let model = Model::new();
    let agent = agent::react()
        .model(model.clone())
        .toolkit(toolkit(&log))
        .permission(policy.clone())
        .build()
        .unwrap();
    assert!(
        block_on(collect(&agent, 1, Cancellation::new()))
            .iter()
            .all(Result::is_ok)
    );
    assert_eq!(policy.requests.lock().unwrap().len(), 1);
    assert_eq!(*log.lock().unwrap(), ["policy"]);
    assert_eq!(
        model.requests.lock().unwrap()[1]
            .messages
            .0
            .last()
            .unwrap()
            .values["error"],
        json!({"code":"UNAVAILABLE","message":"policy backend unavailable","details":"original backend error"})
    );
}

#[test]
fn after_permission_cannot_redirect_an_already_checked_call() {
    let log = Log::default();
    let policy = Policy::new(&log, Mode::Allow);
    let mut hook = Hook::new("A", &log);
    hook.change = Some("resource");
    hook.change_at = Some("after");
    let agent = agent::react()
        .model(Model::new())
        .toolkit(toolkit(&log))
        .permission(policy.clone())
        .middleware(hook)
        .build()
        .unwrap();
    assert_eq!(
        block_on(collect(&agent, 1, Cancellation::new()))
            .last()
            .unwrap()
            .as_ref()
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert_eq!(policy.requests.lock().unwrap().len(), 1);
    assert_eq!(*log.lock().unwrap(), ["A:before", "policy", "A:after"]);
}

#[tokio::test]
async fn pending_policy_and_permission_hooks_are_cancellable() {
    for target in ["policy", "A:before", "A:after"] {
        let log = Log::default();
        let mode = if target == "policy" {
            Mode::Pending
        } else {
            Mode::Allow
        };
        let policy = Policy::new(&log, mode);
        let mut hook = Hook::new("A", &log);
        hook.wait = match target {
            "A:before" => Some("before"),
            "A:after" => Some("after"),
            _ => None,
        };
        let agent = agent::react()
            .model(Model::new())
            .toolkit(toolkit(&log))
            .permission(policy)
            .middleware(hook)
            .build()
            .unwrap();
        let c = Cancellation::new();
        let token = c.clone();
        let watched = log.clone();
        let task = tokio::spawn(async move {
            while !watched.lock().unwrap().iter().any(|s| s == target) {
                tokio::task::yield_now().await;
            }
            token.cancel();
        });
        let events = tokio::time::timeout(Duration::from_secs(2), collect(&agent, 1, c))
            .await
            .unwrap();
        task.await.unwrap();
        assert!(
            matches!(events.last().unwrap(),Ok(Event::Complete{finish_reason:Some(reason),..}) if reason=="cancelled")
        );
        assert!(!log.lock().unwrap().iter().any(|s| s.starts_with("tool:")));
    }
}

#[derive(Clone, Copy)]
enum Before {
    Rewrite,
    Fail,
    Rename,
    Cancel,
}

struct RewritingLookup {
    inner: Lookup,
    city: Value,
    before: Before,
}

impl Tool for RewritingLookup {
    fn definition(&self) -> &ToolDefinition {
        &self.inner.definition
    }

    async fn before(&self, call: &mut Call, cancellation: &Cancellation) -> Result<(), Error> {
        self.inner.before(call, cancellation).await?;
        match self.before {
            Before::Rewrite => {
                call.arguments.insert("city".into(), self.city.clone());
            }
            Before::Fail => return Err(Error::new("INVALID_ARGUMENTS", "before rejected")),
            Before::Rename => call.name = "another_tool".into(),
            Before::Cancel => cancellation.cancel(),
        }
        Ok(())
    }

    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        self.inner.call(call, cancellation).await?;
        Ok(Message::function(json!({"city":call.arguments["city"]})))
    }

    async fn after(
        &self,
        call: &Call,
        message: &mut Message,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        self.inner.after(call, message, cancellation).await
    }

    async fn eval(
        &self,
        call: &Call,
        message: &Message,
        cancellation: &Cancellation,
    ) -> Result<Evaluation, Error> {
        assert_eq!(message.values["city"], call.arguments["city"]);
        self.inner.eval(call, message, cancellation).await
    }
}

fn rewriting_toolkit(log: &Log, city: Value, before: Before) -> Toolkit {
    Toolkit::new().tools(RewritingLookup {
        inner: Lookup {
            definition: definition(),
            log: log.clone(),
        },
        city,
        before,
    })
}

#[test]
fn before_cannot_bypass_policy_by_changing_authorized_arguments() {
    let log = Log::default();
    let model = Model::new();
    let policy = Policy::new(&log, Mode::DenyChanged);
    let agent = agent::react()
        .model(model.clone())
        .toolkit(rewriting_toolkit(&log, json!("Paris"), Before::Rewrite))
        .permission(policy.clone())
        .middleware(Hook::new("A", &log))
        .build()
        .unwrap();
    let events = block_on(collect(&agent, 1, Cancellation::new()));
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(
        *log.lock().unwrap(),
        [
            "A:before",
            "policy",
            "A:after",
            "tool:before",
            "A:before",
            "policy",
            "A:after"
        ]
    );
    let requests = policy.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].metadata["call"]["arguments"]["city"], "Beijing");
    assert_eq!(requests[1].metadata["call"]["arguments"]["city"], "Paris");
    assert_eq!(
        requests[0].metadata["call"]["call_id"],
        requests[1].metadata["call"]["call_id"]
    );
    assert_eq!(
        model.requests.lock().unwrap()[1]
            .messages
            .0
            .last()
            .unwrap()
            .values["error"],
        json!({"code":"PERMISSION_DENIED","message":"no access","details":null})
    );
    assert!(
        !model.requests.lock().unwrap()[1]
            .messages
            .0
            .last()
            .unwrap()
            .values
            .contains_key("tool_result")
    );
}

#[test]
fn changed_arguments_are_authorized_then_executed_without_repeating_before() {
    let log = Log::default();
    let policy = Policy::new(&log, Mode::Allow);
    let agent = agent::react()
        .model(Model::new())
        .toolkit(rewriting_toolkit(&log, json!("Paris"), Before::Rewrite))
        .permission(policy.clone())
        .middleware(Hook::new("A", &log))
        .build()
        .unwrap();
    let events = block_on(collect(&agent, 1, Cancellation::new()));
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(
        *log.lock().unwrap(),
        [
            "A:before",
            "policy",
            "A:after",
            "tool:before",
            "A:before",
            "policy",
            "A:after",
            "tool:call",
            "tool:after",
            "tool:eval"
        ]
    );
    assert_eq!(policy.requests.lock().unwrap().len(), 2);
    assert!(
        matches!(&events[1], Ok(Event::Complete { message, .. }) if message.values["city"] == "Paris")
    );
}

#[test]
fn invalid_or_failed_before_never_reauthorizes_or_executes() {
    for (city, before) in [
        (json!(123), Before::Rewrite),
        (json!("Paris"), Before::Rename),
        (json!("Paris"), Before::Fail),
    ] {
        let log = Log::default();
        let model = Model::new();
        let policy = Policy::new(&log, Mode::Allow);
        let agent = agent::react()
            .model(model.clone())
            .toolkit(rewriting_toolkit(&log, city, before))
            .permission(policy.clone())
            .build()
            .unwrap();
        assert!(
            block_on(collect(&agent, 1, Cancellation::new()))
                .iter()
                .all(Result::is_ok)
        );
        assert_eq!(*log.lock().unwrap(), ["policy", "tool:before"]);
        assert_eq!(policy.requests.lock().unwrap().len(), 1);
        assert_eq!(
            model.requests.lock().unwrap()[1]
                .messages
                .0
                .last()
                .unwrap()
                .values["error"]["code"],
            "INVALID_ARGUMENTS"
        );
    }
}

#[test]
fn changed_arguments_without_policy_do_not_enable_permission_hooks() {
    let log = Log::default();
    let mut hook = Hook::new("A", &log);
    hook.fail = Some("before");
    let agent = agent::react()
        .model(Model::new())
        .toolkit(rewriting_toolkit(&log, json!("Paris"), Before::Rewrite))
        .middleware(hook)
        .build()
        .unwrap();
    let events = block_on(collect(&agent, 1, Cancellation::new()));
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(
        *log.lock().unwrap(),
        ["tool:before", "tool:call", "tool:after", "tool:eval"]
    );
}

#[test]
fn cancellation_during_before_or_second_check_does_not_execute_call() {
    for (before, mode, checks) in [
        (Before::Cancel, Mode::Allow, 1),
        (Before::Rewrite, Mode::CancelSecond, 2),
    ] {
        let log = Log::default();
        let policy = Policy::new(&log, mode);
        let agent = agent::react()
            .model(Model::new())
            .toolkit(rewriting_toolkit(&log, json!("Paris"), before))
            .permission(policy.clone())
            .middleware(Hook::new("A", &log))
            .build()
            .unwrap();
        let events = block_on(collect(&agent, 1, Cancellation::new()));
        assert!(
            matches!(events.last().unwrap(), Ok(Event::Complete {finish_reason:Some(reason),..}) if reason == "cancelled")
        );
        assert_eq!(policy.requests.lock().unwrap().len(), checks);
        assert!(
            !log.lock()
                .unwrap()
                .iter()
                .any(|entry| matches!(entry.as_str(), "tool:call" | "tool:after" | "tool:eval"))
        );
    }
}

#[tokio::test]
async fn pending_reauthorization_is_cancellable() {
    let log = Log::default();
    let policy = Policy::new(&log, Mode::PendingSecond);
    let agent = agent::react()
        .model(Model::new())
        .toolkit(rewriting_toolkit(&log, json!("Paris"), Before::Rewrite))
        .permission(policy.clone())
        .build()
        .unwrap();
    let cancellation = Cancellation::new();
    let token = cancellation.clone();
    let requests = policy.requests.clone();
    let cancel = async move {
        loop {
            if requests.lock().unwrap().len() == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
        token.cancel();
    };
    let (events, ()) = tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(collect(&agent, 1, cancellation), cancel)
    })
    .await
    .unwrap();
    assert!(
        matches!(events.last().unwrap(), Ok(Event::Complete {finish_reason:Some(reason),..}) if reason == "cancelled")
    );
    assert_eq!(*log.lock().unwrap(), ["policy", "tool:before", "policy"]);
}

struct ChangedHook {
    stage: &'static str,
    log: Log,
}
impl ChangedHook {
    fn check(&self, stage: &str, request: &Value) -> Result<(), Error> {
        self.log.lock().unwrap().push(format!("A:{stage}"));
        if self.stage == stage && request["metadata"]["call"]["arguments"]["city"] == "Paris" {
            Err(denied())
        } else {
            Ok(())
        }
    }
}
impl Middleware for ChangedHook {
    async fn before_permission(&self, request: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.check("before", request)
    }
    async fn after_permission(&self, request: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.check("after", request)
    }
}

#[test]
fn second_permission_hook_failure_stops_agent_not_tool_error_context() {
    for stage in ["before", "after"] {
        let log = Log::default();
        let model = Model::new();
        let policy = Policy::new(&log, Mode::Allow);
        let agent = agent::react()
            .model(model.clone())
            .toolkit(rewriting_toolkit(&log, json!("Paris"), Before::Rewrite))
            .permission(policy.clone())
            .middleware(ChangedHook {
                stage,
                log: log.clone(),
            })
            .build()
            .unwrap();
        let events = block_on(collect(&agent, 1, Cancellation::new()));
        assert_eq!(events.last().unwrap().as_ref().unwrap_err(), &denied());
        assert_eq!(model.requests.lock().unwrap().len(), 1);
        assert_eq!(
            policy.requests.lock().unwrap().len(),
            if stage == "before" { 1 } else { 2 }
        );
        assert!(
            !log.lock()
                .unwrap()
                .iter()
                .any(|entry| matches!(entry.as_str(), "tool:call" | "tool:after" | "tool:eval"))
        );
    }
}

#[test]
fn chat_executes_tool_calls_through_the_configured_toolkit() {
    let log = Log::default();
    let model = Model::new();
    let mut hook = Hook::new("A", &log);
    hook.fail = Some("before");
    let agent = agent::chat()
        .model(model.clone())
        .toolkit(toolkit(&log))
        .middleware(hook)
        .build()
        .unwrap();
    let events = block_on(collect(&agent, 1, Cancellation::new()));
    assert!(!events.is_empty());
    assert!(events.last().unwrap().is_ok());
    assert_eq!(model.requests.lock().unwrap()[0].tools.len(), 1);
    assert!(
        log.lock()
            .unwrap()
            .iter()
            .any(|entry| entry.ends_with(":before"))
    );
}

async fn check_ask<A: Agent>(
    agent: A,
    checkpoint: Arc<halo_agents::core::checkpoint::InMemory>,
    policy: Policy,
    model: Model,
    log: Log,
    changed: bool,
) {
    use halo_agents::checkpoint::Checkpoint as _;
    let events = collect(&agent, 70, Cancellation::new()).await;
    assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
    assert_eq!(model.requests.lock().unwrap().len(), 1);
    let saved = checkpoint.load(70, &Cancellation::new()).await.unwrap();
    assert_eq!(saved["status"], "waiting");
    assert_eq!(saved["query"], serde_json::to_value(input(70)).unwrap());
    assert_eq!(
        saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["status"],
        "waiting"
    );
    assert!(
        saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["error"]
            .is_null(),
        "ask is not a technical Tool error"
    );
    assert_eq!(
        saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["call"]["arguments"]["city"],
        if changed { "Paris" } else { "Beijing" },
        "waiting retains the actual checked operation"
    );
    assert_eq!(
        saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["call"]["call_id"],
        "c1"
    );
    assert!(
        !log.lock()
            .unwrap()
            .iter()
            .any(|entry| matches!(entry.as_str(), "tool:call" | "tool:after" | "tool:eval"))
    );
    assert_eq!(
        log.lock()
            .unwrap()
            .iter()
            .filter(|entry| entry.as_str() == "tool:before")
            .count(),
        usize::from(changed)
    );
    if !changed {
        // Until the external-reply consumer is implemented, arbitrary input
        // cannot turn the policy's ask into allow or bypass another check.
        let mut reply = input(70);
        reply.payload = vec![halo_agents::agent::Payload {
            r#type: "json".into(),
            data: json!({"value":{"approved":true}}),
            mime_type: None,
        }];
        let again = agent
            .run(reply, true, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(again.last(),Some(Err(error)) if error.code=="WAITING"));
        assert_eq!(model.requests.lock().unwrap().len(), 1);
        assert_eq!(policy.requests.lock().unwrap().len(), 2);
        assert_eq!(
            checkpoint.load(70, &Cancellation::new()).await.unwrap()["query"],
            saved["query"]
        );
        assert!(
            !log.lock()
                .unwrap()
                .iter()
                .any(|entry| entry.starts_with("tool:"))
        );
    }
    for request in &*policy.requests.lock().unwrap() {
        assert_eq!(request.resource, "lookup");
        assert!(request.metadata["call"].get("response").is_none());
        assert!(request.metadata.get("response").is_none());
    }
}

#[tokio::test]
async fn normal_ask_pauses_both_agents_without_running_tools_or_granting_approval() {
    for react in [false, true] {
        for changed in [false, true] {
            let log = Log::default();
            let model = Model::new();
            let policy = Policy::new(&log, if changed { Mode::AskChanged } else { Mode::Ask });
            let checkpoint = Arc::new(halo_agents::core::checkpoint::InMemory::new());
            let toolkit = if changed {
                rewriting_toolkit(&log, json!("Paris"), Before::Rewrite)
            } else {
                toolkit(&log)
            };
            if react {
                check_ask(
                    agent::react()
                        .model(model.clone())
                        .toolkit(toolkit)
                        .permission(policy.clone())
                        .middleware(Hook::new("A", &log))
                        .checkpoint(checkpoint.clone())
                        .build()
                        .unwrap(),
                    checkpoint,
                    policy,
                    model,
                    log,
                    changed,
                )
                .await;
            } else {
                check_ask(
                    agent::chat()
                        .model(model.clone())
                        .toolkit(toolkit)
                        .permission(policy.clone())
                        .middleware(Hook::new("A", &log))
                        .checkpoint(checkpoint.clone())
                        .build()
                        .unwrap(),
                    checkpoint,
                    policy,
                    model,
                    log,
                    changed,
                )
                .await;
            }
        }
    }
}

#[test]
fn chat_permission_hook_failures_are_not_tool_error_messages() {
    for mode in [Mode::Allow, Mode::Deny, Mode::Ask] {
        for stage in ["before", "after"] {
            let log = Log::default();
            let model = Model::new();
            let mut hook = Hook::new("A", &log);
            hook.fail = Some(stage);
            let agent = agent::chat()
                .model(model.clone())
                .toolkit(toolkit(&log))
                .permission(Policy::new(&log, mode))
                .middleware(hook)
                .build()
                .unwrap();
            let events = block_on(collect(&agent, 71, Cancellation::new()));
            assert_eq!(events.last().unwrap().as_ref().unwrap_err(), &denied());
            assert_eq!(model.requests.lock().unwrap().len(), 1);
            assert!(
                !log.lock()
                    .unwrap()
                    .iter()
                    .any(|entry| entry.starts_with("tool:"))
            );
        }
    }
}
