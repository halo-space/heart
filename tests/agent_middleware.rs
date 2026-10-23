use std::{
    future::pending,
    sync::{Arc, Mutex},
    time::Duration,
};

use futures_executor::block_on;
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::{self, Agent, Input, Payload, toolkit::Toolkit},
    event::Event,
    message::Role,
    middleware::Middleware,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::{Value, json};

type Log = Arc<Mutex<Vec<String>>>;

#[derive(Clone)]
struct Recorder {
    name: &'static str,
    log: Log,
    fail: Option<&'static str>,
    wait: Option<&'static str>,
    cancel: Option<&'static str>,
    invalid: bool,
}
impl Recorder {
    fn new(name: &'static str, log: &Log) -> Self {
        Self {
            name,
            log: log.clone(),
            fail: None,
            wait: None,
            cancel: None,
            invalid: false,
        }
    }
    async fn record(&self, stage: &'static str, cancellation: &Cancellation) -> Result<(), Error> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}:{stage}", self.name));
        if self.fail == Some(stage) {
            return Err(Error::new("HOOK_FAILED", "original").with_details("unchanged"));
        }
        if self.wait == Some(stage) {
            pending::<()>().await;
        }
        if self.cancel == Some(stage) {
            cancellation.cancel();
        }
        Ok(())
    }
    fn annotate(&self, value: &mut Value, field: &str) {
        let previous = value["values"][field].as_str().unwrap_or("").to_owned();
        value["values"][field] = json!(format!("{previous}{}", self.name));
        value["usage"]["input"] = json!(999);
        value["usage"]["output"] = json!(999);
        value["usage"]["total"] = json!(1998);
    }
}
impl Middleware for Recorder {
    async fn before_execute(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        self.record("before_execute", c).await?;
        value["payload"][0]["data"] = json!("modified question");
        Ok(())
    }
    async fn after_execute(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        self.record("after_execute", c).await?;
        self.annotate(value, "execute_order");
        Ok(())
    }
    async fn before_system_prompt(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        self.record("before_prompt", c).await?;
        *value = json!(format!("{}{}b", value.as_str().unwrap_or(""), self.name));
        Ok(())
    }
    async fn after_system_prompt(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        self.record("after_prompt", c).await?;
        *value = json!(format!("{}{}a", value.as_str().unwrap_or(""), self.name));
        Ok(())
    }
    async fn before_model(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        self.record("before_model", c).await?;
        if self.invalid {
            *value = json!([]);
        } else {
            value["temperature"] = json!(0.5);
        }
        Ok(())
    }
    async fn after_model(&self, value: &mut Value, c: &Cancellation) -> Result<(), Error> {
        self.record("after_model", c).await?;
        self.annotate(value, "model_order");
        Ok(())
    }
    async fn before_permission(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        panic!("no permission stage without permission operation")
    }
    async fn before_compress_context(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        panic!("no compression stage without compression operation")
    }
}

#[derive(Clone)]
struct Model {
    requests: Arc<Mutex<Vec<Request>>>,
    with_tool: bool,
    wait: bool,
}
impl Model {
    fn new() -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            with_tool: false,
            wait: false,
        }
    }
}
fn usage() -> Usage {
    Usage {
        input: Some(10),
        output: Some(3),
        total: Some(13),
        ..Default::default()
    }
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let first = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request);
            requests.len() == 1
        };
        if self.wait {
            pending::<()>().await;
        }
        let mut delta = Message::new(Role::Assistant);
        delta.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"partial"}).as_object().unwrap().clone(),
        });
        let mut message = Message::new(Role::Assistant);
        message.values.insert("answer".into(), json!("complete"));
        if self.with_tool && first {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"c1","name":"lookup","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        }
        Ok(stream::iter(vec![
            Ok(Event::Delta { message: delta }),
            Ok(Event::Complete {
                message,
                usage: Some(usage()),
                finish_reason: Some("stop".into()),
            }),
        ]))
    }
}

struct Lookup;
impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: std::sync::LazyLock<ToolDefinition> =
            std::sync::LazyLock::new(|| ToolDefinition {
                name: "lookup".into(),
                description: "lookup".into(),
                parameters: serde_json::from_value(json!({"type":"object","properties":{}}))
                    .unwrap(),
            });
        &DEFINITION
    }
    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        Ok(Message::function(json!({"found":true})))
    }
}

fn input() -> Input {
    Input {
        metadata: serde_json::from_value(json!({"chat_id":1})).unwrap(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"question"}),
            mime_type: None,
        }],
    }
}

async fn collect<A: Agent>(agent: &A, cancellation: Cancellation) -> Vec<Result<Event, Error>> {
    match agent.run(input(), true, cancellation).await {
        Ok(events) => events.collect().await,
        Err(error) => vec![Err(error)],
    }
}

async fn run(
    react: bool,
    model: Model,
    hooks: Vec<Recorder>,
    c: Cancellation,
) -> Vec<Result<Event, Error>> {
    if react {
        let mut builder = agent::react()
            .model(model)
            .system_prompt("system")
            .toolkit(Toolkit::new().tools(Lookup));
        for hook in hooks {
            builder = builder.middleware(hook);
        }
        collect(&builder.build().unwrap(), c).await
    } else {
        let mut builder = agent::chat().model(model).system_prompt("system");
        for hook in hooks {
            builder = builder.middleware(hook);
        }
        collect(&builder.build().unwrap(), c).await
    }
}

#[test]
fn both_agents_apply_onion_order_and_sequential_prompt_before_commit() {
    for react in [false, true] {
        let log = Log::default();
        let model = Model::new();
        let events = block_on(run(
            react,
            model.clone(),
            vec![Recorder::new("A", &log), Recorder::new("B", &log)],
            Cancellation::new(),
        ));
        assert_eq!(
            *log.lock().unwrap(),
            [
                "A:before_execute",
                "B:before_execute",
                "A:before_prompt",
                "B:before_prompt",
                "A:after_prompt",
                "B:after_prompt",
                "A:before_model",
                "B:before_model",
                "B:after_model",
                "A:after_model",
                "B:after_execute",
                "A:after_execute"
            ]
        );
        let requests = model.requests.lock().unwrap();
        assert_eq!(requests[0].temperature, Some(0.5));
        assert_eq!(
            requests[0].messages.0[0].content[0].data["value"],
            "systemAbBbAaBa"
        );
        assert_eq!(
            requests[0].messages.0[1].content[0].data["value"],
            "modified question"
        );
        assert!(!requests[0].messages.0[1].metadata.contains_key("chat_id"));
        assert_eq!(events.len(), 2);
        let Ok(Event::Complete {
            message,
            usage: actual,
            ..
        }) = &events[1]
        else {
            panic!("Complete after hooks")
        };
        assert_eq!(message.values["model_order"], "BA");
        assert_eq!(message.values["execute_order"], "BA");
        assert_eq!(message.usage, Some(usage()));
        assert_eq!(*actual, Some(usage()));
    }
}

#[test]
fn react_model_hooks_wrap_each_model_not_tool_completes() {
    let log = Log::default();
    let mut model = Model::new();
    model.with_tool = true;
    let events = block_on(run(
        true,
        model.clone(),
        vec![Recorder::new("A", &log)],
        Cancellation::new(),
    ));
    let log = log.lock().unwrap();
    assert_eq!(log.iter().filter(|s| *s == "A:before_model").count(), 2);
    assert_eq!(log.iter().filter(|s| *s == "A:after_model").count(), 2);
    assert_eq!(log.iter().filter(|s| *s == "A:after_execute").count(), 1);
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Ok(Event::Complete { .. })))
            .count(),
        3
    );
    let Ok(Event::Complete { message, .. }) = &events[2] else {
        panic!("Tool result")
    };
    assert_eq!(message.values["found"], true);
    assert!(!message.values.contains_key("model_order"));
    assert_eq!(
        model.requests.lock().unwrap()[1].messages.0[2].values["model_order"],
        "A"
    );
}

#[test]
fn each_stage_error_is_direct_and_stops_later_hooks_without_retry() {
    for react in [false, true] {
        for stage in [
            "before_execute",
            "before_prompt",
            "after_prompt",
            "before_model",
            "after_model",
            "after_execute",
        ] {
            let log = Log::default();
            let model = Model::new();
            let mut hook = Recorder::new("A", &log);
            hook.fail = Some(stage);
            let events = block_on(run(react, model.clone(), vec![hook], Cancellation::new()));
            assert_eq!(
                events.last().unwrap().as_ref().unwrap_err(),
                &Error::new("HOOK_FAILED", "original").with_details("unchanged")
            );
            assert_eq!(log.lock().unwrap().last().unwrap(), &format!("A:{stage}"));
            assert!(
                !events
                    .iter()
                    .any(|e| matches!(e, Ok(Event::Complete { .. })))
            );
            assert_eq!(
                model.requests.lock().unwrap().len(),
                usize::from(matches!(stage, "after_model" | "after_execute"))
            );
        }
    }
}

#[test]
fn malformed_hook_request_is_rejected_before_model_work() {
    for react in [false, true] {
        let log = Log::default();
        let model = Model::new();
        let mut hook = Recorder::new("A", &log);
        hook.invalid = true;
        let events = block_on(run(react, model.clone(), vec![hook], Cancellation::new()));
        assert_eq!(
            events.last().unwrap().as_ref().unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
        assert!(model.requests.lock().unwrap().is_empty());
    }
}

#[test]
fn reused_agents_do_not_share_per_run_hook_results() {
    let log = Log::default();
    let model = Model::new();
    let agent = agent::chat()
        .middleware(Recorder::new("A", &log))
        .model(model.clone())
        .build()
        .unwrap();
    for _ in 0..2 {
        let events = block_on(collect(&agent, Cancellation::new()));
        let Ok(Event::Complete { message, .. }) = &events[1] else {
            panic!("Complete")
        };
        assert_eq!(message.values["model_order"], "A");
    }
    assert_eq!(model.requests.lock().unwrap().len(), 2);
    assert_eq!(
        log.lock()
            .unwrap()
            .iter()
            .filter(|s| *s == "A:before_execute")
            .count(),
        2
    );
}

struct Other;
impl Middleware for Other {
    async fn before_model(&self, value: &mut Value, _: &Cancellation) -> Result<(), Error> {
        value["max_tokens"] = json!(25);
        Ok(())
    }
}
#[test]
fn heterogeneous_middlewares_attach_without_public_adapter() {
    let model = Model::new();
    let agent = agent::react()
        .middleware(Other)
        .model(model.clone())
        .middleware(Recorder::new("A", &Log::default()))
        .build()
        .unwrap();
    assert!(
        block_on(collect(&agent, Cancellation::new()))
            .iter()
            .all(Result::is_ok)
    );
    assert_eq!(model.requests.lock().unwrap()[0].max_tokens, Some(25));
}

#[test]
fn synchronous_after_cancel_keeps_received_complete_and_original_usage() {
    for react in [false, true] {
        let log = Log::default();
        let model = Model::new();
        let mut hook = Recorder::new("A", &log);
        hook.cancel = Some("after_model");
        let events = block_on(run(react, model, vec![hook], Cancellation::new()));
        let Ok(Event::Complete {
            message,
            usage: actual,
            finish_reason,
        }) = events.last().unwrap()
        else {
            panic!("cancelled Complete")
        };
        assert_eq!(finish_reason.as_deref(), Some("cancelled"));
        assert_eq!(message.values["answer"], "complete");
        assert!(!message.values.contains_key("model_order"));
        assert_eq!(*actual, Some(usage()));
        assert!(!log.lock().unwrap().iter().any(|s| s == "A:after_execute"));
    }
}

#[tokio::test]
async fn pending_before_and_after_hooks_are_woken_by_cancellation() {
    for react in [false, true] {
        for stage in ["before_model", "after_model"] {
            let log = Log::default();
            let model = Model::new();
            let mut hook = Recorder::new("A", &log);
            hook.wait = Some(stage);
            let c = Cancellation::new();
            let token = c.clone();
            let watched = log.clone();
            let task = tokio::spawn(async move {
                while !watched
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|s| s == &format!("A:{stage}"))
                {
                    tokio::task::yield_now().await;
                }
                token.cancel();
            });
            let events = tokio::time::timeout(
                Duration::from_secs(2),
                run(react, model.clone(), vec![hook], c),
            )
            .await
            .unwrap();
            task.await.unwrap();
            assert!(
                matches!(events.last().unwrap(),Ok(Event::Complete{finish_reason:Some(reason),..}) if reason == "cancelled")
            );
            assert_eq!(
                model.requests.lock().unwrap().len(),
                usize::from(stage == "after_model")
            );
        }
    }
}

#[tokio::test]
async fn pending_model_creation_is_cancellable_without_middleware() {
    for react in [false, true] {
        let mut model = Model::new();
        model.wait = true;
        let requests = model.requests.clone();
        let c = Cancellation::new();
        let token = c.clone();
        let task = tokio::spawn(async move {
            while requests.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
            token.cancel();
        });
        let events = tokio::time::timeout(Duration::from_secs(2), run(react, model, vec![], c))
            .await
            .unwrap();
        task.await.unwrap();
        assert!(
            matches!(events.last().unwrap(),Ok(Event::Complete{finish_reason:Some(reason),..}) if reason == "cancelled")
        );
    }
}

struct InvalidToolSchema;
impl Middleware for InvalidToolSchema {
    async fn before_model(&self, request: &mut Value, _: &Cancellation) -> Result<(), Error> {
        request["tools"] =
            json!([{"name":"invalid","description":"bad root","parameters":{"type":"array"}}]);
        Ok(())
    }
}

#[test]
fn request_changed_by_hooks_is_revalidated_before_model_dispatch() {
    for react in [false, true] {
        let model = Model::new();
        let events = if react {
            let agent = agent::react()
                .model(model.clone())
                .middleware(InvalidToolSchema)
                .build()
                .unwrap();
            block_on(collect(&agent, Cancellation::new()))
        } else {
            let agent = agent::chat()
                .model(model.clone())
                .middleware(InvalidToolSchema)
                .build()
                .unwrap();
            block_on(collect(&agent, Cancellation::new()))
        };
        assert_eq!(
            events.last().unwrap().as_ref().unwrap_err().code,
            "INVALID_ARGUMENTS"
        );
        assert!(model.requests.lock().unwrap().is_empty());
    }
}

struct MessageUsage;
impl Chat for MessageUsage {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        message.usage = Some(usage());
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: None,
        })]))
    }
}

#[test]
fn known_message_usage_survives_when_event_usage_is_absent() {
    for react in [false, true] {
        let events = if react {
            let agent = agent::react().model(MessageUsage).build().unwrap();
            block_on(collect(&agent, Cancellation::new()))
        } else {
            let agent = agent::chat().model(MessageUsage).build().unwrap();
            block_on(collect(&agent, Cancellation::new()))
        };
        let Ok(Event::Complete {
            message,
            usage: actual,
            ..
        }) = &events[0]
        else {
            panic!("Complete")
        };
        assert_eq!(message.usage, Some(usage()));
        assert_eq!(*actual, Some(usage()));
    }
}
