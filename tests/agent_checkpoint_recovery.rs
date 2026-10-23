//! Frozen Agent run entry: original query, identity and same planned execution.
use futures_util::{StreamExt, stream};
use halo::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, Payload},
    checkpoint::Checkpoint as _,
    core,
    event::Event,
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
};
use serde_json::json;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

type Responses = VecDeque<Vec<Result<Event, Error>>>;

#[derive(Clone)]
struct Model {
    requests: Arc<Mutex<Vec<Request>>>,
    responses: Arc<Mutex<Responses>>,
}
impl Model {
    fn new(responses: Vec<Vec<Result<Event, Error>>>) -> Self {
        Self {
            requests: Default::default(),
            responses: Arc::new(Mutex::new(responses.into())),
        }
    }
}
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.requests.lock().unwrap().push(request);
        Ok(stream::iter(
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected model replay"),
        ))
    }
}
fn input(text: &str) -> Input {
    Input {
        metadata: json!({"chat_id":1001,"tenant_id":1,"user_id":2,"agent_id":3,"session_id":4})
            .as_object()
            .unwrap()
            .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":text}),
            mime_type: None,
        }],
    }
}
fn message(text: &str) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":text}).as_object().unwrap().clone(),
    });
    message
}
fn usage(total: u64) -> Usage {
    Usage {
        input: Some(total),
        output: Some(0),
        total: Some(total),
        ..Default::default()
    }
}
fn complete(message: Message, total: u64) -> Result<Event, Error> {
    Ok(Event::Complete {
        message,
        usage: Some(usage(total)),
        finish_reason: Some("stop".into()),
    })
}
fn waiting() -> Result<Event, Error> {
    Err(Error::new("WAITING", "approval required"))
}

#[derive(Clone)]
struct OpeningErrorModel {
    error: Error,
}

impl Chat for OpeningErrorModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        Err(self.error.clone())
    }
}

struct CancellingModel(Cancellation);

impl Chat for CancellingModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        self.0.cancel();
        std::future::pending().await
    }
}

#[tokio::test]
async fn chat_cancelled_during_provider_preparation_saves_a_replayable_complete() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let original = input("write the original report");
    let cancellation = Cancellation::new();
    let first = agent::chat()
        .model(CancellingModel(cancellation.clone()))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = first
        .run(original.clone(), true, cancellation)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let [
        Ok(Event::Complete {
            message,
            usage,
            finish_reason,
        }),
    ] = events.as_slice()
    else {
        panic!("cancellation must close with the received complete: {events:?}");
    };
    assert!(message.content.is_empty());
    assert_eq!(finish_reason.as_deref(), Some("cancelled"));
    assert!(usage.as_ref().unwrap().input.unwrap() > 0);
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["status"], "stopped");
    assert_eq!(scene["finish_reason"], "cancelled");
    assert!(scene["final_message"].is_object());
    assert_eq!(scene["query"], json!(original));
    drop(first);
    let model = Model::new(Vec::new());
    let restored = agent::chat()
        .model(model.clone())
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let replay = restored
        .run(original, true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(replay, events);
    assert!(model.requests.lock().unwrap().is_empty());
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        scene
    );
}

#[tokio::test]
async fn chat_waiting_before_provider_stream_saves_scene_and_resumes_original_run() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let original = input("write the original report");
    let first = agent::chat()
        .model(OpeningErrorModel {
            error: Error::new("WAITING", "approval before model dispatch"),
        })
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = first
        .run(original.clone(), true, Cancellation::new())
        .await
        .expect("preparation waiting must close through the Agent event stream")
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.as_slice(), [Err(error)] if error.code == "WAITING"));
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["status"], "waiting");
    assert_eq!(scene["query"], json!(original));
    assert!(scene["usage"].is_null());
    assert_eq!(scene["messages"].as_array().unwrap().len(), 1);
    drop(first);

    let model = Model::new(vec![vec![complete(message("approved report"), 7)]]);
    let restored = agent::chat()
        .model(model.clone())
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = restored
        .run(input("approved"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(model.requests.lock().unwrap().len(), 1);
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["status"], "succeeded");
    assert_eq!(scene["query"], json!(original));
    assert_eq!(scene["usage"]["total"], 7);
}

#[tokio::test]
async fn chat_provider_preparation_failure_saves_terminal_scene_before_returning_error() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let original = input("write the original report");
    let first = agent::chat()
        .model(OpeningErrorModel {
            error: Error::new("NETWORK", "dispatch failed"),
        })
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let error = first
        .run(original.clone(), true, Cancellation::new())
        .await
        .unwrap_err();
    assert_eq!(error.code, "NETWORK");
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["status"], "failed");
    assert_eq!(scene["query"], json!(original));
    assert_eq!(scene["error"]["code"], "NETWORK");
    assert!(scene["usage"].is_null());

    let model = Model::new(Vec::new());
    let restored = agent::chat()
        .model(model.clone())
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = restored
        .run(original, true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.as_slice(), [Err(error)] if error.code == "NETWORK"));
    assert!(model.requests.lock().unwrap().is_empty());
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        scene
    );
}

#[tokio::test]
async fn chat_repeated_preparation_wait_preserves_saved_usage_without_double_settlement() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let original = input("write the original report");
    let mut partial = message("partial report");
    partial.usage = Some(usage(2));
    let first = agent::chat()
        .model(Model::new(vec![vec![
            Ok(Event::Delta { message: partial }),
            waiting(),
        ]]))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = first
        .run(original.clone(), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap()["usage"]["total"],
        2
    );
    drop(first);

    let second = agent::chat()
        .model(OpeningErrorModel {
            error: Error::new("WAITING", "approval still required"),
        })
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = second
        .run(input("approved"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.as_slice(), [Err(error)] if error.code == "WAITING"));
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["usage"]["total"], 2);
    assert_eq!(scene["query"], json!(original));
    drop(second);

    let third = agent::chat()
        .model(Model::new(vec![vec![complete(message("final report"), 5)]]))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = third
        .run(input("approved now"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["usage"]["total"], 7);
    assert_eq!(scene["query"], json!(original));
}

#[tokio::test]
async fn chat_continuation_preserves_original_query_and_terminal_idempotence() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let original = input("write the original report");
    let first_model = Model::new(vec![vec![waiting()]]);
    let first = agent::chat()
        .model(first_model)
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = first
        .run(original.clone(), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.as_slice(), [Err(error)] if error.code == "WAITING"));
    drop(first);

    let model = Model::new(vec![vec![complete(message("approved report"), 7)]]);
    let restored = agent::chat()
        .model(model.clone())
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = restored
        .run(input("approved"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(saved["query"], serde_json::to_value(&original).unwrap());
    let request = model.requests.lock().unwrap()[0].clone();
    assert!(request.messages.0.iter().any(|m| {
        m.content
            .iter()
            .any(|p| p.data["value"] == "write the original report")
    }));
    assert!(
        request
            .messages
            .0
            .iter()
            .any(|m| m.content.iter().any(|p| p.data["value"] == "approved"))
    );
    let events = restored
        .run(original, true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(model.requests.lock().unwrap().len(), 1);
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        saved
    );
}

#[tokio::test]
async fn react_waiting_restore_rejects_changed_run_ownership_before_model_or_writes() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let first = agent::react()
        .model(Model::new(vec![vec![waiting()]]))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = first
        .run(input("original"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.as_slice(), [Err(error)] if error.code == "WAITING"));
    let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    drop(first);
    let model = Model::new(vec![vec![complete(message("must not run"), 99)]]);
    let restored = agent::react()
        .model(model.clone())
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    for field in ["tenant_id", "user_id", "agent_id", "session_id"] {
        let mut changed = input("approval");
        changed.metadata.insert(field.into(), json!(99));
        let error = match restored.run(changed, true, Cancellation::new()).await {
            Ok(_) => panic!("changed {field} was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.code, "CONFLICT");
    }
    assert!(model.requests.lock().unwrap().is_empty());
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        saved
    );
}

fn first_id() -> i64 {
    6001
}
fn no_new_id() -> i64 {
    panic!("waiting continuation must keep its execution ID")
}

#[tokio::test]
async fn chat_waiting_reuses_memory_chat_and_final_trace_is_an_independent_scene_copy() {
    use chat::Store as _;
    use halo::memory::{
        self,
        short::{chat, session, trace},
    };
    use session::Store as _;
    use trace::Store as _;
    let memory = memory::Config::new(
        core::memory::short::session::InMemory::new(),
        core::memory::short::chat::InMemory::new(),
        core::memory::short::trace::InMemory::new(),
        core::memory::InMemory::new(),
    );
    memory
        .session
        .create(
            session::Session {
                id: 4,
                tenant_id: 1,
                user_id: 2,
                r#type: memory::short::Type::Agent,
                title: None,
                summary: None,
                usage: None,
                status: session::Status::Active,
                metadata: Default::default(),
                created_time: 0,
                updated_time: 0,
            },
            &Cancellation::new(),
        )
        .await
        .unwrap();
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let original = input("original task");
    let first = agent::chat()
        .model(OpeningErrorModel {
            error: Error::new("WAITING", "approval before model dispatch"),
        })
        .memory(memory.clone())
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = first
        .run(original.clone(), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
    let open = memory.chat.read(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(open.status, chat::Status::Running);
    assert!(open.completed_time.is_none());
    assert_eq!(
        memory
            .trace
            .read(1001, &Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    drop(first);

    let model = Model::new(vec![vec![complete(message("done"), 7)]]);
    let restored = agent::chat()
        .model(model.clone())
        .memory(memory.clone())
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = restored
        .run(input("approved"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    let closed = memory.chat.read(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(closed.status, chat::Status::Completed);
    assert_eq!(closed.created_time, open.created_time);
    assert_eq!(closed.input, open.input);
    assert_eq!(closed.usage, Some(usage(7)));
    let snapshot = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    let audit = memory.trace.read(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(audit.state, snapshot);
    assert_eq!(audit.state["chat_id"], 1001);
    assert_eq!(
        audit.state["query"],
        serde_json::to_value(&original).unwrap()
    );
    checkpoint.delete(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        memory.trace.read(1001, &Cancellation::new()).await.unwrap(),
        audit
    );
    // The Trace remains unchanged when the independent recovery store changes.
    checkpoint
        .save(1001, snapshot, &Cancellation::new())
        .await
        .unwrap();
    let events = restored
        .run(original, true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(model.requests.lock().unwrap().len(), 1);
    assert_eq!(
        memory.trace.read(1001, &Cancellation::new()).await.unwrap(),
        audit
    );
}

#[tokio::test]
async fn real_planned_waiting_keeps_execution_attempt_and_settles_only_new_usage() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let mut trigger = Message::new(Role::Assistant);
    trigger.content.push(Part {
        r#type: "tool_call".into(),
        data: json!({"call_id":"plan_1","name":"plan","arguments":{}})
            .as_object()
            .unwrap()
            .clone(),
    });
    let mut plan = Message::new(Role::Assistant);
    plan.values = json!({"version":1,"nodes":{"a":{"name":"research","objective":"write the report"}},"edges":[]}).as_object().unwrap().clone();
    let mut partial = message("waiting for approval");
    partial.usage = Some(usage(2));
    let first = agent::react()
        .model(Model::new(vec![
            vec![complete(trigger, 3)],
            vec![complete(plan, 3)],
            vec![Ok(Event::Delta { message: partial }), waiting()],
        ]))
        .checkpoint(checkpoint.clone())
        .with_id_fn(first_id)
        .build()
        .unwrap();
    let events = first
        .run(input("original"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
    let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(saved["nodes"]["a"]["status"], "waiting");
    assert_eq!(
        saved["nodes"]["a"]["execution"]["attempts"][0]["status"],
        "waiting"
    );
    assert!(saved["nodes"]["a"]["execution"]["error"].is_null());
    assert_eq!(saved["usage"]["total"], 8);
    let created = saved["nodes"]["a"]["execution"]["attempts"][0]["created_time"].clone();
    drop(first);

    let model = Model::new(vec![
        vec![complete(message("task report"), 4)],
        vec![complete(message("final report"), 5)],
    ]);
    let restored = agent::react()
        .model(model.clone())
        .checkpoint(checkpoint.clone())
        .with_id_fn(no_new_id)
        .build()
        .unwrap();
    let events = restored
        .run(input("approved"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    let completed = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    let execution = &completed["nodes"]["a"]["execution"];
    assert_eq!(completed["status"], "succeeded");
    assert_eq!(execution["exec_id"], 6001);
    assert_eq!(execution["attempts"].as_array().unwrap().len(), 1);
    assert_eq!(execution["attempts"][0]["created_time"], created);
    assert_eq!(execution["attempts"][0]["usage"]["total"], 6);
    assert_eq!(execution["usage"]["total"], 6);
    assert_eq!(completed["usage"]["total"], 17);
    assert_eq!(completed["query"], saved["query"]);
    assert_eq!(model.requests.lock().unwrap().len(), 2);
    let terminal = restored
        .run(input("original"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(terminal.iter().all(Result::is_ok));
    assert_eq!(model.requests.lock().unwrap().len(), 2);
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        completed
    );
}
