//! Successful Tool phases survive waiting; unfinished hook mutations do not.
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicI64, Ordering},
};

use futures_util::{StreamExt, poll, stream};
use halo_agents::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, toolkit::Toolkit},
    checkpoint::Checkpoint,
    core,
    event::Event,
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
    tool::{Call, Evaluation, Tool, ToolDefinition},
};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    Before,
    Call,
    After,
    Eval,
}

#[derive(Default, Debug)]
struct Log {
    before: usize,
    calls: usize,
    after: usize,
    eval: usize,
    requests: Vec<Request>,
}

#[derive(Clone)]
struct Model(Arc<Mutex<Log>>);
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let completed = request.messages.as_slice().iter().any(|message| {
            message.metadata.get("name") == Some(&json!("operation"))
                && message.values.get("n") == Some(&json!(3))
                || (message.role == Role::Tool
                    && message
                        .values
                        .get("error")
                        .is_some_and(|error| error["code"] == "BACKEND"))
        });
        self.0.lock().unwrap().requests.push(request);
        let mut message = Message::new(Role::Assistant);
        if completed {
            message.values.insert("answer".into(), json!("done"));
        } else {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"op_1","name":"operation","arguments":{"n":1}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                input: Some(3),
                output: Some(2),
                total: Some(5),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}

struct Operation {
    definition: ToolDefinition,
    log: Arc<Mutex<Log>>,
    pause: Option<Phase>,
    human: bool,
}
impl Operation {
    fn new(log: Arc<Mutex<Log>>, pause: Option<Phase>, human: bool) -> Self {
        Self { log, pause, human, definition: ToolDefinition {
            name:"operation".into(), description:"exercise native lifecycle facts".into(),
            parameters: json!({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"],"additionalProperties":false})
                .as_object().unwrap().clone(),
        }}
    }
}
impl Tool for Operation {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn before(&self, call: &mut Call, _: &Cancellation) -> Result<(), Error> {
        self.log.lock().unwrap().before += 1;
        assert_eq!(
            call.arguments["n"], 1,
            "never repeat a successful parameter rewrite"
        );
        if self.pause == Some(Phase::Before) {
            call.arguments.insert("n".into(), json!(999));
            return Err(Error::new("WAITING", "before unfinished"));
        }
        call.arguments.insert("n".into(), json!(2));
        Ok(())
    }
    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.log.lock().unwrap().calls += 1;
        assert_eq!(call.arguments["n"], 2);
        if self.pause == Some(Phase::Call) {
            return Err(Error::new("WAITING", "call unfinished"));
        }
        Ok(Message::function(json!({"n":2})))
    }
    async fn after(
        &self,
        call: &Call,
        message: &mut Message,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        self.log.lock().unwrap().after += 1;
        assert_eq!(call.arguments["n"], 2);
        assert_eq!(
            message.values["n"], 2,
            "resume after with the saved raw result"
        );
        message.values.insert("n".into(), json!(999));
        if self.human {
            let feedback = halo_agents::runtime::agent::wait::ask(
                Message::function(json!({"question":"confirm processing"})),
                2,
                cancellation,
            )
            .await?;
            assert_eq!(feedback.as_slice().len(), 2);
        }
        if self.pause == Some(Phase::After) {
            return Err(Error::new("WAITING", "after unfinished"));
        }
        message.values.insert("n".into(), json!(3));
        Ok(())
    }
    async fn eval(
        &self,
        call: &Call,
        message: &Message,
        _: &Cancellation,
    ) -> Result<Evaluation, Error> {
        self.log.lock().unwrap().eval += 1;
        assert_eq!(call.arguments["n"], 2);
        assert_eq!(message.values["n"], 3);
        if self.pause == Some(Phase::Eval) {
            return Err(Error::new("WAITING", "eval unfinished"));
        }
        Ok(Evaluation::passed())
    }
}

fn next_id() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(81_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
fn input() -> Input {
    serde_json::from_value(
        json!({"metadata":{"chat_id":1001,"session_id":2001,"agent_id":3001},
        "payload":[{"type":"text","data":{"value":"process original request"}}]}),
    )
    .unwrap()
}
fn answer(id: i64, text: &str) -> Input {
    let mut input = input();
    input.metadata.insert("reply_id".into(), id.into());
    input.payload[0].data = json!({"value":text});
    input
}
async fn run(
    react: bool,
    checkpoint: Arc<core::checkpoint::InMemory>,
    log: Arc<Mutex<Log>>,
    pause: Option<Phase>,
    mode: bool,
) -> Vec<Result<Event, Error>> {
    let toolkit = Toolkit::new().tools(Operation::new(log.clone(), pause, false));
    if react {
        agent::react()
            .model(Model(log))
            .toolkit(toolkit)
            .checkpoint(checkpoint)
            .with_id_fn(next_id)
            .build()
            .unwrap()
            .run(input(), mode, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await
    } else {
        agent::chat()
            .model(Model(log))
            .toolkit(toolkit)
            .checkpoint(checkpoint)
            .with_id_fn(next_id)
            .build()
            .unwrap()
            .run(input(), mode, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await
    }
}

fn check_saved(saved: &Value, phase: Phase) {
    let step = &saved["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["kind"] == "tool")
        .expect("actual Tool Step");
    assert_eq!(saved["status"], "waiting");
    assert_eq!(step["stage"], format!("{phase:?}").to_lowercase());
    assert_eq!(
        step["call"]["arguments"]["n"],
        if phase == Phase::Before { 1 } else { 2 }
    );
    if matches!(phase, Phase::After | Phase::Eval) {
        assert_eq!(
            step["message"]["values"]["n"],
            if phase == Phase::Eval { 3 } else { 2 }
        );
    } else {
        assert!(step.get("message").is_none());
    }
}
fn check_done(log: &Arc<Mutex<Log>>, phase: Phase, human: bool) {
    let log = log.lock().unwrap();
    assert_eq!(
        (log.before, log.calls, log.after, log.eval),
        (
            1 + usize::from(phase == Phase::Before),
            1 + usize::from(phase == Phase::Call),
            1 + usize::from(phase == Phase::After),
            1 + usize::from(phase == Phase::Eval)
        ),
        "{log:?}"
    );
    assert_eq!(log.requests.len(), 2, "do not replay the selecting model");
    if !human {
        assert!(
            log.requests[1]
                .messages
                .as_slice()
                .iter()
                .any(|message| message.values.get("n") == Some(&json!(3)))
        );
    }
}

#[tokio::test]
async fn each_unfinished_phase_restores_without_repeating_successful_predecessors() {
    for react in [false, true] {
        for mode in [false, true] {
            for phase in [Phase::Before, Phase::Call, Phase::After, Phase::Eval] {
                let store = Arc::new(core::checkpoint::InMemory::new());
                let log = Arc::new(Mutex::new(Log::default()));
                let events = run(react, store.clone(), log.clone(), Some(phase), mode).await;
                assert!(
                    matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
                    "{events:?}"
                );
                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                check_saved(&saved, phase);
                let restored = Arc::new(core::checkpoint::InMemory::new());
                // Only native JSON is handed to a newly constructed backend/Agent.
                restored
                    .save(
                        1001,
                        serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap(),
                        &Cancellation::new(),
                    )
                    .await
                    .unwrap();
                let events = run(react, restored.clone(), log.clone(), None, !mode).await;
                assert!(
                    events.iter().all(Result::is_ok),
                    "{react}/{phase:?}: {events:?}"
                );
                let final_state = restored.load(1001, &Cancellation::new()).await.unwrap();
                assert_eq!(final_state["status"], "succeeded");
                assert_eq!(final_state["usage"]["total"], 10);
                assert_eq!(final_state["stream"], mode);
                assert_eq!(final_state["query"], saved["query"]);
                assert_eq!(
                    final_state["steps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|step| step["kind"] == "tool")
                        .expect("actual Tool Step")["id"],
                    saved["steps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|step| step["kind"] == "tool")
                        .expect("actual Tool Step")["id"]
                );
                check_done(&log, phase, false);
            }
        }
    }
}

async fn human_recovery<A: Agent>(
    build: impl Fn() -> A,
    store: Arc<core::checkpoint::InMemory>,
    log: Arc<Mutex<Log>>,
    mode: bool,
) {
    let first = build();
    let mut events = Box::pin(first.run(input(), mode, Cancellation::new()).await.unwrap());
    let id = loop {
        match events.next().await.unwrap().unwrap() {
            Event::Complete { message, .. } => {
                if let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64) {
                    break id;
                }
                assert_ne!(
                    message.values.get("n"),
                    Some(&json!(2)),
                    "raw call is not delivered before after"
                );
            }
            Event::Delta { .. } => {}
        }
    };
    first
        .reply(answer(id, "first"), Cancellation::new())
        .await
        .unwrap();
    let saved = tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            if let std::task::Poll::Ready(Some(result)) = poll!(events.next()) {
                result.unwrap();
            }
            let saved = store.load(1001, &Cancellation::new()).await.unwrap();
            if saved["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["waits"][id.to_string()]["feedback"]
                .as_array()
                .unwrap()
                .len()
                == 1
            {
                break saved;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    check_saved(&saved, Phase::After);
    drop(events);
    drop(first);
    let restored = build();
    let events = restored
        .run(answer(id, "second"), !mode, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    let finished = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(finished["status"], "succeeded");
    assert_eq!(finished["usage"]["total"], 10);
    assert_eq!(
        finished["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][id.to_string()]["feedback"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        finished["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["waits"][id.to_string()]["status"],
        "resolved"
    );
    check_done(&log, Phase::After, true);
}

#[tokio::test]
async fn after_feedback_restores_partial_collection_and_the_raw_call_result() {
    for mode in [false, true] {
        for react in [false, true] {
            let store = Arc::new(core::checkpoint::InMemory::new());
            let log = Arc::new(Mutex::new(Log::default()));
            let toolkit = || Toolkit::new().tools(Operation::new(log.clone(), None, true));
            if react {
                human_recovery(
                    || {
                        agent::react()
                            .model(Model(log.clone()))
                            .toolkit(toolkit())
                            .checkpoint(store.clone())
                            .with_id_fn(next_id)
                            .build()
                            .unwrap()
                    },
                    store.clone(),
                    log.clone(),
                    mode,
                )
                .await;
            } else {
                human_recovery(
                    || {
                        agent::chat()
                            .model(Model(log.clone()))
                            .toolkit(toolkit())
                            .checkpoint(store.clone())
                            .with_id_fn(next_id)
                            .build()
                            .unwrap()
                    },
                    store.clone(),
                    log.clone(),
                    mode,
                )
                .await;
            }
        }
    }
}

fn native_step(value: &Value) -> Option<&Value> {
    if let Some(step) = value["steps"].as_array().and_then(|steps| {
        steps
            .iter()
            .find(|step| step["kind"] == "tool" && step["call_id"] == "op_1")
    }) {
        return Some(step);
    }
    for field in ["agent", "graph", "execution"] {
        if let Some(step) = value.get(field).and_then(native_step) {
            return Some(step);
        }
    }
    for field in ["nodes", "executions"] {
        if let Some(entries) = value[field].as_object() {
            for child in entries.values() {
                if let Some(step) = native_step(child) {
                    return Some(step);
                }
            }
        }
    }
    value["attempts"]
        .as_array()
        .and_then(|attempts| attempts.last())
        .and_then(native_step)
}

fn embedded<A: Agent + 'static>(
    agent: A,
    depth: usize,
) -> halo_agents::runtime::workflow::CompiledWorkflow {
    use halo_agents::runtime::workflow::{
        WorkflowBuilder,
        graph::{END, START},
    };
    let mut builder = WorkflowBuilder::new("hook-agent");
    builder.add_agent_node("worker", agent).unwrap();
    builder.add_edge(START, "worker").unwrap();
    builder.add_edge("worker", END).unwrap();
    let mut graph = builder.compile().unwrap();
    for level in 0..depth {
        let mut builder = WorkflowBuilder::new(format!("hook-wrapper-{level}"));
        builder.add_graph_node("inside", &graph).unwrap();
        builder.add_edge(START, "inside").unwrap();
        builder.add_edge("inside", END).unwrap();
        graph = builder.compile().unwrap();
    }
    graph
}
fn graph(
    react: bool,
    depth: usize,
    pause: Option<Phase>,
    human: bool,
    log: Arc<Mutex<Log>>,
) -> halo_agents::runtime::workflow::CompiledWorkflow {
    let toolkit = Toolkit::new().tools(Operation::new(log.clone(), pause, human));
    if react {
        embedded(
            agent::react()
                .model(Model(log))
                .toolkit(toolkit)
                .with_id_fn(next_id)
                .build()
                .unwrap(),
            depth,
        )
    } else {
        embedded(
            agent::chat()
                .model(Model(log))
                .toolkit(toolkit)
                .with_id_fn(next_id)
                .build()
                .unwrap(),
            depth,
        )
    }
}
fn initial() -> halo_agents::Messages {
    let mut message = Message::new(Role::User);
    message.metadata = input().metadata;
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"original task"})
            .as_object()
            .unwrap()
            .clone(),
    });
    halo_agents::Messages::new([message])
}

#[tokio::test]
async fn direct_and_nested_workflows_restore_hook_phases_from_the_outer_scene() {
    use halo_agents::runtime::workflow::Runtime;
    for react in [false, true] {
        for depth in [0, 2] {
            for phase in [Phase::Before, Phase::Call, Phase::After] {
                let log = Arc::new(Mutex::new(Log::default()));
                let first =
                    Runtime::new(1001, graph(react, depth, Some(phase), false, log.clone()))
                        .unwrap()
                        .with_max_concurrency(1)
                        .unwrap();
                let events = first
                    .run(initial(), Arc::new(next_id), Cancellation::new())
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                assert!(
                    matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
                    "{events:?}"
                );
                let saved = first.snapshot().unwrap();
                let step = native_step(&saved).unwrap();
                assert_eq!(step["stage"], format!("{phase:?}").to_lowercase());
                if phase == Phase::After {
                    assert_eq!(step["message"]["values"]["n"], 2);
                }
                let mut restored = Runtime::from_snapshot(
                    1001,
                    graph(react, depth, None, false, log.clone()),
                    serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap(),
                )
                .unwrap()
                .with_max_concurrency(1)
                .unwrap();
                restored
                    .resume(
                        if depth == 0 { "worker" } else { "inside" },
                        halo_agents::Messages::new([Message::function(json!({"continue":true}))]),
                    )
                    .unwrap();
                let events = restored
                    .run(initial(), Arc::new(next_id), Cancellation::new())
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                assert!(
                    events.iter().all(Result::is_ok),
                    "{react}/{depth}/{phase:?}: {events:?}"
                );
                let final_state = restored.snapshot().unwrap();
                assert_eq!(final_state["status"], "succeeded");
                assert_eq!(final_state["usage"]["total"], 10);
                assert_eq!(native_step(&final_state).unwrap()["id"], step["id"]);
                check_done(&log, phase, false);
            }
        }
    }
}

#[tokio::test]
async fn nested_workflow_after_feedback_uses_the_saved_result_and_original_question() {
    use halo_agents::runtime::workflow::Runtime;
    for react in [false, true] {
        let log = Arc::new(Mutex::new(Log::default()));
        let store = Arc::new(core::checkpoint::InMemory::new());
        let first = Runtime::new(1001, graph(react, 2, None, true, log.clone()))
            .unwrap()
            .with_checkpoint(store.clone())
            .with_max_concurrency(1)
            .unwrap();
        let mut events = Box::pin(
            first
                .run(initial(), Arc::new(next_id), Cancellation::new())
                .await
                .unwrap(),
        );
        let id = loop {
            match events.next().await.unwrap().unwrap() {
                Event::Complete { message, .. } => {
                    if let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64) {
                        break id;
                    }
                }
                Event::Delta { .. } => {}
            }
        };
        let saved = store.load(1001, &Cancellation::new()).await.unwrap();
        let step = native_step(&saved).unwrap();
        assert_eq!(step["stage"], "after");
        assert_eq!(step["message"]["values"]["n"], 2);
        drop(events);
        drop(first);
        let restored =
            Runtime::from_snapshot(1001, graph(react, 2, None, true, log.clone()), saved)
                .unwrap()
                .with_checkpoint(store.clone())
                .with_max_concurrency(1)
                .unwrap();
        let feedback = |text: &str| {
            let input = answer(id, text);
            let mut message = Message::new(Role::User);
            message.metadata = input.metadata;
            message.content.push(Part {
                r#type: "text".into(),
                data: input.payload[0].data.as_object().unwrap().clone(),
            });
            halo_agents::Messages::new([message])
        };
        restored
            .reply(feedback("first"), Cancellation::new())
            .await
            .unwrap();
        let mut events = Box::pin(
            restored
                .run(initial(), Arc::new(next_id), Cancellation::new())
                .await
                .unwrap(),
        );
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            loop {
                if let std::task::Poll::Ready(Some(result)) = poll!(events.next()) {
                    result.unwrap();
                }
                let saved = store.load(1001, &Cancellation::new()).await.unwrap();
                if native_step(&saved).unwrap()["waits"][id.to_string()]["feedback"]
                    .as_array()
                    .unwrap()
                    .len()
                    == 1
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        restored
            .reply(feedback("second"), Cancellation::new())
            .await
            .unwrap();
        let events = events.collect::<Vec<_>>().await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert_eq!(restored.snapshot().unwrap()["status"], "succeeded");
        check_done(&log, Phase::After, true);
    }
}

#[derive(Clone)]
struct ParentModel(Arc<Mutex<usize>>);
impl Chat for ParentModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        *self.0.lock().unwrap() += 1;
        let mut message = Message::new(Role::Assistant);
        if request
            .messages
            .as_slice()
            .iter()
            .any(|message| message.metadata.get("name") == Some(&json!("agent_spawn")))
        {
            message.values.insert("answer".into(), json!("parent done"));
        } else {
            message.content.push(Part{r#type:"tool_call".into(),data:json!({"call_id":"spawn_1","name":"agent_spawn",
                "arguments":{"name":"worker","input":{"payload":[{"type":"text","data":{"value":"child task"}}]}}}).as_object().unwrap().clone()});
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                input: Some(3),
                output: Some(2),
                total: Some(5),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}
fn child(react: bool, pause: Option<Phase>, log: Arc<Mutex<Log>>) -> agent::SubAgent {
    let toolkit = Toolkit::new().tools(Operation::new(log.clone(), pause, false));
    if react {
        agent::SubAgent::new(
            "worker",
            agent::react()
                .model(Model(log))
                .toolkit(toolkit)
                .with_id_fn(next_id)
                .build()
                .unwrap(),
        )
        .unwrap()
    } else {
        agent::SubAgent::new(
            "worker",
            agent::chat()
                .model(Model(log))
                .toolkit(toolkit)
                .with_id_fn(next_id)
                .build()
                .unwrap(),
        )
        .unwrap()
    }
}

#[tokio::test]
async fn delegated_agents_keep_successful_hook_phases_in_their_actual_child_scene() {
    for parent_react in [false, true] {
        for child_react in [false, true] {
            for phase in [Phase::Before, Phase::Call, Phase::After] {
                let store = Arc::new(core::checkpoint::InMemory::new());
                let log = Arc::new(Mutex::new(Log::default()));
                let requests = Arc::new(Mutex::new(0));
                let mut saved_step = Value::Null;
                for pause in [Some(phase), None] {
                    let child = child(child_react, pause, log.clone());
                    let model = ParentModel(requests.clone());
                    let events = if parent_react {
                        agent::react()
                            .model(model)
                            .subagents([child])
                            .checkpoint(store.clone())
                            .with_id_fn(next_id)
                            .build()
                            .unwrap()
                            .run(input(), true, Cancellation::new())
                            .await
                            .unwrap()
                            .collect::<Vec<_>>()
                            .await
                    } else {
                        agent::chat()
                            .model(model)
                            .subagents([child])
                            .checkpoint(store.clone())
                            .with_id_fn(next_id)
                            .build()
                            .unwrap()
                            .run(input(), true, Cancellation::new())
                            .await
                            .unwrap()
                            .collect::<Vec<_>>()
                            .await
                    };
                    let scene = store.load(1001, &Cancellation::new()).await.unwrap();
                    let step = native_step(&scene).unwrap();
                    if pause.is_some() {
                        assert!(
                            matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
                            "{events:?}"
                        );
                        assert_eq!(step["stage"], format!("{phase:?}").to_lowercase());
                        saved_step = step["id"].clone();
                    } else {
                        assert!(events.iter().all(Result::is_ok), "{events:?}");
                        assert_eq!(scene["status"], "succeeded");
                        assert_eq!(scene["usage"]["total"], 20);
                        assert_eq!(step["id"], saved_step);
                    }
                }
                assert_eq!(
                    *requests.lock().unwrap(),
                    2,
                    "do not replay parent delegation"
                );
                check_done(&log, phase, false);
            }
        }
    }
}

#[tokio::test]
async fn inconsistent_phase_results_fail_before_any_component_is_invoked() {
    for react in [false, true] {
        let store = Arc::new(core::checkpoint::InMemory::new());
        let log = Arc::new(Mutex::new(Log::default()));
        run(react, store.clone(), log.clone(), Some(Phase::After), true).await;
        let saved = store.load(1001, &Cancellation::new()).await.unwrap();
        for stage in ["after", "before", "call"] {
            let mut invalid = saved.clone();
            invalid["steps"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["stage"] = stage.into();
            if stage == "after" {
                invalid["steps"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|step| step["kind"] == "tool")
                    .expect("actual Tool Step")
                    .as_object_mut()
                    .unwrap()
                    .remove("message");
            }
            let malformed = Arc::new(core::checkpoint::InMemory::new());
            malformed
                .save(1001, invalid.clone(), &Cancellation::new())
                .await
                .unwrap();
            let toolkit = Toolkit::new().tools(Operation::new(log.clone(), None, false));
            let result = if react {
                agent::react()
                    .model(Model(log.clone()))
                    .toolkit(toolkit)
                    .checkpoint(malformed.clone())
                    .build()
                    .unwrap()
                    .run(input(), true, Cancellation::new())
                    .await
                    .map(|_| ())
            } else {
                agent::chat()
                    .model(Model(log.clone()))
                    .toolkit(toolkit)
                    .checkpoint(malformed.clone())
                    .build()
                    .unwrap()
                    .run(input(), true, Cancellation::new())
                    .await
                    .map(|_| ())
            };
            assert!(matches!(result,Err(error) if error.code=="INVALID_STATE"));
            assert_eq!(
                malformed.load(1001, &Cancellation::new()).await.unwrap(),
                invalid
            );
        }
        let log = log.lock().unwrap();
        assert_eq!(
            (
                log.before,
                log.calls,
                log.after,
                log.eval,
                log.requests.len()
            ),
            (1, 1, 1, 0, 1)
        );
    }
}

#[derive(Default)]
struct CountingStore {
    inner: core::checkpoint::InMemory,
    writes: std::sync::atomic::AtomicUsize,
}
impl Checkpoint for CountingStore {
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        self.inner.save(id, state, cancellation).await
    }
    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        self.inner.load(id, cancellation).await
    }
    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, cancellation).await
    }
}
struct PendingAfter(Operation, Arc<std::sync::atomic::AtomicBool>);
impl Tool for PendingAfter {
    fn definition(&self) -> &ToolDefinition {
        self.0.definition()
    }
    async fn before(&self, call: &mut Call, cancellation: &Cancellation) -> Result<(), Error> {
        self.0.before(call, cancellation).await
    }
    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        self.0.call(call, cancellation).await
    }
    async fn after(&self, _: &Call, message: &mut Message, _: &Cancellation) -> Result<(), Error> {
        self.0.log.lock().unwrap().after += 1;
        assert_eq!(message.values["n"], 2);
        message.values.insert("n".into(), json!(999));
        self.1.store(true, Ordering::SeqCst);
        std::future::pending().await
    }
}

struct FailingAfter(Operation);
impl Tool for FailingAfter {
    fn definition(&self) -> &ToolDefinition {
        self.0.definition()
    }
    async fn before(&self, call: &mut Call, cancellation: &Cancellation) -> Result<(), Error> {
        self.0.before(call, cancellation).await
    }
    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        self.0.call(call, cancellation).await
    }
    async fn after(&self, _: &Call, message: &mut Message, _: &Cancellation) -> Result<(), Error> {
        self.0.log.lock().unwrap().after += 1;
        message.values.insert("n".into(), json!(999));
        Err(Error::new("BACKEND", "after unavailable"))
    }
}

#[tokio::test]
async fn after_technical_failure_preserves_raw_success_and_delivers_only_the_original_error() {
    for react in [false, true] {
        let store = Arc::new(core::checkpoint::InMemory::new());
        let log = Arc::new(Mutex::new(Log::default()));
        let toolkit = Toolkit::new().tools(FailingAfter(Operation::new(log.clone(), None, false)));
        let events = if react {
            agent::react()
                .model(Model(log.clone()))
                .toolkit(toolkit)
                .checkpoint(store.clone())
                .build()
                .unwrap()
                .run(input(), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        } else {
            agent::chat()
                .model(Model(log.clone()))
                .toolkit(toolkit)
                .checkpoint(store.clone())
                .build()
                .unwrap()
                .run(input(), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        };
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        let saved = store.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(saved["status"], "succeeded");
        assert_eq!(
            saved["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["status"],
            "failed"
        );
        assert_eq!(
            saved["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["error"]["code"],
            "BACKEND"
        );
        assert_eq!(
            saved["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["message"]["values"]["n"],
            2
        );
        let log = log.lock().unwrap();
        assert_eq!(
            (
                log.before,
                log.calls,
                log.after,
                log.eval,
                log.requests.len()
            ),
            (1, 1, 1, 0, 2)
        );
        let context = log.requests[1]
            .messages
            .as_slice()
            .iter()
            .find(|message| message.role == Role::Tool)
            .unwrap();
        assert_eq!(context.values["error"]["code"], "BACKEND");
        assert!(
            !context.values.contains_key("n"),
            "saved raw success is not a normal downstream result"
        );
    }
}

struct MixedAfter {
    inner: Operation,
    sent: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    received: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}
impl Tool for MixedAfter {
    fn definition(&self) -> &ToolDefinition {
        self.inner.definition()
    }
    async fn before(&self, call: &mut Call, cancellation: &Cancellation) -> Result<(), Error> {
        self.inner.before(call, cancellation).await
    }
    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        self.inner.call(call, cancellation).await
    }
    async fn after(
        &self,
        call: &Call,
        message: &mut Message,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        if call.call_id == "op_1" {
            self.inner.log.lock().unwrap().after += 1;
            if let Some(sent) = self.sent.lock().unwrap().take() {
                let _ = sent.send(());
            }
            return Err(Error::new("BACKEND", "first after failed"));
        }
        if self.inner.pause.is_some() {
            self.inner.log.lock().unwrap().after += 1;
            let received = self.received.lock().unwrap().take();
            if let Some(received) = received {
                received.await.unwrap();
            }
            return Err(Error::new("WAITING", "second after unfinished"));
        }
        self.inner.after(call, message, cancellation).await
    }
    async fn eval(
        &self,
        call: &Call,
        message: &Message,
        cancellation: &Cancellation,
    ) -> Result<Evaluation, Error> {
        self.inner.eval(call, message, cancellation).await
    }
}
#[derive(Clone)]
struct MixedModel(Arc<Mutex<Log>>);
impl Chat for MixedModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let done = request
            .messages
            .as_slice()
            .iter()
            .any(|message| message.values.get("n") == Some(&json!(3)));
        if done {
            assert_eq!(
                request
                    .messages
                    .as_slice()
                    .iter()
                    .filter(|message| message.values.contains_key("error"))
                    .count(),
                1
            );
            assert!(
                !request
                    .messages
                    .as_slice()
                    .iter()
                    .any(|message| message.values.get("n") == Some(&json!(2)))
            );
        }
        self.0.lock().unwrap().requests.push(request);
        let mut message = Message::new(Role::Assistant);
        if done {
            message.values.insert("answer".into(), json!("done"));
        } else {
            for id in ["op_1", "op_2"] {
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: json!({"call_id":id,"name":"operation","arguments":{"n":1}})
                        .as_object()
                        .unwrap()
                        .clone(),
                });
            }
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                input: Some(3),
                output: Some(2),
                total: Some(5),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}

#[tokio::test]
async fn a_failed_after_is_reused_as_error_context_while_a_sibling_after_resumes() {
    for react in [false, true] {
        let store = Arc::new(core::checkpoint::InMemory::new());
        let log = Arc::new(Mutex::new(Log::default()));
        for waiting in [true, false] {
            let (sent, received) = tokio::sync::oneshot::channel();
            let toolkit = Toolkit::new().tools(MixedAfter {
                inner: Operation::new(log.clone(), waiting.then_some(Phase::After), false),
                sent: Mutex::new(Some(sent)),
                received: Mutex::new(Some(received)),
            });
            let events = if react {
                agent::react()
                    .model(MixedModel(log.clone()))
                    .toolkit(toolkit)
                    .checkpoint(store.clone())
                    .build()
                    .unwrap()
                    .run(input(), true, Cancellation::new())
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await
            } else {
                agent::chat()
                    .model(MixedModel(log.clone()))
                    .toolkit(toolkit)
                    .checkpoint(store.clone())
                    .build()
                    .unwrap()
                    .run(input(), true, Cancellation::new())
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await
            };
            let saved = store.load(1001, &Cancellation::new()).await.unwrap();
            if waiting {
                assert!(
                    matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
                    "{events:?}"
                );
            } else {
                assert!(events.iter().all(Result::is_ok), "{events:?}");
            }
            assert_eq!(
                saved["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|step| step["kind"] == "tool")
                    .expect("actual Tool Step")["status"],
                "failed"
            );
            assert_eq!(
                saved["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|step| step["kind"] == "tool")
                    .expect("actual Tool Step")["message"]["values"]["n"],
                2
            );
        }
        let log = log.lock().unwrap();
        assert_eq!(
            (
                log.before,
                log.calls,
                log.after,
                log.eval,
                log.requests.len()
            ),
            (2, 2, 3, 1, 2)
        );
    }
}
async fn cancel_pending_after<A: Agent>(
    agent: A,
    store: Arc<CountingStore>,
    entered: Arc<std::sync::atomic::AtomicBool>,
    log: Arc<Mutex<Log>>,
) {
    let cancellation = Cancellation::new();
    let mut events = Box::pin(
        agent
            .run(input(), true, cancellation.clone())
            .await
            .unwrap(),
    );
    assert!(matches!(
        events.next().await,
        Some(Ok(Event::Complete { .. }))
    ));
    let writes = store.writes.load(Ordering::SeqCst);
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while !entered.load(Ordering::SeqCst) {
            if let std::task::Poll::Ready(Some(result)) = poll!(events.next()) {
                result.unwrap();
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        store.writes.load(Ordering::SeqCst),
        writes,
        "successful phases update memory only"
    );
    cancellation.cancel();
    let events = events.collect::<Vec<_>>().await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    assert!(events.iter().any(|event|matches!(event,Ok(Event::Complete{finish_reason:Some(reason),..}) if reason=="cancelled")));
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(saved["status"], "stopped");
    assert_eq!(
        saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["stage"],
        "after"
    );
    assert_eq!(
        saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["status"],
        "cancelled"
    );
    assert_eq!(
        saved["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["message"]["values"]["n"],
        2
    );
    assert_eq!(saved["usage"]["total"], 5);
    let log = log.lock().unwrap();
    assert_eq!((log.before, log.calls, log.after, log.eval), (1, 1, 1, 0));
}

#[tokio::test]
async fn cancellation_keeps_raw_success_without_new_phase_checkpoint_writes() {
    for react in [false, true] {
        let store = Arc::new(CountingStore::default());
        let log = Arc::new(Mutex::new(Log::default()));
        let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let toolkit = Toolkit::new().tools(PendingAfter(
            Operation::new(log.clone(), None, false),
            entered.clone(),
        ));
        if react {
            let agent = agent::react()
                .model(Model(log.clone()))
                .toolkit(toolkit)
                .checkpoint(store.clone())
                .build()
                .unwrap();
            cancel_pending_after(agent, store, entered, log).await;
        } else {
            let agent = agent::chat()
                .model(Model(log.clone()))
                .toolkit(toolkit)
                .checkpoint(store.clone())
                .build()
                .unwrap();
            cancel_pending_after(agent, store, entered, log).await;
        }
    }
}

#[derive(Clone)]
struct PlanModel(Arc<Mutex<Log>>);
impl Chat for PlanModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        if request
            .messages
            .as_slice()
            .iter()
            .any(|message| message.values.contains_key("plan_node"))
        {
            return Model(self.0.clone()).stream(request).await;
        }
        let mut message = Message::new(Role::Assistant);
        if request
            .messages
            .as_slice()
            .iter()
            .any(|message| message.values.contains_key("plan_complete"))
        {
            message.values.insert("answer".into(), json!("plan done"));
        } else if self.0.lock().unwrap().requests.is_empty() {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"plan_1","name":"plan","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        } else {
            message.values=json!({"version":1,"nodes":{"a":{"name":"process","objective":"process original task"}},"edges":[]}).as_object().unwrap().clone();
        }
        self.0.lock().unwrap().requests.push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(Usage {
                input: Some(3),
                output: Some(2),
                total: Some(5),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}

#[tokio::test]
async fn planned_tasks_restore_phases_in_the_same_attempt_independently_and_inside_graphs() {
    use halo_agents::runtime::workflow::Runtime;
    for workflow in [false, true] {
        for phase in [Phase::Before, Phase::Call, Phase::After] {
            let store = Arc::new(core::checkpoint::InMemory::new());
            let log = Arc::new(Mutex::new(Log::default()));
            let mut previous: Option<Value> = None;
            let mut original = Value::Null;
            for pause in [Some(phase), None] {
                let agent = agent::react()
                    .model(PlanModel(log.clone()))
                    .toolkit(Toolkit::new().tools(Operation::new(log.clone(), pause, false)))
                    .checkpoint(store.clone())
                    .with_id_fn(next_id)
                    .with_max_concurrency(1)
                    .build()
                    .unwrap();
                let (events, saved) = if workflow {
                    let compiled = embedded(agent, 2);
                    let runtime = if let Some(saved) = previous.take() {
                        let mut runtime = Runtime::from_snapshot(1001, compiled, saved).unwrap();
                        runtime
                            .resume(
                                "inside",
                                halo_agents::Messages::new([Message::function(
                                    json!({"continue":true}),
                                )]),
                            )
                            .unwrap();
                        runtime
                    } else {
                        Runtime::new(1001, compiled).unwrap()
                    };
                    let events = runtime
                        .run(initial(), Arc::new(next_id), Cancellation::new())
                        .await
                        .unwrap()
                        .collect::<Vec<_>>()
                        .await;
                    (events, runtime.snapshot().unwrap())
                } else {
                    let events = agent
                        .run(input(), true, Cancellation::new())
                        .await
                        .unwrap()
                        .collect::<Vec<_>>()
                        .await;
                    (
                        events,
                        store.load(1001, &Cancellation::new()).await.unwrap(),
                    )
                };
                let step = native_step(&saved).unwrap();
                if pause.is_some() {
                    assert!(
                        matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
                        "{events:?}"
                    );
                    assert_eq!(step["stage"], format!("{phase:?}").to_lowercase());
                    original = step["id"].clone();
                    previous = Some(saved);
                } else {
                    assert!(
                        events.iter().all(Result::is_ok),
                        "{workflow}/{phase:?}: {events:?}"
                    );
                    assert_eq!(saved["status"], "succeeded");
                    assert_eq!(step["id"], original);
                    assert_eq!(saved["usage"]["total"], 25);
                }
            }
            let log = log.lock().unwrap();
            assert_eq!(
                (log.before, log.calls, log.after),
                (
                    1 + usize::from(phase == Phase::Before),
                    1 + usize::from(phase == Phase::Call),
                    1 + usize::from(phase == Phase::After)
                )
            );
            assert_eq!(log.eval, 1);
            assert_eq!(
                log.requests.len(),
                5,
                "do not replay planning or selecting models"
            );
        }
    }
}

#[tokio::test]
#[ignore = "explicitly invoked by hook_phases_restore_in_fresh_os_processes"]
async fn hook_stage_worker() {
    let phase = match std::env::var("HEART_HOOK_STAGE").unwrap().as_str() {
        "before" => Phase::Before,
        "call" => Phase::Call,
        "after" => Phase::After,
        "eval" => Phase::Eval,
        _ => panic!("invalid worker phase"),
    };
    let restore = std::env::var("HEART_HOOK_RESTORE").unwrap() == "true";
    let react = std::env::var("HEART_HOOK_REACT").unwrap() == "true";
    let path = std::path::PathBuf::from(std::env::var_os("HEART_HOOK_SCENE").unwrap());
    let store = Arc::new(core::checkpoint::InMemory::new());
    if restore {
        store
            .save(
                1001,
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap(),
                &Cancellation::new(),
            )
            .await
            .unwrap();
    }
    let log = Arc::new(Mutex::new(Log::default()));
    let events = run(
        react,
        store.clone(),
        log.clone(),
        (!restore).then_some(phase),
        true,
    )
    .await;
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    if restore {
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert_eq!(saved["status"], "succeeded");
        assert_eq!(saved["usage"]["total"], 10);
        let log = log.lock().unwrap();
        assert_eq!(
            (
                log.before,
                log.calls,
                log.after,
                log.eval,
                log.requests.len()
            ),
            (
                usize::from(phase == Phase::Before),
                usize::from(matches!(phase, Phase::Before | Phase::Call)),
                usize::from(phase != Phase::Eval),
                1,
                1
            )
        );
    } else {
        assert!(
            matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
            "{events:?}"
        );
        check_saved(&saved, phase);
        std::fs::write(path, serde_json::to_vec(&saved).unwrap()).unwrap();
    }
}

#[test]
fn hook_phases_restore_in_fresh_os_processes() {
    let root = std::env::temp_dir().join(format!(
        "heart-hook-stage-{}-{}",
        std::process::id(),
        next_id()
    ));
    std::fs::create_dir(&root).unwrap();
    for react in [false, true] {
        for phase in ["before", "call", "after", "eval"] {
            for restore in [false, true] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "hook_stage_worker", "--ignored", "--nocapture"])
                    .env("HEART_HOOK_STAGE", phase)
                    .env("HEART_HOOK_RESTORE", restore.to_string())
                    .env("HEART_HOOK_REACT", react.to_string())
                    .env("HEART_HOOK_SCENE", root.join(format!("{phase}.json")))
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{phase}/{restore}: {} {}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
