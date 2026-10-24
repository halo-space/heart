//! I01/I02: confirmed Tool phases and real Agent continuation under parent scenes.
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::checkpoint::Checkpoint as _,
    agent::event::Event,
    agent::message::Role,
    agent::model::{
        chat::{Chat, Request},
        content::part::Part,
    },
    agent::tool::{Call, Evaluation, Tool, ToolDefinition},
    agent::{self, Agent, Input, Payload},
    core,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
};

fn embedded_workflow<A: Agent + 'static>(agent: A) -> halo::runtime::workflow::CompiledWorkflow {
    use halo::runtime::workflow::{
        WorkflowBuilder,
        graph::{END, START},
    };
    let mut builder = WorkflowBuilder::new("builtin-agent-eval-continuation");
    builder.add_agent_node("agent", agent).unwrap();
    builder.add_edge(START, "agent").unwrap();
    builder.add_edge("agent", END).unwrap();
    builder.compile().unwrap()
}

fn wrap_workflow(
    mut compiled: halo::runtime::workflow::CompiledWorkflow,
    depth: usize,
) -> halo::runtime::workflow::CompiledWorkflow {
    use halo::runtime::workflow::{
        WorkflowBuilder,
        graph::{END, START},
    };
    for level in 0..depth {
        let mut builder = WorkflowBuilder::new(format!("embedded-agent-wrapper-{level}"));
        builder.add_graph_node("child", &compiled).unwrap();
        builder.add_edge(START, "child").unwrap();
        builder.add_edge("child", END).unwrap();
        compiled = builder.compile().unwrap();
    }
    compiled
}

fn agent_execution(
    state: &halo::runtime::workflow::State,
) -> &halo::runtime::workflow::node::Execution {
    if let Some(child) = state.nodes.get("child") {
        agent_execution(
            child
                .executions
                .values()
                .next()
                .unwrap()
                .attempts
                .last()
                .unwrap()
                .graph
                .as_ref()
                .unwrap(),
        )
    } else {
        state.nodes["agent"].executions.values().next().unwrap()
    }
}

struct WaitingModel;
impl Chat for WaitingModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        message.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"received before waiting"})
                .as_object()
                .unwrap()
                .clone(),
        });
        message.usage = Some(halo::agent::model::token::Usage {
            input: Some(3),
            output: Some(0),
            total: Some(3),
            ..Default::default()
        });
        Ok(stream::iter(vec![
            Ok(Event::Delta { message }),
            Err(Error::new("WAITING", "model waiting")),
        ]))
    }
}

#[derive(Clone, Default)]
struct PlannedModel(Arc<Mutex<Vec<Request>>>);

impl Chat for PlannedModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let node = request.messages.0.iter().rev().find_map(|m| {
            m.values
                .get("plan_node")
                .and_then(serde_json::Value::as_str)
        });
        let mut message = Message::new(Role::Assistant);
        match node {
            Some("a" | "b") => {
                if request
                    .messages
                    .0
                    .iter()
                    .any(|m| m.metadata.get("name") == Some(&json!("weather")))
                {
                    message.values.insert("answer".into(), json!(node.unwrap()));
                } else {
                    message = calls(&[if node == Some("a") {
                        "weather_001"
                    } else {
                        "weather_002"
                    }]);
                }
            }
            Some("c") => {
                message.values.insert("answer".into(), json!("merged"));
            }
            None if request
                .messages
                .0
                .last()
                .unwrap()
                .values
                .contains_key("plan_complete") =>
            {
                message.values.insert("answer".into(), json!("final"));
            }
            None if request.tools.is_empty() => {
                message.values = json!({"version":1,"nodes":{
                    "a":{"name":"research","objective":"first task"},
                    "b":{"name":"research","objective":"second task"},
                    "c":{"name":"combine","objective":"combine both tasks"}
                },"edges":[{"from":"a","to":"c"},{"from":"b","to":"c"}]})
                .as_object()
                .unwrap()
                .clone();
            }
            None => {
                message.content.push(Part {
                    r#type: "tool_call".into(),
                    data: json!({"call_id":"plan_1","name":"plan","arguments":{}})
                        .as_object()
                        .unwrap()
                        .clone(),
                });
            }
            Some(other) => panic!("unexpected planned task {other}"),
        }
        self.0.lock().unwrap().push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(halo::agent::model::token::Usage {
                input: Some(2),
                output: Some(1),
                total: Some(3),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}

fn next_plan_id() -> i64 {
    static NEXT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(70000);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn root_child<'a>(scene: &'a serde_json::Value, call_id: &str) -> &'a serde_json::Value {
    let id = scene["steps"]
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .find(|step| step["kind"] == "agent" && step["call_id"] == call_id)
        .unwrap()["to"][0]
        .as_str()
        .unwrap();
    &scene["execution"]["attempts"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()["nodes"][id]
}

fn root_child_mut<'a>(
    scene: &'a mut serde_json::Value,
    call_id: &str,
) -> &'a mut serde_json::Value {
    let id = scene["steps"]
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .find(|step| step["kind"] == "agent" && step["call_id"] == call_id)
        .unwrap()["to"][0]
        .as_str()
        .unwrap()
        .to_owned();
    &mut scene["execution"]["attempts"]
        .as_array_mut()
        .unwrap()
        .last_mut()
        .unwrap()["nodes"][id]
}

fn root_child_scene<'a>(scene: &'a serde_json::Value, call_id: &str) -> &'a serde_json::Value {
    &root_child(scene, call_id)["execution"]["attempts"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()["agent"]
}

#[tokio::test]
async fn graphless_roots_keep_canonical_child_identity_and_do_not_duplicate_parent_context() {
    for react in [false, true] {
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let counts = Arc::new(Mutex::new(Counts::default()));
        let child = agent::chat()
            .model(ContextualChildModel)
            .tools(Weather::new(counts.clone()))
            .build()
            .unwrap();
        let child = agent::SubAgent::new("reviewer", child).unwrap();
        let model = Metered(Model::new(spawn_message()), 3);
        let events = if react {
            agent::react()
                .model(model)
                .subagents([child])
                .with_id_fn(next_plan_id)
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("original parent question"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        } else {
            agent::chat()
                .model(model)
                .subagents([child])
                .with_id_fn(next_plan_id)
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("original parent question"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        };
        assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
        let waiting = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert!(waiting.get("pending_subagents").is_none());
        assert!(waiting.get("graph").is_none_or(serde_json::Value::is_null));
        assert!(waiting.get("nodes").is_none());
        let root = &waiting["execution"];
        assert!(root["exec_id"].is_i64());
        assert_eq!(root["attempts"].as_array().unwrap().len(), 1);
        assert_eq!(root["attempts"][0]["status"], "waiting");
        assert!(root["attempts"][0]["error"].is_null());
        assert_eq!(root["messages"], json!([]));
        assert_eq!(root["attempts"][0]["messages"], json!([]));
        assert_eq!(root["attempts"][0]["steps"], json!([]));
        assert_eq!(waiting["usage"]["total"], 6);
        assert_eq!(root["usage"], waiting["usage"]);
        let child = root_child(&waiting, "delegate_1");
        let execution = &child["execution"];
        assert_ne!(root["exec_id"], execution["exec_id"]);
        assert_eq!(
            root_child_scene(&waiting, "delegate_1")["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["stage"],
            "eval"
        );

        let child = agent::chat()
            .model(ContextualChildModel)
            .tools(Weather::new(counts.clone()))
            .build()
            .unwrap();
        let child = agent::SubAgent::new("reviewer", child).unwrap();
        let model = Metered(Model::new(Message::new(Role::Assistant)), 5);
        let events = if react {
            agent::react()
                .model(model)
                .subagents([child])
                .with_id_fn(|| panic!("root/child waiting restore must not allocate IDs"))
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("approved"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        } else {
            agent::chat()
                .model(model)
                .subagents([child])
                .with_id_fn(|| panic!("root/child waiting restore must not allocate IDs"))
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("approved"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        };
        assert!(
            events.iter().all(Result::is_ok),
            "react={react}: {events:?}"
        );
        let done = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(done["execution"]["exec_id"], root["exec_id"]);
        assert_eq!(
            done["execution"]["attempts"][0]["created_time"],
            root["attempts"][0]["created_time"]
        );
        assert_eq!(done["execution"]["attempts"].as_array().unwrap().len(), 1);
        assert_eq!(done["execution"]["attempts"][0]["status"], "succeeded");
        let retained = root_child(&done, "delegate_1");
        assert_eq!(retained["status"], "succeeded");
        assert_eq!(retained["execution"]["exec_id"], execution["exec_id"]);
        assert_eq!(
            retained["execution"]["attempts"][0]["created_time"],
            execution["attempts"][0]["created_time"]
        );
        assert_eq!(root_child_scene(&done, "delegate_1")["status"], "succeeded");
        assert_eq!(done["usage"]["total"], 14);
        assert_eq!(done["execution"]["usage"], done["usage"]);
        assert_eq!(
            done["query"]["payload"][0]["data"]["value"],
            "original parent question"
        );
        let counts = counts.lock().unwrap();
        assert_eq!(
            (counts.before, counts.call, counts.after, counts.eval),
            (1, 1, 1, 2)
        );
    }
}

#[tokio::test]
async fn ordinary_agent_checkpoint_needs_no_root_execution_or_id_function_call() {
    for react in [false, true] {
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let model = Model::new(Message::new(Role::Assistant));
        let events = if react {
            agent::react()
                .model(model)
                .with_id_fn(|| panic!("ordinary run needs no root identity"))
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("hello"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        } else {
            agent::chat()
                .model(model)
                .with_id_fn(|| panic!("ordinary run needs no root identity"))
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("hello"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        };
        assert!(events.iter().all(Result::is_ok));
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(saved["status"], "succeeded");
        assert!(
            saved
                .get("execution")
                .is_none_or(serde_json::Value::is_null)
        );
    }
}

#[tokio::test]
async fn cached_root_child_and_waiting_tool_restore_in_original_call_order() {
    for react in [false, true] {
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let counts = Arc::new(Mutex::new(Counts::default()));
        let mut selection = calls(&["weather_001"]);
        selection.content.extend(spawn_message().content);
        let child = agent::chat()
            .model(Metered(Model::new(Message::new(Role::Assistant)), 4))
            .build()
            .unwrap();
        let child = agent::SubAgent::new("reviewer", child).unwrap();
        let capabilities = Weather::new(counts.clone());
        let model = Metered(Model::new(selection), 3);
        let events = if react {
            agent::react()
                .model(model)
                .tools(capabilities)
                .subagents([child])
                .with_id_fn(next_plan_id)
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("mixed task"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        } else {
            agent::chat()
                .model(model)
                .tools(capabilities)
                .subagents([child])
                .with_id_fn(next_plan_id)
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("mixed task"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        };
        assert!(matches!(events.last(),Some(Err(error)) if error.code=="WAITING"));
        let waiting = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(root_child(&waiting, "delegate_1")["status"], "succeeded");
        let child_model = Model::new(Message::new(Role::Assistant));
        let child_requests = child_model.requests.clone();
        let child = agent::SubAgent::new(
            "reviewer",
            agent::chat().model(child_model).build().unwrap(),
        )
        .unwrap();
        let model = Model::new(Message::new(Role::Assistant));
        let requests = model.requests.clone();
        let capabilities = Weather::new(counts.clone());
        let events = if react {
            agent::react()
                .model(model)
                .tools(capabilities)
                .subagents([child])
                .with_id_fn(|| panic!("cached child must not allocate IDs"))
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("approved"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        } else {
            agent::chat()
                .model(model)
                .tools(capabilities)
                .subagents([child])
                .with_id_fn(|| panic!("cached child must not allocate IDs"))
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("approved"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        };
        assert!(
            events.iter().all(Result::is_ok),
            "react={react}: {events:?}"
        );
        assert!(child_requests.lock().unwrap().is_empty());
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let ids = requests[0]
            .messages
            .0
            .iter()
            .filter_map(|message| {
                message
                    .metadata
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
            })
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["weather_001", "delegate_1"]);
        assert_eq!(counts.lock().unwrap().call, 1);
        assert_eq!(counts.lock().unwrap().eval, 2);
    }
}

#[tokio::test]
async fn planned_pending_eval_cancellation_keeps_attempt_result_and_never_admits_queued_nodes() {
    use halo::runtime::workflow::{Runtime, Status};
    let counts = Arc::new(Mutex::new(Counts::default()));
    let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let compiled = wrap_workflow(
        embedded_workflow(
            agent::react()
                .model(PlannedModel::default())
                .tools(PendingEval(Weather::new(counts.clone()), entered.clone()))
                .with_id_fn(next_plan_id)
                .with_max_concurrency(1)
                .build()
                .unwrap(),
        ),
        2,
    );
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let mut runtime = Runtime::new(1001, compiled)
        .unwrap()
        .with_checkpoint(checkpoint.clone());
    let token = Cancellation::new();
    let cancel_token = token.clone();
    let cancel = tokio::spawn(async move {
        while !entered.load(std::sync::atomic::Ordering::Relaxed) {
            tokio::task::yield_now().await;
        }
        cancel_token.cancel();
    });
    let next = Arc::new(std::sync::atomic::AtomicI64::new(9001));
    let events = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        runtime
            .run(
                halo::agent::Messages::default(),
                Arc::new(move || next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)),
                token,
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    })
    .await
    .expect("nested planned eval must not block cancellation");
    cancel.await.unwrap();
    assert!(
        matches!(events.last(),Some(Err(error)) if error.code=="CANCELLED"),
        "{events:?}"
    );
    assert_eq!(runtime.state().status, Status::Cancelled);
    let execution = agent_execution(runtime.state());
    assert_eq!(execution.attempts.len(), 1);
    let scene = execution.attempts[0].agent.as_ref().unwrap();
    assert_eq!(scene["status"], "stopped");
    assert_eq!(scene["nodes"]["a"]["status"], "cancelled");
    let attempt = &scene["nodes"]["a"]["execution"]["attempts"][0];
    assert_eq!(attempt["status"], "cancelled");
    assert_eq!(
        attempt["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["status"],
        "cancelled"
    );
    assert_eq!(
        attempt["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["message"]["values"]["temperature"],
        23
    );
    assert!(scene["nodes"]["b"]["execution"].is_null());
    assert!(scene["nodes"]["c"]["execution"].is_null());
    assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(9));
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        runtime.snapshot().unwrap()
    );
    let counts = counts.lock().unwrap();
    assert_eq!((counts.before, counts.call, counts.after), (1, 1, 1));
}

async fn cancel_after_child<A: Agent>(agent: A, token: Cancellation) {
    let mut events = Box::pin(
        agent
            .run(input("parent task"), true, token.clone())
            .await
            .unwrap(),
    );
    assert!(events.next().await.unwrap().is_ok());
    assert!(
        matches!(events.next().await.unwrap().unwrap(),Event::Complete {usage:Some(usage),..} if usage.total==Some(7))
    );
    token.cancel();
    let rest = events.collect::<Vec<_>>().await;
    assert!(
        matches!(rest.last(),Some(Ok(Event::Complete {finish_reason:Some(reason),..})) if reason=="cancelled")
    );
}

#[tokio::test]
async fn terminal_child_scene_delivery_is_not_billed_as_another_model_call() {
    use halo::agent::SubAgent;
    for react in [false, true] {
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let counts = Arc::new(Mutex::new(Counts::default()));
        let child = agent::chat()
            .model(Metered(Model::new(calls(&["weather_001"])), 3))
            .tools(Weather::new(counts.clone()))
            .build()
            .unwrap();
        let child = SubAgent::new("reviewer", child).unwrap();
        let model = Metered(Model::new(spawn_message()), 3);
        if react {
            let agent = agent::react()
                .model(model)
                .subagents(vec![child])
                .with_id_fn(next_plan_id)
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap();
            let events = agent
                .run(input("parent task"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert!(matches!(events.last(),Some(Err(error)) if error.code=="WAITING"));
        } else {
            let agent = agent::chat()
                .model(model)
                .subagents(vec![child])
                .with_id_fn(next_plan_id)
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap();
            let events = agent
                .run(input("parent task"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert!(matches!(events.last(),Some(Err(error)) if error.code=="WAITING"));
        }
        let mut parent = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        let child_scene = root_child_scene(&parent, "delegate_1").clone();
        let mut child_input: Input = serde_json::from_value(child_scene["query"].clone()).unwrap();
        child_input.payload = input("approved").payload;
        let child_checkpoint = Arc::new(core::checkpoint::InMemory::new());
        child_checkpoint
            .save(1001, child_scene, &Cancellation::new())
            .await
            .unwrap();
        let child = agent::chat()
            .model(Metered(Model::new(Message::new(Role::Assistant)), 4))
            .tools(Weather::new(counts.clone()))
            .checkpoint(child_checkpoint.clone())
            .build()
            .unwrap();
        assert!(
            child
                .run(child_input, true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
                .iter()
                .all(Result::is_ok)
        );
        let child_scene = child_checkpoint
            .load(1001, &Cancellation::new())
            .await
            .unwrap();
        assert_eq!(child_scene["usage"]["total"], 7);
        // Model a consistent commit: child is complete and its consumption
        // already settled, while delivery/wakeup of the parent is pending.
        let mut message: Message =
            serde_json::from_value(child_scene["final_message"].clone()).unwrap();
        message.role = Role::Function;
        message
            .metadata
            .insert("call_id".into(), json!("delegate_1"));
        message.metadata.insert("name".into(), json!("agent_spawn"));
        let child = root_child_mut(&mut parent, "delegate_1");
        child["status"] = json!("succeeded");
        child["usage"] = child_scene["usage"].clone();
        child["messages"] = json!([message]);
        child["execution"]["usage"] = child_scene["usage"].clone();
        child["execution"]["messages"] = json!([message]);
        child["execution"]["attempts"][0]["status"] = json!("succeeded");
        child["execution"]["attempts"][0]["usage"] = child_scene["usage"].clone();
        child["execution"]["attempts"][0]["agent"] = child_scene;
        parent["usage"] = json!({"input":10,"output":0,"total":10,"input_details":null,"output_details":null,"details":{}});
        checkpoint
            .save(1001, parent, &Cancellation::new())
            .await
            .unwrap();
        let model = Model::new(Message::new(Role::Assistant));
        let child_requests = model.requests.clone();
        let child = SubAgent::new(
            "reviewer",
            agent::chat()
                .model(model)
                .tools(Weather::new(counts.clone()))
                .build()
                .unwrap(),
        )
        .unwrap();
        let model = Metered(Model::new(Message::new(Role::Assistant)), 5);
        let events = if react {
            agent::react()
                .model(model)
                .subagents(vec![child])
                .with_id_fn(next_plan_id)
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("approved"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        } else {
            agent::chat()
                .model(model)
                .subagents(vec![child])
                .with_id_fn(next_plan_id)
                .checkpoint(checkpoint.clone())
                .build()
                .unwrap()
                .run(input("approved"), true, Cancellation::new())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        };
        assert!(events.iter().all(Result::is_ok));
        assert!(child_requests.lock().unwrap().is_empty());
        assert_eq!(
            checkpoint.load(1001, &Cancellation::new()).await.unwrap()["usage"]["total"],
            15,
            "cached child Complete must not add its old 4 again"
        );
        let counts = counts.lock().unwrap();
        assert_eq!(
            (counts.before, counts.call, counts.after, counts.eval),
            (1, 1, 1, 2)
        );
    }
}

#[tokio::test]
async fn cancelling_parent_after_successful_child_keeps_child_usage_separate_from_model_refinement()
{
    use halo::agent::SubAgent;
    for react in [false, true] {
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let child = agent::chat()
            .model(Metered(Model::new(Message::new(Role::Assistant)), 7))
            .build()
            .unwrap();
        let child = SubAgent::new("reviewer", child).unwrap();
        let model = Metered(Model::new(spawn_message()), 3);
        let token = Cancellation::new();
        if react {
            cancel_after_child(
                agent::react()
                    .model(model)
                    .subagents(vec![child])
                    .with_id_fn(next_plan_id)
                    .checkpoint(checkpoint.clone())
                    .build()
                    .unwrap(),
                token,
            )
            .await;
        } else {
            cancel_after_child(
                agent::chat()
                    .model(model)
                    .subagents(vec![child])
                    .with_id_fn(next_plan_id)
                    .checkpoint(checkpoint.clone())
                    .build()
                    .unwrap(),
                token,
            )
            .await;
        }
        let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(
            scene["usage"]["total"], 10,
            "react={react}: child 7 must not be refined as the parent's model 3"
        );
    }
}

fn spawn_message() -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {r#type:"tool_call".into(),data:json!({
        "call_id":"delegate_1","name":"agent_spawn","arguments":{"name":"reviewer","input":{
            "metadata":{},"payload":[{"type":"text","data":{"value":"child weather task"},"mime_type":null}]
        }}
    }).as_object().unwrap().clone()});
    message
}

struct SequenceModel {
    first: Message,
    requests: Arc<Mutex<Vec<Request>>>,
}
impl Chat for SequenceModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut requests = self.requests.lock().unwrap();
        let message = if requests.is_empty() {
            self.first.clone()
        } else {
            Message::new(Role::Assistant)
        };
        requests.push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(halo::agent::model::token::Usage {
                input: Some(3),
                output: Some(0),
                total: Some(3),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}

#[tokio::test]
async fn subagent_cannot_override_shared_run_session_or_user_ownership() {
    use halo::{agent::SubAgent, runtime::workflow::Runtime};
    for react in [false, true] {
        for key in ["chat_id", "session_id", "user_id", "tenant_id"] {
            let child = Model::new(Message::new(Role::Assistant));
            let requests = child.requests.clone();
            let child =
                SubAgent::new("reviewer", agent::chat().model(child).build().unwrap()).unwrap();
            let mut selection = spawn_message();
            selection.content[0].data["arguments"]["input"]["metadata"][key] = json!(999);
            let model = SequenceModel {
                first: selection,
                requests: Default::default(),
            };
            let compiled = if react {
                embedded_workflow(
                    agent::react()
                        .model(model)
                        .subagents(vec![child])
                        .with_id_fn(next_plan_id)
                        .build()
                        .unwrap(),
                )
            } else {
                embedded_workflow(
                    agent::chat()
                        .model(model)
                        .subagents(vec![child])
                        .with_id_fn(next_plan_id)
                        .build()
                        .unwrap(),
                )
            };
            let mut user = Message::new(Role::User);
            user.metadata = input("parent task").metadata;
            user.metadata.insert("tenant_id".into(), json!(1));
            let mut runtime = Runtime::new(1001, compiled).unwrap();
            let events = runtime
                .run(
                    halo::agent::Messages::new([user]),
                    Arc::new(|| 9001),
                    Cancellation::new(),
                )
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert!(events.iter().all(Result::is_ok));
            assert!(
                requests.lock().unwrap().is_empty(),
                "child invoked with conflicting {key}"
            );
            let execution = agent_execution(runtime.state());
            assert!(
                execution.attempts[0]
                    .messages
                    .0
                    .iter()
                    .any(|message| message.role == Role::Function
                        && message.values["error"]["code"] == "CONFLICT")
            );
            assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(6));
        }
    }
}

struct TaskDelegationModel;
impl Chat for TaskDelegationModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let node = request
            .messages
            .0
            .iter()
            .rev()
            .find_map(|message| message.values.get("plan_node"));
        let mut message = Message::new(Role::Assistant);
        if node.is_some() {
            if request
                .messages
                .0
                .iter()
                .any(|message| message.metadata.get("name") == Some(&json!("agent_spawn")))
            {
                message
                    .values
                    .insert("answer".into(), json!("task completed"));
            } else {
                message = spawn_message();
            }
        } else if request
            .messages
            .0
            .last()
            .unwrap()
            .values
            .contains_key("plan_complete")
        {
            message.values.insert("answer".into(), json!("final"));
        } else if request.tools.is_empty() {
            message.values = json!({"version":1,"nodes":{"a":{"name":"research","objective":"delegate child task"}},"edges":[]}).as_object().unwrap().clone();
        } else {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"plan_1","name":"plan","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(halo::agent::model::token::Usage {
                input: Some(3),
                output: Some(0),
                total: Some(3),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}

#[tokio::test]
async fn planned_subagent_wait_must_not_replay_child_tool_after_checkpoint_restore() {
    use halo::agent::SubAgent;
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let counts = Arc::new(Mutex::new(Counts::default()));
    let child = agent::chat()
        .model(ContextualChildModel)
        .tools(Weather::new(counts.clone()))
        .build()
        .unwrap();
    let first = agent::react()
        .model(TaskDelegationModel)
        .subagents(vec![SubAgent::new("reviewer", child).unwrap()])
        .with_id_fn(next_plan_id)
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = first
        .run(input("parent task"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.last(),Some(Err(error)) if error.code=="WAITING"));
    let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    let saved: halo::runtime::agent::State = serde_json::from_value(saved).unwrap();
    let attempt = &saved.nodes["a"].execution.as_ref().unwrap().attempts[0];
    assert_eq!(
        attempt.nodes.len(),
        1,
        "parent Attempt must retain its child"
    );
    let (child_id, child) = attempt.nodes.iter().next().unwrap();
    assert_ne!(child_id, "reviewer");
    let child_execution = child.execution.as_ref().unwrap();
    assert_ne!(child_id, &child_execution.exec_id.to_string());
    let child_scene = child_execution.attempts[0].agent.as_ref().unwrap();
    assert_eq!(
        child_scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["stage"],
        "eval"
    );
    assert_eq!(
        child_scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["message"]["values"]["temperature"],
        23
    );
    assert!(
        attempt
            .steps
            .iter()
            .any(|step| step.kind == "agent" && step.to == vec![child_id.clone()])
    );
    let child = agent::chat()
        .model(ContextualChildModel)
        .tools(Weather::new(counts.clone()))
        .build()
        .unwrap();
    let restored = agent::react()
        .model(TaskDelegationModel)
        .subagents(vec![SubAgent::new("reviewer", child).unwrap()])
        .with_id_fn(|| panic!("waiting restoration must not allocate new identities"))
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
    assert_eq!(
        counts.lock().unwrap().eval,
        2,
        "resume must complete the pending eval"
    );
    let final_scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    let final_scene: halo::runtime::agent::State = serde_json::from_value(final_scene).unwrap();
    let final_attempt = &final_scene.nodes["a"].execution.as_ref().unwrap().attempts[0];
    let retained = &final_attempt.nodes[child_id];
    assert_eq!(
        retained.status,
        halo::runtime::agent::node::Status::Succeeded
    );
    let retained_execution = retained.execution.as_ref().unwrap();
    assert_eq!(retained_execution.exec_id, child_execution.exec_id);
    assert_eq!(retained_execution.attempts.len(), 1);
    assert_eq!(
        retained_execution.attempts[0].created_time,
        child_execution.attempts[0].created_time
    );
    assert_eq!(
        retained_execution.attempts[0].agent.as_ref().unwrap()["status"],
        "succeeded"
    );
    assert_eq!(
        counts.lock().unwrap().call,
        1,
        "saved successful child Tool must not be executed twice"
    );
}

/// A fresh model instance decides from the supplied context, not an in-memory
/// call counter. With a lost child scene it requests the Tool again; with a
/// restored eval result it produces the final answer.
struct ContextualChildModel;
impl Chat for ContextualChildModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let completed = request.messages.0.iter().any(|message| {
            message.metadata.get("name") == Some(&json!("weather"))
                && message.values.get("temperature") == Some(&json!(23))
        });
        let mut message = if completed {
            Message::new(Role::Assistant)
        } else {
            calls(&["weather_001"])
        };
        if completed {
            message
                .values
                .insert("answer".into(), json!("approved weather"));
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(halo::agent::model::token::Usage {
                input: Some(3),
                output: Some(0),
                total: Some(3),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}

struct MultipleDelegationModel;
struct DelegateThenPlanModel;
impl Chat for DelegateThenPlanModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let mut message = Message::new(Role::Assistant);
        if request
            .messages
            .0
            .iter()
            .any(|message| message.values.contains_key("plan_node"))
        {
            message
                .values
                .insert("answer".into(), json!("planned task"));
        } else if request
            .messages
            .0
            .iter()
            .any(|message| message.values.contains_key("plan_complete"))
        {
            message.values.insert("answer".into(), json!("final"));
        } else if request.tools.is_empty() {
            message.values = json!({"version":1,"nodes":{"a":{"name":"finish","objective":"use prior research"}},"edges":[]}).as_object().unwrap().clone();
        } else if request
            .messages
            .0
            .iter()
            .any(|message| message.metadata.get("name") == Some(&json!("agent_spawn")))
        {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"plan_later","name":"plan","arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        } else {
            message = spawn_message();
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: Some(halo::agent::model::token::Usage {
                input: Some(3),
                output: Some(0),
                total: Some(3),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}

#[tokio::test]
async fn direct_child_survives_later_plan_creation_without_entering_static_nodes() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let child = agent::chat()
        .model(Metered(Model::new(Message::new(Role::Assistant)), 4))
        .build()
        .unwrap();
    let parent = agent::react()
        .model(DelegateThenPlanModel)
        .subagents([agent::SubAgent::new("reviewer", child).unwrap()])
        .with_id_fn(next_plan_id)
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = parent
        .run(input("delegate then plan"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    let child = root_child(&saved, "delegate_1");
    assert_eq!(child["status"], "succeeded");
    assert_eq!(
        root_child_scene(&saved, "delegate_1")["status"],
        "succeeded"
    );
    assert_eq!(
        saved["execution"]["attempts"][0]["nodes"]
            .as_object()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        saved["nodes"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        vec!["a"]
    );
    assert_eq!(
        saved["graph"]["plan"]["nodes"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        vec!["a"]
    );
    assert_ne!(
        saved["execution"]["exec_id"],
        saved["nodes"]["a"]["execution"]["exec_id"]
    );
    assert_eq!(saved["usage"]["total"], 19);
    assert_eq!(saved["execution"]["usage"], saved["usage"]);
}
impl Chat for MultipleDelegationModel {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let events = TaskDelegationModel
            .stream(request)
            .await?
            .collect::<Vec<_>>()
            .await;
        let events = events
            .into_iter()
            .map(|event| {
                event.map(|mut event| {
                    if let Event::Complete { message, .. } = &mut event
                        && let Some(part) = message
                            .content
                            .iter()
                            .find(|part| part.data.get("name") == Some(&json!("agent_spawn")))
                    {
                        let mut second = part.clone();
                        second.data["call_id"] = json!("delegate_2");
                        message.content.push(second);
                    }
                    event
                })
            })
            .collect::<Vec<_>>();
        Ok(stream::iter(events))
    }
}

#[tokio::test]
async fn planned_subagent_inherits_id_function_for_its_own_plan_and_restores_that_scene() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let counts = Arc::new(Mutex::new(Counts::default()));
    for phase in 0..2 {
        // No child-local with_id_fn: its plan uses the enclosing application's
        // function. Memory has no role in execution identity allocation.
        let child = agent::react()
            .model(PlannedModel::default())
            .tools(Weather::new(counts.clone()))
            .build()
            .unwrap();
        let parent = agent::react()
            .model(TaskDelegationModel)
            .subagents([agent::SubAgent::new("reviewer", child).unwrap()])
            .with_id_fn(next_plan_id)
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap();
        let events = parent
            .run(
                input(if phase == 0 {
                    "parent task"
                } else {
                    "approved"
                }),
                true,
                Cancellation::new(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        if phase == 0 {
            assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
        } else {
            assert!(events.iter().all(Result::is_ok), "{events:?}");
        }
        let state: halo::runtime::agent::State =
            serde_json::from_value(checkpoint.load(1001, &Cancellation::new()).await.unwrap())
                .unwrap();
        let attempt = &state.nodes["a"].execution.as_ref().unwrap().attempts[0];
        let child = attempt.nodes.values().next().unwrap();
        let scene = child.execution.as_ref().unwrap().attempts[0]
            .agent
            .as_ref()
            .unwrap();
        assert!(scene["graph"]["plan"]["nodes"]["a"].is_object());
        if phase == 1 {
            assert_eq!(scene["status"], "succeeded");
            assert_eq!(
                scene["nodes"]["a"]["execution"]["attempts"]
                    .as_array()
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(scene["nodes"]["b"]["status"], "succeeded");
        }
    }
    assert_eq!(
        counts.lock().unwrap().call,
        2,
        "two child plan Tools must each execute once"
    );
    assert_eq!(
        counts.lock().unwrap().eval,
        3,
        "only the waiting Tool eval resumes"
    );
}

#[tokio::test]
async fn repeated_registered_subagent_has_distinct_nodes_and_retains_successful_scenes() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let child = agent::chat()
        .model(Metered(
            Model {
                response: Message::new(Role::Assistant),
                requests: requests.clone(),
            },
            4,
        ))
        .build()
        .unwrap();
    let parent = agent::react()
        .model(MultipleDelegationModel)
        .subagents([agent::SubAgent::new("reviewer", child).unwrap()])
        .with_id_fn(next_plan_id)
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = parent
        .run(input("parent task"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    let state: halo::runtime::agent::State = serde_json::from_value(saved).unwrap();
    assert_eq!(state.graph.as_ref().unwrap().plan.nodes.len(), 1);
    assert_eq!(
        state.nodes.len(),
        1,
        "children must not enter static Plan nodes"
    );
    let parent_execution = state.nodes["a"].execution.as_ref().unwrap();
    let attempt = &parent_execution.attempts[0];
    assert_eq!(attempt.nodes.len(), 2);
    assert_eq!(requests.lock().unwrap().len(), 2);
    let mut identities = std::collections::BTreeSet::new();
    identities.insert(parent_execution.exec_id);
    for (id, node) in &attempt.nodes {
        assert!(identities.insert(id.parse::<i64>().unwrap()));
        let execution = node.execution.as_ref().unwrap();
        assert!(identities.insert(execution.exec_id));
        assert_eq!(node.status, halo::runtime::agent::node::Status::Succeeded);
        assert_eq!(
            execution.attempts[0].agent.as_ref().unwrap()["status"],
            "succeeded"
        );
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(4));
        assert!(
            attempt
                .steps
                .iter()
                .any(|step| step.kind == "agent" && step.to == vec![id.clone()])
        );
    }
    assert_eq!(parent_execution.usage.as_ref().unwrap().total, Some(14));
    assert_eq!(state.usage.as_ref().unwrap().total, Some(23));
    let old_children = attempt.nodes.clone();
    let mut graph = halo::runtime::agent::graph::Graph::new(state.graph.unwrap().plan).unwrap();
    graph
        .start_batch(
            1,
            &[halo::runtime::agent::graph::Start {
                node_id: "a".into(),
                exec_id: 900001,
            }],
        )
        .unwrap();
    graph
        .nodes
        .get_mut("a")
        .unwrap()
        .execution
        .as_mut()
        .unwrap()
        .attempts[0]
        .nodes = old_children.clone();
    graph
        .fail_with("a", Some(Error::new("FAILED", "later parent failure")))
        .unwrap();
    graph.retry("a").unwrap();
    let execution = graph.nodes["a"].execution.as_ref().unwrap();
    assert_eq!(execution.exec_id, 900001);
    assert_eq!(execution.attempts[0].nodes, old_children);
    assert!(
        execution.attempts[1].nodes.is_empty(),
        "new Attempt must not overwrite historical children"
    );
}

#[tokio::test]
async fn real_subagent_eval_wait_retains_child_consumption_and_does_not_reselect_delegation() {
    use halo::{
        agent::SubAgent,
        runtime::workflow::{Runtime, Status},
    };
    for react in [false, true] {
        let counts = Arc::new(Mutex::new(Counts::default()));
        let child = agent::chat()
            .model(Metered(Model::new(calls(&["weather_001"])), 3))
            .tools(Weather::new(counts.clone()))
            .build()
            .unwrap();
        let child = SubAgent::new("reviewer", child).unwrap();
        let model = Metered(Model::new(spawn_message()), 3);
        let compiled = if react {
            embedded_workflow(
                agent::react()
                    .model(model)
                    .subagents(vec![child])
                    .with_id_fn(next_plan_id)
                    .build()
                    .unwrap(),
            )
        } else {
            embedded_workflow(
                agent::chat()
                    .model(model)
                    .subagents(vec![child])
                    .with_id_fn(next_plan_id)
                    .build()
                    .unwrap(),
            )
        };
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let mut user = Message::new(Role::User);
        user.metadata = input("parent task").metadata;
        user.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"parent task"}).as_object().unwrap().clone(),
        });
        let initial = halo::agent::Messages::new([user]);
        let mut runtime = Runtime::new(1001, compiled)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        let events = runtime
            .run(initial.clone(), Arc::new(|| 9001), Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(
            matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
            "{events:?}"
        );
        let scene = agent_execution(runtime.state()).attempts[0]
            .agent
            .as_ref()
            .unwrap();
        let child = root_child_scene(scene, "delegate_1");
        assert_eq!(child["status"], "waiting");
        assert_eq!(
            child["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["stage"],
            "eval"
        );
        assert_eq!(child["query"]["metadata"]["depth"], 2);
        assert_eq!(child["query"]["metadata"]["chat_id"], 1001);
        assert_eq!(child["query"]["metadata"]["session_id"], 4);
        assert_eq!(child["usage"]["total"], 3);
        assert_eq!(scene["query"]["payload"][0]["data"]["value"], "parent task");
        assert_eq!(
            runtime.state().usage.as_ref().unwrap().total,
            Some(6),
            "react={react}: parent 3 plus waiting child 3"
        );
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        let child = agent::chat()
            .model(Metered(Model::new(Message::new(Role::Assistant)), 5))
            .tools(Weather::new(counts.clone()))
            .build()
            .unwrap();
        let child = SubAgent::new("reviewer", child).unwrap();
        let model = Metered(Model::new(Message::new(Role::Assistant)), 5);
        let requests = model.0.requests.clone();
        let compiled = if react {
            embedded_workflow(
                agent::react()
                    .model(model)
                    .subagents(vec![child])
                    .with_id_fn(next_plan_id)
                    .build()
                    .unwrap(),
            )
        } else {
            embedded_workflow(
                agent::chat()
                    .model(model)
                    .subagents(vec![child])
                    .with_id_fn(next_plan_id)
                    .build()
                    .unwrap(),
            )
        };
        let mut restored = Runtime::from_snapshot(1001, compiled, saved)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        restored
            .resume(
                "agent",
                halo::agent::Messages::new([Message::function(json!({"approved":true}))]),
            )
            .unwrap();
        let events = restored
            .run(
                initial,
                Arc::new(|| panic!("parent continuation must retain execution ID")),
                Cancellation::new(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert_eq!(restored.state().status, Status::Succeeded);
        assert_eq!(restored.state().usage.as_ref().unwrap().total, Some(16));
        assert_eq!(
            checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
            restored.snapshot().unwrap()
        );
        assert_eq!(
            requests.lock().unwrap().len(),
            1,
            "parent must not select delegation again"
        );
        let counts = counts.lock().unwrap();
        assert_eq!(
            (counts.before, counts.call, counts.after, counts.eval),
            (1, 1, 1, 2),
            "real child eval continuation, react={react}"
        );
    }
}

#[tokio::test]
async fn planned_eval_in_nested_workflow_keeps_successful_sibling_and_current_attempt() {
    use halo::runtime::workflow::{Runtime, Status};
    let counts = Arc::new(Mutex::new(Counts::default()));
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let model = PlannedModel::default();
    let compiled = wrap_workflow(
        embedded_workflow(
            agent::react()
                .model(model)
                .tools(Weather::new(counts.clone()))
                .with_id_fn(next_plan_id)
                .with_max_concurrency(2)
                .build()
                .unwrap(),
        ),
        2,
    );
    let mut user = Message::new(Role::User);
    user.metadata = input("weather?").metadata;
    user.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"weather?"}).as_object().unwrap().clone(),
    });
    let initial = halo::agent::Messages::new([user]);
    let next = Arc::new(std::sync::atomic::AtomicI64::new(9001));
    let mut runtime = Runtime::new(1001, compiled)
        .unwrap()
        .with_checkpoint(checkpoint.clone());
    let events = runtime
        .run(
            initial.clone(),
            Arc::new(move || next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)),
            Cancellation::new(),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
        "{events:?}"
    );
    let original = agent_execution(runtime.state());
    let outer_exec = original.exec_id;
    let scene = original.attempts[0].agent.as_ref().unwrap();
    assert_eq!(scene["nodes"]["a"]["status"], "waiting");
    assert_eq!(scene["nodes"]["b"]["status"], "succeeded");
    assert_eq!(
        scene["nodes"]["a"]["execution"]["attempts"][0]["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["stage"],
        "eval"
    );
    let sibling = scene["nodes"]["b"].clone();
    let planned_exec = scene["nodes"]["a"]["execution"]["exec_id"].clone();
    let created_time = scene["nodes"]["a"]["execution"]["attempts"][0]["created_time"].clone();
    assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(15));
    let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    let model = PlannedModel::default();
    let requests = model.0.clone();
    let compiled = wrap_workflow(
        embedded_workflow(
            agent::react()
                .model(model)
                .tools(Weather::new(counts.clone()))
                .with_id_fn(next_plan_id)
                .with_max_concurrency(2)
                .build()
                .unwrap(),
        ),
        2,
    );
    let mut restored = Runtime::from_snapshot(1001, compiled, saved)
        .unwrap()
        .with_checkpoint(checkpoint.clone());
    restored
        .resume(
            "child",
            halo::agent::Messages::new([Message::function(json!({"approved":true}))]),
        )
        .unwrap();
    let events = restored
        .run(
            initial,
            Arc::new(|| panic!("parent continuation must retain execution IDs")),
            Cancellation::new(),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    assert_eq!(restored.state().status, Status::Succeeded);
    let execution = agent_execution(restored.state());
    assert_eq!(execution.exec_id, outer_exec);
    assert_eq!(execution.attempts.len(), 1);
    let scene = execution.attempts[0].agent.as_ref().unwrap();
    assert_eq!(scene["nodes"]["b"], sibling);
    assert_eq!(scene["nodes"]["a"]["execution"]["exec_id"], planned_exec);
    assert_eq!(
        scene["nodes"]["a"]["execution"]["attempts"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        scene["nodes"]["a"]["execution"]["attempts"][0]["created_time"],
        created_time
    );
    assert_eq!(
        scene["nodes"]["a"]["execution"]["attempts"][0]["usage"]["total"],
        6
    );
    assert_eq!(scene["nodes"]["c"]["status"], "succeeded");
    assert_eq!(restored.state().usage.as_ref().unwrap().total, Some(24));
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        restored.snapshot().unwrap()
    );
    let requests = requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        3,
        "only a, c and the final model are invoked"
    );
    let counts = counts.lock().unwrap();
    assert_eq!(
        (counts.before, counts.call, counts.after, counts.eval),
        (2, 2, 2, 3)
    );
}

#[tokio::test]
async fn nested_builtin_agents_resume_model_wait_from_the_parent_checkpoint() {
    use halo::runtime::workflow::{Runtime, Status};
    for react in [false, true] {
        let compiled = if react {
            embedded_workflow(agent::react().model(WaitingModel).build().unwrap())
        } else {
            embedded_workflow(agent::chat().model(WaitingModel).build().unwrap())
        };
        let compiled = wrap_workflow(compiled, 2);
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let mut user = Message::new(Role::User);
        user.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"original question"})
                .as_object()
                .unwrap()
                .clone(),
        });
        let initial = halo::agent::Messages::new([user]);
        let next = Arc::new(std::sync::atomic::AtomicI64::new(9001));
        let mut runtime = Runtime::new(1001, compiled)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        let events = runtime
            .run(
                initial.clone(),
                Arc::new(move || next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)),
                Cancellation::new(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(events.first(), Some(Ok(Event::Delta { .. }))));
        assert!(matches!(events.last(), Some(Err(error)) if error.code=="WAITING"));
        assert_eq!(runtime.state().status, Status::Paused);
        assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(3));
        let original = agent_execution(runtime.state());
        let exec_id = original.exec_id;
        assert_eq!(exec_id, 9003);
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        let model = Metered(Model::new(Message::new(Role::Assistant)), 5);
        let requests = model.0.requests.clone();
        let compiled = if react {
            embedded_workflow(agent::react().model(model).build().unwrap())
        } else {
            embedded_workflow(agent::chat().model(model).build().unwrap())
        };
        let compiled = wrap_workflow(compiled, 2);
        let mut restored = Runtime::from_snapshot(1001, compiled.clone(), saved)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        restored
            .resume(
                "child",
                halo::agent::Messages::new([Message::function(json!({"approved":true}))]),
            )
            .unwrap();
        let events = restored
            .run(
                initial.clone(),
                Arc::new(|| panic!("nested continuation must retain all execution IDs")),
                Cancellation::new(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(
            events.iter().all(Result::is_ok),
            "react={react}: {events:?}"
        );
        assert_eq!(restored.state().status, Status::Succeeded);
        assert_eq!(restored.state().usage.as_ref().unwrap().total, Some(8));
        let execution = agent_execution(restored.state());
        assert_eq!(execution.exec_id, exec_id);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(8));
        assert_eq!(
            execution.attempts[0].agent.as_ref().unwrap()["query"]["payload"][0]["data"]["value"],
            "original question"
        );
        assert_eq!(requests.lock().unwrap().len(), 1);
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(saved, restored.snapshot().unwrap());
        let terminal = Runtime::from_snapshot(1001, compiled, saved.clone()).unwrap();
        assert!(
            terminal
                .run(
                    initial,
                    Arc::new(|| panic!("terminal graph does not allocate IDs")),
                    Cancellation::new()
                )
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
                .iter()
                .all(Result::is_ok)
        );
        assert_eq!(terminal.snapshot().unwrap(), saved);
    }
}

#[tokio::test]
async fn embedded_reply_identity_conflict_keeps_previous_scene_and_consumption() {
    use halo::runtime::workflow::{Runtime, Status};
    for react in [false, true] {
        let counts = Arc::new(Mutex::new(Counts::default()));
        let model = Metered(Model::new(calls(&["weather_001"])), 3);
        let capabilities = Weather::new(counts.clone());
        let compiled = if react {
            embedded_workflow(
                agent::react()
                    .model(model)
                    .tools(capabilities)
                    .build()
                    .unwrap(),
            )
        } else {
            embedded_workflow(
                agent::chat()
                    .model(model)
                    .tools(capabilities)
                    .build()
                    .unwrap(),
            )
        };
        let mut user = Message::new(Role::User);
        user.metadata = input("weather?").metadata;
        user.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"weather?"}).as_object().unwrap().clone(),
        });
        let initial = halo::agent::Messages::new([user]);
        let runtime = Runtime::new(1001, compiled.clone()).unwrap();
        let events = runtime
            .run(initial.clone(), Arc::new(|| 9001), Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(events.last(), Some(Err(error)) if error.code=="WAITING"));
        let saved = runtime.snapshot().unwrap();
        let scene = saved["nodes"]["agent"]["executions"]["9001"]["attempts"][0]["agent"].clone();
        let mut runtime = Runtime::from_snapshot(1001, compiled, saved).unwrap();
        let mut reply = Message::function(json!({"approved":true}));
        reply.metadata.insert("user_id".into(), json!(999));
        runtime
            .resume("agent", halo::agent::Messages::new([reply]))
            .unwrap();
        let events = runtime
            .run(
                initial,
                Arc::new(|| panic!("invalid reply must not allocate another execution")),
                Cancellation::new(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(events.last(), Some(Err(error)) if error.code=="CONFLICT"));
        assert_eq!(runtime.state().status, Status::Failed);
        let execution = &runtime.state().nodes["agent"].executions[&9001];
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].agent.as_ref().unwrap(), &scene);
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(3));
        assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(3));
        let counts = counts.lock().unwrap();
        assert_eq!(
            (counts.before, counts.call, counts.after, counts.eval),
            (1, 1, 1, 1)
        );
    }
}

#[tokio::test]
async fn embedded_chat_and_react_resume_eval_in_the_original_workflow_attempt() {
    use halo::runtime::workflow::{Runtime, Status, node};
    for react in [false, true] {
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let child_checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let counts = Arc::new(Mutex::new(Counts::default()));
        let model = Metered(Model::new(calls(&["weather_002", "weather_001"])), 3);
        let capabilities = Weather::new(counts.clone());
        let compiled = if react {
            embedded_workflow(
                agent::react()
                    .model(model)
                    .tools(capabilities)
                    .checkpoint(child_checkpoint.clone())
                    .build()
                    .unwrap(),
            )
        } else {
            embedded_workflow(
                agent::chat()
                    .model(model)
                    .tools(capabilities)
                    .checkpoint(child_checkpoint.clone())
                    .build()
                    .unwrap(),
            )
        };
        let mut user = Message::new(Role::User);
        user.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"weather?"}).as_object().unwrap().clone(),
        });
        user.metadata = input("weather?").metadata;
        let initial = halo::agent::Messages::new([user]);
        let mut runtime = Runtime::new(1001, compiled)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        let events = runtime
            .run(initial.clone(), Arc::new(|| 9001), Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
        assert_eq!(runtime.state().status, Status::Paused);
        let execution = &runtime.state().nodes["agent"].executions[&9001];
        assert_eq!(execution.status, node::ExecutionStatus::Waiting);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(3));
        let created_time = execution.attempts[0].created_time;
        let scene = execution.attempts[0].agent.as_ref().unwrap();
        assert_eq!(
            scene["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool" && step["call_id"] == "weather_001")
                .expect("waiting weather_001 Tool Step")["stage"],
            "eval"
        );
        assert_eq!(
            scene["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool" && step["call_id"] == "weather_001")
                .expect("waiting weather_001 Tool Step")["message"]["values"]["temperature"],
            23
        );
        assert_eq!(scene["query"]["payload"][0]["data"]["value"], "weather?");
        assert_eq!(
            child_checkpoint
                .load(1001, &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(saved, runtime.snapshot().unwrap());
        drop(runtime);

        let mut answer = Message::new(Role::Assistant);
        answer.values.insert("answer".into(), json!(42));
        let model = Metered(Model::new(answer), 5);
        let requests = model.0.requests.clone();
        let capabilities = Weather::new(counts.clone());
        let compiled = if react {
            embedded_workflow(
                agent::react()
                    .model(model)
                    .tools(capabilities)
                    .checkpoint(child_checkpoint.clone())
                    .build()
                    .unwrap(),
            )
        } else {
            embedded_workflow(
                agent::chat()
                    .model(model)
                    .tools(capabilities)
                    .checkpoint(child_checkpoint.clone())
                    .build()
                    .unwrap(),
            )
        };
        let mut restored = Runtime::from_snapshot(1001, compiled.clone(), saved)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        restored
            .resume(
                "agent",
                halo::agent::Messages::new([Message::function(json!({"approved":true}))]),
            )
            .unwrap();
        let events = restored
            .run(
                initial.clone(),
                Arc::new(|| panic!("continuation must not allocate another execution")),
                Cancellation::new(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok), "{react}: {events:?}");
        assert_eq!(restored.state().status, Status::Succeeded);
        let state = restored.state();
        let execution = &state.nodes["agent"].executions[&9001];
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].created_time, created_time);
        assert_eq!(execution.attempts[0].idx, 0);
        assert_eq!(execution.status, node::ExecutionStatus::Succeeded);
        assert_eq!(execution.messages.0[0].values["answer"], 42);
        assert_eq!(
            execution.usage.as_ref().unwrap().total,
            Some(8),
            "react={react}"
        );
        assert_eq!(execution.attempts[0].usage.as_ref().unwrap().total, Some(8));
        assert_eq!(
            execution.attempts[0].messages.0[0].content[0].r#type,
            "tool_call"
        );
        assert_eq!(
            execution.attempts[0].steps.len(),
            execution.attempts[0].messages.0.len()
        );
        let step_ids = execution.attempts[0]
            .steps
            .iter()
            .map(|step| &step.id)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(step_ids.len(), execution.attempts[0].steps.len());
        assert_eq!(state.usage.as_ref().unwrap().total, Some(8));
        {
            let counts = counts.lock().unwrap();
            assert_eq!(
                (counts.before, counts.call, counts.after, counts.eval),
                (2, 2, 2, 3)
            );
            assert_eq!(counts.eval_calls["weather_002"], 1);
        }
        {
            let requests = requests.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert!(
                !serde_json::to_string(&requests[0].messages)
                    .unwrap()
                    .contains("approved")
            );
        }
        let scene = execution.attempts[0].agent.as_ref().unwrap();
        assert_eq!(scene["usage"]["total"], 8);
        assert_eq!(scene["query"]["payload"][0]["data"]["value"], "weather?");
        assert_eq!(
            child_checkpoint
                .load(1001, &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "NOT_FOUND"
        );
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(saved, restored.snapshot().unwrap());
        let terminal = Runtime::from_snapshot(1001, compiled, saved.clone()).unwrap();
        let events = terminal
            .run(
                initial,
                Arc::new(|| panic!("terminal restore must allocate no execution")),
                Cancellation::new(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(events.iter().all(Result::is_ok));
        assert_eq!(terminal.snapshot().unwrap(), saved);
    }
}

#[derive(Clone)]
struct Model {
    response: Message,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl Model {
    fn new(response: Message) -> Self {
        Self {
            response,
            requests: Default::default(),
        }
    }
}

impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.requests.lock().unwrap().push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: self.response.clone(),
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

struct Metered(Model, u64);
impl Chat for Metered {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.0.requests.lock().unwrap().push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: self.0.response.clone(),
            usage: Some(halo::agent::model::token::Usage {
                input: Some(self.1),
                output: Some(0),
                total: Some(self.1),
                ..Default::default()
            }),
            finish_reason: Some("stop".into()),
        })]))
    }
}

struct PendingFinal(Arc<std::sync::atomic::AtomicUsize>);
impl Chat for PendingFinal {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        if self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed) > 0 {
            std::future::pending().await
        } else {
            Ok(stream::iter(vec![Ok(Event::Complete {
                message: calls(&["weather_002"]),
                usage: Some(halo::agent::model::token::Usage {
                    input: Some(3),
                    output: Some(0),
                    total: Some(3),
                    ..Default::default()
                }),
                finish_reason: Some("stop".into()),
            })]))
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Counts {
    before: usize,
    call: usize,
    after: usize,
    eval: usize,
    eval_calls: BTreeMap<String, usize>,
}

struct Weather {
    definition: ToolDefinition,
    counts: Arc<Mutex<Counts>>,
}

struct PendingEval(Weather, Arc<std::sync::atomic::AtomicBool>);
impl Tool for PendingEval {
    fn definition(&self) -> &ToolDefinition {
        self.0.definition()
    }
    async fn before(&self, call: &mut Call, token: &Cancellation) -> Result<(), Error> {
        self.0.before(call, token).await
    }
    async fn call(&self, call: &Call, token: &Cancellation) -> Result<Message, Error> {
        self.0.call(call, token).await
    }
    async fn after(
        &self,
        call: &Call,
        message: &mut Message,
        token: &Cancellation,
    ) -> Result<(), Error> {
        self.0.after(call, message, token).await
    }
    async fn eval(
        &self,
        _: &Call,
        message: &Message,
        _: &Cancellation,
    ) -> Result<Evaluation, Error> {
        assert_eq!(message.values["temperature"], 23);
        self.1.store(true, std::sync::atomic::Ordering::Relaxed);
        std::future::pending().await
    }
}

impl Weather {
    fn new(counts: Arc<Mutex<Counts>>) -> Self {
        Self {
            definition: ToolDefinition {
                name: "weather".into(),
                description: "weather lookup".into(),
                parameters: json!({"type":"object", "properties":{"city":{"type":"string"}}, "required":["city"]})
                    .as_object().unwrap().clone(),
            },
            counts,
        }
    }
}

impl Tool for Weather {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn before(&self, call: &mut Call, _: &Cancellation) -> Result<(), Error> {
        self.counts.lock().unwrap().before += 1;
        call.arguments.insert("city".into(), json!("Beijing"));
        Ok(())
    }

    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.counts.lock().unwrap().call += 1;
        assert_eq!(call.arguments["city"], "Beijing");
        Ok(Message::function(json!({"temperature":22})))
    }

    async fn after(&self, _: &Call, result: &mut Message, _: &Cancellation) -> Result<(), Error> {
        self.counts.lock().unwrap().after += 1;
        result.values.insert("temperature".into(), json!(23));
        Ok(())
    }

    async fn eval(
        &self,
        call: &Call,
        result: &Message,
        _: &Cancellation,
    ) -> Result<Evaluation, Error> {
        let mut counts = self.counts.lock().unwrap();
        counts.eval += 1;
        assert_eq!(call.arguments["city"], "Beijing");
        assert_eq!(result.values["temperature"], 23);
        let attempt = counts.eval_calls.entry(call.call_id.clone()).or_default();
        *attempt += 1;
        if call.call_id == "weather_001" && *attempt == 1 {
            Err(Error::new("WAITING", "evaluation approval required"))
        } else {
            Ok(Evaluation::passed())
        }
    }
}

fn input(value: &str) -> Input {
    Input {
        metadata: json!({"chat_id":1001, "user_id":2, "agent_id":3, "session_id":4})
            .as_object()
            .unwrap()
            .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":value}),
            mime_type: None,
        }],
    }
}

#[tokio::test]
async fn react_waiting_eval_resumes_eval_without_repeating_before_or_call() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let counts = Arc::new(Mutex::new(Counts::default()));
    let mut response = Message::new(Role::Assistant);
    response.content.push(Part {
        r#type: "tool_call".into(),
        data: json!({"call_id":"weather_001", "name":"weather", "arguments":{"city":"北京"}})
            .as_object()
            .unwrap()
            .clone(),
    });
    let first = agent::react()
        .model(Model::new(response))
        .tools(Weather::new(counts.clone()))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = first
        .run(input("weather?"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["status"], "waiting");
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["stage"],
        "eval"
    );
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["status"],
        "waiting"
    );
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["call"]["arguments"]["city"],
        "Beijing"
    );
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["message"]["values"]["temperature"],
        23
    );
    assert!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")
            .get("eval")
            .is_none()
    );
    drop(first);

    let model = Model::new(Message::new(Role::Assistant));
    let restored = agent::react()
        .model(model.clone())
        .tools(Weather::new(counts.clone()))
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
    {
        let counts = counts.lock().unwrap();
        assert_eq!(counts.before, 1, "successful before must not be repeated");
        assert_eq!(
            counts.call, 1,
            "successful side effect must not be repeated"
        );
        assert_eq!(counts.after, 1);
        assert_eq!(
            counts.eval, 2,
            "resume must finish eval before asking the model again"
        );
    }
    assert_eq!(events.len(), 1, "previous Tool Complete is not replayed");
    {
        let requests = model.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(!requests[0].messages.0.iter().any(|message| {
            message
                .content
                .iter()
                .any(|part| part.data.get("value") == Some(&json!("approved")))
        }));
    }
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["stage"],
        "committed"
    );
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["eval"]["passed"],
        true
    );
}

fn calls(ids: &[&str]) -> Message {
    let mut response = Message::new(Role::Assistant);
    for id in ids {
        response.content.push(Part {
            r#type: "tool_call".into(),
            data: json!({"call_id":id,"name":"weather","arguments":{"city":"北京"}})
                .as_object()
                .unwrap()
                .clone(),
        });
    }
    response
}

#[tokio::test]
async fn completed_sibling_eval_is_reused_and_context_keeps_selection_order() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let counts = Arc::new(Mutex::new(Counts::default()));
    let first = agent::react()
        .model(Model::new(calls(&["weather_002", "weather_001"])))
        .tools(Weather::new(counts.clone()))
        .with_max_concurrency(1)
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let events = first
        .run(input("weather?"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["stage"],
        "complete"
    );
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool" && step["call_id"] == "weather_001")
            .expect("waiting weather_001 Tool Step")["stage"],
        "eval"
    );
    drop(first);
    let model = Model::new(Message::new(Role::Assistant));
    let restored = agent::react()
        .model(model.clone())
        .tools(Weather::new(counts.clone()))
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
    let counts = counts.lock().unwrap();
    assert_eq!(
        (counts.before, counts.call, counts.after, counts.eval),
        (2, 2, 2, 3)
    );
    assert_eq!(counts.eval_calls["weather_002"], 1);
    let requests = model.requests.lock().unwrap();
    let ids = requests[0]
        .messages
        .0
        .iter()
        .filter_map(|m| m.metadata.get("call_id"))
        .collect::<Vec<_>>();
    assert_eq!(ids, [&json!("weather_002"), &json!("weather_001")]);
}

#[tokio::test]
async fn chat_batch_restores_results_and_evals_without_parent_model_reselection() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let counts = Arc::new(Mutex::new(Counts::default()));
    let first = agent::chat()
        .model(Model::new(calls(&["weather_002", "weather_001"])))
        .tools(Weather::new(counts.clone()))
        .with_max_concurrency(2)
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let original = input("weather?");
    let events = first
        .run(original.clone(), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
    assert_eq!(
        events.len(),
        2,
        "failed batch has not delivered Tool Complete yet"
    );
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["stage"],
        "complete"
    );
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool" && step["call_id"] == "weather_001")
            .expect("waiting weather_001 Tool Step")["stage"],
        "eval"
    );
    drop(first);
    let model = Model::new(Message::new(Role::Assistant));
    let restored = agent::chat()
        .model(model.clone())
        .tools(Weather::new(counts.clone()))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let mut stream = restored
        .run(input("approved"), true, Cancellation::new())
        .await
        .unwrap();
    assert!(
        model.requests.lock().unwrap().is_empty(),
        "do not open parent model before batch recovery"
    );
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    assert_eq!(events.len(), 3);
    {
        let counts = counts.lock().unwrap();
        assert_eq!(
            (counts.before, counts.call, counts.after, counts.eval),
            (2, 2, 2, 3)
        );
        assert_eq!(counts.eval_calls["weather_002"], 1);
        let requests = model.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let ids = requests[0]
            .messages
            .0
            .iter()
            .filter_map(|m| m.metadata.get("call_id"))
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![json!("weather_002"), json!("weather_001")]);
        assert!(!requests[0].messages.0.iter().any(|m| {
            m.content
                .iter()
                .any(|p| p.data.get("value") == Some(&json!("approved")))
        }));
    }
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["query"], json!(original));
    assert!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s["stage"] == "committed")
    );
}

#[tokio::test]
async fn chat_invalid_eval_scene_is_not_overwritten_or_given_to_model() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let counts = Arc::new(Mutex::new(Counts::default()));
    let first = agent::chat()
        .model(Model::new(calls(&["weather_001"])))
        .tools(Weather::new(counts.clone()))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let _ = first
        .run(input("weather?"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let mut scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    scene["steps"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|step| step["kind"] == "tool")
        .expect("actual Tool Step")
        .as_object_mut()
        .unwrap()
        .remove("message");
    checkpoint
        .save(1001, scene.clone(), &Cancellation::new())
        .await
        .unwrap();
    let model = Model::new(Message::new(Role::Assistant));
    let restored = agent::chat()
        .model(model.clone())
        .tools(Weather::new(counts))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let error = restored
        .run(input("approved"), true, Cancellation::new())
        .await
        .unwrap_err();
    assert_eq!(error.code, "INVALID_STATE");
    assert!(model.requests.lock().unwrap().is_empty());
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        scene
    );
}

#[tokio::test]
async fn chat_running_batch_checkpoint_reuses_success_and_retains_usage() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let counts = Arc::new(Mutex::new(Counts::default()));
    let first = agent::chat()
        .model(Metered(Model::new(calls(&["weather_002"])), 3))
        .tools(Weather::new(counts.clone()))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let original = input("weather?");
    let mut first_stream = first
        .run(original.clone(), true, Cancellation::new())
        .await
        .unwrap();
    assert!(first_stream.next().await.unwrap().is_ok());
    assert!(first_stream.next().await.unwrap().is_ok());
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["status"], "running");
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["stage"],
        "complete"
    );
    assert_eq!(scene["usage"]["total"], 3);
    drop(first_stream);
    drop(first);
    let model = Model::new(Message::new(Role::Assistant));
    let restored = agent::chat()
        .model(Metered(model.clone(), 5))
        .tools(Weather::new(counts.clone()))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let stream = restored
        .run(input("continue"), true, Cancellation::new())
        .await
        .unwrap();
    assert!(model.requests.lock().unwrap().is_empty());
    let events = stream.collect::<Vec<_>>().await;
    assert!(events.iter().all(Result::is_ok));
    {
        let counts = counts.lock().unwrap();
        assert_eq!(
            (counts.before, counts.call, counts.after, counts.eval),
            (1, 1, 1, 1)
        );
        let requests = model.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0]
                .messages
                .0
                .iter()
                .filter(|m| m.metadata.get("call_id") == Some(&json!("weather_002")))
                .count(),
            1
        );
    }
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["usage"]["total"], 8);
    assert_eq!(scene["query"], json!(original));
}

#[tokio::test]
async fn chat_cancellation_drops_pending_followup_provider_and_keeps_previous_usage() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let counts = Arc::new(Mutex::new(Counts::default()));
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let first = agent::chat()
        .model(PendingFinal(calls.clone()))
        .tools(Weather::new(counts))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let token = Cancellation::new();
    let mut stream = first
        .run(input("weather?"), true, token.clone())
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().is_ok());
    assert!(stream.next().await.unwrap().is_ok());
    let cancel = tokio::spawn(async move {
        while calls.load(std::sync::atomic::Ordering::Relaxed) < 2 {
            tokio::task::yield_now().await;
        }
        token.cancel();
    });
    let event = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("pending Provider must not block cancellation closeout")
        .unwrap()
        .unwrap();
    cancel.await.unwrap();
    let Event::Complete {
        message,
        usage: Some(usage),
        finish_reason,
        ..
    } = event
    else {
        panic!("cancelled Complete expected")
    };
    assert!(message.content.is_empty());
    assert_eq!(finish_reason.as_deref(), Some("cancelled"));
    assert!(usage.input.unwrap() > 0);
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(scene["status"], "stopped");
    assert_eq!(scene["usage"]["total"], 3 + usage.total.unwrap());
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["stage"],
        "committed"
    );
}

#[tokio::test]
async fn cancelling_pending_chat_eval_keeps_call_result_and_marks_only_unfinished_step() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let counts = Arc::new(Mutex::new(Counts::default()));
    let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let agent = agent::chat()
        .model(Metered(Model::new(calls(&["weather_002"])), 3))
        .tools(PendingEval(Weather::new(counts.clone()), entered.clone()))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let token = Cancellation::new();
    let mut events = agent
        .run(input("weather?"), true, token.clone())
        .await
        .unwrap();
    assert!(events.next().await.unwrap().is_ok());
    let cancel = tokio::spawn(async move {
        while !entered.load(std::sync::atomic::Ordering::Relaxed) {
            tokio::task::yield_now().await;
        }
        token.cancel();
    });
    let event = tokio::time::timeout(std::time::Duration::from_secs(2), events.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    cancel.await.unwrap();
    assert!(
        matches!(event, Event::Complete { finish_reason:Some(reason), .. } if reason=="cancelled")
    );
    let scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["status"],
        "cancelled"
    );
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["stage"],
        "eval"
    );
    assert_eq!(
        scene["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("actual Tool Step")["message"]["values"]["temperature"],
        23
    );
    assert_eq!(scene["usage"]["total"], 3);
    let counts = counts.lock().unwrap();
    assert_eq!((counts.before, counts.call, counts.after), (1, 1, 1));
}

#[tokio::test]
async fn missing_eval_result_is_rejected_before_model_or_state_writes() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let counts = Arc::new(Mutex::new(Counts::default()));
    let first = agent::react()
        .model(Model::new(calls(&["weather_001"])))
        .tools(Weather::new(counts.clone()))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let _ = first
        .run(input("weather?"), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let mut scene = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    scene["steps"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|step| step["kind"] == "tool")
        .expect("actual Tool Step")
        .as_object_mut()
        .unwrap()
        .remove("message");
    checkpoint
        .save(1001, scene.clone(), &Cancellation::new())
        .await
        .unwrap();
    let model = Model::new(Message::new(Role::Assistant));
    let restored = agent::react()
        .model(model.clone())
        .tools(Weather::new(counts))
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let error = match restored
        .run(input("approved"), true, Cancellation::new())
        .await
    {
        Ok(_) => panic!("invalid eval scene was accepted"),
        Err(error) => error,
    };
    assert_eq!(error.code, "INVALID_STATE");
    assert!(model.requests.lock().unwrap().is_empty());
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        scene
    );
}

#[derive(Clone)]
struct Disk(PathBuf);
impl halo::agent::checkpoint::Checkpoint for Disk {
    async fn delete(&self, _: i64, _: &Cancellation) -> Result<(), Error> {
        std::fs::remove_file(&self.0).map_err(|e| Error::new("IO", e.to_string()))
    }
    async fn save(&self, _: i64, state: serde_json::Value, _: &Cancellation) -> Result<(), Error> {
        let next = self.0.with_extension("next");
        std::fs::write(&next, serde_json::to_vec(&state).unwrap())
            .map_err(|e| Error::new("IO", e.to_string()))?;
        std::fs::rename(next, &self.0).map_err(|e| Error::new("IO", e.to_string()))
    }
    async fn load(&self, _: i64, _: &Cancellation) -> Result<serde_json::Value, Error> {
        let data = std::fs::read(&self.0).map_err(|e| {
            Error::new(
                if e.kind() == std::io::ErrorKind::NotFound {
                    "NOT_FOUND"
                } else {
                    "IO"
                },
                e.to_string(),
            )
        })?;
        serde_json::from_slice(&data).map_err(|e| Error::new("INVALID_STATE", e.to_string()))
    }
}

#[tokio::test]
#[ignore = "subprocess fixture, invoked by independent_process_eval_recovery"]
async fn eval_worker() {
    let root = PathBuf::from(std::env::var_os("HEART_EVAL_SCENE").unwrap());
    let phase = std::env::var("HEART_EVAL_PHASE").unwrap();
    let chat = std::env::var("HEART_EVAL_AGENT").ok().as_deref() == Some("chat");
    let count_path = root.join("counts.json");
    let counts = Arc::new(Mutex::new(if phase == "wait" {
        Counts::default()
    } else {
        serde_json::from_slice(&std::fs::read(&count_path).unwrap()).unwrap()
    }));
    let checkpoint = Disk(root.join("scene.json"));
    let model = Model::new(if phase == "wait" {
        calls(&["weather_001"])
    } else {
        Message::new(Role::Assistant)
    });
    let request = input(if phase == "wait" {
        "weather?"
    } else {
        "approved"
    });
    let events = if chat {
        agent::chat()
            .model(model)
            .tools(Weather::new(counts.clone()))
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap()
            .run(request, true, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    } else {
        agent::react()
            .model(model)
            .tools(Weather::new(counts.clone()))
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap()
            .run(request, true, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    };
    if phase == "wait" {
        assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
    } else {
        assert!(events.iter().all(Result::is_ok));
        {
            let counts = counts.lock().unwrap();
            assert_eq!(
                (counts.before, counts.call, counts.after, counts.eval),
                (1, 1, 1, 2)
            );
        }
        assert_eq!(
            checkpoint.load(1001, &Cancellation::new()).await.unwrap()["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["stage"],
            "committed"
        );
    }
    std::fs::write(
        &count_path,
        serde_json::to_vec(&*counts.lock().unwrap()).unwrap(),
    )
    .unwrap();
}

#[test]
fn independent_process_eval_recovery() {
    recover_process("react");
    recover_process("chat");
}

#[test]
fn independent_process_embedded_eval_recovery() {
    recover_process("workflow_chat");
    recover_process("workflow_react");
    recover_process("workflow_nested_chat");
    recover_process("workflow_nested_react");
}

#[test]
fn independent_process_real_subagent_recovery() {
    recover_process("spawn_chat");
    recover_process("spawn_react");
}

#[tokio::test]
#[ignore = "subprocess fixture, invoked by independent_process_real_subagent_recovery"]
async fn subagent_worker() {
    use halo::{
        agent::SubAgent,
        runtime::workflow::{Runtime, Status},
    };
    let root = PathBuf::from(std::env::var_os("HEART_EVAL_SCENE").unwrap());
    let wait = std::env::var("HEART_EVAL_PHASE").unwrap() == "wait";
    let react = std::env::var("HEART_EVAL_AGENT").unwrap() == "spawn_react";
    let count_path = root.join("counts.json");
    let counts = Arc::new(Mutex::new(if wait {
        Counts::default()
    } else {
        serde_json::from_slice(&std::fs::read(&count_path).unwrap()).unwrap()
    }));
    let child_checkpoint = Disk(root.join("child.json"));
    let child = agent::chat()
        .model(Metered(
            Model::new(if wait {
                calls(&["weather_001"])
            } else {
                Message::new(Role::Assistant)
            }),
            if wait { 3 } else { 5 },
        ))
        .tools(Weather::new(counts.clone()))
        .checkpoint(child_checkpoint)
        .build()
        .unwrap();
    let child = SubAgent::new("reviewer", child).unwrap();
    let model = Metered(
        Model::new(if wait {
            spawn_message()
        } else {
            Message::new(Role::Assistant)
        }),
        if wait { 3 } else { 5 },
    );
    let requests = model.0.requests.clone();
    let compiled = if react {
        embedded_workflow(
            agent::react()
                .model(model)
                .subagents(vec![child])
                .with_id_fn(next_plan_id)
                .build()
                .unwrap(),
        )
    } else {
        embedded_workflow(
            agent::chat()
                .model(model)
                .subagents(vec![child])
                .with_id_fn(next_plan_id)
                .build()
                .unwrap(),
        )
    };
    let checkpoint = Disk(root.join("scene.json"));
    let mut user = Message::new(Role::User);
    user.metadata = input("parent task").metadata;
    user.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"parent task"}).as_object().unwrap().clone(),
    });
    let initial = halo::agent::Messages::new([user]);
    let mut runtime = if wait {
        Runtime::new(1001, compiled.clone()).unwrap()
    } else {
        let mut runtime = Runtime::from_snapshot(
            1001,
            compiled.clone(),
            checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        )
        .unwrap();
        runtime
            .resume(
                "agent",
                halo::agent::Messages::new([Message::function(json!({"approved":true}))]),
            )
            .unwrap();
        runtime
    }
    .with_checkpoint(checkpoint.clone());
    let events = runtime
        .run(
            initial.clone(),
            Arc::new(move || {
                assert!(wait, "restore must not allocate parent execution");
                9001
            }),
            Cancellation::new(),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    if wait {
        assert!(matches!(events.last(),Some(Err(error)) if error.code=="WAITING"));
        assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(6));
    } else {
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert_eq!(runtime.state().status, Status::Succeeded);
        assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(16));
        let execution = agent_execution(runtime.state());
        assert_eq!(execution.exec_id, 9001);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(requests.lock().unwrap().len(), 1);
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(saved, runtime.snapshot().unwrap());
        let terminal = Runtime::from_snapshot(1001, compiled, saved.clone()).unwrap();
        assert!(
            terminal
                .run(
                    initial,
                    Arc::new(|| panic!("terminal restore allocates no execution")),
                    Cancellation::new()
                )
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
                .iter()
                .all(Result::is_ok)
        );
        assert_eq!(terminal.snapshot().unwrap(), saved);
        let counts = counts.lock().unwrap();
        assert_eq!(
            (counts.before, counts.call, counts.after, counts.eval),
            (1, 1, 1, 2)
        );
    }
    assert!(!root.join("child.json").exists());
    std::fs::write(
        &count_path,
        serde_json::to_vec(&*counts.lock().unwrap()).unwrap(),
    )
    .unwrap();
}

#[tokio::test]
#[ignore = "subprocess fixture, invoked by independent_process_embedded_eval_recovery"]
async fn embedded_eval_worker() {
    use halo::runtime::workflow::{Runtime, Status};
    let root = PathBuf::from(std::env::var_os("HEART_EVAL_SCENE").unwrap());
    let phase = std::env::var("HEART_EVAL_PHASE").unwrap();
    let kind = std::env::var("HEART_EVAL_AGENT").unwrap();
    let react = kind.ends_with("react");
    let depth = if kind.contains("nested") { 2 } else { 0 };
    let count_path = root.join("counts.json");
    let counts = Arc::new(Mutex::new(if phase == "wait" {
        Counts::default()
    } else {
        serde_json::from_slice(&std::fs::read(&count_path).unwrap()).unwrap()
    }));
    let checkpoint = Disk(root.join("scene.json"));
    let child_checkpoint = Disk(root.join("child.json"));
    let model = Metered(
        Model::new(if phase == "wait" {
            calls(&["weather_001"])
        } else {
            Message::new(Role::Assistant)
        }),
        if phase == "wait" { 3 } else { 5 },
    );
    let capabilities = Weather::new(counts.clone());
    let compiled = if react {
        embedded_workflow(
            agent::react()
                .model(model)
                .tools(capabilities)
                .checkpoint(child_checkpoint)
                .build()
                .unwrap(),
        )
    } else {
        embedded_workflow(
            agent::chat()
                .model(model)
                .tools(capabilities)
                .checkpoint(child_checkpoint)
                .build()
                .unwrap(),
        )
    };
    let mut user = Message::new(Role::User);
    user.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"weather?"}).as_object().unwrap().clone(),
    });
    let compiled = wrap_workflow(compiled, depth);
    // The enclosing Workflow supplies chat_id; user metadata is not required
    // to duplicate it just to restore the embedded Agent's original scene.
    let initial = halo::agent::Messages::new([user]);
    let mut runtime = if phase == "wait" {
        Runtime::new(1001, compiled.clone()).unwrap()
    } else {
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        let mut runtime = Runtime::from_snapshot(1001, compiled.clone(), saved).unwrap();
        runtime
            .resume(
                if depth == 0 { "agent" } else { "child" },
                halo::agent::Messages::new([Message::function(json!({"approved":true}))]),
            )
            .unwrap();
        runtime
    }
    .with_checkpoint(checkpoint.clone());
    let wait = phase == "wait";
    let next = Arc::new(std::sync::atomic::AtomicI64::new(9001));
    let events = runtime
        .run(
            initial.clone(),
            Arc::new(move || {
                assert!(wait, "restoration must not allocate an exec_id");
                next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            }),
            Cancellation::new(),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    if wait {
        assert!(matches!(events.last(), Some(Err(error)) if error.code=="WAITING"));
        assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(3));
        // Model a persisted nonempty creation time without changing the
        // production clock or adding an ID/timestamp generator contract.
        let mut saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        let mut state = &mut saved;
        for index in 0..depth {
            state = &mut state["nodes"]["child"]["executions"][(9001 + index).to_string()]["attempts"]
                [0]["graph"];
        }
        state["nodes"]["agent"]["executions"][(9001 + depth).to_string()]["attempts"][0]["created_time"] =
            json!(1234);
        checkpoint
            .save(1001, saved, &Cancellation::new())
            .await
            .unwrap();
    } else {
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert_eq!(runtime.state().status, Status::Succeeded);
        let state = runtime.state();
        let execution = agent_execution(state);
        assert_eq!(execution.exec_id, 9001 + depth as i64);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(execution.attempts[0].created_time, Some(1234));
        assert_eq!(execution.usage.as_ref().unwrap().total, Some(8));
        assert_eq!(state.usage.as_ref().unwrap().total, Some(8));
        let scene = execution.attempts[0].agent.as_ref().unwrap();
        assert_eq!(
            scene["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["stage"],
            "committed"
        );
        assert_eq!(scene["query"]["payload"][0]["data"]["value"], "weather?");
        assert_eq!(scene["query"]["metadata"]["chat_id"], 1001);
        let saved = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
        assert_eq!(saved, runtime.snapshot().unwrap());
        let terminal = Runtime::from_snapshot(1001, compiled, saved.clone()).unwrap();
        assert!(
            terminal
                .run(
                    initial,
                    Arc::new(|| panic!("terminal restore must allocate no execution")),
                    Cancellation::new()
                )
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
                .iter()
                .all(Result::is_ok)
        );
        assert_eq!(terminal.snapshot().unwrap(), saved);
        let counts = counts.lock().unwrap();
        assert_eq!(
            (counts.before, counts.call, counts.after, counts.eval),
            (1, 1, 1, 2)
        );
    }
    assert!(!root.join("child.json").exists());
    std::fs::write(
        &count_path,
        serde_json::to_vec(&*counts.lock().unwrap()).unwrap(),
    )
    .unwrap();
}

#[tokio::test]
async fn embedded_chat_and_react_cancel_pending_eval_preserving_the_agent_scene() {
    use halo::runtime::workflow::{Runtime, Status, node};
    for react in [false, true] {
        let counts = Arc::new(Mutex::new(Counts::default()));
        let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let model = Metered(Model::new(calls(&["weather_002"])), 3);
        let capabilities = PendingEval(Weather::new(counts.clone()), entered.clone());
        let compiled = if react {
            embedded_workflow(
                agent::react()
                    .model(model)
                    .tools(capabilities)
                    .build()
                    .unwrap(),
            )
        } else {
            embedded_workflow(
                agent::chat()
                    .model(model)
                    .tools(capabilities)
                    .build()
                    .unwrap(),
            )
        };
        let checkpoint = Arc::new(core::checkpoint::InMemory::new());
        let mut runtime = Runtime::new(1001, compiled)
            .unwrap()
            .with_checkpoint(checkpoint.clone());
        let mut user = Message::new(Role::User);
        user.content.push(Part {
            r#type: "text".into(),
            data: json!({"value":"weather?"}).as_object().unwrap().clone(),
        });
        let token = Cancellation::new();
        let cancel_token = token.clone();
        let cancel = tokio::spawn(async move {
            while !entered.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::task::yield_now().await;
            }
            cancel_token.cancel();
        });
        let events = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            runtime
                .run(halo::agent::Messages::new([user]), Arc::new(|| 9001), token)
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        })
        .await
        .expect("embedded eval cancellation must close without waiting for the Tool future");
        cancel.await.unwrap();
        assert!(matches!(events.last(), Some(Err(error)) if error.code=="CANCELLED"));
        assert_eq!(events.iter().filter(|event| matches!(event, Ok(Event::Complete {finish_reason:Some(reason),..}) if reason=="cancelled")).count(), 1);
        assert_eq!(runtime.state().status, Status::Cancelled);
        let execution = &runtime.state().nodes["agent"].executions[&9001];
        assert_eq!(execution.status, node::ExecutionStatus::Cancelled);
        let scene = execution.attempts[0].agent.as_ref().unwrap();
        assert_eq!(
            scene["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["status"],
            "cancelled"
        );
        assert_eq!(
            scene["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["stage"],
            "eval"
        );
        assert_eq!(
            scene["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["message"]["values"]["temperature"],
            23
        );
        assert_eq!(scene["usage"]["total"], 3);
        assert_eq!(runtime.state().usage.as_ref().unwrap().total, Some(3));
        assert_eq!(
            checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
            runtime.snapshot().unwrap()
        );
        let counts = counts.lock().unwrap();
        assert_eq!((counts.before, counts.call, counts.after), (1, 1, 1));
    }
}

#[test]
fn independent_process_planned_subagent_recovery() {
    recover_process("planned_child");
    recover_process("planned_child_nested");
}

#[tokio::test]
#[ignore = "subprocess fixture, invoked by independent_process_planned_subagent_recovery"]
async fn planned_child_worker() {
    use halo::runtime::workflow::{Runtime, Status};
    let root = PathBuf::from(std::env::var_os("HEART_EVAL_SCENE").unwrap());
    let wait = std::env::var("HEART_EVAL_PHASE").unwrap() == "wait";
    let nested = std::env::var("HEART_EVAL_AGENT").unwrap() == "planned_child_nested";
    let count_path = root.join("counts.json");
    let counts = Arc::new(Mutex::new(if wait {
        Counts::default()
    } else {
        serde_json::from_slice(&std::fs::read(&count_path).unwrap()).unwrap()
    }));
    let checkpoint = Disk(root.join("scene.json"));
    let child = agent::chat()
        .model(ContextualChildModel)
        .tools(Weather::new(counts.clone()))
        .checkpoint(Disk(root.join("child.json")))
        .build()
        .unwrap();
    let parent = agent::react()
        .model(TaskDelegationModel)
        .subagents([agent::SubAgent::new("reviewer", child).unwrap()])
        .with_id_fn(if wait {
            next_plan_id
        } else {
            || panic!("restore must retain all Agent execution identities")
        })
        .checkpoint(checkpoint.clone())
        .build()
        .unwrap();
    let previous = if wait {
        None
    } else {
        Some(checkpoint.load(1001, &Cancellation::new()).await.unwrap())
    };
    let (events, scene) = if nested {
        let compiled = wrap_workflow(embedded_workflow(parent), 2);
        let mut runtime = if let Some(previous) = previous.clone() {
            let mut runtime = Runtime::from_snapshot(1001, compiled, previous).unwrap();
            runtime
                .resume(
                    "child",
                    halo::agent::Messages::new([Message::function(json!({"approved":true}))]),
                )
                .unwrap();
            runtime
        } else {
            Runtime::new(1001, compiled).unwrap()
        }
        .with_checkpoint(checkpoint.clone());
        let mut user = Message::new(Role::User);
        user.metadata = input("parent task").metadata;
        let events = runtime
            .run(
                halo::agent::Messages::new([user]),
                Arc::new(move || {
                    assert!(wait, "restore must retain Workflow execution identities");
                    next_plan_id()
                }),
                Cancellation::new(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert_eq!(
            runtime.state().status,
            if wait {
                Status::Paused
            } else {
                Status::Succeeded
            }
        );
        let scene = agent_execution(runtime.state()).attempts[0]
            .agent
            .clone()
            .unwrap();
        (events, scene)
    } else {
        let events = parent
            .run(
                input(if wait { "parent task" } else { "approved" }),
                true,
                Cancellation::new(),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        (
            events,
            checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        )
    };
    let state: halo::runtime::agent::State = serde_json::from_value(scene).unwrap();
    let parent_execution = state.nodes["a"].execution.as_ref().unwrap();
    let attempt = &parent_execution.attempts[0];
    assert_eq!(attempt.nodes.len(), 1);
    let (child_id, child) = attempt.nodes.iter().next().unwrap();
    let execution = child.execution.as_ref().unwrap();
    let child_attempt = &execution.attempts[0];
    let identity = json!({"node_id":child_id,"exec_id":execution.exec_id,"created_time":child_attempt.created_time,"parent":{"exec_id":parent_execution.exec_id,"created_time":attempt.created_time}});
    if wait {
        assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
        assert_eq!(
            child_attempt.agent.as_ref().unwrap()["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["stage"],
            "eval"
        );
        std::fs::write(
            root.join("identities.json"),
            serde_json::to_vec(&identity).unwrap(),
        )
        .unwrap();
    } else {
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert_eq!(child.status, halo::runtime::agent::node::Status::Succeeded);
        assert_eq!(execution.attempts.len(), 1);
        assert_eq!(child_attempt.agent.as_ref().unwrap()["status"], "succeeded");
        let original: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("identities.json")).unwrap()).unwrap();
        assert_eq!(identity, original);
        let counts = counts.lock().unwrap();
        assert_eq!(
            (counts.before, counts.call, counts.after, counts.eval),
            (1, 1, 1, 2)
        );
    }
    assert!(
        !root.join("child.json").exists(),
        "child must not open an independent checkpoint"
    );
    std::fs::write(
        &count_path,
        serde_json::to_vec(&*counts.lock().unwrap()).unwrap(),
    )
    .unwrap();
}

fn recover_process(kind: &str) {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let root = loop {
        // Concurrent tests can observe the same clock value even when its
        // unit is nanoseconds. Reserve a unique directory atomically; leave
        // any pre-existing directory untouched, including failed-run data.
        let candidate = std::env::temp_dir().join(format!(
            "heart-eval-{}-{}-{}",
            std::process::id(),
            kind,
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        match std::fs::create_dir(&candidate) {
            Ok(()) => break candidate,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("cannot reserve recovery test directory: {error}"),
        }
    };
    for phase in ["wait", "restore"] {
        let result = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                if kind.starts_with("planned_child") {
                    "planned_child_worker"
                } else if kind.starts_with("spawn_") {
                    "subagent_worker"
                } else if kind.starts_with("workflow_") {
                    "embedded_eval_worker"
                } else {
                    "eval_worker"
                },
                "--nocapture",
            ])
            .env("HEART_EVAL_SCENE", &root)
            .env("HEART_EVAL_PHASE", phase)
            .env("HEART_EVAL_AGENT", kind)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{phase}: {} {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }
    std::fs::remove_dir_all(&root).unwrap();
}
