//! Each worker has fresh models, logs and storage. Only canonical Checkpoint
//! JSON crosses processes; provider connections and request logs never do.
use std::{
    collections::BTreeMap,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicI64, AtomicUsize, Ordering},
    },
};

use futures_core::Stream;
use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, SubAgent},
    checkpoint::Checkpoint,
    core,
    event::Event,
    message::Role,
    middleware::Middleware,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
};
use serde_json::{Value, json};

#[derive(Default)]
struct Store {
    backend: core::checkpoint::InMemory,
    loads: AtomicUsize,
    saves: AtomicUsize,
}
impl Checkpoint for Store {
    async fn save(&self, id: i64, state: Value, c: &Cancellation) -> Result<(), Error> {
        self.saves.fetch_add(1, Ordering::SeqCst);
        self.backend.save(id, state, c).await
    }
    async fn load(&self, id: i64, c: &Cancellation) -> Result<Value, Error> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.backend.load(id, c).await
    }
    async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
        self.backend.delete(id, c).await
    }
}
#[derive(Default)]
struct Log {
    calls: Mutex<Vec<String>>,
    before: AtomicUsize,
    after: AtomicUsize,
}
struct Hooks(Arc<Log>);
impl Middleware for Hooks {
    async fn before_model(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.before.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn after_model(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.after.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Save,
    Restore,
    Terminal,
}
#[derive(Clone, Copy)]
enum ModelKind {
    Planned,
    PlannedParent,
    Parent,
    SiblingFirstParent,
    Child,
    Sibling,
}
struct Model {
    kind: ModelKind,
    phase: Phase,
    log: Arc<Log>,
}
fn part(kind: &str, data: Value) -> Part {
    Part {
        r#type: kind.into(),
        data: data.as_object().unwrap().clone(),
    }
}
fn text(value: &str) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(part("text", json!({"value":value})));
    message
}
fn complete(message: Message, tokens: u64) -> Result<Event, Error> {
    Ok(Event::Complete {
        message,
        usage: Some(Usage {
            total: Some(tokens),
            ..Default::default()
        }),
        finish_reason: Some("stop".into()),
    })
}
fn partial() -> Message {
    let mut message = text("unfinished answer");
    message
        .content
        .insert(0, part("think", json!({"value":"unfinished thinking"})));
    message.content.push(part(
        "tool_call",
        json!({"name":"must_not_execute","call_id":"incomplete","arguments":"{"}),
    ));
    message.usage = Some(Usage {
        input: Some(30),
        output: Some(10),
        total: Some(40),
        ..Default::default()
    });
    message
}
fn waiting() -> Vec<Result<Event, Error>> {
    vec![
        Ok(Event::Delta { message: partial() }),
        Err(Error::new("WAITING", "provider waits")),
    ]
}
impl Chat for Model {
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        assert!(
            self.phase != Phase::Terminal,
            "terminal restore must not call any model"
        );
        let messages = serde_json::to_string(&request.messages).unwrap();
        assert!(
            !messages.contains("unfinished answer"),
            "partial output is not prompt context"
        );
        assert!(!messages.contains("unfinished thinking"));
        let task = request
            .messages
            .as_slice()
            .iter()
            .rev()
            .find_map(|m| m.values.get("plan_node").and_then(Value::as_str));
        let (label, events) = match self.kind {
            ModelKind::Planned | ModelKind::PlannedParent => match (task, self.phase) {
                (Some("a"), Phase::Save) if matches!(self.kind, ModelKind::PlannedParent) => {
                    assert!(
                        !self
                            .log
                            .calls
                            .lock()
                            .unwrap()
                            .iter()
                            .any(|c| c == "a:delegate")
                    );
                    let mut message = Message::new(Role::Assistant);
                    message.content.push(part("tool_call", json!({
                        "name":"agent_spawn","call_id":"delegate_a",
                        "arguments":{"name":"a","input":{"payload":[{"type":"text","data":{"value":"child query"}}]}}
                    })));
                    ("a:delegate", vec![complete(message, 2)])
                }
                (Some("a"), Phase::Restore) if matches!(self.kind, ModelKind::PlannedParent) => {
                    assert!(
                        messages.contains("child authoritative")
                            || messages.contains("final authoritative")
                    );
                    ("a:final", vec![complete(text("a authoritative"), 7)])
                }
                (Some("a"), Phase::Save) => ("a", waiting()),
                (Some("a"), Phase::Restore) => ("a", vec![complete(text("a authoritative"), 7)]),
                (Some("b"), Phase::Save) => ("b", vec![complete(text("b authoritative"), 5)]),
                (Some(_), _) => panic!("successful task must not execute again"),
                (None, Phase::Save) if request.response_format.is_some() => {
                    let mut message = Message::new(Role::Assistant);
                    message.values = json!({"version":1,"nodes":{
                        "a":{"name":"research","objective":"first task"},
                        "b":{"name":"research","objective":"second task"}
                    },"edges":[]})
                    .as_object()
                    .unwrap()
                    .clone();
                    ("plan", vec![complete(message, 3)])
                }
                (None, Phase::Save) => {
                    assert!(
                        self.log.calls.lock().unwrap().is_empty(),
                        "no final inference before tasks finish"
                    );
                    let mut message = Message::new(Role::Assistant);
                    message.content.push(part(
                        "tool_call",
                        json!({"name":"plan","call_id":"plan","arguments":{}}),
                    ));
                    ("select", vec![complete(message, 2)])
                }
                (None, Phase::Restore) => {
                    assert!(
                        request.response_format.is_none(),
                        "planning must not replay"
                    );
                    assert!(
                        messages.contains("a authoritative")
                            && messages.contains("b authoritative")
                    );
                    ("final", vec![complete(text("final authoritative"), 11)])
                }
                _ => unreachable!(),
            },
            ModelKind::Child => {
                if self.phase == Phase::Save {
                    ("a", waiting())
                } else {
                    ("a", vec![complete(text("child authoritative"), 20)])
                }
            }
            ModelKind::Sibling => {
                assert!(
                    self.phase == Phase::Save,
                    "successful sibling must not execute again"
                );
                ("b", vec![complete(text("sibling authoritative"), 5)])
            }
            ModelKind::Parent | ModelKind::SiblingFirstParent => {
                if self.phase == Phase::Save {
                    assert!(
                        self.log
                            .calls
                            .lock()
                            .unwrap()
                            .iter()
                            .all(|label| label != "select")
                    );
                    let mut message = Message::new(Role::Assistant);
                    // In the capacity-one nested fixture the sibling must
                    // actually finish before we assert successful reuse.
                    let order = if matches!(self.kind, ModelKind::SiblingFirstParent) {
                        ["b", "a"]
                    } else {
                        ["a", "b"]
                    };
                    for name in order {
                        message.content.push(part("tool_call", json!({
                        "name":"agent_spawn","call_id":format!("delegate_{name}"),
                        "arguments":{"name":name,"input":{"payload":[{"type":"text","data":{"value":"child query"}}]}}
                    })));
                    }
                    ("select", vec![complete(message, 2)])
                } else {
                    assert!(
                        (messages.contains("child authoritative")
                            || messages.contains("final authoritative"))
                            && messages.contains("sibling authoritative")
                    );
                    ("final", vec![complete(text("parent authoritative"), 2)])
                }
            }
        };
        self.log.calls.lock().unwrap().push(label.into());
        Ok(Box::pin(stream::iter(events)))
    }
}
fn ids() -> i64 {
    static NEXT: AtomicI64 = AtomicI64::new(91_000_000);
    NEXT.fetch_add(1, Ordering::SeqCst)
}
fn no_ids() -> i64 {
    panic!("restoration must not allocate replacement identities")
}
fn input() -> Input {
    serde_json::from_value(json!({
        "metadata":{"chat_id":1001,"session_id":2001,"agent_id":3001},
        "payload":[{"type":"text","data":{"value":"original"}}]
    }))
    .unwrap()
}
fn current(node: &Value) -> &Value {
    node["execution"]["attempts"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
}
fn models(scene: &Value) -> Vec<&Value> {
    scene["steps"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["kind"] == "model")
        .collect()
}
fn nodes(scene: &Value, plan: bool) -> BTreeMap<String, &Value> {
    let nodes = if plan {
        &scene["nodes"]
    } else {
        &scene["execution"]["attempts"][0]["nodes"]
    };
    nodes
        .as_object()
        .unwrap()
        .iter()
        .map(|(id, node)| (id.clone(), node))
        .collect()
}
fn native(node: &Value, plan: bool) -> &Value {
    if plan {
        current(node)
    } else {
        &current(node)["agent"]
    }
}

#[allow(clippy::too_many_arguments)]
async fn run(
    plan: bool,
    parent_react: bool,
    child_react: bool,
    phase: Phase,
    mode: bool,
    store: Arc<Store>,
    child_store: Arc<Store>,
    log: Arc<Log>,
) -> Vec<Result<Event, Error>> {
    let id_fn: fn() -> i64 = if phase == Phase::Save { ids } else { no_ids };
    let make = |kind| Model {
        kind,
        phase,
        log: log.clone(),
    };
    if plan {
        return agent::react()
            .model(make(ModelKind::Planned))
            .middleware(Hooks(log.clone()))
            .checkpoint(store)
            .with_id_fn(id_fn)
            .with_max_iters(6)
            .with_max_concurrency(2)
            .build()
            .unwrap()
            .run(input(), mode, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
    }
    let a = if child_react {
        SubAgent::new(
            "a",
            agent::react()
                .model(make(ModelKind::Child))
                .middleware(Hooks(log.clone()))
                .checkpoint(child_store.clone())
                .build()
                .unwrap(),
        )
        .unwrap()
    } else {
        SubAgent::new(
            "a",
            agent::chat()
                .model(make(ModelKind::Child))
                .middleware(Hooks(log.clone()))
                .checkpoint(child_store.clone())
                .build()
                .unwrap(),
        )
        .unwrap()
    };
    let b = SubAgent::new(
        "b",
        agent::chat()
            .model(make(ModelKind::Sibling))
            .middleware(Hooks(log.clone()))
            .checkpoint(child_store)
            .build()
            .unwrap(),
    )
    .unwrap();
    if parent_react {
        agent::react()
            .model(make(ModelKind::Parent))
            .middleware(Hooks(log.clone()))
            .subagents(vec![a, b])
            .checkpoint(store)
            .with_id_fn(id_fn)
            .with_max_concurrency(2)
            .build()
            .unwrap()
            .run(input(), mode, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await
    } else {
        agent::chat()
            .model(make(ModelKind::Parent))
            .middleware(Hooks(log.clone()))
            .subagents(vec![a, b])
            .checkpoint(store)
            .with_id_fn(id_fn)
            .with_max_concurrency(2)
            .build()
            .unwrap()
            .run(input(), mode, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await
    }
}

#[tokio::test]
#[ignore = "explicitly invoked by the OS-process regression"]
async fn stream_scene_worker() {
    let path = std::path::PathBuf::from(std::env::var_os("HEART_MODEL_PROCESS_SCENE").unwrap());
    let phase = match std::env::var("HEART_MODEL_PROCESS_PHASE").unwrap().as_str() {
        "save" => Phase::Save,
        "restore" => Phase::Restore,
        "terminal" => Phase::Terminal,
        _ => panic!("invalid fixture phase"),
    };
    let plan = std::env::var("HEART_MODEL_PROCESS_KIND").unwrap() == "plan";
    let parent_react = std::env::var("HEART_MODEL_PROCESS_PARENT").unwrap() == "true";
    let child_react = std::env::var("HEART_MODEL_PROCESS_CHILD").unwrap() == "true";
    let mode = std::env::var("HEART_MODEL_PROCESS_STREAM").unwrap() == "true";
    let store = Arc::new(Store::default());
    let child_store = Arc::new(Store::default());
    let log = Arc::new(Log::default());
    let before: Option<Value> = if phase == Phase::Save {
        None
    } else {
        let scene = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        store
            .backend
            .save(1001, scene, &Cancellation::new())
            .await
            .unwrap();
        Some(
            store
                .backend
                .load(1001, &Cancellation::new())
                .await
                .unwrap(),
        )
    };
    // A different argument on restoration must not replace the saved mode.
    let actual_mode = if phase == Phase::Save { mode } else { !mode };
    let events = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        run(
            plan,
            parent_react,
            child_react,
            phase,
            actual_mode,
            store.clone(),
            child_store.clone(),
            log.clone(),
        ),
    )
    .await
    .unwrap();
    let saved = store
        .backend
        .load(1001, &Cancellation::new())
        .await
        .unwrap();
    assert_eq!(saved["query"], serde_json::to_value(input()).unwrap());
    assert_eq!(saved["stream"], mode);
    assert_eq!(child_store.loads.load(Ordering::SeqCst), 0);
    assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
    if !mode {
        assert!(!events.iter().any(|e| matches!(e, Ok(Event::Delta { .. }))));
    }
    let mut calls = log.calls.lock().unwrap().clone();
    calls.sort();
    if phase == Phase::Terminal {
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert!(
            events
                .iter()
                .any(|event| matches!(event, Ok(Event::Complete { .. })))
        );
        assert_eq!(saved, before.unwrap());
        assert!(calls.is_empty());
        assert_eq!(log.before.load(Ordering::SeqCst), 0);
        assert_eq!(log.after.load(Ordering::SeqCst), 0);
        assert_eq!(store.saves.load(Ordering::SeqCst), 0);
        return;
    }
    if phase == Phase::Save {
        assert!(
            matches!(events.last(),Some(Err(e)) if e.code == "WAITING"),
            "{events:?}"
        );
        assert_eq!(saved["status"], "waiting");
        assert_eq!(saved["usage"]["total"], if plan { 50 } else { 47 });
        assert_eq!(
            calls,
            if plan {
                vec!["a", "b", "plan", "select"]
            } else {
                vec!["a", "b", "select"]
            }
        );
        assert_eq!(log.before.load(Ordering::SeqCst), if plan { 4 } else { 3 });
        assert_eq!(log.after.load(Ordering::SeqCst), if plan { 3 } else { 2 });
        let nodes = nodes(&saved, plan);
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes
                .values()
                .filter(|n| n["status"] == "succeeded")
                .count(),
            1
        );
        let a = nodes.values().find(|n| n["status"] == "waiting").unwrap();
        let native = native(a, plan);
        let step = models(native)[0];
        assert_eq!(step["attempts"][0]["status"], "waiting");
        assert_eq!(step["attempts"][0]["usage"]["total"], 40);
        assert_eq!(
            step["attempts"][0]["message"]["content"],
            serde_json::to_value(partial().content).unwrap()
        );
        assert!(step["message"].is_null());
    } else {
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert_eq!(saved["status"], "succeeded");
        assert_eq!(saved["usage"]["total"], if plan { 68 } else { 69 });
        assert_eq!(calls, vec!["a", "final"]);
        assert_eq!(log.before.load(Ordering::SeqCst), 2);
        assert_eq!(log.after.load(Ordering::SeqCst), 2);
        let old = before.unwrap();
        let old_nodes = nodes(&old, plan);
        let new_nodes = nodes(&saved, plan);
        assert_eq!(
            old_nodes.keys().collect::<Vec<_>>(),
            new_nodes.keys().collect::<Vec<_>>()
        );
        if !plan {
            assert_eq!(saved["execution"]["exec_id"], old["execution"]["exec_id"]);
        }
        for (id, old_node) in old_nodes {
            let new_node = new_nodes[&id];
            if old_node["status"] == "succeeded" {
                assert_eq!(new_node, old_node);
                continue;
            }
            assert_eq!(
                new_node["execution"]["exec_id"],
                old_node["execution"]["exec_id"]
            );
            assert_eq!(
                new_node["execution"]["attempts"].as_array().unwrap().len(),
                1
            );
            assert_eq!(
                current(new_node)["created_time"],
                current(old_node)["created_time"]
            );
            assert_eq!(current(new_node)["idx"], current(old_node)["idx"]);
            let old_native = native(old_node, plan);
            let new_native = native(new_node, plan);
            let old_step = models(old_native)[0];
            let new_step = models(new_native)[0];
            assert_eq!(new_step["id"], old_step["id"]);
            assert_eq!(new_step["attempts"][0], old_step["attempts"][0]);
            assert_eq!(new_step["attempts"].as_array().unwrap().len(), 2);
            assert_eq!(new_step["attempts"][1]["status"], "succeeded");
            if !plan {
                assert_eq!(new_native["query"], old_native["query"]);
            }
        }
    }
    std::fs::write(path, serde_json::to_vec(&saved).unwrap()).unwrap();
}

#[test]
fn planned_and_delegated_scenes_restore_using_only_json_in_fresh_processes() {
    let root = std::env::temp_dir().join(format!(
        "heart-model-process-{}-{}",
        std::process::id(),
        ids()
    ));
    std::fs::create_dir(&root).unwrap();
    for mode in [false, true] {
        for (plan, parent, child) in [
            (true, true, false),
            (false, false, false),
            (false, false, true),
            (false, true, false),
            (false, true, true),
        ] {
            let path = root.join(format!("{plan}-{parent}-{child}-{mode}.json"));
            for phase in ["save", "restore", "terminal"] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "stream_scene_worker", "--ignored", "--nocapture"])
                    .env("HEART_MODEL_PROCESS_SCENE", &path)
                    .env("HEART_MODEL_PROCESS_PHASE", phase)
                    .env(
                        "HEART_MODEL_PROCESS_KIND",
                        if plan { "plan" } else { "delegation" },
                    )
                    .env("HEART_MODEL_PROCESS_PARENT", parent.to_string())
                    .env("HEART_MODEL_PROCESS_CHILD", child.to_string())
                    .env("HEART_MODEL_PROCESS_STREAM", mode.to_string())
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{plan}/{parent}/{child}/{mode}/{phase}: {} {}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[derive(Clone, Copy)]
struct Nested {
    parent_plan: bool,
    child_plan: bool,
    parent_react: bool,
    child_react: bool,
    capacity: usize,
}

#[allow(clippy::too_many_arguments)]
async fn run_nested(
    spec: Nested,
    phase: Phase,
    mode: bool,
    store: Arc<Store>,
    child_store: Arc<Store>,
    outer: Arc<Log>,
    child: Arc<Log>,
) -> Vec<Result<Event, Error>> {
    let id_fn: fn() -> i64 = if phase == Phase::Save { ids } else { no_ids };
    let child_model = Model {
        kind: if spec.child_plan {
            ModelKind::Planned
        } else {
            ModelKind::Child
        },
        phase,
        log: child.clone(),
    };
    let a = if spec.child_react {
        SubAgent::new(
            "a",
            agent::react()
                .model(child_model)
                .middleware(Hooks(child.clone()))
                .checkpoint(child_store.clone())
                .with_id_fn(id_fn)
                .with_max_iters(6)
                .with_max_concurrency(spec.capacity)
                .build()
                .unwrap(),
        )
        .unwrap()
    } else {
        SubAgent::new(
            "a",
            agent::chat()
                .model(child_model)
                .middleware(Hooks(child.clone()))
                .checkpoint(child_store.clone())
                .with_id_fn(id_fn)
                .with_max_concurrency(spec.capacity)
                .build()
                .unwrap(),
        )
        .unwrap()
    };
    let mut children = vec![a];
    if !spec.parent_plan {
        children.push(
            SubAgent::new(
                "b",
                agent::chat()
                    .model(Model {
                        kind: ModelKind::Sibling,
                        phase,
                        log: outer.clone(),
                    })
                    .middleware(Hooks(outer.clone()))
                    .checkpoint(child_store)
                    .build()
                    .unwrap(),
            )
            .unwrap(),
        );
    }
    let parent_model = Model {
        kind: if spec.parent_plan {
            ModelKind::PlannedParent
        } else {
            ModelKind::SiblingFirstParent
        },
        phase,
        log: outer.clone(),
    };
    if spec.parent_react {
        agent::react()
            .model(parent_model)
            .middleware(Hooks(outer))
            .subagents(children)
            .checkpoint(store)
            .with_id_fn(id_fn)
            .with_max_iters(6)
            .with_max_concurrency(spec.capacity)
            .build()
            .unwrap()
            .run(input(), mode, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await
    } else {
        agent::chat()
            .model(parent_model)
            .middleware(Hooks(outer))
            .subagents(children)
            .checkpoint(store)
            .with_id_fn(id_fn)
            .with_max_concurrency(spec.capacity)
            .build()
            .unwrap()
            .run(input(), mode, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await
    }
}

/// Walk only canonical scene/node containment, never a global node-name index.
fn walk_scenes<'a>(
    scene: &'a Value,
    path: &str,
    scenes: &mut BTreeMap<String, &'a Value>,
    states: &mut BTreeMap<String, &'a Value>,
) {
    scenes.insert(path.to_owned(), scene);
    if let Some(nodes) = scene["nodes"].as_object() {
        for (id, node) in nodes {
            walk_node(node, &format!("{path}/nodes/{id}"), scenes, states);
        }
    }
    if let Some(attempts) = scene["execution"]["attempts"].as_array() {
        for (idx, attempt) in attempts.iter().enumerate() {
            if let Some(nodes) = attempt["nodes"].as_object() {
                for (id, node) in nodes {
                    walk_node(
                        node,
                        &format!("{path}/execution/attempts/{idx}/nodes/{id}"),
                        scenes,
                        states,
                    );
                }
            }
        }
    }
}
fn walk_node<'a>(
    node: &'a Value,
    path: &str,
    scenes: &mut BTreeMap<String, &'a Value>,
    states: &mut BTreeMap<String, &'a Value>,
) {
    states.insert(path.to_owned(), node);
    for (idx, attempt) in node["execution"]["attempts"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let path = format!("{path}/execution/attempts/{idx}");
        if attempt["agent"].is_object() {
            walk_scenes(&attempt["agent"], &format!("{path}/agent"), scenes, states);
        }
        if let Some(nodes) = attempt["nodes"].as_object() {
            for (id, node) in nodes {
                walk_node(node, &format!("{path}/nodes/{id}"), scenes, states);
            }
        }
    }
}

#[tokio::test]
#[ignore = "explicitly invoked by the nested OS-process regression"]
async fn nested_stream_scene_worker() {
    let path = std::path::PathBuf::from(std::env::var_os("HEART_MODEL_PROCESS_SCENE").unwrap());
    let phase = match std::env::var("HEART_MODEL_PROCESS_PHASE").unwrap().as_str() {
        "save" => Phase::Save,
        "restore" => Phase::Restore,
        "terminal" => Phase::Terminal,
        _ => panic!("invalid fixture phase"),
    };
    let flag = |name| std::env::var(name).unwrap() == "true";
    let spec = Nested {
        parent_plan: flag("HEART_MODEL_NESTED_PARENT_PLAN"),
        child_plan: flag("HEART_MODEL_NESTED_CHILD_PLAN"),
        parent_react: flag("HEART_MODEL_PROCESS_PARENT"),
        child_react: flag("HEART_MODEL_PROCESS_CHILD"),
        capacity: std::env::var("HEART_MODEL_NESTED_CAPACITY")
            .unwrap()
            .parse()
            .unwrap(),
    };
    let mode = flag("HEART_MODEL_PROCESS_STREAM");
    let store = Arc::new(Store::default());
    let child_store = Arc::new(Store::default());
    let outer = Arc::new(Log::default());
    let child = Arc::new(Log::default());
    let before: Option<Value> = if phase == Phase::Save {
        None
    } else {
        let saved = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        store
            .backend
            .save(1001, saved, &Cancellation::new())
            .await
            .unwrap();
        Some(
            store
                .backend
                .load(1001, &Cancellation::new())
                .await
                .unwrap(),
        )
    };
    let events = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        run_nested(
            spec,
            phase,
            if phase == Phase::Save { mode } else { !mode },
            store.clone(),
            child_store.clone(),
            outer.clone(),
            child.clone(),
        ),
    )
    .await
    .unwrap();
    let saved = store
        .backend
        .load(1001, &Cancellation::new())
        .await
        .unwrap();
    assert_eq!(saved["query"], serde_json::to_value(input()).unwrap());
    assert_eq!(saved["stream"], mode);
    assert_eq!(child_store.loads.load(Ordering::SeqCst), 0);
    assert_eq!(child_store.saves.load(Ordering::SeqCst), 0);
    if !mode {
        assert!(!events.iter().any(|e| matches!(e, Ok(Event::Delta { .. }))));
    }
    let sorted = |log: &Log| {
        let mut calls = log.calls.lock().unwrap().clone();
        calls.sort();
        calls
    };
    if phase == Phase::Terminal {
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert!(
            events
                .iter()
                .any(|event| matches!(event, Ok(Event::Complete { .. })))
        );
        assert_eq!(saved, before.unwrap());
        for log in [&outer, &child] {
            assert!(sorted(log).is_empty());
            assert_eq!(log.before.load(Ordering::SeqCst), 0);
            assert_eq!(log.after.load(Ordering::SeqCst), 0);
        }
        assert_eq!(store.saves.load(Ordering::SeqCst), 0);
        return;
    }
    let mut scenes = BTreeMap::new();
    let mut states = BTreeMap::new();
    walk_scenes(&saved, "root", &mut scenes, &mut states);
    assert_eq!(scenes.len(), if spec.parent_plan { 2 } else { 3 });
    if phase == Phase::Save {
        assert!(
            matches!(events.last(),Some(Err(e)) if e.code == "WAITING"),
            "{events:?}"
        );
        let expected = match (spec.parent_plan, spec.child_plan) {
            (true, false) => 52,
            (false, true) => 57,
            (true, true) => 62,
            _ => unreachable!(),
        };
        assert_eq!(saved["usage"]["total"], expected);
        assert_eq!(
            sorted(&outer),
            if spec.parent_plan {
                vec!["a:delegate", "b", "plan", "select"]
            } else {
                vec!["b", "select"]
            }
        );
        assert_eq!(
            sorted(&child),
            if spec.child_plan {
                vec!["a", "b", "plan", "select"]
            } else {
                vec!["a"]
            }
        );
        assert_eq!(
            outer.before.load(Ordering::SeqCst),
            if spec.parent_plan { 4 } else { 2 }
        );
        assert_eq!(
            outer.after.load(Ordering::SeqCst),
            if spec.parent_plan { 4 } else { 2 }
        );
        assert_eq!(
            child.before.load(Ordering::SeqCst),
            if spec.child_plan { 4 } else { 1 }
        );
        assert_eq!(
            child.after.load(Ordering::SeqCst),
            if spec.child_plan { 3 } else { 0 }
        );
        let mut partials = 0;
        for scene in scenes
            .values()
            .copied()
            .chain(states.values().map(|node| current(node)))
        {
            if scene["steps"].is_array() {
                for model in models(scene) {
                    if model["stage"] == "call" {
                        assert_eq!(model["attempts"][0]["status"], "waiting");
                        assert_eq!(
                            model["attempts"][0]["message"]["content"],
                            serde_json::to_value(partial().content).unwrap()
                        );
                        assert_eq!(model["attempts"][0]["usage"]["total"], 40);
                        assert!(model["message"].is_null());
                        partials += 1;
                    }
                }
            }
        }
        assert_eq!(
            partials, 1,
            "one actual partial Model owner, not copied into containers"
        );
    } else {
        assert!(events.iter().all(Result::is_ok), "{events:?}");
        assert_eq!(saved["status"], "succeeded");
        assert_eq!(
            saved["usage"]["total"],
            match (spec.parent_plan, spec.child_plan) {
                (true, false) => 90,
                (false, true) => 77,
                (true, true) => 98,
                _ => unreachable!(),
            }
        );
        assert_eq!(
            sorted(&outer),
            if spec.parent_plan {
                vec!["a:final", "final"]
            } else {
                vec!["final"]
            }
        );
        assert_eq!(
            sorted(&child),
            if spec.child_plan {
                vec!["a", "final"]
            } else {
                vec!["a"]
            }
        );
        for (log, count) in [
            (&outer, if spec.parent_plan { 2 } else { 1 }),
            (&child, if spec.child_plan { 2 } else { 1 }),
        ] {
            assert_eq!(log.before.load(Ordering::SeqCst), count);
            assert_eq!(log.after.load(Ordering::SeqCst), count);
        }
        let old = before.unwrap();
        let mut old_scenes = BTreeMap::new();
        let mut old_states = BTreeMap::new();
        walk_scenes(&old, "root", &mut old_scenes, &mut old_states);
        assert_eq!(
            scenes.keys().collect::<Vec<_>>(),
            old_scenes.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            states.keys().collect::<Vec<_>>(),
            old_states.keys().collect::<Vec<_>>()
        );
        for (path, old_node) in &old_states {
            let new_node = states[path];
            if old_node["status"] == "succeeded" {
                assert_eq!(new_node, *old_node);
            }
            assert_eq!(
                new_node["execution"]["exec_id"],
                old_node["execution"]["exec_id"]
            );
            assert_eq!(
                new_node["execution"]["attempts"].as_array().unwrap().len(),
                old_node["execution"]["attempts"].as_array().unwrap().len()
            );
            assert_eq!(current(new_node)["idx"], current(old_node)["idx"]);
            assert_eq!(
                current(new_node)["created_time"],
                current(old_node)["created_time"]
            );
        }
        for (path, old_scene) in &old_scenes {
            let new_scene = scenes[path];
            assert_eq!(new_scene["query"], old_scene["query"]);
            assert_eq!(new_scene["stream"], old_scene["stream"]);
            assert_eq!(
                new_scene["execution"]["exec_id"],
                old_scene["execution"]["exec_id"]
            );
        }
        let mut continued = 0;
        for (old_owner, new_owner) in old_scenes
            .iter()
            .map(|(path, scene)| (*scene, scenes[path]))
            .chain(
                old_states
                    .iter()
                    .map(|(path, node)| (current(node), current(states[path]))),
            )
        {
            if !old_owner["steps"].is_array() {
                continue;
            }
            for old_model in models(old_owner) {
                let new_model = models(new_owner)
                    .into_iter()
                    .find(|m| m["id"] == old_model["id"])
                    .unwrap();
                if old_model["stage"] == "call" {
                    assert_eq!(new_model["attempts"][0], old_model["attempts"][0]);
                    assert_eq!(new_model["attempts"].as_array().unwrap().len(), 2);
                    assert_eq!(new_model["attempts"][1]["status"], "succeeded");
                    continued += 1;
                } else {
                    assert_eq!(new_model, old_model);
                }
            }
        }
        assert_eq!(continued, 1);
    }
    std::fs::write(path, serde_json::to_vec(&saved).unwrap()).unwrap();
}

#[test]
fn planned_delegations_and_child_plans_restore_in_fresh_processes() {
    let root = std::env::temp_dir().join(format!(
        "heart-nested-model-process-{}-{}",
        std::process::id(),
        ids()
    ));
    std::fs::create_dir(&root).unwrap();
    for mode in [false, true] {
        for capacity in [1, 2] {
            for (parent_plan, child_plan, parent, child) in [
                (true, false, true, false),
                (true, false, true, true),
                (false, true, false, true),
                (false, true, true, true),
                (true, true, true, true),
            ] {
                let path = root.join(format!(
                    "{parent_plan}-{child_plan}-{parent}-{child}-{mode}-{capacity}.json"
                ));
                for phase in ["save", "restore", "terminal"] {
                    let output = std::process::Command::new(std::env::current_exe().unwrap())
                        .args([
                            "--exact",
                            "nested_stream_scene_worker",
                            "--ignored",
                            "--nocapture",
                        ])
                        .env("HEART_MODEL_PROCESS_SCENE", &path)
                        .env("HEART_MODEL_PROCESS_PHASE", phase)
                        .env("HEART_MODEL_PROCESS_PARENT", parent.to_string())
                        .env("HEART_MODEL_PROCESS_CHILD", child.to_string())
                        .env("HEART_MODEL_PROCESS_STREAM", mode.to_string())
                        .env("HEART_MODEL_NESTED_PARENT_PLAN", parent_plan.to_string())
                        .env("HEART_MODEL_NESTED_CHILD_PLAN", child_plan.to_string())
                        .env("HEART_MODEL_NESTED_CAPACITY", capacity.to_string())
                        .output()
                        .unwrap();
                    assert!(
                        output.status.success(),
                        "{parent_plan}/{child_plan}/{parent}/{child}/{mode}/{capacity}/{phase}: {} {}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
            }
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
