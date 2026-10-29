//! ToolNode recovery uses native Workflow Attempts, not a public state API.
use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::checkpoint::Checkpoint,
    agent::event::Event,
    agent::message::Role,
    agent::model::content::part::Part,
    agent::tool::{Call, Evaluation, Tool, ToolDefinition},
    agent::workflow::{
        CompiledWorkflow, RetryPolicy, Runtime, WorkflowBuilder,
        graph::{END, START},
        tool::ToolNode,
    },
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
    },
};

#[derive(Clone, Copy, Debug)]
enum Phase {
    Before,
    Call,
    After,
}
struct Log {
    phase: Phase,
    gate: AtomicBool,
    technical: bool,
    cancel: bool,
    hold: bool,
    counts: Mutex<BTreeMap<String, [usize; 3]>>,
}
impl Log {
    fn new(phase: Phase) -> Self {
        Self {
            phase,
            gate: AtomicBool::new(true),
            technical: false,
            cancel: false,
            hold: false,
            counts: Default::default(),
        }
    }
    fn counts(&self, owner: &str) -> [usize; 3] {
        *self.counts.lock().unwrap().get(owner).unwrap_or(&[0; 3])
    }
    fn count(&self, owner: &str, phase: usize) {
        self.counts.lock().unwrap().entry(owner.into()).or_default()[phase] += 1;
    }
    async fn stop(
        &self,
        phase: Phase,
        call: &Call,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        if call.arguments["owner"] != "A"
            || std::mem::discriminant(&phase) != std::mem::discriminant(&self.phase)
            || !self.gate.load(Ordering::SeqCst)
        {
            return Ok(());
        }
        if self.cancel {
            cancellation.cancel();
            return std::future::pending().await;
        }
        if self.hold {
            return std::future::pending().await;
        }
        if self.technical {
            self.gate.store(false, Ordering::SeqCst);
            return Err(
                Error::new("TIMEOUT", "original tool hook failure").with_details("original detail")
            );
        }
        Err(Error::new("WAITING", "unfinished actual Tool method"))
    }
}
#[derive(Clone)]
struct Lookup {
    definition: ToolDefinition,
    log: Arc<Log>,
}
impl Lookup {
    fn new(log: Arc<Log>) -> Self {
        Self{log,definition:ToolDefinition{name:"lookup".into(),description:"lookup".into(),parameters:json!({"type":"object","properties":{"owner":{"type":"string"},"n":{"type":"integer"}},"required":["owner","n"],"additionalProperties":false}).as_object().unwrap().clone()}}
    }
}
impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn before(&self, call: &mut Call, cancellation: &Cancellation) -> Result<(), Error> {
        self.log.count(call.arguments["owner"].as_str().unwrap(), 0);
        assert_eq!(call.arguments["n"], 1);
        // A failed working copy must not replace the native original input.
        call.arguments.insert("n".into(), json!(99));
        self.log.stop(Phase::Before, call, cancellation).await?;
        call.arguments.insert("n".into(), json!(2));
        Ok(())
    }
    async fn call(&self, call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        self.log.count(call.arguments["owner"].as_str().unwrap(), 1);
        assert_eq!(call.arguments["n"], 2);
        self.log.stop(Phase::Call, call, cancellation).await?;
        Ok(Message::function(
            json!({"owner":call.arguments["owner"],"n":2}),
        ))
    }
    async fn after(
        &self,
        call: &Call,
        message: &mut Message,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        self.log.count(call.arguments["owner"].as_str().unwrap(), 2);
        assert_eq!(call.arguments["n"], 2);
        assert_eq!(message.values["n"], 2);
        message.values.insert("n".into(), json!(99));
        self.log.stop(Phase::After, call, cancellation).await?;
        message.values.insert("n".into(), json!(3));
        Ok(())
    }
    async fn eval(&self, _: &Call, _: &Message, _: &Cancellation) -> Result<Evaluation, Error> {
        panic!("Workflow must not enable Tool eval")
    }
}
fn input(owners: &[&str]) -> Messages {
    let mut message = Message::new(Role::Assistant);
    for (idx, owner) in owners.iter().enumerate() {
        message.content.push(Part{r#type:"tool_call".into(),data:json!({"call_id":format!("call_{idx}"),"name":"lookup","arguments":{"owner":owner,"n":1}}).as_object().unwrap().clone()});
    }
    Messages::new([message])
}
fn workflow(log: Arc<Log>, parallel: bool, depth: usize, retries: u32) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("native-tool-phases");
    let node = ToolNode::new(Lookup::new(log));
    builder
        .add_node("tools", if parallel { node.parallel() } else { node })
        .unwrap();
    builder
        .set_retry(
            "tools",
            RetryPolicy {
                max_retries: retries,
                min_delay_ms: 0,
                max_delay_ms: 0,
            },
        )
        .unwrap();
    builder.add_edge(START, "tools").unwrap();
    builder.add_edge("tools", END).unwrap();
    let mut graph = builder.compile().unwrap();
    for idx in 0..depth {
        let mut builder = WorkflowBuilder::new(format!("parent-{idx}"));
        builder.add_graph_node("child", &graph).unwrap();
        builder.add_edge(START, "child").unwrap();
        builder.add_edge("child", END).unwrap();
        graph = builder.compile().unwrap();
    }
    graph
}
fn ids() -> Arc<dyn Fn() -> i64 + Send + Sync> {
    static IDS: AtomicI64 = AtomicI64::new(91_000_000);
    Arc::new(|| IDS.fetch_add(1, Ordering::SeqCst))
}
fn attempt(snapshot: &Value) -> &Value {
    let node = snapshot["nodes"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap();
    let execution = node["executions"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap();
    let current = execution["attempts"].as_array().unwrap().last().unwrap();
    if current["graph"].is_object() {
        attempt(&current["graph"])
    } else {
        current
    }
}
fn execution(snapshot: &Value) -> &Value {
    let node = snapshot["nodes"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap();
    let execution = node["executions"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap();
    let current = execution["attempts"].as_array().unwrap().last().unwrap();
    if current["graph"].is_object() {
        self::execution(&current["graph"])
    } else {
        execution
    }
}
async fn run(runtime: &Runtime, input: Messages) -> Vec<Result<Event, Error>> {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        runtime
            .run(input, ids(), Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await
    })
    .await
    .expect("native Tool phases must yield")
}
#[tokio::test]
async fn before_call_after_waits_resume_only_unfinished_methods() {
    for phase in [Phase::Before, Phase::Call, Phase::After] {
        for parallel in [false, true] {
            for depth in [0, 2] {
                let log = Arc::new(Log::new(phase));
                let runtime =
                    Runtime::new(1001, workflow(log.clone(), parallel, depth, 0)).unwrap();
                let events = run(&runtime, input(&["A", "B"])).await;
                assert!(
                    matches!(events.last(),Some(Err(error)) if error.code=="WAITING"),
                    "{phase:?}/{parallel}/{depth}/{events:?}"
                );
                let saved = runtime.snapshot().unwrap();
                let actual = attempt(&saved);
                let step = actual["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|step| step["call_id"] == "call_0")
                    .unwrap();
                assert_eq!(
                    step["stage"],
                    match phase {
                        Phase::Before => "before",
                        Phase::Call => "call",
                        Phase::After => "after",
                    }
                );
                assert_eq!(
                    step["call"]["arguments"]["n"],
                    if matches!(phase, Phase::Before) { 1 } else { 2 }
                );
                if matches!(phase, Phase::After) {
                    assert_eq!(step["message"]["values"]["n"], 2);
                    assert_eq!(step["attempts"][0]["status"], "succeeded");
                }
                assert!(saved["waits"].as_array().unwrap().is_empty());
                assert!(execution(&saved)["messages"].as_array().unwrap().is_empty());
                let sibling = if parallel {
                    Some(
                        actual["steps"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|step| step["call_id"] == "call_1")
                            .unwrap()
                            .clone(),
                    )
                } else {
                    None
                };
                let restored = Runtime::from_snapshot(
                    1001,
                    workflow(log.clone(), parallel, depth, 0),
                    saved.clone(),
                )
                .unwrap();
                let again = run(&restored, input(&["A", "B"])).await;
                assert!(matches!(again.last(),Some(Err(error)) if error.code=="WAITING"));
                log.gate.store(false, Ordering::SeqCst);
                let runtime = Runtime::from_snapshot(
                    1001,
                    workflow(log.clone(), parallel, depth, 0),
                    restored.snapshot().unwrap(),
                )
                .unwrap();
                let done = run(&runtime, input(&["A", "B"])).await;
                assert!(done.iter().all(Result::is_ok), "{done:?}");
                let final_scene = runtime.snapshot().unwrap();
                assert_eq!(final_scene["status"], "succeeded");
                assert_eq!(
                    execution(&final_scene)["exec_id"],
                    execution(&saved)["exec_id"]
                );
                assert_eq!(
                    execution(&final_scene)["attempts"]
                        .as_array()
                        .unwrap()
                        .len(),
                    1
                );
                assert_eq!(
                    log.counts("A"),
                    match phase {
                        Phase::Before => [3, 1, 1],
                        Phase::Call => [1, 3, 1],
                        Phase::After => [1, 1, 3],
                    }
                );
                assert_eq!(log.counts("B"), [1, 1, 1]);
                let messages = execution(&final_scene)["messages"].as_array().unwrap();
                assert_eq!(
                    messages
                        .iter()
                        .map(|m| m["values"]["owner"].as_str().unwrap())
                        .collect::<Vec<_>>(),
                    ["A", "B"]
                );
                assert!(messages.iter().all(|m| m["values"]["n"] == 3));
                if let Some(sibling) = sibling {
                    assert_eq!(
                        attempt(&final_scene)["steps"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|step| step["call_id"] == "call_1")
                            .unwrap(),
                        &sibling
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn node_retry_keeps_successful_before_and_call_facts() {
    for phase in [Phase::Before, Phase::Call, Phase::After] {
        for parallel in [false, true] {
            let log = Arc::new(Log {
                technical: true,
                ..Log::new(phase)
            });
            let runtime = Runtime::new(1001, workflow(log.clone(), parallel, 0, 1)).unwrap();
            let events = run(&runtime, input(&["A", "B"])).await;
            assert!(events.iter().all(Result::is_ok), "{phase:?}/{events:?}");
            assert_eq!(
                log.counts("A"),
                match phase {
                    Phase::Before => [2, 1, 1],
                    Phase::Call => [1, 2, 1],
                    Phase::After => [1, 1, 2],
                }
            );
            assert_eq!(log.counts("B"), [1, 1, 1]);
            let saved = runtime.snapshot().unwrap();
            let attempts = execution(&saved)["attempts"].as_array().unwrap();
            assert_eq!(attempts.len(), 2);
            assert_eq!(attempts[0]["error"]["code"], "TIMEOUT");
            assert_eq!(attempts[0]["error"]["details"], "original detail");
            assert_eq!(attempts[1]["idx"], 1);
            assert_eq!(
                attempts[1]["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|s| s["call_id"] == "call_0")
                    .unwrap()["attempts"]
                    .as_array()
                    .unwrap()
                    .len(),
                if matches!(phase, Phase::Call) { 2 } else { 1 }
            );
        }
    }
}

#[tokio::test]
async fn cancelled_hook_preserves_last_success_without_normal_node_output() {
    for phase in [Phase::Before, Phase::Call, Phase::After] {
        let log = Arc::new(Log {
            cancel: true,
            ..Log::new(phase)
        });
        let runtime = Runtime::new(1001, workflow(log.clone(), false, 0, 0)).unwrap();
        let events = run(&runtime, input(&["A"])).await;
        assert!(matches!(events.last(),Some(Err(error)) if error.code=="CANCELLED"));
        let saved = runtime.snapshot().unwrap();
        assert_eq!(saved["status"], "cancelled");
        assert!(execution(&saved)["messages"].as_array().unwrap().is_empty());
        let step = &attempt(&saved)["steps"][0];
        assert_eq!(step["status"], "cancelled");
        assert_ne!(step["call"]["arguments"]["n"], 99);
        if matches!(phase, Phase::After) {
            assert_eq!(step["message"]["values"]["n"], 2);
            assert_eq!(step["attempts"][0]["status"], "succeeded");
        }
    }
}

#[tokio::test]
async fn invalid_native_tool_scene_fails_before_component_work() {
    let log = Arc::new(Log::new(Phase::After));
    let runtime = Runtime::new(1001, workflow(log.clone(), false, 0, 0)).unwrap();
    run(&runtime, input(&["A"])).await;
    let saved = runtime.snapshot().unwrap();
    let before = log.counts("A");
    for field in [
        "stage",
        "call",
        "message",
        "kind",
        "call_id",
        "duplicate",
        "result",
    ] {
        let mut bad = saved.clone();
        let actual = bad["nodes"]["tools"]["executions"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap();
        let native = &mut actual["attempts"][0];
        match field {
            "stage" => native["steps"][0]["stage"] = json!("eval"),
            "call" => native["steps"][0]["call"]["name"] = json!("unregistered"),
            "message" => native["steps"][0]["message"] = Value::Null,
            "kind" => native["steps"][0]["kind"] = json!("model"),
            "call_id" => native["steps"][0]["call_id"] = json!("wrong"),
            "duplicate" => {
                let step = native["steps"][0].clone();
                native["steps"].as_array_mut().unwrap().push(step);
            }
            _ => native["steps"][0]["attempts"][0]["status"] = json!("failed"),
        }
        assert!(
            Runtime::from_snapshot(1001, workflow(log.clone(), false, 0, 0), bad).is_err(),
            "{field}"
        );
        assert_eq!(log.counts("A"), before);
        assert_eq!(runtime.snapshot().unwrap(), saved);
    }
}

struct Store {
    inner: halo::agent::core::checkpoint::InMemory,
    saves: AtomicUsize,
}
impl Checkpoint for Store {
    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        self.saves.fetch_add(1, Ordering::SeqCst);
        self.inner.save(id, state, cancellation).await
    }
    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        self.inner.load(id, cancellation).await
    }
    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, cancellation).await
    }
}
#[tokio::test]
async fn phase_acknowledgements_do_not_add_checkpoint_writes() {
    let log = Arc::new(Log::new(Phase::After));
    let store = Arc::new(Store {
        inner: halo::agent::core::checkpoint::InMemory::new(),
        saves: AtomicUsize::new(0),
    });
    let runtime = Runtime::new(1001, workflow(log.clone(), true, 2, 0))
        .unwrap()
        .with_checkpoint(store.clone());
    run(&runtime, input(&["A", "B"])).await;
    assert_eq!(store.saves.load(Ordering::SeqCst), 1);
    let saved = store.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(saved, runtime.snapshot().unwrap());
    log.gate.store(false, Ordering::SeqCst);
    let restored = Runtime::from_snapshot(1001, workflow(log.clone(), true, 2, 0), saved)
        .unwrap()
        .with_checkpoint(store.clone());
    let events = run(&restored, input(&["A", "B"])).await;
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(store.saves.load(Ordering::SeqCst), 2);
}

fn shared_nodes(log: Arc<Log>) -> CompiledWorkflow {
    let mut builder = WorkflowBuilder::new("shared-tool-node-owners");
    let tool = Lookup::new(log);
    for (source, target, owner) in [("source_a", "a", "A"), ("source_b", "b", "B")] {
        builder
            .add_node(source, move |_: Messages, _: Cancellation| async move {
                Ok::<_, Error>(input(&[owner]))
            })
            .unwrap();
        builder
            .add_node(target, ToolNode::new(tool.clone()))
            .unwrap();
        builder.add_edge(START, source).unwrap();
        builder.add_edge(source, target).unwrap();
    }
    builder.add_node("merge", halo::agent::node::Echo).unwrap();
    builder.add_join(["a", "b"], "merge").unwrap();
    builder.add_edge("merge", END).unwrap();
    builder.compile().unwrap()
}
#[tokio::test]
async fn shared_component_and_call_id_keep_distinct_node_scenes() {
    let log = Arc::new(Log::new(Phase::After));
    let runtime = Runtime::new(1001, shared_nodes(log.clone())).unwrap();
    let events = run(&runtime, Messages::default()).await;
    assert!(matches!(events.last(),Some(Err(error)) if error.code=="WAITING"));
    let saved = runtime.snapshot().unwrap();
    let sibling = saved["nodes"]["b"].clone();
    assert_eq!(log.counts("B"), [1, 1, 1]);
    log.gate.store(false, Ordering::SeqCst);
    let restored = Runtime::from_snapshot(1001, shared_nodes(log.clone()), saved).unwrap();
    assert!(
        run(&restored, Messages::default())
            .await
            .iter()
            .all(Result::is_ok)
    );
    assert_eq!(restored.snapshot().unwrap()["nodes"]["b"], sibling);
    assert_eq!(log.counts("A"), [1, 1, 2]);
    assert_eq!(log.counts("B"), [1, 1, 1]);
}

#[tokio::test]
async fn dropping_stream_in_after_retains_the_acknowledged_call_result() {
    let log = Arc::new(Log {
        hold: true,
        ..Log::new(Phase::After)
    });
    let runtime = Runtime::new(1001, workflow(log.clone(), false, 2, 0)).unwrap();
    let mut output = Box::pin(
        runtime
            .run(input(&["A"]), ids(), Cancellation::new())
            .await
            .unwrap(),
    );
    for _ in 0..100 {
        assert!(matches!(
            futures_util::poll!(output.next()),
            std::task::Poll::Pending
        ));
        if log.counts("A")[2] == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(log.counts("A"), [1, 1, 1]);
    drop(output);
    let saved = runtime.snapshot().unwrap();
    assert_eq!(attempt(&saved)["steps"][0]["stage"], "after");
    assert_eq!(attempt(&saved)["steps"][0]["message"]["values"]["n"], 2);
    log.gate.store(false, Ordering::SeqCst);
    let restored = Runtime::from_snapshot(1001, workflow(log.clone(), false, 2, 0), saved).unwrap();
    assert!(
        run(&restored, input(&["A"]))
            .await
            .iter()
            .all(Result::is_ok)
    );
    assert_eq!(log.counts("A"), [1, 1, 2]);
}

#[tokio::test]
async fn tool_hook_phase_recovers_in_a_fresh_process() {
    for parallel in [false, true] {
        for depth in [0, 2] {
            let log = Arc::new(Log::new(Phase::After));
            let runtime = Runtime::new(1001, workflow(log, parallel, depth, 0)).unwrap();
            run(&runtime, input(&["A", "B"])).await;
            let saved = runtime.snapshot().unwrap();
            let path = std::env::temp_dir().join(format!(
                "heart-tool-phase-{}-{parallel}-{depth}.json",
                std::process::id()
            ));
            std::fs::write(&path, serde_json::to_vec(&saved).unwrap()).unwrap();
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    "tool_phase_process_worker",
                    "--nocapture",
                ])
                .env("HEART_TOOL_PHASE_SNAPSHOT", &path)
                .env("HEART_TOOL_PHASE_PARALLEL", parallel.to_string())
                .env("HEART_TOOL_PHASE_DEPTH", depth.to_string())
                .output();
            std::fs::remove_file(&path).unwrap();
            let result = result.unwrap();
            assert!(
                result.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}

#[tokio::test]
#[ignore = "explicit subprocess worker; invoked by tool_hook_phase_recovers_in_a_fresh_process"]
async fn tool_phase_process_worker() {
    let saved: Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("HEART_TOOL_PHASE_SNAPSHOT").expect("worker fixture"))
            .unwrap(),
    )
    .unwrap();
    let parallel = std::env::var("HEART_TOOL_PHASE_PARALLEL").unwrap() == "true";
    let depth = std::env::var("HEART_TOOL_PHASE_DEPTH")
        .unwrap()
        .parse()
        .unwrap();
    let log = Arc::new(Log::new(Phase::After));
    log.gate.store(false, Ordering::SeqCst);
    let runtime = Runtime::from_snapshot(
        1001,
        workflow(log.clone(), parallel, depth, 0),
        saved.clone(),
    )
    .unwrap();
    let output = runtime
        .run(
            input(&["A", "B"]),
            Arc::new(|| panic!("restore must reuse original execution IDs")),
            Cancellation::new(),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().all(Result::is_ok), "{output:?}");
    assert_eq!(log.counts("A"), [0, 0, 1]);
    assert_eq!(
        log.counts("B"),
        if parallel { [0, 0, 0] } else { [1, 1, 1] }
    );
    let done = runtime.snapshot().unwrap();
    assert_eq!(execution(&done)["exec_id"], execution(&saved)["exec_id"]);
    assert_eq!(execution(&done)["attempts"].as_array().unwrap().len(), 1);
    let terminal =
        Runtime::from_snapshot(1001, workflow(log.clone(), parallel, depth, 0), done).unwrap();
    assert!(
        run(&terminal, input(&["A", "B"]))
            .await
            .iter()
            .all(Result::is_ok)
    );
    assert_eq!(log.counts("A"), [0, 0, 1]);
}
