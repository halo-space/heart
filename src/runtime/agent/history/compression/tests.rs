use std::{
    collections::VecDeque,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use serde_json::{Map, Value, json};
use tokio::sync::Semaphore;

use super::*;
use crate::memory::short::{chat, chat::Store as _, session};
use crate::{
    agent::{Input, Payload},
    core,
    middleware::Middleware,
    model::content::part::Part,
    runtime::agent::hooks::Hooks,
};

type WithSession<S> = memory::Config<
    memory::short::Pair<S, core::memory::short::chat::InMemory>,
    core::memory::InMemory,
>;
type Config = WithSession<core::memory::short::session::InMemory>;

fn usage(total: u64) -> Usage {
    Usage {
        input: Some(total / 2),
        output: Some(total - total / 2),
        total: Some(total),
        ..Default::default()
    }
}

fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: crate::components::model::content::part::object(json!({"value":value})),
    });
    message
}

fn output(value: &str, total: u64) -> Messages {
    let mut message = text(Role::Assistant, value);
    message.usage = Some(usage(total));
    Messages::new([message])
}

fn tool_call(call_id: &str) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "tool_call".into(),
        data: crate::components::model::content::part::object(json!({
            "call_id": call_id,
            "name": "search",
            "arguments": {}
        })),
    });
    message
}

fn tool_result(call_id: &str) -> Message {
    let mut message = Message::new(Role::Tool);
    message
        .metadata
        .insert("call_id".into(), Value::String(call_id.into()));
    message.content.push(Part {
        r#type: "text".into(),
        data: crate::components::model::content::part::object(json!({
            "value": "result"
        })),
    });
    message
}

#[test]
fn compression_drops_an_unfinished_tool_group_but_keeps_complete_groups() {
    let complete = vec![
        text(Role::User, "before"),
        tool_call("call_001"),
        tool_result("call_001"),
        text(Role::Assistant, "after"),
    ];
    assert_eq!(safe_messages(complete.clone()), complete);

    let incomplete = vec![
        text(Role::User, "before"),
        tool_call("call_002"),
        text(Role::Assistant, "unfinished continuation"),
    ];
    let retained = safe_messages(incomplete);
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].content[0].data["value"], "before");
}

#[test]
fn compression_truncates_only_the_tool_result_view() {
    let mut result = tool_result("call_003");
    let long = "x".repeat(MAX_TOOL_RESULT_CHARS + 100);
    result.content[0].data["value"] = Value::String(long);
    result.values.insert(
        "documents".into(),
        json!({"items":[{"content":"y".repeat(MAX_TOOL_RESULT_CHARS + 100)}]}),
    );

    let original = result.clone();
    let compressed = truncate_tool_results(vec![result]);
    let compressed = &compressed[0];
    assert_eq!(
        original.content[0].data["value"].as_str().unwrap().len(),
        MAX_TOOL_RESULT_CHARS + 100
    );
    assert!(
        compressed.content[0].data["value"]
            .as_str()
            .unwrap()
            .contains("tool result truncated for compression input")
    );
    assert!(
        compressed.values["documents"]["items"][0]["content"]
            .as_str()
            .unwrap()
            .contains("tool result truncated for compression input")
    );
}

#[tokio::test]
async fn compression_window_keeps_a_contiguous_prefix_and_retains_the_suffix() {
    let compressor = Compressor::new(Some(1)).window(Some(4), Some(2));
    let config = config().await.with_compression(compressor);
    let pairs = vec![
        (1, text(Role::User, "u1"), text(Role::Assistant, "a1"), 1),
        (2, text(Role::User, "u2"), text(Role::Assistant, "a2"), 1),
        (3, text(Role::User, "u3"), text(Role::Assistant, "a3"), 1),
    ];

    let selected = super::super::compression_window(&config, pairs);
    assert_eq!(
        selected.iter().map(|pair| pair.0).collect::<Vec<_>>(),
        [1, 2]
    );
}

#[derive(Clone)]
struct Compressor {
    size: Option<u64>,
    limit: Option<u64>,
    retain: Option<u64>,
    calls: Arc<AtomicUsize>,
    inputs: Arc<Mutex<Vec<Messages>>>,
    gate: Option<Arc<Semaphore>>,
    results: Arc<Mutex<VecDeque<Result<Messages, Error>>>>,
}

impl Compressor {
    fn new(size: Option<u64>) -> Self {
        Self {
            size,
            limit: None,
            retain: None,
            calls: Arc::new(AtomicUsize::new(0)),
            inputs: Arc::new(Mutex::new(Vec::new())),
            gate: None,
            results: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
    fn gated(mut self) -> Self {
        self.gate = Some(Arc::new(Semaphore::new(0)));
        self
    }
    fn release(&self) {
        self.gate.as_ref().unwrap().add_permits(1);
    }

    fn window(mut self, limit: Option<u64>, retain: Option<u64>) -> Self {
        self.limit = limit;
        self.retain = retain;
        self
    }
}

impl crate::compression::Compressor for Compressor {
    fn size(&self) -> Option<u64> {
        self.size
    }

    fn limit(&self) -> Option<u64> {
        self.limit
    }

    fn retain(&self) -> Option<u64> {
        self.retain
    }

    fn count(&self, input: &Messages) -> u64 {
        input.as_slice().len() as u64
    }
    async fn compress(
        &self,
        input: Messages,
        cancellation: &Cancellation,
    ) -> Result<Messages, Error> {
        assert!(!cancellation.is_cancelled());
        self.inputs.lock().unwrap().push(input);
        self.calls.fetch_add(1, Ordering::SeqCst);
        // Keep shared state across a real suspension in a Send future.
        let local = Arc::new(7);
        if let Some(gate) = &self.gate {
            gate.acquire().await.unwrap().forget();
        }
        tokio::task::yield_now().await;
        assert_eq!(*local, 7);
        self.results
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Ok(output("new summary", 12)))
    }
}

async fn config() -> Config {
    let config = Config::new(
        memory::short::Pair::new(
            core::memory::short::session::InMemory::new(),
            core::memory::short::chat::InMemory::new(),
        ),
        core::memory::InMemory::new(),
    );
    add_session(&config, 101).await;
    config
}

async fn add_session<S: session::Store>(config: &WithSession<S>, id: i64) {
    config
        .session()
        .create(
            session::Session {
                id,
                tenant_id: 1,
                user_id: 2,
                r#type: memory::short::Type::Agent,
                title: Some("keep title".into()),
                summary: None,
                usage: None,
                status: session::Status::Active,
                metadata: json!({"application":"keep"}).as_object().unwrap().clone(),
                created_time: 0,
                updated_time: 0,
            },
            &Cancellation::new(),
        )
        .await
        .unwrap();
}

async fn seed<S: session::Store>(
    config: &WithSession<S>,
    id: i64,
    session_id: i64,
    status: chat::Status,
    usage: Option<Usage>,
) {
    seed_with_text(
        config,
        id,
        session_id,
        status,
        usage,
        &format!("question {id}"),
    )
    .await;
}

async fn seed_with_text<S: session::Store>(
    config: &WithSession<S>,
    id: i64,
    session_id: i64,
    status: chat::Status,
    usage: Option<Usage>,
    input_text: &str,
) {
    config
        .chat()
        .create(
            chat::Chat {
                id,
                tenant_id: 1,
                user_id: 2,
                agent_id: 3,
                session_id,
                r#type: memory::short::Type::Agent,
                status: status.clone(),
                input: Some(chat::Request {
                    metadata: Map::new(),
                    payload: vec![chat::Payload {
                        r#type: "text".into(),
                        data: json!({"value":input_text}),
                        mime_type: None,
                    }],
                }),
                message: (status == chat::Status::Completed)
                    .then(|| text(Role::Assistant, &format!("reply {id}"))),
                usage,
                graph: None,
                state: None,
                metadata: Map::new(),
                started_time: None,
                completed_time: None,
                created_time: 0,
                updated_time: 0,
            },
            &Cancellation::new(),
        )
        .await
        .unwrap();
}

async fn seed_current<S: session::Store>(
    config: &WithSession<S>,
    id: i64,
    session_id: i64,
    usage: Option<Usage>,
) {
    seed_with_text(
        config,
        id,
        session_id,
        chat::Status::Completed,
        usage,
        &format!("current {id}"),
    )
    .await;
}

fn input(id: i64, session_id: i64) -> Input {
    Input {
        metadata:
            json!({"tenant_id":1,"user_id":2,"agent_id":3,"session_id":session_id,"chat_id":id})
                .as_object()
                .unwrap()
                .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":format!("current {id}")}),
            mime_type: None,
        }],
    }
}

async fn run_with_hooks<S, L>(
    config: memory::Config<S, L>,
    id: i64,
    session_id: i64,
    hooks: Hooks,
) -> Messages
where
    S: memory::short::Store + 'static,
    L: memory::Memory + 'static,
{
    run_with_hooks_and_cancellation(config, id, session_id, hooks, &Cancellation::new()).await
}

async fn run_with_hooks_and_cancellation<S, L>(
    config: memory::Config<S, L>,
    id: i64,
    session_id: i64,
    hooks: Hooks,
    cancellation: &Cancellation,
) -> Messages
where
    S: memory::short::Store + 'static,
    L: memory::Memory + 'static,
{
    let mut history = super::super::History::default();
    history.set(config);
    let messages = history
        .load(input(id, session_id).metadata, hooks, cancellation)
        .await
        .unwrap();
    Messages::new(messages)
}

async fn run<S, L>(config: memory::Config<S, L>, id: i64, session_id: i64) -> Messages
where
    S: memory::short::Store + 'static,
    L: memory::Memory + 'static,
{
    run_with_hooks(config, id, session_id, Hooks::default()).await
}

async fn idle<S>(config: &WithSession<S>, id: i64) {
    let key = (Arc::as_ptr(&config.short) as usize, id);
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if !ACTIVE
                .get_or_init(Default::default)
                .lock()
                .unwrap()
                .contains(&key)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
}

async fn entered(component: &Compressor, count: usize) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while component.calls.load(Ordering::SeqCst) < count {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
}

async fn stored<S: session::Store>(config: &WithSession<S>) -> session::Session {
    config
        .session()
        .read(101, &Cancellation::new())
        .await
        .unwrap()
}

fn contents(messages: &Messages) -> String {
    messages
        .as_slice()
        .iter()
        .flat_map(|m| &m.content)
        .filter_map(|p| p.data.get("value").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("|")
}

#[derive(Clone, Default)]
struct LongExtractor {
    calls: Arc<AtomicUsize>,
    fail: Arc<std::sync::atomic::AtomicBool>,
    inputs: Arc<Mutex<Vec<Messages>>>,
}
impl memory::extraction::Extractor for LongExtractor {
    async fn extract(&self, input: &Messages, _: &Cancellation) -> Result<Messages, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inputs.lock().unwrap().push(input.clone());
        if self.fail.swap(false, Ordering::SeqCst) {
            return Err(Error::new("TRANSPORT", "extract failed"));
        }
        let mut message = Message::new(Role::Assistant);
        message.values = json!({"user":{"fact":"durable fact"}})
            .as_object()
            .unwrap()
            .clone();
        Ok(Messages::new([message]))
    }
    async fn merge(
        &self,
        _: &memory::Item,
        content: &str,
        _: &Cancellation,
    ) -> Result<String, Error> {
        Ok(content.to_owned())
    }
}

#[tokio::test]
async fn long_extraction_retry_uses_same_agent_without_recompressing_successful_summary() {
    let compressor = Compressor::new(Some(100));
    let extractor = LongExtractor::default();
    extractor.fail.store(true, Ordering::SeqCst);
    let config = config()
        .await
        .with_compression(compressor.clone())
        .with_extractor(extractor.clone())
        .with_id_fn(|| 1001);
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(100))).await;
    for id in [202, 203] {
        let _ = run(config.clone(), id, 101).await;
        idle(&config, 101).await;
    }
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(extractor.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        stored(&config).await.summary.as_deref(),
        Some("new summary")
    );
    let fact = config
        .long
        .get(2, None, None, memory::Type::Fact, &Cancellation::new())
        .await
        .unwrap()
        .unwrap();
    assert_eq!((fact.id, fact.from, fact.to), (1001, 201, 202));
    for source in extractor.inputs.lock().unwrap().iter() {
        assert_eq!(contents(source), "question 201|reply 201");
    }
}

#[tokio::test]
async fn successful_summary_without_extractor_does_not_generate_long_memory() {
    let config = config().await.with_compression(Compressor::new(Some(1)));
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(100))).await;
    run(config.clone(), 202, 101).await;
    idle(&config, 101).await;
    assert!(
        config
            .long
            .get(2, None, None, memory::Type::Fact, &Cancellation::new())
            .await
            .unwrap()
            .is_none()
    );
}

struct LostAckSession {
    inner: core::memory::short::session::InMemory,
    fail: std::sync::atomic::AtomicBool,
    malformed: bool,
}
impl session::Store for LostAckSession {
    async fn create(
        &self,
        value: session::Session,
        c: &Cancellation,
    ) -> Result<session::Session, Error> {
        self.inner.create(value, c).await
    }
    async fn read(&self, id: i64, c: &Cancellation) -> Result<session::Session, Error> {
        self.inner.read(id, c).await
    }
    async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, c).await
    }
    async fn update(
        &self,
        value: session::Session,
        c: &Cancellation,
    ) -> Result<session::Session, Error> {
        let changes_summary = self.inner.read(value.id, c).await?.summary != value.summary;
        let mut value = self.inner.update(value, c).await?;
        if changes_summary && self.fail.swap(false, Ordering::SeqCst) {
            if self.malformed {
                value.user_id += 1000;
                return Ok(value);
            }
            return Err(Error::new("BACKEND", "acknowledgement lost"));
        }
        Ok(value)
    }
}

#[tokio::test]
async fn committed_summary_with_lost_ack_retains_extraction_without_recompression() {
    for malformed in [false, true] {
        let compressor = Compressor::new(Some(100));
        let extractor = LongExtractor::default();
        let config = memory::Config::new(
            memory::short::Pair::new(
                LostAckSession {
                    inner: core::memory::short::session::InMemory::new(),
                    fail: std::sync::atomic::AtomicBool::new(true),
                    malformed,
                },
                core::memory::short::chat::InMemory::new(),
            ),
            core::memory::InMemory::new(),
        )
        .with_compression(compressor.clone())
        .with_extractor(extractor.clone())
        .with_id_fn(|| 1001);
        add_session(&config, 101).await;
        seed(&config, 201, 101, chat::Status::Completed, Some(usage(100))).await;
        for id in [202, 203] {
            let _ = run(config.clone(), id, 101).await;
            idle(&config, 101).await;
            assert_eq!(
                stored(&config).await.summary.as_deref(),
                Some("new summary")
            );
            if id == 202 {
                assert_eq!(extractor.calls.load(Ordering::SeqCst), 0);
            }
        }
        assert_eq!(compressor.calls.load(Ordering::SeqCst), 1);
        assert_eq!(extractor.calls.load(Ordering::SeqCst), 1);
        assert!(
            config
                .long
                .get(2, None, None, memory::Type::Fact, &Cancellation::new())
                .await
                .unwrap()
                .is_some()
        );
    }
}

#[tokio::test]
async fn invalid_size_is_rejected_before_io() {
    for size in [None, Some(0)] {
        let compressor = Compressor::new(size);
        let config = config().await.with_compression(compressor.clone());
        let original = stored(&config).await;
        assert_eq!(config.validate().unwrap_err().code, "INVALID_ARGUMENTS");
        assert_eq!(compressor.calls.load(Ordering::SeqCst), 0);
        assert_eq!(stored(&config).await, original);
    }
}

#[tokio::test]
async fn disabled_below_threshold_and_unknown_usage_do_not_start_work() {
    let config = config().await;
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    let original = stored(&config).await;
    run(config.clone(), 301, 101).await;
    let compressor = Compressor::new(Some(100));
    let enabled = config.clone().with_compression(compressor.clone());
    run(enabled.clone(), 302, 101).await;
    idle(&enabled, 101).await;
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(stored(&config).await, original);

    let unknown = self::config()
        .await
        .with_compression(Compressor::new(Some(1)));
    seed(&unknown, 201, 101, chat::Status::Completed, None).await;
    run(unknown.clone(), 301, 101).await;
    idle(&unknown, 101).await;
    assert!(stored(&unknown).await.summary.is_none());
}

#[tokio::test]
async fn background_does_not_block_switch_foreground_or_duplicate_a_session() {
    let compressor = Compressor::new(Some(10)).gated();
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    let original_chat = config.chat().read(201, &Cancellation::new()).await.unwrap();
    let first = run(config.clone(), 301, 101).await;
    entered(&compressor, 1).await;
    let second = run(config.clone(), 302, 101).await;
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 1);
    assert!(stored(&config).await.summary.is_none());
    for messages in [&first, &second] {
        assert!(contents(messages).contains("question 201"));
        assert!(!contents(messages).contains("new summary"));
    }
    compressor.release();
    idle(&config, 101).await;
    let summary = stored(&config).await;
    assert_eq!(summary.summary.as_deref(), Some("new summary"));
    assert_eq!(summary.metadata["from"], 201);
    assert_eq!(summary.metadata["to"], 201);
    assert_eq!(summary.metadata["application"], "keep");
    assert_eq!(summary.title.as_deref(), Some("keep title"));
    assert_eq!(summary.usage, Some(usage(12)));
    assert_eq!(
        config.chat().read(201, &Cancellation::new()).await.unwrap(),
        original_chat
    );
    assert!(
        config
            .long
            .list(2, None, None, memory::Type::Summary, &Cancellation::new())
            .await
            .unwrap()
            .is_empty()
    );
    // Starting later requests is deliberate; the job did not chase Chat 301/302.
    let next = config.clone().with_compression(Compressor::new(Some(1000)));
    seed_current(&next, 301, 101, Some(usage(10))).await;
    seed_current(&next, 302, 101, Some(usage(10))).await;
    let next = contents(&run(next, 303, 101).await);
    assert!(next.contains("new summary"));
    assert!(next.contains("current 301"));
    assert!(next.contains("current 302"));
    assert!(!next.contains("question 201"));
}

#[tokio::test]
async fn rolling_summary_threshold_and_usage_are_replaced_not_accumulated() {
    // Only the uncovered Chat Usage (10) reaches the threshold. The current
    // summary Usage (10) and covered Chat Usage (40) must not be counted again.
    let compressor = Compressor::new(Some(10));
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(40))).await;
    seed(
        &config,
        202,
        101,
        chat::Status::Completed,
        Some(Usage {
            input: Some(5),
            output: Some(5),
            ..Default::default()
        }),
    )
    .await;
    let mut old = stored(&config).await;
    old.summary = Some("old summary".into());
    old.usage = Some(usage(10));
    old.metadata.insert("from".into(), 201.into());
    old.metadata.insert("to".into(), 201.into());
    config
        .session()
        .update(old, &Cancellation::new())
        .await
        .unwrap();
    run(config.clone(), 301, 101).await;
    idle(&config, 101).await;
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 1);
    let stored = stored(&config).await;
    assert_eq!(stored.metadata["from"], 201);
    assert_eq!(stored.metadata["to"], 202);
    assert_eq!(stored.usage, Some(usage(12)));
    let inputs = compressor.inputs.lock().unwrap();
    let input = contents(&inputs[0]);
    assert!(input.contains("old summary"));
    assert!(input.contains("question 202"));
    assert!(!input.contains("question 201"));
    assert!(!input.contains("current 301"));
}

#[tokio::test]
async fn covered_chat_usage_is_not_added_to_the_current_summary_threshold() {
    let compressor = Compressor::new(Some(11));
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(40))).await;
    seed(&config, 202, 101, chat::Status::Completed, Some(usage(10))).await;
    let mut old = stored(&config).await;
    old.summary = Some("old summary".into());
    old.usage = Some(usage(10));
    old.metadata
        .extend([("from".into(), 201.into()), ("to".into(), 201.into())]);
    let old = config
        .session()
        .update(old, &Cancellation::new())
        .await
        .unwrap();

    run(config.clone(), 301, 101).await;
    idle(&config, 101).await;

    // 10 + 10 < 21, even though the already-covered Chat used another 40.
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(stored(&config).await, old);
}

#[tokio::test]
async fn no_new_pairs_never_recompresses_summary_alone() {
    let compressor = Compressor::new(Some(1));
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(100))).await;
    let mut old = stored(&config).await;
    old.summary = Some("old summary".into());
    old.usage = Some(usage(100));
    old.metadata
        .extend([("from".into(), 201.into()), ("to".into(), 201.into())]);
    config
        .session()
        .update(old, &Cancellation::new())
        .await
        .unwrap();
    run(config.clone(), 301, 101).await;
    idle(&config, 101).await;
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn concurrent_writer_wins_without_stale_overwrite_or_retry() {
    let compressor = Compressor::new(Some(1)).gated();
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    run(config.clone(), 301, 101).await;
    entered(&compressor, 1).await;
    let mut latest = stored(&config).await;
    latest.title = Some("concurrent edit".into());
    let latest = config
        .session()
        .update(latest, &Cancellation::new())
        .await
        .unwrap();
    compressor.release();
    idle(&config, 101).await;
    assert_eq!(stored(&config).await, latest);
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 1);
    seed_current(&config, 301, 101, Some(usage(10))).await;
    run(config.clone(), 302, 101).await;
    entered(&compressor, 2).await;
    compressor.release();
    idle(&config, 101).await;
    assert_eq!(
        stored(&config).await.title.as_deref(),
        Some("concurrent edit")
    );
    assert_eq!(stored(&config).await.metadata["to"], 301);
}

#[tokio::test]
async fn failed_component_releases_claim_and_retries_only_on_new_request() {
    let compressor = Compressor::new(Some(1));
    compressor
        .results
        .lock()
        .unwrap()
        .push_back(Err(Error::new("TRANSPORT", "private detail")));
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    let original = stored(&config).await;
    run(config.clone(), 301, 101).await;
    idle(&config, 101).await;
    assert_eq!(stored(&config).await, original);
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 1);
    seed_current(&config, 301, 101, Some(usage(10))).await;
    run(config.clone(), 302, 101).await;
    idle(&config, 101).await;
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 2);
    assert_eq!(stored(&config).await.metadata["to"], 301);
}

#[tokio::test]
async fn earlier_running_chat_is_not_swallowed_by_coverage() {
    let compressor = Compressor::new(Some(1));
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    seed(&config, 202, 101, chat::Status::Running, None).await;
    seed(&config, 203, 101, chat::Status::Completed, Some(usage(10))).await;
    let messages = run(config.clone(), 301, 101).await;
    idle(&config, 101).await;
    assert_eq!(stored(&config).await.metadata["to"], 201);
    assert!(!contents(&compressor.inputs.lock().unwrap()[0]).contains("question 203"));
    assert!(contents(&messages).contains("question 203"));
}

#[tokio::test]
async fn failed_and_stopped_chats_do_not_enter_compression() {
    let compressor = Compressor::new(Some(1));
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Failed, Some(usage(100))).await;
    seed(&config, 202, 101, chat::Status::Stopped, Some(usage(100))).await;
    run(config.clone(), 301, 101).await;
    idle(&config, 101).await;
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 0);
    seed(&config, 401, 101, chat::Status::Completed, Some(usage(10))).await;
    run(config.clone(), 501, 101).await;
    idle(&config, 101).await;
    assert!(!contents(&compressor.inputs.lock().unwrap()[0]).contains("question 201"));
    assert!(!contents(&compressor.inputs.lock().unwrap()[0]).contains("question 202"));
}

#[tokio::test]
async fn independent_sessions_can_compress_in_parallel() {
    let compressor = Compressor::new(Some(1)).gated();
    let config = config().await.with_compression(compressor.clone());
    add_session(&config, 102).await;
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    seed(&config, 202, 102, chat::Status::Completed, Some(usage(10))).await;
    run(config.clone(), 301, 101).await;
    run(config.clone(), 302, 102).await;
    entered(&compressor, 2).await;
    compressor.release();
    compressor.release();
    idle(&config, 101).await;
    idle(&config, 102).await;
    assert!(stored(&config).await.summary.is_some());
    assert!(
        config
            .session()
            .read(102, &Cancellation::new())
            .await
            .unwrap()
            .summary
            .is_some()
    );
}

#[tokio::test]
async fn foreground_cancellation_does_not_cancel_background_compression() {
    let compressor = Compressor::new(Some(1)).gated();
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    let cancellation = Cancellation::new();
    let _ =
        run_with_hooks_and_cancellation(config.clone(), 301, 101, Hooks::default(), &cancellation)
            .await;
    entered(&compressor, 1).await;
    cancellation.cancel();
    compressor.release();
    idle(&config, 101).await;
    assert_eq!(
        stored(&config).await.summary.as_deref(),
        Some("new summary")
    );
}

#[derive(Clone)]
struct Hook {
    name: &'static str,
    log: Arc<Mutex<Vec<String>>>,
    fail: bool,
}
impl Middleware for Hook {
    async fn before_compress_context(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.log
            .lock()
            .unwrap()
            .push(format!("before:{}", self.name));
        let local = Arc::new(1);
        tokio::task::yield_now().await;
        assert_eq!(*local, 1);
        if self.fail {
            return Err(Error::new("INVALID_ARGUMENTS", "hook failed"));
        }
        Ok(())
    }
    async fn after_compress_context(
        &self,
        output: &mut Value,
        _: &Cancellation,
    ) -> Result<(), Error> {
        self.log
            .lock()
            .unwrap()
            .push(format!("after:{}", self.name));
        output[0]["content"][0]["data"]["value"] = json!("hook summary");
        output[0]["usage"] = serde_json::to_value(usage(999)).unwrap();
        Ok(())
    }
}

#[tokio::test]
async fn compression_hooks_run_in_order_without_corrupting_usage() {
    let compressor = Compressor::new(Some(1));
    let config = config().await.with_compression(compressor);
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut hooks = Hooks::default();
    hooks.add(Hook {
        name: "a",
        log: log.clone(),
        fail: false,
    });
    hooks.add(Hook {
        name: "b",
        log: log.clone(),
        fail: false,
    });
    let _ = run_with_hooks(config.clone(), 301, 101, hooks).await;
    idle(&config, 101).await;
    assert_eq!(
        *log.lock().unwrap(),
        ["before:a", "before:b", "after:b", "after:a"]
    );
    assert_eq!(
        stored(&config).await.summary.as_deref(),
        Some("hook summary")
    );
    assert_eq!(stored(&config).await.usage, Some(usage(12)));
}

#[tokio::test]
async fn compression_hook_error_never_fails_foreground_or_commits_summary() {
    let compressor = Compressor::new(Some(1));
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    let original = stored(&config).await;
    let mut hooks = Hooks::default();
    hooks.add(Hook {
        name: "fail",
        log: Arc::new(Mutex::new(Vec::new())),
        fail: true,
    });
    let _ = run_with_hooks(config.clone(), 301, 101, hooks).await;
    idle(&config, 101).await;
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(stored(&config).await, original);
}

#[tokio::test]
async fn invalid_compression_output_leaves_session_unchanged() {
    for output in [
        Messages::default(),
        Messages::new([text(Role::User, "wrong role")]),
        Messages::new([text(Role::Assistant, "  ")]),
        Messages::new([Message::function(json!({"summary":"wrong contract"}))]),
    ] {
        let compressor = Compressor::new(Some(1));
        compressor.results.lock().unwrap().push_back(Ok(output));
        let config = config().await.with_compression(compressor.clone());
        seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
        let original = stored(&config).await;
        run(config.clone(), 301, 101).await;
        idle(&config, 101).await;
        assert_eq!(stored(&config).await, original);
        assert_eq!(compressor.calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn history_load_uses_the_same_background_lifecycle_for_all_agents() {
    let compressor = Compressor::new(Some(1));
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    let _ = run(config.clone(), 301, 101).await;
    idle(&config, 101).await;
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        stored(&config).await.summary.as_deref(),
        Some("new summary")
    );
}

#[test]
fn summary_extracts_text_and_hides_think() {
    let mut result = output("actual summary", 3);
    result.0[0].content.insert(
        0,
        Part {
            r#type: "think".into(),
            data: crate::components::model::content::part::object(
                json!({"value":"private thinking"}),
            ),
        },
    );
    result.0[0]
        .usage
        .as_mut()
        .unwrap()
        .details
        .insert("provider_fact".into(), true.into());
    assert_eq!(summary(&result).unwrap(), "actual summary");
}

#[test]
fn token_threshold_fallback_and_overflow_do_not_fabricate_usage() {
    assert_eq!(tokens(None), 0);
    assert_eq!(
        tokens(Some(&Usage {
            input: Some(7),
            ..Default::default()
        })),
        7
    );
    assert_eq!(
        tokens(Some(&Usage {
            input: Some(u64::MAX),
            output: Some(1),
            ..Default::default()
        })),
        u64::MAX
    );
    assert_eq!(
        tokens(Some(&Usage {
            total: Some(5),
            input: Some(9),
            output: Some(9),
            ..Default::default()
        })),
        5
    );
}

#[tokio::test]
async fn newer_snapshot_is_skipped_before_compressor_io() {
    let compressor = Compressor::new(Some(1));
    let config = config().await.with_compression(compressor.clone());
    let old = stored(&config).await;
    let mut latest = old.clone();
    latest.title = Some("newer".into());
    let latest = config
        .session()
        .update(latest, &Cancellation::new())
        .await
        .unwrap();
    start(
        Arc::new(config.clone()),
        old,
        vec![text(Role::User, "old input")],
        Some((201, 201)),
        10,
        Hooks::default(),
        extraction::Work {
            agent_id: 11,
            input: Messages::default(),
        },
    );
    idle(&config, 101).await;
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(stored(&config).await, latest);
}

#[tokio::test]
async fn replacing_configured_component_cannot_bypass_session_single_flight() {
    let first = Compressor::new(Some(1)).gated();
    let second = Compressor::new(Some(1));
    let config = config().await.with_compression(first.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    run(config.clone(), 301, 101).await;
    entered(&first, 1).await;
    let replaced = config.clone().with_compression(second.clone());
    run(replaced.clone(), 302, 101).await;
    assert_eq!(second.calls.load(Ordering::SeqCst), 0);
    first.release();
    idle(&config, 101).await;
    seed_current(&config, 301, 101, Some(usage(10))).await;
    seed_current(&config, 302, 101, Some(usage(10))).await;
    run(replaced, 303, 101).await;
    idle(&config, 101).await;
    assert_eq!(second.calls.load(Ordering::SeqCst), 1);
    assert_eq!(first.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn incomplete_completed_pair_does_not_create_a_coverage_hole() {
    let compressor = Compressor::new(Some(1));
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    let mut incomplete = config.chat().read(201, &Cancellation::new()).await.unwrap();
    incomplete.message = None;
    config
        .chat()
        .update(incomplete, &Cancellation::new())
        .await
        .unwrap();
    seed(&config, 202, 101, chat::Status::Completed, Some(usage(10))).await;
    let messages = run(config.clone(), 301, 101).await;
    idle(&config, 101).await;
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 0);
    assert!(contents(&messages).contains("question 202"));
}

struct AfterError;
impl Middleware for AfterError {
    async fn after_compress_context(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        Err(Error::new("INVALID_ARGUMENTS", "after hook failed"))
    }
}

#[tokio::test]
async fn after_hook_failure_does_not_commit_partial_summary() {
    let compressor = Compressor::new(Some(1));
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    let original = stored(&config).await;
    let mut hooks = Hooks::default();
    hooks.add(AfterError);
    let _ = run_with_hooks(config.clone(), 301, 101, hooks).await;
    idle(&config, 101).await;
    assert_eq!(compressor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(stored(&config).await, original);
}

#[tokio::test]
async fn long_recall_and_current_prompt_are_not_session_compression_input() {
    let compressor = Compressor::new(Some(1));
    let config = config()
        .await
        .with_compression(compressor.clone())
        .with_scopes(["user"]);
    config
        .long
        .write(
            memory::Request {
                id: 401,
                r#type: memory::Type::Preference,
                user_id: 2,
                agent_id: None,
                session_id: None,
                from: 1,
                to: 2,
                content: "long preference".into(),
                metadata: None,
                vector: None,
            },
            &Cancellation::new(),
        )
        .await
        .unwrap();
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    let messages = run(config.clone(), 301, 101).await;
    idle(&config, 101).await;
    let background = contents(&compressor.inputs.lock().unwrap()[0]);
    assert!(background.contains("question 201"));
    assert!(!background.contains("long preference"));
    assert!(!background.contains("system instruction"));
    assert!(!background.contains("current 301"));
    assert!(contents(&messages).contains("long preference"));
}

#[tokio::test]
async fn current_single_invocation_keeps_provider_usage_details() {
    let compressor = Compressor::new(Some(1));
    let mut result = output("summary", 3);
    result.0[0]
        .usage
        .as_mut()
        .unwrap()
        .details
        .insert("provider_fact".into(), true.into());
    let expected = result.0[0].usage.clone();
    compressor.results.lock().unwrap().push_back(Ok(result));
    let config = config().await.with_compression(compressor.clone());
    seed(&config, 201, 101, chat::Status::Completed, Some(usage(10))).await;
    run(config.clone(), 301, 101).await;
    idle(&config, 101).await;
    assert_eq!(stored(&config).await.usage, expected);
}
