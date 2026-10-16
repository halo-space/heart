//! A test-only application backend persists JSON; no production storage API.
//! Each recovery phase runs in a different OS process with fresh Agent objects.
use chat::Store as _;
use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message, Messages,
    agent::{self, Agent, Input, Payload},
    compression::Compressor,
    core,
    event::Event,
    memory::{
        self, Memory,
        short::{self, chat, session},
    },
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
        token::Usage,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicI64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Deserialize, Serialize)]
struct Data {
    case: String,
    session: session::Session,
    items: Vec<memory::Item>,
    counts: BTreeMap<String, usize>,
    failures: usize,
}
impl Data {
    fn count(&mut self, name: &str) {
        *self.counts.entry(name.into()).or_default() += 1;
    }
    fn amount(&self, name: &str) -> usize {
        self.counts.get(name).copied().unwrap_or(0)
    }
}
#[derive(Clone)]
struct Disk(Arc<(PathBuf, Mutex<Data>)>);
impl Disk {
    fn open(path: PathBuf) -> Self {
        let data = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        Self(Arc::new((path, Mutex::new(data))))
    }
    fn change<R>(&self, action: impl FnOnce(&mut Data) -> R) -> R {
        let mut data = self.0.1.lock().unwrap();
        let result = action(&mut data);
        let temp = self.0.0.with_extension("next");
        std::fs::write(&temp, serde_json::to_vec(&*data).unwrap()).unwrap();
        std::fs::rename(temp, &self.0.0).unwrap();
        result
    }
    fn snapshot(&self) -> Data {
        self.0.1.lock().unwrap().clone()
    }
}
fn failure(data: &mut Data) -> Error {
    data.failures += 1;
    Error::new("BACKEND", "injected failure")
}
fn unsupported<T>() -> Result<T, Error> {
    Err(Error::new("UNSUPPORTED", "unused test operation"))
}

impl session::Store for Disk {
    async fn create(
        &self,
        _: session::Session,
        _: &Cancellation,
    ) -> Result<session::Session, Error> {
        unsupported()
    }
    async fn delete(&self, _: i64, _: &Cancellation) -> Result<(), Error> {
        unsupported()
    }
    async fn read(&self, id: i64, _: &Cancellation) -> Result<session::Session, Error> {
        let data = self.snapshot();
        assert_eq!(id, data.session.id);
        Ok(data.session)
    }
    async fn update(
        &self,
        mut value: session::Session,
        _: &Cancellation,
    ) -> Result<session::Session, Error> {
        let result = self.change(|data| {
            if value.updated_time != data.session.updated_time {
                return Err(Error::new("CONFLICT", "stale scene"));
            }
            let next = value.metadata.get("memory");
            let previous = data.session.metadata.get("memory");
            let summary_commit = value.summary != data.session.summary;
            let prepared = next
                .and_then(|v| v.pointer("/job/tasks/0/prepared"))
                .is_some_and(|v| !v.is_null());
            let done = next
                .and_then(|v| v.pointer("/job/tasks/0/done"))
                .and_then(Value::as_bool)
                == Some(true);
            let inject = data.failures == 0
                && match data.case.as_str() {
                    "summary_commit_fail" | "superseded_summary" => summary_commit,
                    "prepare_save_fail" => prepared,
                    "done_save_fail" => done,
                    "cleanup_fail" => next.is_none() && previous.is_some(),
                    _ => false,
                };
            if inject {
                return Err(failure(data));
            }
            if data.case == "progress_conflict" && prepared && data.failures == 0 {
                data.session.title = Some("concurrent".into());
                data.session.updated_time += 1;
                data.failures += 1;
                return Err(Error::new("CONFLICT", "concurrent writer"));
            }
            value.updated_time += 1;
            data.session = value.clone();
            if data.failures == 0
                && ((data.case == "summary_lost_ack" && summary_commit)
                    || (data.case == "metadata_lost_ack" && !summary_commit))
            {
                return Err(failure(data));
            }
            Ok(value)
        });
        let data = self.snapshot();
        let first = std::env::var("HEART_MEMORY_PHASE").unwrap() == "1";
        if first
            && data.case == "crash_after_summary"
            && data.session.summary.is_some()
            && data.amount("extract") == 0
        {
            std::process::exit(74);
        }
        if first
            && data.case == "crash_after_extract"
            && data
                .session
                .metadata
                .get("memory")
                .and_then(|v| v.pointer("/job/tasks"))
                .is_some_and(Value::is_array)
            && data.amount("fact_write") == 0
        {
            std::process::exit(75);
        }
        result
    }
}
impl Memory for Disk {
    async fn write(
        &self,
        request: memory::Request,
        _: &Cancellation,
    ) -> Result<memory::Item, Error> {
        let fact = request.r#type == memory::Type::Fact;
        let result = self.change(|data| {
            data.count(if fact {
                "fact_write"
            } else {
                "preference_write"
            });
            if fact && data.case == "write_fail" && data.failures == 0 {
                return Err(failure(data));
            }
            if data.items.iter().any(|i| i.id == request.id) {
                return Err(Error::new("CONFLICT", "duplicate ID"));
            }
            let item = item(request, 1);
            data.items.push(item.clone());
            if fact && data.case == "lost_ack" && data.failures == 0 {
                return Err(failure(data));
            }
            Ok(item)
        });
        if fact
            && self.snapshot().case == "crash_after_write"
            && std::env::var("HEART_MEMORY_PHASE").unwrap() == "1"
        {
            std::process::exit(73);
        }
        result
    }
    async fn update(
        &self,
        request: memory::Request,
        version: u64,
        _: &Cancellation,
    ) -> Result<memory::Item, Error> {
        self.change(|data| {
            data.count("fact_update");
            if data.case == "merge_write_fail" && data.failures == 0 {
                return Err(failure(data));
            }
            let position = data.items.iter().position(|i| i.id == request.id).unwrap();
            if data.items[position].version != version {
                return Err(Error::new("CONFLICT", "stale item"));
            }
            let updated = item(request, version + 1);
            data.items[position] = updated.clone();
            Ok(updated)
        })
    }
    async fn delete(&self, _: i64, _: &Cancellation) -> Result<(), Error> {
        unsupported()
    }
    async fn get(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        kind: memory::Type,
        _: &Cancellation,
    ) -> Result<Option<memory::Item>, Error> {
        Ok(self.snapshot().items.into_iter().find(|i| {
            i.user_id == user_id
                && i.agent_id == agent_id
                && i.session_id == session_id
                && i.r#type == kind
        }))
    }
    async fn list(
        &self,
        _: i64,
        _: Option<i64>,
        _: Option<i64>,
        _: memory::Type,
        _: &Cancellation,
    ) -> Result<Vec<memory::Item>, Error> {
        unsupported()
    }
    async fn keyword(
        &self,
        _: memory::keyword::Request,
        _: &Cancellation,
    ) -> Result<Vec<memory::Item>, Error> {
        unsupported()
    }
    async fn vector(
        &self,
        _: memory::vector::Request,
        _: &Cancellation,
    ) -> Result<Vec<memory::Item>, Error> {
        unsupported()
    }
    async fn hybrid(
        &self,
        _: memory::hybrid::Request,
        _: &Cancellation,
    ) -> Result<Vec<memory::Item>, Error> {
        unsupported()
    }
}
fn item(request: memory::Request, version: u64) -> memory::Item {
    memory::Item {
        id: request.id,
        r#type: request.r#type,
        user_id: request.user_id,
        agent_id: request.agent_id,
        session_id: request.session_id,
        from: request.from,
        to: request.to,
        content: request.content,
        version,
        created_time: 1,
        updated_time: version as i64,
        metadata: request.metadata,
    }
}
fn text(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":value}).as_object().unwrap().clone(),
    });
    message
}
fn usage() -> Usage {
    Usage {
        input: Some(50),
        output: Some(50),
        total: Some(100),
        ..Default::default()
    }
}
struct Model;
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        assert!(
            request
                .messages
                .0
                .iter()
                .all(|m| !m.metadata.contains_key("memory"))
        );
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: text(Role::Assistant, "reply"),
            usage: Some(usage()),
            finish_reason: None,
        })]))
    }
}
struct Compression(Disk);
impl Compressor for Compression {
    fn size(&self) -> Option<u64> {
        Some(1)
    }
    async fn compress(&self, _: Messages, _: &Cancellation) -> Result<Messages, Error> {
        self.0.change(|data| data.count("compress"));
        let mut message = text(Role::Assistant, "new summary");
        message.usage = Some(usage());
        Ok(Messages::new([message]))
    }
}
struct Extraction(Disk);
impl memory::extraction::Extractor for Extraction {
    async fn extract(&self, input: &Messages, _: &Cancellation) -> Result<Messages, Error> {
        assert!(
            input
                .0
                .iter()
                .all(|m| m.usage.is_none() && m.metadata.is_empty() && m.values.is_empty())
        );
        self.0.change(|data| {
            data.count("extract");
            if data.case == "extract_fail" && data.failures == 0 {
                return Err(failure(data));
            }
            Ok(())
        })?;
        let mut message = Message::new(Role::Assistant);
        message.values = json!({"user":{"fact":"new fact"},"agent":{"preference":"concise"}})
            .as_object()
            .unwrap()
            .clone();
        Ok(Messages::new([message]))
    }
    async fn merge(
        &self,
        current: &memory::Item,
        new: &str,
        _: &Cancellation,
    ) -> Result<String, Error> {
        self.0.change(|data| data.count("merge"));
        Ok(format!("{} + {new}", current.content))
    }
}
static IDS: AtomicI64 = AtomicI64::new(7001);
fn next_id() -> i64 {
    let phase = std::env::var("HEART_MEMORY_PHASE").unwrap();
    let case = std::env::var("HEART_MEMORY_CASE").unwrap();
    if phase == "2"
        && matches!(
            case.as_str(),
            "write_fail" | "lost_ack" | "merge_write_fail" | "cleanup_fail"
        )
    {
        panic!("all IDs were already prepared; recovery must reuse them");
    }
    IDS.fetch_add(1, Ordering::SeqCst)
}

#[tokio::test]
#[ignore = "subprocess fixture; invoked by the recovery tests"]
async fn memory_worker() {
    let phase: usize = std::env::var("HEART_MEMORY_PHASE")
        .unwrap()
        .parse()
        .unwrap();
    let path = PathBuf::from(std::env::var_os("HEART_MEMORY_FILE").unwrap());
    IDS.store(if phase == 1 { 7001 } else { 9001 }, Ordering::SeqCst);
    let disk = Disk::open(path);
    let config = memory::Config::new(
        disk.clone(),
        core::memory::short::chat::InMemory::new(),
        core::memory::short::trace::InMemory::new(),
        disk.clone(),
    )
    .with_compression(Compression(disk.clone()))
    .with_extractor(Extraction(disk.clone()))
    .with_id_fn(next_id);
    config
        .chat
        .create(
            chat::Chat {
                id: 201,
                tenant_id: 1,
                user_id: 2,
                agent_id: 3,
                session_id: 101,
                r#type: short::Type::Agent,
                status: chat::Status::Completed,
                input: Some(chat::Request {
                    metadata: Map::new(),
                    payload: vec![chat::Payload {
                        r#type: "text".into(),
                        data: json!({"value":"question"}),
                        mime_type: None,
                    }],
                }),
                message: Some(text(Role::Assistant, "historical answer")),
                usage: Some(usage()),
                graph: None,
                metadata: Map::new(),
                started_time: None,
                completed_time: None,
                created_time: 1,
                updated_time: 1,
            },
            &Cancellation::new(),
        )
        .await
        .unwrap();
    let agent = agent::chat().model(Model).memory(config).build().unwrap();
    let events = agent.run(
    Input {
        metadata: json!({"tenant_id":1,"user_id":2,"agent_id":3,"session_id":101,"chat_id":300 + phase}).as_object().unwrap().clone(),
        payload: vec![Payload { r#type: "text".into(), data: json!({"value":"current"}), mime_type: None }],
    },
    true,
    Cancellation::new(),
    ).await.unwrap().collect::<Vec<_>>().await;
    assert!(events.iter().all(Result::is_ok));
    tokio::time::timeout(Duration::from_secs(4), async {
        loop {
            let data = disk.snapshot();
            if phase == 2 && !data.session.metadata.contains_key("memory") {
                break;
            }
            if phase == 1 && data.failures > 0 {
                let needs_preference = matches!(
                    data.case.as_str(),
                    "write_fail" | "lost_ack" | "merge_write_fail"
                );
                let preference_done = data
                    .session
                    .metadata
                    .get("memory")
                    .and_then(|v| v.pointer("/job/tasks/1/done"))
                    == Some(&Value::Bool(true));
                if !needs_preference || preference_done {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

fn exercise(case: &str) -> Data {
    let root = std::env::temp_dir().join(format!(
        "heart-memory-{}-{}-{case}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let file = root.join("scene.json");
    let mut data = Data {
        case: case.into(),
        session: session::Session {
            id: 101,
            tenant_id: 1,
            user_id: 2,
            r#type: short::Type::Agent,
            title: Some("keep".into()),
            summary: None,
            usage: None,
            status: session::Status::Active,
            metadata: json!({"application":"keep"}).as_object().unwrap().clone(),
            created_time: 1,
            updated_time: 1,
        },
        items: vec![],
        counts: BTreeMap::new(),
        failures: 0,
    };
    if case == "merge_write_fail" {
        data.items.push(item(
            memory::Request {
                id: 6001,
                r#type: memory::Type::Fact,
                user_id: 2,
                agent_id: None,
                session_id: None,
                from: 100,
                to: 150,
                content: "old fact".into(),
                metadata: None,
                vector: None,
            },
            1,
        ));
    }
    std::fs::write(&file, serde_json::to_vec(&data).unwrap()).unwrap();
    for phase in [1, 2] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "memory_worker", "--ignored", "--nocapture"])
            .env("HEART_MEMORY_FILE", &file)
            .env("HEART_MEMORY_CASE", case)
            .env("HEART_MEMORY_PHASE", phase.to_string())
            .output()
            .unwrap();
        let expected = if phase == 1 {
            match case {
                "crash_after_write" => 73,
                "crash_after_summary" => 74,
                "crash_after_extract" => 75,
                _ => 0,
            }
        } else {
            0
        };
        assert_eq!(
            output.status.code(),
            Some(expected),
            "case={case}, phase={phase}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if phase == 1 {
            let mut saved: Data = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
            assert!(saved.session.metadata.contains_key("memory"));
            if matches!(case, "prepare_save_fail" | "summary_commit_fail") {
                assert_eq!(saved.amount("fact_write"), 0);
            }
            if case == "superseded_summary" {
                saved.session.summary = Some("external summary".into());
                saved.session.usage = Some(usage());
                saved.session.metadata.insert("from".into(), 201.into());
                saved.session.metadata.insert("to".into(), 201.into());
                saved.session.updated_time += 1;
                std::fs::write(&file, serde_json::to_vec(&saved).unwrap()).unwrap();
            }
        }
    }
    let result: Data = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    assert!(!result.session.metadata.contains_key("memory"));
    assert_eq!(result.session.metadata["application"], "keep");
    assert_eq!(result.session.metadata["from"], 201);
    assert_eq!(result.session.metadata["to"], 201);
    assert_eq!(
        result.session.title.as_deref(),
        Some(if case == "progress_conflict" {
            "concurrent"
        } else {
            "keep"
        })
    );
    assert_eq!(
        result.session.summary.as_deref(),
        Some(if case == "superseded_summary" {
            "external summary"
        } else {
            "new summary"
        })
    );
    assert_eq!(result.amount("compress"), 1);
    assert_eq!(
        result.amount("extract"),
        match case {
            "extract_fail" => 2,
            "superseded_summary" => 0,
            _ => 1,
        }
    );
    assert_eq!(
        result.amount("preference_write"),
        usize::from(case != "superseded_summary")
    );
    std::fs::remove_dir_all(root).unwrap();
    result
}

#[test]
fn independently_restarted_agents_resume_each_durable_stage() {
    for case in [
        "extract_fail",
        "write_fail",
        "merge_write_fail",
        "lost_ack",
        "summary_commit_fail",
        "summary_lost_ack",
        "metadata_lost_ack",
        "prepare_save_fail",
        "progress_conflict",
        "done_save_fail",
        "cleanup_fail",
    ] {
        let data = exercise(case);
        assert_eq!(data.items.len(), 2, "{case}");
        assert_eq!(
            data.amount("fact_write"),
            match case {
                "write_fail" => 2,
                "merge_write_fail" => 0,
                _ => 1,
            },
            "{case}"
        );
        if case == "merge_write_fail" {
            assert_eq!(data.amount("merge"), 1);
            assert_eq!(data.amount("fact_update"), 2);
            assert_eq!(data.items[0].content, "old fact + new fact");
        }
    }
}

#[test]
fn hard_process_exit_recovers_committed_summary_extraction_and_write_facts() {
    for case in [
        "crash_after_write",
        "crash_after_summary",
        "crash_after_extract",
    ] {
        let data = exercise(case);
        assert_eq!(data.amount("fact_write"), 1);
        assert_eq!(data.items.len(), 2);
        if case == "crash_after_write" {
            assert_eq!(data.items[0].id, 7001);
        }
    }
}

#[test]
fn prepared_summary_cannot_overwrite_a_superseding_summary() {
    let data = exercise("superseded_summary");
    assert_eq!(data.amount("fact_write"), 0);
    assert!(data.items.is_empty());
    assert_eq!(data.session.usage, Some(usage()));
}
