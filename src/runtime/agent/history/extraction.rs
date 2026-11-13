//! Stage-preserving long Memory processing. Production progress is saved in
//! Session metadata before external writes and after acknowledged completion.
use crate::cancellation::cancellable;
use crate::runtime::observation::observe_usage;
use crate::runtime::trace::fastrace::Root;
use crate::{
    Cancellation, Error, Messages,
    memory::{self, Item, Memory, Request},
    message::Role,
};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[cfg(test)]
#[derive(Default)]
pub(super) struct Jobs(Mutex<BTreeMap<i64, Job>>);

pub(super) struct Work {
    pub(super) agent_id: i64,
    pub(super) input: Messages,
}

#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Job {
    user_id: i64,
    agent_id: i64,
    from: i64,
    to: i64,
    input: Messages,
    tasks: Option<Vec<Task>>,
}
#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Task {
    agent_id: Option<i64>,
    kind: memory::Type,
    content: String,
    prepared: Option<Prepared>,
    done: bool,
}
#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Prepared {
    previous: Option<Item>,
    request: Request,
}

#[cfg(test)]
impl Jobs {
    pub(super) fn contains(&self, session_id: i64) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&session_id)
    }
    pub(super) fn insert(&self, session_id: i64, job: Job) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(session_id)
            .or_insert(job);
    }
}
impl Job {
    pub(super) fn new(
        user_id: i64,
        agent_id: i64,
        from: i64,
        to: i64,
        input: Messages,
    ) -> Result<Self, Error> {
        // Session coverage endpoints are inclusive IDs; long-memory provenance
        // is the established half-open interval. Do not generate a new ID here.
        let to = to
            .checked_add(1)
            .filter(|to| *to > from)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "invalid long memory source range"))?;
        let input = Messages::new(input.0.into_iter().filter_map(|mut message| {
            if !matches!(message.role, Role::User | Role::Assistant) {
                return None;
            }
            message.content.retain(|part| part.r#type == "text");
            message.values.clear();
            message.metadata.clear();
            message.usage = None;
            (!message.content.is_empty()).then_some(message)
        }));
        Ok(Self {
            user_id,
            agent_id,
            from,
            to,
            input,
            tasks: None,
        })
    }
    pub(super) fn covers(&self, from: i64, to: i64) -> bool {
        self.from == from && to.checked_add(1) == Some(self.to)
    }
    pub(super) fn validate(&self, user_id: i64) -> Result<(), Error> {
        if self.user_id != user_id
            || user_id <= 0
            || self.agent_id <= 0
            || self.from <= 0
            || self.to <= self.from
        {
            return Err(invalid());
        }
        if let Some(tasks) = &self.tasks {
            if !self.input.0.is_empty() || tasks.len() > 6 {
                return Err(invalid());
            }
            let mut seen = std::collections::BTreeSet::new();
            for task in tasks {
                if task.agent_id.is_some_and(|id| id != self.agent_id)
                    || task.content.trim().is_empty()
                    || !seen.insert((task.agent_id, format!("{:?}", task.kind)))
                    || (task.done && task.prepared.is_none())
                {
                    return Err(invalid());
                }
                if let Some(prepared) = &task.prepared {
                    let request = &prepared.request;
                    if request.id <= 0
                        || request.user_id != user_id
                        || request.agent_id != task.agent_id
                        || request.session_id.is_some()
                        || request.r#type != task.kind
                        || request.content.trim().is_empty()
                        || request.from > self.from
                        || request.from < 0
                        || request.to < self.to
                        || request.vector.as_ref().is_some_and(|vector| {
                            vector.is_empty() || vector.iter().any(|value| !value.is_finite())
                        })
                    {
                        return Err(invalid());
                    }
                    if let Some(previous) = &prepared.previous
                        && (previous.id != request.id
                            || previous.version == 0
                            || previous.user_id != user_id
                            || previous.agent_id != task.agent_id
                            || previous.session_id.is_some()
                            || previous.r#type != task.kind
                            || previous.content.trim().is_empty()
                            || previous.from < 0
                            || previous.to <= previous.from)
                    {
                        return Err(invalid());
                    }
                }
            }
        } else if self.input.0.iter().any(|message| {
            !matches!(message.role, Role::User | Role::Assistant)
                || !message.values.is_empty()
                || !message.metadata.is_empty()
                || message.usage.is_some()
                || message.content.iter().any(|part| part.r#type != "text")
        }) {
            return Err(invalid());
        }
        Ok(())
    }
}

pub(super) trait Progress {
    fn save(
        &mut self,
        job: &Job,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<(), Error>> + Send;
}

/// Drop requeues failures/panics; no mutex is held while a component awaits.
#[cfg(test)]
struct Ticket {
    jobs: Arc<Jobs>,
    session_id: i64,
    job: Option<Job>,
}
#[cfg(test)]
impl Drop for Ticket {
    fn drop(&mut self) {
        if let Some(job) = self.job.take() {
            self.jobs.insert(self.session_id, job);
        }
    }
}

#[cfg(test)]
async fn resume<L: Memory>(
    jobs: Arc<Jobs>,
    session_id: i64,
    component: &memory::extraction::Configured,
    store: &L,
    next_id: fn() -> i64,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    let job = jobs
        .0
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&session_id);
    let Some(job) = job else {
        return Ok(());
    };
    let mut ticket = Ticket {
        jobs,
        session_id,
        job: Some(job),
    };
    struct Local;
    impl Progress for Local {
        async fn save(&mut self, _: &Job, _: &Cancellation) -> Result<(), Error> {
            Ok(())
        }
    }
    resume_job(
        ticket.job.as_mut().unwrap(),
        &mut Local,
        component,
        store,
        None,
        next_id,
        cancellation,
    )
    .await?;
    ticket.job = None;
    Ok(())
}

pub(super) async fn resume_job<L: Memory, P: Progress>(
    job: &mut Job,
    progress: &mut P,
    component: &memory::extraction::Configured,
    store: &L,
    vectorizer: Option<memory::Vectorizer>,
    next_id: fn() -> i64,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    // Extraction is a background model call without a foreground Event. Its
    // Usage is audited separately and is never added to the Chat ledger.
    let trace = Root::start("memory.extraction");
    if job.tasks.is_none() {
        let output = cancellable(
            component.extract(job.input.clone(), cancellation),
            cancellation,
        )
        .await?;
        let [message] = output.as_slice() else {
            return Err(invalid());
        };
        observe_usage(&trace, "memory.extraction.usage", message.usage.as_ref());
        if message.role != Role::Assistant || !message.content.is_empty() {
            return Err(invalid());
        }
        memory::extraction::validate(&message.values)?;
        let mut tasks = Vec::new();
        for (level, types) in &message.values {
            for (kind, content) in types.as_object().unwrap() {
                let kind = match kind.as_str() {
                    "summary" => memory::Type::Summary,
                    "fact" => memory::Type::Fact,
                    "preference" => memory::Type::Preference,
                    _ => unreachable!("validated kind"),
                };
                tasks.push(Task {
                    agent_id: (level == "agent").then_some(job.agent_id),
                    kind,
                    content: content.as_str().unwrap().to_owned(),
                    prepared: None,
                    done: false,
                });
            }
        }
        job.tasks = Some(tasks);
        // A successful extraction is never repeated on a write retry. Retain
        // the smaller extracted content, not another copy of raw history.
        job.input = Messages::default();
        progress.save(job, cancellation).await?;
    }
    let mut error = None;
    for index in 0..job.tasks.as_ref().unwrap().len() {
        if job.tasks.as_ref().unwrap()[index].done {
            continue;
        }
        let prepared = prepare(
            &mut job.tasks.as_mut().unwrap()[index],
            job.user_id,
            job.from,
            job.to,
            component,
            store,
            vectorizer.as_ref(),
            &trace,
            next_id,
            cancellation,
        )
        .await;
        match prepared {
            Ok(already_committed) => {
                if already_committed {
                    job.tasks.as_mut().unwrap()[index].done = true;
                }
                // Retain successful merge and the allocated ID before writing.
                // A failed progress save must stop before the external effect.
                progress.save(job, cancellation).await?;
                if !already_committed {
                    let result =
                        write_prepared(&job.tasks.as_ref().unwrap()[index], store, cancellation)
                            .await;
                    match result {
                        Ok(()) => {
                            job.tasks.as_mut().unwrap()[index].done = true;
                            progress.save(job, cancellation).await?;
                        }
                        Err(value) => {
                            error.get_or_insert(value);
                        }
                    }
                }
            }
            Err(value) => {
                error.get_or_insert(value);
            }
        }
        if cancellation.is_cancelled() {
            break;
        }
    }
    if let Some(error) = error {
        return Err(error);
    }
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "memory writeback cancelled"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn prepare<L: Memory>(
    task: &mut Task,
    user_id: i64,
    from: i64,
    to: i64,
    component: &memory::extraction::Configured,
    store: &L,
    vectorizer: Option<&memory::Vectorizer>,
    trace: &Root,
    next_id: fn() -> i64,
    cancellation: &Cancellation,
) -> Result<bool, Error> {
    let current = cancellable(
        store.get(
            user_id,
            task.agent_id,
            None,
            task.kind.clone(),
            cancellation,
        ),
        cancellation,
    )
    .await?;
    if let Some(item) = &current
        && (item.id <= 0
            || item.version == 0
            || item.user_id != user_id
            || item.agent_id != task.agent_id
            || item.session_id.is_some()
            || item.r#type != task.kind
            || item.from < 0
            || item.to <= item.from
            || item.content.trim().is_empty())
    {
        return Err(invalid());
    }
    if let Some(prepared) = &task.prepared {
        let version = match &prepared.previous {
            Some(previous) => previous.version.checked_add(1),
            None => Some(1),
        };
        // A lost acknowledgement must not merge the same extracted text twice.
        if current
            .as_ref()
            .is_some_and(|item| Some(item.version) == version && matches(item, &prepared.request))
        {
            return Ok(true);
        }
    }
    if let (Some(prepared), Some(vectorizer)) = (task.prepared.as_mut(), vectorizer)
        && prepared.previous == current
        && prepared.request.vector.is_none()
    {
        let output = vectorizer
            .embed(prepared.request.content.clone(), cancellation)
            .await?;
        observe_usage(trace, "memory.embedding.usage", output.usage.as_ref());
        prepared.request.vector = Some(parse_vector(&output)?);
        return Ok(false);
    }
    if task
        .prepared
        .as_ref()
        .is_none_or(|prepared| prepared.previous != current)
    {
        let (id, content, from, to, metadata) = match &current {
            Some(item) => {
                let content = cancellable(
                    component.merge(item.clone(), task.content.clone(), cancellation),
                    cancellation,
                )
                .await?;
                if content.trim().is_empty() {
                    return Err(invalid());
                }
                (
                    item.id,
                    content,
                    item.from.min(from),
                    item.to.max(to),
                    item.metadata.clone(),
                )
            }
            None => {
                // Reuse an allocated new-record ID after a failed write.
                let id = if let Some(prepared) = &task.prepared {
                    prepared.request.id
                } else {
                    let id = next_id();
                    if id <= 0 {
                        return Err(Error::new(
                            "INVALID_ARGUMENTS",
                            "memory ID function must return a positive i64",
                        ));
                    }
                    id
                };
                (id, task.content.clone(), from, to, None)
            }
        };
        let vector = if let Some(vectorizer) = vectorizer {
            let output = vectorizer.embed(content.clone(), cancellation).await?;
            observe_usage(trace, "memory.embedding.usage", output.usage.as_ref());
            Some(parse_vector(&output)?)
        } else {
            None
        };
        task.prepared = Some(Prepared {
            previous: current,
            request: Request {
                id,
                r#type: task.kind.clone(),
                user_id,
                agent_id: task.agent_id,
                session_id: None,
                from,
                to,
                content,
                metadata,
                vector,
            },
        });
    }
    Ok(false)
}

fn parse_vector(message: &crate::Message) -> Result<Vec<f32>, Error> {
    let values = message
        .values
        .get("vector")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::new("INVALID_RESPONSE", "embedding Message has no vector"))?;
    if values.is_empty() {
        return Err(Error::new("INVALID_RESPONSE", "embedding vector is empty"));
    }
    values
        .iter()
        .map(|value| {
            let number = value
                .as_f64()
                .ok_or_else(|| Error::new("INVALID_RESPONSE", "embedding vector is not numeric"))?
                as f32;
            number
                .is_finite()
                .then_some(number)
                .ok_or_else(|| Error::new("INVALID_RESPONSE", "embedding vector is not finite"))
        })
        .collect()
}

async fn write_prepared<L: Memory>(
    task: &Task,
    store: &L,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    let prepared = task.prepared.as_ref().unwrap();
    let version = match &prepared.previous {
        Some(previous) => previous.version.checked_add(1),
        None => Some(1),
    }
    .ok_or_else(|| Error::new("CONFLICT", "memory version exhausted"))?;
    let result = if let Some(previous) = &prepared.previous {
        cancellable(
            store.update(prepared.request.clone(), previous.version, cancellation),
            cancellation,
        )
        .await?
    } else {
        cancellable(
            store.write(prepared.request.clone(), cancellation),
            cancellation,
        )
        .await?
    };
    if result.version != version || !matches(&result, &prepared.request) {
        return Err(invalid());
    }
    Ok(())
}

fn matches(item: &Item, request: &Request) -> bool {
    item.id == request.id
        && item.r#type == request.r#type
        && item.user_id == request.user_id
        && item.agent_id == request.agent_id
        && item.session_id == request.session_id
        && item.from == request.from
        && item.to == request.to
        && item.content == request.content
        && item.metadata == request.metadata
}
fn invalid() -> Error {
    Error::new(
        "INVALID_RESPONSE",
        "invalid long memory transformation or backend response",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, core::memory::InMemory, model::content::part::Part};
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[allow(clippy::too_many_arguments)]
    async fn write<L: Memory>(
        task: &mut Task,
        user_id: i64,
        from: i64,
        to: i64,
        component: &memory::extraction::Configured,
        store: &L,
        next_id: fn() -> i64,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        let trace = crate::runtime::trace::fastrace::Root::noop();
        if !prepare(
            task,
            user_id,
            from,
            to,
            component,
            store,
            None,
            &trace,
            next_id,
            cancellation,
        )
        .await?
        {
            write_prepared(task, store, cancellation).await?;
        }
        Ok(())
    }

    #[derive(Clone, Default)]
    struct Extractor {
        calls: Arc<AtomicUsize>,
        merges: Arc<AtomicUsize>,
        fail: Arc<AtomicBool>,
        fail_merge: Arc<AtomicBool>,
        invalid: bool,
    }
    impl memory::extraction::Extractor for Extractor {
        async fn extract(&self, input: &Messages, _: &Cancellation) -> Result<Messages, Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(
                input
                    .0
                    .iter()
                    .all(|m| m.content.iter().all(|p| p.r#type == "text")
                        && m.values.is_empty()
                        && m.metadata.is_empty()
                        && m.usage.is_none())
            );
            let local = Arc::new(1);
            tokio::task::yield_now().await;
            assert_eq!(*local, 1);
            if self.fail.swap(false, Ordering::SeqCst) {
                return Err(Error::new("TRANSPORT", "extract failed"));
            }
            let mut message = Message::new(Role::Assistant);
            message.values = if self.invalid {
                json!({"user":{"unknown":"bad"}})
            } else {
                json!({"user":{"fact":"new fact"},"agent":{"preference":"concise"}})
            }
            .as_object()
            .unwrap()
            .clone();
            Ok(Messages::new([message]))
        }
        async fn merge(
            &self,
            current: &Item,
            content: &str,
            _: &Cancellation,
        ) -> Result<String, Error> {
            self.merges.fetch_add(1, Ordering::SeqCst);
            if self.fail_merge.swap(false, Ordering::SeqCst) {
                return Err(Error::new("TRANSPORT", "merge failed"));
            }
            Ok(format!("{} + {}", current.content, content))
        }
    }

    #[derive(Default)]
    struct Store {
        inner: InMemory,
        fail_write: AtomicBool,
        fail_get: AtomicBool,
        lost_ack: AtomicBool,
        conflict: AtomicBool,
        writes: AtomicUsize,
    }
    impl Memory for Store {
        async fn write(&self, request: Request, c: &Cancellation) -> Result<Item, Error> {
            self.writes.fetch_add(1, Ordering::SeqCst);
            if self.fail_write.swap(false, Ordering::SeqCst) {
                return Err(Error::new("BACKEND", "write failed"));
            }
            let result = self.inner.write(request, c).await?;
            if self.lost_ack.swap(false, Ordering::SeqCst) {
                return Err(Error::new("BACKEND", "ack lost"));
            }
            Ok(result)
        }
        async fn update(
            &self,
            request: Request,
            version: u64,
            c: &Cancellation,
        ) -> Result<Item, Error> {
            self.writes.fetch_add(1, Ordering::SeqCst);
            if self.fail_write.swap(false, Ordering::SeqCst) {
                return Err(Error::new("BACKEND", "update failed"));
            }
            if self.conflict.swap(false, Ordering::SeqCst) {
                let mut other = request.clone();
                other.content = "concurrent fact".into();
                self.inner.update(other, version, c).await?;
                return Err(Error::new("CONFLICT", "concurrent writer"));
            }
            let result = self.inner.update(request, version, c).await?;
            if self.lost_ack.swap(false, Ordering::SeqCst) {
                return Err(Error::new("BACKEND", "ack lost"));
            }
            Ok(result)
        }
        async fn get(
            &self,
            user_id: i64,
            agent_id: Option<i64>,
            session_id: Option<i64>,
            kind: memory::Type,
            c: &Cancellation,
        ) -> Result<Option<Item>, Error> {
            if self.fail_get.swap(false, Ordering::SeqCst) {
                return Err(Error::new("BACKEND", "read failed"));
            }
            self.inner.get(user_id, agent_id, session_id, kind, c).await
        }
        async fn delete(&self, id: i64, c: &Cancellation) -> Result<(), Error> {
            self.inner.delete(id, c).await
        }
        async fn list(
            &self,
            u: i64,
            a: Option<i64>,
            s: Option<i64>,
            t: memory::Type,
            c: &Cancellation,
        ) -> Result<Vec<Item>, Error> {
            self.inner.list(u, a, s, t, c).await
        }
        async fn keyword(
            &self,
            r: memory::keyword::Request,
            c: &Cancellation,
        ) -> Result<Vec<Item>, Error> {
            self.inner.keyword(r, c).await
        }
        async fn vector(
            &self,
            r: memory::vector::Request,
            c: &Cancellation,
        ) -> Result<Vec<Item>, Error> {
            self.inner.vector(r, c).await
        }
        async fn hybrid(
            &self,
            r: memory::hybrid::Request,
            c: &Cancellation,
        ) -> Result<Vec<Item>, Error> {
            self.inner.hybrid(r, c).await
        }
    }

    thread_local! {
        static IDS: std::cell::RefCell<Arc<AtomicUsize>> = std::cell::RefCell::new(Arc::new(AtomicUsize::new(1000)));
    }
    fn next_id() -> i64 {
        IDS.with(|ids| ids.borrow().fetch_add(1, Ordering::SeqCst) as i64)
    }

    fn setup(
        extractor: Extractor,
    ) -> (Arc<Jobs>, memory::extraction::Configured, Arc<AtomicUsize>) {
        let ids = Arc::new(AtomicUsize::new(1000));
        IDS.with(|counter| *counter.borrow_mut() = ids.clone());
        let component = memory::extraction::Configured::new(extractor);
        let jobs = Arc::new(Jobs::default());
        let mut message = Message::new(Role::Assistant);
        message.content = vec![
            Part {
                r#type: "text".into(),
                data: crate::components::model::content::part::object(json!({"value":"answer"})),
            },
            Part {
                r#type: "think".into(),
                data: crate::components::model::content::part::object(
                    json!({"value":"private thinking"}),
                ),
            },
        ];
        message.values.insert("private".into(), json!(1));
        jobs.insert(
            9,
            Job::new(42, 7, 100, 110, Messages::new([message])).unwrap(),
        );
        (jobs, component, ids)
    }
    async fn seed(store: &Store) {
        store
            .inner
            .write(
                Request {
                    id: 99,
                    r#type: memory::Type::Fact,
                    user_id: 42,
                    agent_id: None,
                    session_id: None,
                    from: 1,
                    to: 50,
                    content: "old fact".into(),
                    metadata: Some(json!({"keep":1}).as_object().unwrap().clone()),
                    vector: None,
                },
                &Cancellation::new(),
            )
            .await
            .unwrap();
    }
    async fn fact(store: &Store) -> Item {
        store
            .inner
            .get(42, None, None, memory::Type::Fact, &Cancellation::new())
            .await
            .unwrap()
            .unwrap()
    }

    #[tokio::test]
    async fn writes_both_levels_without_session_records_or_recall_configuration() {
        let extractor = Extractor::default();
        let (jobs, c, ids) = setup(extractor.clone());
        let store = Store::default();
        resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
            .await
            .unwrap();
        assert!(!jobs.contains(9));
        let item = fact(&store).await;
        assert_eq!((item.from, item.to), (100, 111));
        assert_eq!(item.content, "new fact");
        assert!(
            store
                .inner
                .get(
                    42,
                    Some(7),
                    None,
                    memory::Type::Preference,
                    &Cancellation::new()
                )
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            store
                .inner
                .get(
                    42,
                    Some(7),
                    Some(9),
                    memory::Type::Fact,
                    &Cancellation::new()
                )
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(ids.load(Ordering::SeqCst), 1002);
        assert_eq!(extractor.calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn extraction_failure_retries_only_extraction_from_retained_input() {
        let extractor = Extractor::default();
        extractor.fail.store(true, Ordering::SeqCst);
        let (jobs, c, ids) = setup(extractor.clone());
        let store = Store::default();
        assert_eq!(
            resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "TRANSPORT"
        );
        assert_eq!(ids.load(Ordering::SeqCst), 1000);
        assert!(jobs.contains(9));
        resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
            .await
            .unwrap();
        assert_eq!(extractor.calls.load(Ordering::SeqCst), 2);
    }
    #[tokio::test]
    async fn write_failure_preserves_extraction_id_and_successful_other_type() {
        let extractor = Extractor::default();
        let (jobs, c, ids) = setup(extractor.clone());
        let store = Store::default();
        store.fail_write.store(true, Ordering::SeqCst);
        assert!(
            resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
                .await
                .is_err()
        );
        assert!(jobs.contains(9));
        assert_eq!(ids.load(Ordering::SeqCst), 1002);
        assert_eq!(store.writes.load(Ordering::SeqCst), 2);
        resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
            .await
            .unwrap();
        assert_eq!(extractor.calls.load(Ordering::SeqCst), 1);
        assert_eq!(ids.load(Ordering::SeqCst), 1002);
        assert_eq!(store.writes.load(Ordering::SeqCst), 3);
    }
    #[tokio::test]
    async fn merge_failure_does_not_reextract_or_rollback_other_type() {
        let extractor = Extractor::default();
        extractor.fail_merge.store(true, Ordering::SeqCst);
        let (jobs, c, _) = setup(extractor.clone());
        let store = Store::default();
        seed(&store).await;
        assert!(
            resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
                .await
                .is_err()
        );
        assert_eq!(fact(&store).await.content, "old fact");
        resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
            .await
            .unwrap();
        let item = fact(&store).await;
        assert_eq!(item.content, "old fact + new fact");
        assert_eq!((item.id, item.from, item.to, item.version), (99, 1, 111, 2));
        assert_eq!(item.metadata.unwrap().get("keep"), Some(&json!(1)));
        assert_eq!(extractor.calls.load(Ordering::SeqCst), 1);
        assert_eq!(extractor.merges.load(Ordering::SeqCst), 2);
    }
    #[tokio::test]
    async fn conflict_rereads_latest_and_remerges_without_reextracting() {
        let extractor = Extractor::default();
        let (jobs, c, _) = setup(extractor.clone());
        let store = Store::default();
        seed(&store).await;
        store.conflict.store(true, Ordering::SeqCst);
        assert_eq!(
            resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "CONFLICT"
        );
        resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
            .await
            .unwrap();
        let item = fact(&store).await;
        assert_eq!(item.content, "concurrent fact + new fact");
        assert_eq!(item.version, 3);
        assert_eq!(extractor.calls.load(Ordering::SeqCst), 1);
        assert_eq!(extractor.merges.load(Ordering::SeqCst), 2);
    }
    #[tokio::test]
    async fn failed_update_reuses_already_successful_merge() {
        let extractor = Extractor::default();
        let (jobs, c, _) = setup(extractor.clone());
        let store = Store::default();
        seed(&store).await;
        // Agent preference is processed first. Exercise a prepared user update directly.
        let mut task = Task {
            agent_id: None,
            kind: memory::Type::Fact,
            content: "new fact".into(),
            prepared: None,
            done: false,
        };
        store.fail_write.store(true, Ordering::SeqCst);
        assert!(
            write(
                &mut task,
                42,
                100,
                111,
                &c,
                &store,
                next_id,
                &Cancellation::new()
            )
            .await
            .is_err()
        );
        write(
            &mut task,
            42,
            100,
            111,
            &c,
            &store,
            next_id,
            &Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(extractor.merges.load(Ordering::SeqCst), 1);
        assert!(jobs.contains(9));
    }
    #[tokio::test]
    async fn lost_ack_is_recognized_without_a_second_write() {
        let extractor = Extractor::default();
        let (jobs, c, _) = setup(extractor.clone());
        let store = Store::default();
        store.lost_ack.store(true, Ordering::SeqCst);
        assert!(
            resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
                .await
                .is_err()
        );
        resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
            .await
            .unwrap();
        assert_eq!(store.writes.load(Ordering::SeqCst), 2);
        assert_eq!(extractor.calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn invalid_shape_never_writes_any_type() {
        let (jobs, c, ids) = setup(Extractor {
            invalid: true,
            ..Default::default()
        });
        let store = Store::default();
        assert_eq!(
            resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
        assert_eq!(store.writes.load(Ordering::SeqCst), 0);
        assert_eq!(ids.load(Ordering::SeqCst), 1000);
    }
    #[tokio::test]
    async fn cancellation_preserves_work_without_allocating_ids() {
        let extractor = Extractor::default();
        let (jobs, c, ids) = setup(extractor.clone());
        let store = Store::default();
        let cancel = Cancellation::new();
        cancel.cancel();
        assert_eq!(
            resume(jobs.clone(), 9, &c, &store, next_id, &cancel)
                .await
                .unwrap_err()
                .code,
            "CANCELLED"
        );
        assert_eq!(extractor.calls.load(Ordering::SeqCst), 0);
        assert_eq!(ids.load(Ordering::SeqCst), 1000);
        assert!(jobs.contains(9));
        resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
            .await
            .unwrap();
    }
    #[tokio::test]
    async fn invalid_id_fails_without_store_writes() {
        let (jobs, _, _) = setup(Extractor::default());
        let c = memory::extraction::Configured::new(Extractor::default());
        let store = Store::default();
        assert_eq!(
            resume(jobs, 9, &c, &store, || 0, &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        assert_eq!(store.writes.load(Ordering::SeqCst), 0);
    }
    #[tokio::test]
    async fn no_implicit_embed_or_vector_downgrade() {
        let (jobs, c, _) = setup(Extractor::default());
        let store = InMemory::with_vector_dimension(3).unwrap();
        assert_eq!(
            resume(jobs.clone(), 9, &c, &store, next_id, &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        assert!(jobs.contains(9));
        assert!(
            store
                .get(42, None, None, memory::Type::Fact, &Cancellation::new())
                .await
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn invalid_source_range_rejected() {
        assert!(Job::new(1, 2, 100, 99, Messages::default()).is_err());
        assert!(Job::new(1, 2, 100, i64::MAX, Messages::default()).is_err());
    }
}
