use super::*;
use crate::{Message, core::memory::InMemory, model::content::Part};
use serde_json::json;
use std::{
    rc::Rc,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

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
        let local = Rc::new(1);
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
    async fn delete(&self, id: i64, version: u64, c: &Cancellation) -> Result<(), Error> {
        self.inner.delete(id, version, c).await
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

fn setup(extractor: Extractor) -> (Arc<Jobs>, memory::extraction::Configured, Arc<AtomicUsize>) {
    let ids = Arc::new(AtomicUsize::new(1000));
    IDS.with(|counter| *counter.borrow_mut() = ids.clone());
    let component = memory::extraction::Configured::new(extractor);
    let jobs = Arc::new(Jobs::default());
    let mut message = Message::new(Role::Assistant);
    message.content = vec![
        Part {
            r#type: "text".into(),
            data: json!({"value":"answer"}),
        },
        Part {
            r#type: "think".into(),
            data: json!({"value":"private thinking"}),
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
