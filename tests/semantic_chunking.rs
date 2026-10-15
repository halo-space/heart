use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use futures_executor::block_on;
use halo_agents::{
    Cancellation, Error, Message,
    document::{self, Chunker as _, Document},
    model::embed::{Embed, Request},
};
use serde_json::{Value, json};

#[derive(Clone, Default)]
struct Model {
    calls: Arc<Mutex<Vec<String>>>,
}

impl Embed for Model {
    async fn embed(&self, request: Request) -> Result<Message, Error> {
        self.calls.lock().unwrap().push(request.input.clone());
        let vector = if request.input.contains('B') {
            vec![0.0, 1.0]
        } else {
            vec![1.0, 0.0]
        };
        halo_agents::model::embed::message(vector, "test", "embed", None)
    }
}

fn options(max: usize) -> Value {
    json!({"max_tokens":max, "min_tokens":1, "threshold":0.5})
}

fn document(content: &str) -> Document {
    Document {
        content: content.into(),
        doc_id: Some(55),
        ..Default::default()
    }
}

fn split<C: document::Chunker>(
    chunker: &C,
    documents: Vec<Document>,
) -> Result<Vec<Document>, Error> {
    let mut id = 100;
    block_on(chunker.chunk(documents, &Cancellation::new(), &mut || {
        id += 1;
        id
    }))
}

#[test]
fn semantic_boundary_changes_without_exceeding_size() {
    let model = Model::default();
    let chunker = document::chunker::semantic::new(options(50), model.clone()).unwrap();
    let chunks = split(&chunker, vec![document("A. A! B? B.")]).unwrap();
    assert_eq!(
        chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>(),
        ["A. A!", " B? B."]
    );
    assert_eq!(*model.calls.lock().unwrap(), ["A.", " A!", " B?", " B."]);
    assert_eq!(chunks[0].doc_id, Some(55));
    assert_eq!(chunks[0].id, Some(101));
}

#[test]
fn default_length_is_utf8_byte_estimate_not_unicode_count() {
    for (input, max) in [("abcdefghij", 1), ("北京🙂abcd\r\n", 2), ("  \n\n  ", 1)] {
        let chunker = document::chunker::semantic::new(options(max), Model::default()).unwrap();
        let chunks = split(&chunker, vec![document(input)]).unwrap();
        assert_eq!(
            chunks
                .iter()
                .map(|c| c.content.as_str())
                .collect::<String>(),
            input
        );
        assert!(chunks.iter().all(|c| c.content.len().div_ceil(4) <= max));
    }
}

#[test]
fn optional_length_function_changes_only_the_cutting_scale() {
    let chunker = document::chunker::semantic::new(options(2), Model::default())
        .unwrap()
        .with_length(|text: &str| Ok(text.chars().count()));
    let chunks = split(&chunker, vec![document("北京🙂abcd")]).unwrap();
    assert_eq!(
        chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>(),
        ["北京", "🙂a", "bc", "d"]
    );
    assert!(chunks.iter().all(|c| c.content.chars().count() <= 2));
}

#[test]
fn complete_merged_text_is_measured_instead_of_summing_units() {
    let chunker = document::chunker::semantic::new(options(1), Model::default()).unwrap();
    let chunks = split(&chunker, vec![document("A.A.")]).unwrap();
    assert_eq!(chunks.len(), 1); // ceil(2/4)+ceil(2/4) would incorrectly split.
    assert_eq!(chunks[0].content, "A.A.");
}

#[test]
fn minimum_is_a_merge_preference_not_a_reason_to_drop_tail() {
    let chunker = document::chunker::semantic::new(
        json!({"max_tokens":10,"min_tokens":8,"threshold":0.5}),
        Model::default(),
    )
    .unwrap();
    let chunks = split(&chunker, vec![document("A.B.")]).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content, "A.B.");
}

#[test]
fn all_whitespace_and_empty_content_do_not_call_embed() {
    let model = Model::default();
    let chunker = document::chunker::semantic::new(options(10), model.clone()).unwrap();
    let chunks = split(&chunker, vec![document(""), document(" \t\r\n ")]).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content, " \t\r\n ");
    assert!(model.calls.lock().unwrap().is_empty());
}

#[test]
fn documents_remain_separate_and_slices_drop_stale_vectors() {
    let mut first = document("abcdefghij");
    first.metadata.insert("vector".into(), json!([1, 0]));
    first
        .metadata
        .insert("source".into(), json!({"uri":"test"}));
    let second = document("A.");
    let chunker = document::chunker::semantic::new(options(1), Model::default()).unwrap();
    let chunks = split(&chunker, vec![first, second]).unwrap();
    assert_eq!(chunks.len(), 4);
    assert!(
        chunks[..3]
            .iter()
            .all(|c| c.metadata.get("vector").is_none())
    );
    assert_eq!(chunks[0].metadata["source"]["uri"], "test");
    assert_eq!(chunks[3].content, "A.");
}

#[test]
fn invalid_options_and_length_rules_fail_before_ids() {
    for value in [
        json!({}),
        json!({"max_tokens":0,"min_tokens":1,"threshold":0.5}),
        json!({"max_tokens":2,"min_tokens":3,"threshold":0.5}),
        json!({"max_tokens":2,"min_tokens":1,"threshold":2}),
        json!({"max_tokens":2,"min_tokens":1,"threshold":0.5,"max_concurrency":0}),
        json!({"max_tokens":2,"min_tokens":1,"threshold":0.5,"counter":"hidden"}),
    ] {
        assert!(document::chunker::semantic::new(value, Model::default()).is_err());
    }
    let chunker = document::chunker::semantic::new(options(2), Model::default())
        .unwrap()
        .with_length(|_: &str| Ok(0));
    assert_eq!(
        split(&chunker, vec![document("hello")]).unwrap_err().code,
        "INVALID_RESPONSE"
    );
    let chunker = document::chunker::semantic::new(options(1), Model::default())
        .unwrap()
        .with_length(|text: &str| Ok(text.len()));
    assert_eq!(
        split(&chunker, vec![document("🙂")]).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn user_length_error_is_returned_unchanged() {
    let chunker = document::chunker::semantic::new(options(10), Model::default())
        .unwrap()
        .with_length(
            |_: &str| Err(Error::new("CUSTOM", "failed length rule").with_details("test")),
        );
    assert_eq!(
        split(&chunker, vec![document("hello")]).unwrap_err(),
        Error::new("CUSTOM", "failed length rule").with_details("test")
    );
}

struct Failing {
    calls: AtomicUsize,
}
impl Embed for Failing {
    async fn embed(&self, _: Request) -> Result<Message, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::new("TRANSPORT", "offline"))
    }
}

#[test]
fn failed_embedding_returns_no_partial_chunks_and_consumes_no_ids() {
    let chunker = document::chunker::semantic::new(
        options(10),
        Failing {
            calls: AtomicUsize::new(0),
        },
    )
    .unwrap();
    let mut ids = 0;
    let result = block_on(
        chunker.chunk(vec![document("A.B.")], &Cancellation::new(), &mut || {
            ids += 1;
            ids
        }),
    );
    assert_eq!(result.unwrap_err().code, "TRANSPORT");
    assert_eq!(ids, 0);
}

#[test]
fn reused_ids_are_rejected() {
    let chunker = document::chunker::semantic::new(options(1), Model::default()).unwrap();
    let error = block_on(chunker.chunk(
        vec![document("abcdefghij")],
        &Cancellation::new(),
        &mut || 5,
    ))
    .unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
}

struct CancelModel {
    cancellation: Cancellation,
}
impl Embed for CancelModel {
    async fn embed(&self, _: Request) -> Result<Message, Error> {
        self.cancellation.cancel();
        std::future::pending().await
    }
}

#[test]
fn cancellation_wakes_a_pending_embed_without_allocating_ids() {
    let cancellation = Cancellation::new();
    let chunker = document::chunker::semantic::new(
        options(10),
        CancelModel {
            cancellation: cancellation.clone(),
        },
    )
    .unwrap();
    let result = block_on(chunker.chunk(vec![document("A.")], &cancellation, &mut || {
        panic!("must not allocate")
    }));
    assert_eq!(result.unwrap_err().code, "CANCELLED");
}

#[test]
fn already_cancelled_does_not_invoke_embed() {
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let model = Model::default();
    let chunker = document::chunker::semantic::new(options(10), model.clone()).unwrap();
    assert_eq!(
        block_on(chunker.chunk(vec![document("A.")], &cancellation, &mut || 1))
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert!(model.calls.lock().unwrap().is_empty());
}

struct InvalidVector(Value);
impl Embed for InvalidVector {
    async fn embed(&self, _: Request) -> Result<Message, Error> {
        let mut message = Message::new(halo_agents::message::Role::Assistant);
        message.values.insert("vector".into(), self.0.clone());
        Ok(message)
    }
}

#[test]
fn empty_zero_and_malformed_vectors_are_rejected() {
    for vector in [json!([]), json!([0, 0]), json!(["x"]), Value::Null] {
        let chunker = document::chunker::semantic::new(options(10), InvalidVector(vector)).unwrap();
        assert_eq!(
            split(&chunker, vec![document("A.")]).unwrap_err().code,
            "INVALID_RESPONSE"
        );
    }
}

struct Parallel {
    active: Arc<AtomicUsize>,
    maximum: Arc<AtomicUsize>,
}
impl Embed for Parallel {
    async fn embed(&self, request: Request) -> Result<Message, Error> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.maximum.fetch_max(active, Ordering::SeqCst);
        let delay = if request.input.contains('A') { 15 } else { 1 };
        tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        halo_agents::model::embed::message(vec![1.0, 0.0], "test", "test", None)
    }
}

#[tokio::test]
async fn bounded_parallel_embed_preserves_source_order() {
    let maximum = Arc::new(AtomicUsize::new(0));
    let model = Parallel {
        active: Arc::new(AtomicUsize::new(0)),
        maximum: maximum.clone(),
    };
    let mut opts = options(1);
    opts["max_concurrency"] = json!(2);
    let chunker = document::chunker::semantic::new(opts, model).unwrap();
    let mut id = 0;
    let chunks = chunker
        .chunk(
            vec![document("A.B.C.D.")],
            &Cancellation::new(),
            &mut || {
                id += 1;
                id
            },
        )
        .await
        .unwrap();
    assert_eq!(
        chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<String>(),
        "A.B.C.D."
    );
    assert_eq!(maximum.load(Ordering::SeqCst), 2);
}
