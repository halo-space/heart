//! End-to-end tests for `Inline`.

use rag::index::source::{Base, Error, Inline, Item, Scenario};
use serde_json::{Map, json};

fn item(doc_id: &str, text: &str) -> Item {
    let mut md = Map::new();
    md.insert("doc_id".into(), json!(doc_id));
    Item {
        text: text.into(),
        scenario: Scenario::Manual,
        metadata: md,
    }
}

#[tokio::test]
async fn inline_full_loop_batch_then_eos() {
    let mut s = Inline::new();
    for i in 0..5 {
        s.push(item(&format!("doc_{i}"), &format!("t{i}"))).await;
    }

    let mut seen = 0usize;
    loop {
        let batch = s.read(2, Scenario::General).await.unwrap();
        if batch.is_empty() {
            break;
        }
        for it in &batch {
            assert_eq!(it.scenario, Scenario::General);
        }
        seen += batch.len();
    }
    assert_eq!(seen, 5);
}

#[tokio::test]
async fn inline_one_at_a_time() {
    let mut s = Inline::new();
    s.push(item("a", "x")).await;
    s.push(item("b", "y")).await;

    let one = s.read(1, Scenario::Qa).await.unwrap();
    assert_eq!(one.len(), 1);
    assert_eq!(one[0].text, "x");
    let one = s.read(1, Scenario::Qa).await.unwrap();
    assert_eq!(one.len(), 1);
    assert_eq!(one[0].text, "y");
    let none = s.read(1, Scenario::Qa).await.unwrap();
    assert!(none.is_empty());
}

#[tokio::test]
async fn inline_batch_size_zero_is_noop_keeps_queue() {
    let mut s = Inline::new();
    s.push(item("a", "x")).await;
    let none = s.read(0, Scenario::General).await.unwrap();
    assert!(none.is_empty());
    assert_eq!(s.len(), 1);
}

#[tokio::test]
async fn inline_partial_batch_failure_rolls_back() {
    let mut s = Inline::new();
    s.push(item("a", "x")).await;
    // bad item — no doc_id
    s.push(Item {
        text: "y".into(),
        scenario: Scenario::General,
        metadata: Map::new(),
    })
    .await;
    s.push(item("c", "z")).await;

    let err = s.read(3, Scenario::General).await.unwrap_err();
    assert!(matches!(err, Error::MissingDocId));
    // None consumed.
    assert_eq!(s.len(), 3);
}
