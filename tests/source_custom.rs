//! External data sources must integrate via `impl Base`, with no need for a
//! `DbSource` / `DbAdapter` shim. This test demonstrates that path with a
//! pseudo "DB" backed by an in-memory `Vec`.

use std::collections::VecDeque;

use rag::index::source::{Base, Error, Item, Scenario};
use serde_json::{Map, json};

/// Pretend we're reading rows out of a DB.
struct MyDbReader {
    rows: VecDeque<(String, String)>, // (doc_id, text)
}

impl MyDbReader {
    fn new(rows: impl IntoIterator<Item = (String, String)>) -> Self {
        Self {
            rows: rows.into_iter().collect(),
        }
    }
}

impl Base for MyDbReader {
    async fn read(&mut self, batch_size: usize, scenario: Scenario) -> Result<Vec<Item>, Error> {
        if batch_size == 0 {
            return Ok(Vec::new());
        }
        let take = batch_size.min(self.rows.len());
        if take == 0 {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            let (doc_id, text) = self.rows.pop_front().unwrap();
            let mut metadata = Map::new();
            metadata.insert("doc_id".into(), json!(doc_id));
            metadata.insert("source_table".into(), json!("articles"));
            let item = Item {
                text,
                scenario,
                metadata,
            };
            self.validate_doc_id(&item.metadata).await?;
            out.push(item);
        }
        Ok(out)
    }
}

#[tokio::test]
async fn custom_source_round_trip() {
    let mut s = MyDbReader::new([
        ("doc_1".into(), "alpha".into()),
        ("doc_2".into(), "beta".into()),
        ("doc_3".into(), "gamma".into()),
    ]);

    let mut all: Vec<Item> = Vec::new();
    loop {
        let batch = s.read(2, Scenario::Qa).await.unwrap();
        if batch.is_empty() {
            break;
        }
        all.extend(batch);
    }
    assert_eq!(all.len(), 3);
    for it in &all {
        assert_eq!(it.scenario, Scenario::Qa);
        assert!(it.metadata.get("source_table").is_some());
    }
}

#[tokio::test]
async fn custom_source_doc_id_validation_kicks_in() {
    // Reader that emits a record with no doc_id; the impl must surface
    // `MissingDocId` via the default `validate_doc_id` helper.
    struct Bad;
    impl Base for Bad {
        async fn read(
            &mut self,
            _batch_size: usize,
            scenario: Scenario,
        ) -> Result<Vec<Item>, Error> {
            let item = Item {
                text: "x".into(),
                scenario,
                metadata: Map::new(),
            };
            self.validate_doc_id(&item.metadata).await?;
            Ok(vec![item])
        }
    }
    let err = Bad.read(1, Scenario::General).await.unwrap_err();
    assert!(matches!(err, Error::MissingDocId));
}
