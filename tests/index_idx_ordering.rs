//! Integration tests: idx ordering and spans-based sorting.

use rag::index::pagewiki::Base;
use rag::index::pagewiki::{self, PageWiki, Span};
use rag::index::source::{Item, Scenario};
use rag::index::{Builder, NoopTokenizer};
use serde_json::json;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

struct MultiPageCutter;

impl pagewiki::Base for MultiPageCutter {
    fn cut<'a>(
        &'a self,
        _text: &'a str,
    ) -> Pin<Box<dyn Future<Output = pagewiki::Result<Vec<PageWiki>>> + Send + 'a>> {
        Box::pin(async move {
            // Return 3 pages with spans in reverse order
            Ok(vec![
                PageWiki {
                    content: "third".into(),
                    spans: vec![Span {
                        start: 200,
                        end: 205,
                        original_text: "third".into(),
                        extra: Default::default(),
                    }],
                    ..Default::default()
                },
                PageWiki {
                    content: "first".into(),
                    spans: vec![Span {
                        start: 0,
                        end: 5,
                        original_text: "first".into(),
                        extra: Default::default(),
                    }],
                    ..Default::default()
                },
                PageWiki {
                    content: "second".into(),
                    spans: vec![Span {
                        start: 100,
                        end: 106,
                        original_text: "second".into(),
                        extra: Default::default(),
                    }],
                    ..Default::default()
                },
            ])
        })
    }
}

fn make_item() -> Item {
    let mut metadata = serde_json::Map::new();
    metadata.insert("doc_id".into(), json!("doc_idx_test"));
    Item {
        text: "dummy".into(),
        scenario: Scenario::General,
        metadata,
    }
}

#[tokio::test]
async fn pages_sorted_by_spans_start() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(MultiPageCutter));

    let builder = Builder::new(pw_map, vec![], Box::new(NoopTokenizer), None).unwrap();
    let pages = builder.build(vec![make_item()]).await.unwrap();

    assert_eq!(pages.len(), 3);
    assert_eq!(pages[0].content, "first");
    assert_eq!(pages[1].content, "second");
    assert_eq!(pages[2].content, "third");
}

#[tokio::test]
async fn idx_assigned_after_sorting() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(MultiPageCutter));

    let builder = Builder::new(pw_map, vec![], Box::new(NoopTokenizer), None).unwrap();
    let pages = builder.build(vec![make_item()]).await.unwrap();

    assert_eq!(pages[0].idx, Some(0));
    assert_eq!(pages[1].idx, Some(1));
    assert_eq!(pages[2].idx, Some(2));
}

struct NoSpansCutter;

impl pagewiki::Base for NoSpansCutter {
    fn cut<'a>(
        &'a self,
        _text: &'a str,
    ) -> Pin<Box<dyn Future<Output = pagewiki::Result<Vec<PageWiki>>> + Send + 'a>> {
        Box::pin(async move {
            Ok(vec![
                PageWiki {
                    content: "page1".into(),
                    spans: vec![],
                    ..Default::default()
                },
                PageWiki {
                    content: "page2".into(),
                    spans: vec![],
                    ..Default::default()
                },
            ])
        })
    }
}

#[tokio::test]
async fn no_spans_pages_stay_at_end() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(NoSpansCutter));

    let builder = Builder::new(pw_map, vec![], Box::new(NoopTokenizer), None).unwrap();
    let pages = builder.build(vec![make_item()]).await.unwrap();

    // No spans means they stay in original order (both have usize::MAX as sort key)
    assert_eq!(pages.len(), 2);
    assert_eq!(pages[0].idx, Some(0));
    assert_eq!(pages[1].idx, Some(1));
}
