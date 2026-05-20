//! Integration tests: tokenize / embed pipeline steps.

use rag::index::pagewiki::Base;
use rag::index::pagewiki::{self, PageWiki};
use rag::index::source::{Item, Scenario};
use rag::index::{Builder, NoopEmbedder, NoopTokenizer};
use serde_json::json;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

struct SingleCutter;

impl pagewiki::Base for SingleCutter {
    fn cut<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = pagewiki::Result<Vec<PageWiki>>> + Send + 'a>> {
        Box::pin(async move {
            Ok(vec![PageWiki {
                content: text.to_string(),
                keywords: vec!["foo".into(), "bar".into()],
                questions: vec!["what?".into()],
                ..Default::default()
            }])
        })
    }
}

fn make_item(text: &str) -> Item {
    let mut metadata = serde_json::Map::new();
    metadata.insert("doc_id".into(), json!("doc_tok_test"));
    Item {
        text: text.into(),
        scenario: Scenario::General,
        metadata,
    }
}

#[tokio::test]
async fn noop_tokenizer_produces_empty_tokens() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(SingleCutter));

    let builder = Builder::new(pw_map, vec![], Box::new(NoopTokenizer), None).unwrap();
    let pages = builder.build(vec![make_item("hello world")]).await.unwrap();

    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].content_tokens, "");
    assert_eq!(pages[0].keyword_tokens, "");
    assert_eq!(pages[0].question_tokens, "");
}

#[tokio::test]
async fn noop_embedder_produces_empty_embedding() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(SingleCutter));

    let builder = Builder::new(
        pw_map,
        vec![],
        Box::new(NoopTokenizer),
        Some(Box::new(NoopEmbedder)),
    )
    .unwrap();
    let pages = builder.build(vec![make_item("embed me")]).await.unwrap();

    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].embedding, Some(vec![]));
}

#[tokio::test]
async fn no_embedder_leaves_embedding_none() {
    let mut pw_map: HashMap<Scenario, Box<dyn Base>> = HashMap::new();
    pw_map.insert(Scenario::General, Box::new(SingleCutter));

    let builder = Builder::new(pw_map, vec![], Box::new(NoopTokenizer), None).unwrap();
    let pages = builder.build(vec![make_item("no embed")]).await.unwrap();

    assert_eq!(pages.len(), 1);
    assert!(pages[0].embedding.is_none());
}
