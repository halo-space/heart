//! 集成测试：pagewiki::Qa（JSONL 切分）。

use rag::index::pagewiki::{Base, Qa};

#[tokio::test]
async fn qa_two_valid_pairs() {
    let jsonl = concat!(
        "{\"question\":\"q1\",\"answer\":\"a1\",\"keywords\":[\"k\"]}\n",
        "{\"question\":\"q2\",\"answer\":\"a2\"}\n",
    );
    let pages = Qa::new().cut(jsonl).await.unwrap();
    assert_eq!(pages.len(), 2);
    assert_eq!(pages[0].questions, vec!["q1".to_string()]);
    assert_eq!(pages[0].content, "a1");
    assert_eq!(pages[0].keywords, vec!["k".to_string()]);
    assert!(!pages[0].spans.is_empty());
    assert!(pages[0].spans[0].end > pages[0].spans[0].start);
    assert_eq!(pages[1].questions, vec!["q2".to_string()]);
    assert_eq!(pages[1].content, "a2");
}

#[tokio::test]
async fn qa_blank_lines_skipped() {
    let jsonl = "\n{\"question\":\"q\",\"answer\":\"a\"}\n\n";
    let pages = Qa::new().cut(jsonl).await.unwrap();
    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].content, "a");
}

#[tokio::test]
async fn qa_malformed_line_returns_error() {
    use rag::index::pagewiki::Error;
    let jsonl = "{\"question\":\"q\",\"answer\":\"a\"}\nnot-json";
    let err = Qa::new().cut(jsonl).await.unwrap_err();
    match err {
        Error::QaParse { line, .. } => assert_eq!(line, 2),
        other => panic!("expected QaParse, got {other:?}"),
    }
}

#[tokio::test]
async fn qa_extra_fields_ignored() {
    // JSONL 行里有 QaLine 不认识的字段，不应报错。
    let jsonl = "{\"question\":\"q\",\"answer\":\"a\",\"unknown_field\":42}\n";
    let pages = Qa::new().cut(jsonl).await.unwrap();
    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].content, "a");
}

#[tokio::test]
async fn qa_option_fields_are_none() {
    let jsonl = "{\"question\":\"q\",\"answer\":\"a\"}\n";
    let pages = Qa::new().cut(jsonl).await.unwrap();
    assert_eq!(pages.len(), 1);
    let p = &pages[0];
    assert!(p.id.is_none());
    assert!(p.doc_id.is_none());
    assert!(p.version.is_none());
    assert!(p.scenario.is_none());
    assert!(p.idx.is_none());
    assert!(p.embedding.is_none());
}
