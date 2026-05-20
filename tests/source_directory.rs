//! End-to-end tests for `Directory` using a `tempfile::TempDir`.

use std::fs;

use rag::index::source::{Base, Directory, Error, Scenario};
use tempfile::TempDir;

#[tokio::test]
async fn directory_reads_json_files() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("a.json"),
        br#"{"doc_id":"doc_a","content":"hello","title":"A"}"#,
    )
    .unwrap();
    fs::write(
        tmp.path().join("b.json"),
        b"{\"doc_id\":\"doc_b\",\"content\":\"heading\",\"scenario\":\"qa\"}",
    )
    .unwrap();

    let mut s = Directory::new(tmp.path());
    let got = s.read(10, Scenario::General).await.unwrap();
    assert_eq!(got.len(), 2);

    let docs: Vec<&str> = got
        .iter()
        .map(|it| {
            it.metadata
                .get("doc_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
        })
        .collect();
    assert!(docs.contains(&"doc_a"));
    assert!(docs.contains(&"doc_b"));

    // 验证 content 字段被正确提取到 text
    let a_item = got
        .iter()
        .find(|it| it.metadata.get("doc_id").and_then(|v| v.as_str()) == Some("doc_a"))
        .unwrap();
    assert_eq!(a_item.text, "hello");

    // 验证 scenario 字段被正确解析
    let b_item = got
        .iter()
        .find(|it| it.metadata.get("doc_id").and_then(|v| v.as_str()) == Some("doc_b"))
        .unwrap();
    assert_eq!(b_item.scenario, Scenario::Qa);
}

#[tokio::test]
async fn directory_missing_doc_id_fails_batch() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("a.json"),
        br#"{"doc_id":"doc_a","content":"hi"}"#,
    )
    .unwrap();
    fs::write(
        tmp.path().join("b.json"),
        br#"{"content":"world"}"#, // 缺少 doc_id
    )
    .unwrap();

    let mut s = Directory::new(tmp.path());
    let err = s.read(10, Scenario::General).await.unwrap_err();
    assert!(matches!(err, Error::MissingDocId));
}

#[tokio::test]
async fn directory_missing_content_fails_batch() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("a.json"),
        br#"{"doc_id":"doc_a"}"#, // 缺少 content
    )
    .unwrap();

    let mut s = Directory::new(tmp.path());
    let err = s.read(10, Scenario::General).await.unwrap_err();
    assert!(matches!(err, Error::InvalidMetadata(_)));
}

#[tokio::test]
async fn directory_skips_non_json_files_and_subdirs() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("c.pdf"), b"%PDF").unwrap();
    fs::write(tmp.path().join("d.txt"), b"plain text").unwrap();
    fs::write(
        tmp.path().join("e.json"),
        br#"{"doc_id":"doc_e","content":"good"}"#,
    )
    .unwrap();
    let sub = tmp.path().join("nested");
    fs::create_dir(&sub).unwrap();
    fs::write(
        sub.join("y.json"),
        br#"{"doc_id":"doc_y","content":"nested-text"}"#,
    )
    .unwrap();

    let mut s = Directory::new(tmp.path());
    let got = s.read(10, Scenario::General).await.unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].text, "good");
}

#[tokio::test]
async fn directory_paginates_with_small_batch_size() {
    let tmp = TempDir::new().unwrap();
    for i in 0..3 {
        let filename = format!("doc_{i}.json");
        fs::write(
            tmp.path().join(&filename),
            format!(r#"{{"doc_id":"doc_{i}","content":"t{i}"}}"#),
        )
        .unwrap();
    }

    let mut s = Directory::new(tmp.path());
    let a = s.read(2, Scenario::General).await.unwrap();
    let b = s.read(2, Scenario::General).await.unwrap();
    let c = s.read(2, Scenario::General).await.unwrap();
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 1);
    assert!(c.is_empty());
}

#[tokio::test]
async fn directory_invalid_json_returns_invalid_metadata() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.json"), b"not-json").unwrap();
    let mut s = Directory::new(tmp.path());
    let err = s.read(10, Scenario::General).await.unwrap_err();
    assert!(matches!(err, Error::InvalidMetadata(_)));
}

#[tokio::test]
async fn directory_preserves_extra_metadata_fields() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("a.json"),
        b"{\"doc_id\":\"doc_a\",\"content\":\"text\",\"author\":\"Zhang San\",\"version\":\"1.0\"}",
    )
    .unwrap();

    let mut s = Directory::new(tmp.path());
    let got = s.read(10, Scenario::General).await.unwrap();
    assert_eq!(got.len(), 1);

    let item = &got[0];
    assert_eq!(
        item.metadata.get("author").and_then(|v| v.as_str()),
        Some("Zhang San")
    );
    assert_eq!(
        item.metadata.get("version").and_then(|v| v.as_str()),
        Some("1.0")
    );
    // content 字段也保留在 metadata 中
    assert_eq!(
        item.metadata.get("content").and_then(|v| v.as_str()),
        Some("text")
    );
}
