//! 从本地目录读取 JSON 格式的文档。
//!
//! 每个 `.json` 文件必须包含以下字段：
//! - `"doc_id"`: 字符串，文档唯一标识符
//! - `"content"`: 字符串，文档正文内容
//! - `"scenario"`: 可选字符串，检索场景（如 "qa", "general"）
//! - 其他字段：全部保留在 `metadata` 中

use std::path::{Path, PathBuf};

use crate::index::source::base::Base;
use crate::index::source::types::{Error, Item, Scenario};

/// 以异步方式从本地目录（或单个文件）流式读取 [`Item`]。
pub struct Directory {
    root: PathBuf,
    items: Option<Vec<PathBuf>>,
    cursor: usize,
}

impl Directory {
    /// 以 `path` 为根创建一个 source。
    ///
    /// 路径**不会**被立即扫描；第一次 [`Base::read`] 时才会懒加载顶层
    /// （非递归）的受支持文件列表。
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            root: path.into(),
            items: None,
            cursor: 0,
        }
    }

    /// 当前已枚举并缓冲的条目数，仅测试用。
    #[cfg(test)]
    pub(crate) fn buffered_len(&self) -> Option<usize> {
        self.items.as_ref().map(Vec::len)
    }

    async fn ensure_items(&mut self) -> Result<(), Error> {
        if self.items.is_some() {
            return Ok(());
        }
        let meta = tokio::fs::metadata(&self.root).await?;
        let mut paths: Vec<PathBuf> = Vec::new();
        if meta.is_file() {
            if self.is_ext(&self.root, "json") {
                paths.push(self.root.clone());
            }
        } else {
            // 仅枚举顶层，不递归子目录。
            let mut rd = tokio::fs::read_dir(&self.root).await?;
            while let Some(entry) = rd.next_entry().await? {
                let p = entry.path();
                if entry.file_type().await?.is_file() && self.is_ext(&p, "json") {
                    paths.push(p);
                }
            }
            paths.sort();
        }
        self.items = Some(paths);
        self.cursor = 0;
        Ok(())
    }

    /// 读取单个 JSON 文件并解析为 Item。
    async fn read_path(&self, path: &Path, scenario: Scenario) -> Result<Item, Error> {
        let bytes = tokio::fs::read(path).await?;
        let mut value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| Error::InvalidMetadata(e.to_string()))?;

        let Some(obj) = value.as_object_mut() else {
            return Err(Error::InvalidMetadata(format!(
                "{} must contain a top-level JSON object",
                path.display()
            )));
        };

        // 提取必需字段 content（但保留在 metadata 中）
        let text = obj
            .get("content")
            .and_then(|v| v.as_str().map(String::from))
            .ok_or_else(|| {
                Error::InvalidMetadata(format!(
                    "{} missing required field 'content'",
                    path.display()
                ))
            })?;

        // doc_id 必须存在（但保留在 metadata 中）
        if !obj.contains_key("doc_id") {
            return Err(Error::MissingDocId);
        }

        // scenario 优先使用 JSON 中的值，否则使用传入的默认值
        let scenario = obj
            .get("scenario")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or(scenario);

        // 所有字段（包括 doc_id、content、scenario）都保留在 metadata 中
        let metadata = obj.clone();

        Ok(Item {
            text,
            scenario,
            metadata,
        })
    }

    /// 检查文件扩展名是否匹配指定值（不区分大小写）。
    fn is_ext(&self, path: &Path, ext: &str) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case(ext))
            .unwrap_or(false)
    }
}

impl Base for Directory {
    async fn read(&mut self, batch_size: usize, scenario: Scenario) -> Result<Vec<Item>, Error> {
        let _span = tracing::debug_span!("source.directory.read", batch_size, ?scenario).entered();

        if batch_size == 0 {
            tracing::debug!("source.read.noop");
            return Ok(Vec::new());
        }
        self.ensure_items().await?;

        let items = self.items.as_ref().expect("ensured");
        let remaining = items.len().saturating_sub(self.cursor);
        if remaining == 0 {
            tracing::debug!("source.read.eos");
            return Ok(Vec::new());
        }
        let take = batch_size.min(remaining);

        let mut out: Vec<Item> = Vec::with_capacity(take);
        for offset in 0..take {
            let path = items[self.cursor + offset].clone();
            let item = match self.read_path(&path, scenario).await {
                Ok(it) => it,
                Err(e) => {
                    tracing::warn!(error = %e, path = %path.display(), "source.read.read_path failed");
                    return Err(e);
                }
            };
            if let Err(e) = self.validate_doc_id(&item.metadata).await {
                tracing::warn!(error = %e, path = %path.display(), "source.read.validate_doc_id failed");
                return Err(e);
            }
            out.push(item);
        }
        // 整个批次全部读完并校验通过后才推进 cursor，避免半成功状态。
        self.cursor += take;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn put(dir: &Path, name: &str, content: &[u8]) {
        tokio::fs::write(dir.join(name), content).await.unwrap();
    }

    #[tokio::test]
    async fn read_path_json_with_all_fields() {
        let tmp = TempDir::new().unwrap();
        put(
            tmp.path(),
            "a.json",
            br#"{"doc_id":"doc_a","content":"hello","title":"A"}"#,
        )
        .await;

        let mut s = Directory::new(tmp.path());
        let got = s.read(10, Scenario::General).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].text, "hello");
        assert_eq!(got[0].scenario, Scenario::General);
        assert_eq!(
            got[0].metadata.get("doc_id").and_then(|v| v.as_str()),
            Some("doc_a")
        );
        assert_eq!(
            got[0].metadata.get("title").and_then(|v| v.as_str()),
            Some("A")
        );
    }

    #[tokio::test]
    async fn missing_doc_id_fails_whole_batch() {
        let tmp = TempDir::new().unwrap();
        put(
            tmp.path(),
            "a.json",
            br#"{"doc_id":"doc_a","content":"hello"}"#,
        )
        .await;
        put(tmp.path(), "b.json", br#"{"content":"world"}"#).await; // 缺少 doc_id

        let mut s = Directory::new(tmp.path());
        let err = s.read(10, Scenario::General).await.unwrap_err();
        assert!(matches!(err, Error::MissingDocId));
    }

    #[tokio::test]
    async fn missing_content_fails_whole_batch() {
        let tmp = TempDir::new().unwrap();
        put(tmp.path(), "a.json", br#"{"doc_id":"doc_a"}"#).await; // 缺少 content

        let mut s = Directory::new(tmp.path());
        let err = s.read(10, Scenario::General).await.unwrap_err();
        assert!(matches!(err, Error::InvalidMetadata(_)));
    }

    #[tokio::test]
    async fn non_json_extensions_are_skipped() {
        let tmp = TempDir::new().unwrap();
        put(tmp.path(), "c.pdf", b"%PDF-1.4").await;
        put(tmp.path(), "d.txt", b"plain-text").await;
        put(
            tmp.path(),
            "e.json",
            br#"{"doc_id":"doc_e","content":"only-this"}"#,
        )
        .await;

        let mut s = Directory::new(tmp.path());
        let got = s.read(10, Scenario::General).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].text, "only-this");
    }

    #[tokio::test]
    async fn subdirectories_are_ignored() {
        let tmp = TempDir::new().unwrap();
        put(
            tmp.path(),
            "x.json",
            br#"{"doc_id":"doc_x","content":"top"}"#,
        )
        .await;
        let sub = tmp.path().join("sub");
        tokio::fs::create_dir(&sub).await.unwrap();
        put(&sub, "y.json", br#"{"doc_id":"doc_y","content":"nested"}"#).await;

        let mut s = Directory::new(tmp.path());
        let got = s.read(10, Scenario::General).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].text, "top");
    }

    #[tokio::test]
    async fn case_insensitive_extension() {
        let tmp = TempDir::new().unwrap();
        put(
            tmp.path(),
            "a.JSON",
            br#"{"doc_id":"doc_a","content":"hi"}"#,
        )
        .await;
        let mut s = Directory::new(tmp.path());
        let got = s.read(10, Scenario::General).await.unwrap();
        assert_eq!(got.len(), 1);
    }

    #[tokio::test]
    async fn invalid_json_returns_invalid_metadata() {
        let tmp = TempDir::new().unwrap();
        put(tmp.path(), "a.json", b"not-json").await;
        let mut s = Directory::new(tmp.path());
        let err = s.read(10, Scenario::General).await.unwrap_err();
        assert!(matches!(err, Error::InvalidMetadata(_)));
    }

    #[tokio::test]
    async fn batch_size_zero_is_noop() {
        let tmp = TempDir::new().unwrap();
        put(tmp.path(), "a.json", br#"{"doc_id":"doc_a","content":"x"}"#).await;
        let mut s = Directory::new(tmp.path());
        let got = s.read(0, Scenario::General).await.unwrap();
        assert!(got.is_empty());
        assert!(s.buffered_len().is_none(), "batch_size=0 时不应触发枚举");
    }

    #[tokio::test]
    async fn scenario_from_json_overrides_default() {
        let tmp = TempDir::new().unwrap();
        put(
            tmp.path(),
            "a.json",
            br#"{"doc_id":"doc_a","content":"question?","scenario":"qa"}"#,
        )
        .await;

        let mut s = Directory::new(tmp.path());
        let got = s.read(10, Scenario::General).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].scenario, Scenario::Qa); // JSON 中的 scenario 覆盖传入的 General
    }
}
