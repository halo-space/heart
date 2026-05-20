//! Builder::new 与 Builder::build——7 步流水线主入口。

use std::collections::HashMap;
use crate::index::builder::types::{Builder, Error};
use crate::index::source;
use crate::index::pagewiki;
use crate::index::builder::tokenize::Tokenizer;
use crate::index::builder::embed::Embedder;
use crate::index::builder::{normalize, promote};
use crate::utils::{idx, timex};

/// 从 `item.metadata` 中读取 doc_id，写入 `pw` 并补齐其余系统字段。
fn assemble_system_fields(
    pw: &mut pagewiki::PageWiki,
    item: &source::Item,
    idx: usize,
    version: &str,
) -> Result<(), Error> {
    // id：UUID v4（调用通用工具）
    pw.id = Some(idx::new_uuid_v4());

    // doc_id：从 metadata 读取，缺失或为空则报错
    let doc_id = item
        .metadata
        .get("doc_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or(Error::MissingDocId)?
        .to_string();
    pw.doc_id = Some(doc_id);

    // 防御性清理：cut 实现不应填 doc_id，但若有则移除
    pw.metadata.remove("doc_id");

    pw.version  = Some(version.to_string());
    pw.scenario = Some(item.scenario);
    pw.idx      = Some(idx);

    Ok(())
}

impl Builder {
    /// 构造 Builder。
    ///
    /// 若 `metadata_promote_fields` 中包含不允许的字段，立即返回
    /// `Err(Error::InvalidPromoteField)`。
    pub fn new(
        pagewikis: HashMap<source::Scenario, Box<dyn pagewiki::Base>>,
        metadata_promote_fields: Vec<String>,
        tokenizer: Box<dyn Tokenizer>,
        embedder: Option<Box<dyn Embedder>>,
    ) -> Result<Self, Error> {
        for field in &metadata_promote_fields {
            if !promote::is_allowed(field) {
                return Err(Error::InvalidPromoteField(field.clone()));
            }
        }
        Ok(Self { pagewikis, metadata_promote_fields, tokenizer, embedder })
    }

    /// 对一批 Item 执行 7 步构建流水线，返回完整 PageWiki 列表。
    ///
    /// 任意步骤失败立即 `?` 透传，不返回部分结果。
    pub async fn build(&self, items: Vec<source::Item>) -> Result<Vec<pagewiki::PageWiki>, Error> {
        let _span = tracing::debug_span!(
            "index.build",
            n_items = items.len(),
            promote_fields = ?self.metadata_promote_fields,
        );

        let version = timex::current_millis_string();
        let mut result = Vec::new();

        for item in items {
            let t0 = std::time::Instant::now();

            // 步骤 1：文本规范化
            let normalized = normalize::normalize_text(&item.text);

            // 步骤 2：查找对应 scenario 的切分实现
            let cut_impl = self.pagewikis.get(&item.scenario)
                .ok_or(Error::MissingScenario(item.scenario))?;

            // 步骤 3：切分
            let mut drafts = cut_impl.cut(&normalized).await
                .map_err(|e| { tracing::warn!(stage = "cut", error = %e, "index.build.failed"); e })?;

            // 步骤 4：按 spans.start 升序排列；无 spans 的 chunk 排到末尾
            drafts.sort_by_key(|pw| pw.spans.first().map(|s| s.start).unwrap_or(usize::MAX));

            let n_pages = drafts.len();
            let mut doc_id_str = String::new();

            for (idx, mut pw) in drafts.into_iter().enumerate() {
                // 步骤 5：补齐系统字段（id / doc_id / version / scenario / idx）
                assemble_system_fields(&mut pw, &item, idx, &version)
                    .map_err(|e| { tracing::warn!(stage = "assemble", error = %e, "index.build.failed"); e })?;

                if doc_id_str.is_empty() {
                    doc_id_str = pw.doc_id.clone().unwrap_or_default();
                }

                // 步骤 6：写入 metadata 并应用 promote 字段提升
                pw.metadata = item.metadata.clone();
                pw.metadata.remove("doc_id");
                promote::apply_promote(&mut pw, &self.metadata_promote_fields)
                    .map_err(|e| { tracing::warn!(stage = "promote", error = %e, "index.build.failed"); e })?;

                // 步骤 7a：分词（content / keywords / questions 三路）
                pw.content_tokens = self.tokenizer.tokenize(&pw.content).await
                    .map_err(|e| { tracing::warn!(stage = "tokenize", error = %e, "index.build.failed"); Error::Tokenize(e.to_string()) })?;
                pw.keyword_tokens = self.tokenizer.tokenize(&pw.keywords.join(" ")).await
                    .map_err(|e| { tracing::warn!(stage = "tokenize", error = %e, "index.build.failed"); Error::Tokenize(e.to_string()) })?;
                pw.question_tokens = self.tokenizer.tokenize(&pw.questions.join(" ")).await
                    .map_err(|e| { tracing::warn!(stage = "tokenize", error = %e, "index.build.failed"); Error::Tokenize(e.to_string()) })?;

                // 步骤 7b：生成向量（可选；None 表示不生成 embedding）
                if let Some(embedder) = &self.embedder {
                    pw.embedding = Some(
                        embedder.embed(&pw.content).await
                            .map_err(|e| { tracing::warn!(stage = "embed", error = %e, "index.build.failed"); Error::Embed(e.to_string()) })?
                    );
                }

                result.push(pw);
            }

            let elapsed_ms = t0.elapsed().as_millis() as u64;
            tracing::debug!(doc_id = %doc_id_str, n_pages, elapsed_ms, "index.build.item.done");
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Map, json};
    use crate::index::builder::tokenize::NoopTokenizer;

    /// 构造带 doc_id 的最简 Item
    fn make_item(text: &str, doc_id: &str) -> source::Item {
        let mut metadata = Map::new();
        metadata.insert("doc_id".into(), json!(doc_id));
        source::Item { text: text.into(), scenario: source::Scenario::General, metadata }
    }

    fn make_builder() -> Builder {
        let mut pw_map: HashMap<source::Scenario, Box<dyn pagewiki::Base>> = HashMap::new();
        pw_map.insert(
            source::Scenario::General,
            Box::new(pagewiki::Fixed::new(200)),
        );
        Builder::new(pw_map, vec![], Box::new(NoopTokenizer), None).unwrap()
    }

    #[tokio::test]
    async fn 空输入返回空结果() {
        let b = make_builder();
        let res = b.build(vec![]).await.unwrap();
        assert!(res.is_empty());
    }

    #[tokio::test]
    async fn 缺少scenario返回错误() {
        // 构造一个 Manual item，但 builder 里没有 Manual 的实现
        let b = make_builder();
        let mut meta = Map::new();
        meta.insert("doc_id".into(), json!("d1"));
        let item = source::Item { text: "x".into(), scenario: source::Scenario::Manual, metadata: meta };
        let err = b.build(vec![item]).await.unwrap_err();
        assert!(matches!(err, Error::MissingScenario(_)));
    }

    #[tokio::test]
    async fn 缺少doc_id返回错误() {
        let mut pw_map: HashMap<source::Scenario, Box<dyn pagewiki::Base>> = HashMap::new();
        pw_map.insert(source::Scenario::General, Box::new(pagewiki::Fixed::new(200)));
        let b = Builder::new(pw_map, vec![], Box::new(NoopTokenizer), None).unwrap();
        let item = source::Item {
            text: "hello".into(),
            scenario: source::Scenario::General,
            metadata: Map::new(), // 无 doc_id
        };
        let err = b.build(vec![item]).await.unwrap_err();
        assert!(matches!(err, Error::MissingDocId));
    }

    #[tokio::test]
    async fn 同批次version一致() {
        let b = make_builder();
        let items = vec![make_item("text one", "d1"), make_item("text two", "d2")];
        let pages = b.build(items).await.unwrap();
        let versions: Vec<_> = pages.iter().map(|p| p.version.as_deref().unwrap()).collect();
        assert!(versions.windows(2).all(|w| w[0] == w[1]), "versions differ: {versions:?}");
    }

    #[tokio::test]
    async fn embedder_none时embedding为none() {
        let b = make_builder(); // embedder = None
        let pages = b.build(vec![make_item("hello world", "d1")]).await.unwrap();
        assert!(pages.iter().all(|p| p.embedding.is_none()));
    }

    #[tokio::test]
    async fn embedder_noop时embedding为some空vec() {
        use crate::index::builder::embed::NoopEmbedder;
        let mut pw_map: HashMap<source::Scenario, Box<dyn pagewiki::Base>> = HashMap::new();
        pw_map.insert(source::Scenario::General, Box::new(pagewiki::Fixed::new(200)));
        let b = Builder::new(pw_map, vec![], Box::new(NoopTokenizer), Some(Box::new(NoopEmbedder))).unwrap();
        let pages = b.build(vec![make_item("hello world", "d1")]).await.unwrap();
        assert!(pages.iter().all(|p| p.embedding == Some(vec![])));
    }
}
