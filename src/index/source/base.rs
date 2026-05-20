//! [`Base`] trait —— 输入层唯一的异步入口。

use serde_json::{Map, Value};

use crate::index::source::types::{Error, Item, Scenario};

/// 异步文本数据源。
#[allow(async_fn_in_trait)]
pub trait Base: Send + Sync {
    /// 读取最多 `batch_size` 个 [`Item`]，统一改写 scenario。
    async fn read(&mut self, batch_size: usize, scenario: Scenario) -> Result<Vec<Item>, Error>;

    /// 校验 `metadata.doc_id` 存在且非空。
    async fn validate_doc_id(&self, metadata: &Map<String, Value>) -> Result<(), Error> {
        let Some(Value::String(raw)) = metadata.get("doc_id") else {
            return Err(Error::MissingDocId);
        };
        if raw.trim().is_empty() {
            return Err(Error::MissingDocId);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Mock;

    impl Base for Mock {
        async fn read(
            &mut self,
            _batch_size: usize,
            _scenario: Scenario,
        ) -> Result<Vec<Item>, Error> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn validate_doc_id_passes_for_non_empty_string() {
        let mut md = Map::new();
        md.insert("doc_id".into(), json!("doc_001"));
        Mock.validate_doc_id(&md).await.unwrap();
    }

    #[tokio::test]
    async fn validate_doc_id_fails_when_key_missing() {
        let md = Map::new();
        let err = Mock.validate_doc_id(&md).await.unwrap_err();
        assert!(matches!(err, Error::MissingDocId));
    }

    #[tokio::test]
    async fn validate_doc_id_fails_for_non_string_values() {
        for v in [
            json!(123),
            json!(true),
            json!(null),
            json!({"x": 1}),
            json!([1, 2]),
        ] {
            let mut md = Map::new();
            md.insert("doc_id".into(), v);
            let err = Mock.validate_doc_id(&md).await.unwrap_err();
            assert!(matches!(err, Error::MissingDocId));
        }
    }

    #[tokio::test]
    async fn validate_doc_id_fails_for_blank_string() {
        let mut md = Map::new();
        md.insert("doc_id".into(), json!("   "));
        let err = Mock.validate_doc_id(&md).await.unwrap_err();
        assert!(matches!(err, Error::MissingDocId));
    }
}
