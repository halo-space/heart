//! `source` 模块对外暴露的数据模型：[`Scenario`] / [`Item`] / [`Error`]。

use serde::{Deserialize, Serialize};

/// 检索场景。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scenario {
    /// 通用文本（默认）。
    #[default]
    General,
    /// 问答对。
    Qa,
    /// 操作手册 / 结构化指南。
    Manual,
}

/// 文本记录。
///
/// 业务字段统一存放在 `metadata` 中。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    /// 原始文本内容（UTF-8）。
    pub text: String,
    /// `Base::read` 注入的检索场景。
    pub scenario: Scenario,
    /// 自由形态元数据；**必须**包含可用的 `doc_id` 字符串。
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// `source` 模块错误类型。
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// `metadata.doc_id` 缺失、不是字符串、或 trim 后为空。
    #[error("metadata.doc_id missing or unusable (expected non-empty string)")]
    MissingDocId,

    /// 底层 I/O 故障（文件读、目录遍历等）。
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// 字节无法解码为 UTF-8。
    #[error("utf-8 decode error: {0}")]
    Decode(#[from] std::string::FromUtf8Error),

    /// 调用方传入的 scenario 值实现无法服务。
    #[error("unsupported scenario value: {0}")]
    UnsupportedScenario(String),

    /// sidecar 元数据文件无法解析。
    #[error("metadata file invalid: {0}")]
    InvalidMetadata(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scenario_serializes_to_lowercase() {
        assert_eq!(
            serde_json::to_string(&Scenario::General).unwrap(),
            "\"general\""
        );
        assert_eq!(serde_json::to_string(&Scenario::Qa).unwrap(), "\"qa\"");
        assert_eq!(
            serde_json::to_string(&Scenario::Manual).unwrap(),
            "\"manual\""
        );
    }

    #[test]
    fn scenario_deserializes_from_lowercase() {
        assert_eq!(
            serde_json::from_str::<Scenario>("\"general\"").unwrap(),
            Scenario::General
        );
        assert_eq!(
            serde_json::from_str::<Scenario>("\"qa\"").unwrap(),
            Scenario::Qa
        );
        assert_eq!(
            serde_json::from_str::<Scenario>("\"manual\"").unwrap(),
            Scenario::Manual
        );
    }

    #[test]
    fn scenario_rejects_unknown_variant() {
        let err = serde_json::from_str::<Scenario>("\"custom\"").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown variant"), "got: {msg}");
    }

    #[test]
    fn scenario_default_is_general() {
        assert_eq!(Scenario::default(), Scenario::General);
    }

    #[test]
    fn scenario_is_copy() {
        let a = Scenario::General;
        let _b = a;
        let _c = a;
        assert_eq!(a, Scenario::General);
    }

    #[test]
    fn item_json_roundtrip_preserves_metadata_order() {
        let mut metadata = serde_json::Map::new();
        metadata.insert("doc_id".into(), json!("doc_001"));
        metadata.insert("title".into(), json!("First"));
        metadata.insert("author".into(), json!("Alice"));

        let item = Item {
            text: "hello".into(),
            scenario: Scenario::Qa,
            metadata: metadata.clone(),
        };
        let s = serde_json::to_string(&item).unwrap();
        let back: Item = serde_json::from_str(&s).unwrap();
        let keys: Vec<&str> = back.metadata.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["doc_id", "title", "author"]);
        assert_eq!(back.text, "hello");
        assert_eq!(back.scenario, Scenario::Qa);
    }

    #[test]
    fn error_missing_doc_id_message_is_informative() {
        let msg = Error::MissingDocId.to_string();
        assert!(msg.contains("doc_id"), "got: {msg}");
        assert!(
            msg.contains("missing") || msg.contains("unusable"),
            "got: {msg}"
        );
    }
}
