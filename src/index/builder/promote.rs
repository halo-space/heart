//! metadata 字段提升辅助函数。

use crate::index::builder::types::Error;
use crate::index::pagewiki;
use serde_json::Value;

const ALLOWED: &[&str] = &[
    "header",
    "content",
    "keywords",
    "questions",
    "tags",
    "attributes",
    "graph",
    "metadata",
    "images",
];

/// 判断 `field` 是否允许出现在 `metadata_promote_fields` 中。
pub fn is_allowed(field: &str) -> bool {
    ALLOWED.contains(&field)
}

/// 把 `pw.metadata` 中指定字段提升到 `PageWiki` 对应顶层字段，并从 metadata 中移除。
pub fn apply_promote(pw: &mut pagewiki::PageWiki, fields: &[String]) -> Result<(), Error> {
    for field in fields {
        if !is_allowed(field) {
            return Err(Error::InvalidPromoteField(field.clone()));
        }
        let Some(val) = pw.metadata.remove(field.as_str()) else {
            continue;
        };
        promote_field(pw, field, val)?;
    }
    Ok(())
}

fn promote_field(pw: &mut pagewiki::PageWiki, field: &str, val: Value) -> Result<(), Error> {
    macro_rules! deser {
        ($target:expr, $T:ty) => {{
            let v: $T = serde_json::from_value(val).map_err(|e| Error::PromoteDeserialize {
                field: field.to_string(),
                reason: e.to_string(),
            })?;
            $target = v;
        }};
    }
    match field {
        "header" => deser!(pw.header, String),
        "content" => deser!(pw.content, String),
        "keywords" => deser!(pw.keywords, Vec<String>),
        "questions" => deser!(pw.questions, Vec<String>),
        "tags" => deser!(pw.tags, Vec<String>),
        "attributes" => deser!(pw.attributes, serde_json::Map<String, Value>),
        "graph" => deser!(pw.graph, pagewiki::Graph),
        "metadata" => deser!(pw.metadata, serde_json::Map<String, Value>),
        "images" => deser!(pw.images, Vec<String>),
        _ => unreachable!("is_allowed 已校验"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn default_pw() -> pagewiki::PageWiki {
        pagewiki::PageWiki::default()
    }

    #[test]
    fn 允许字段正确() {
        for f in ALLOWED {
            assert!(is_allowed(f), "{f} 应允许");
        }
        for f in [
            "id",
            "doc_id",
            "version",
            "scenario",
            "idx",
            "content_tokens",
            "keyword_tokens",
            "question_tokens",
            "embedding",
            "spans",
        ] {
            assert!(!is_allowed(f), "{f} 不应允许");
        }
        assert!(!is_allowed("foo_bar"));
    }

    #[test]
    fn 提升keywords() {
        let mut pw = default_pw();
        pw.metadata.insert("keywords".into(), json!(["x", "y"]));
        apply_promote(&mut pw, &["keywords".into()]).unwrap();
        assert_eq!(pw.keywords, vec!["x", "y"]);
        assert!(!pw.metadata.contains_key("keywords"));
    }

    #[test]
    fn 缺失字段跳过() {
        let mut pw = default_pw();
        pw.tags = vec!["existing".into()];
        apply_promote(&mut pw, &["tags".into()]).unwrap();
        assert_eq!(pw.tags, vec!["existing"]);
    }

    #[test]
    fn 类型不匹配返回错误() {
        let mut pw = default_pw();
        pw.metadata.insert("keywords".into(), json!("not-array"));
        let err = apply_promote(&mut pw, &["keywords".into()]).unwrap_err();
        assert!(matches!(err, Error::PromoteDeserialize { .. }));
    }

    #[test]
    fn 禁止字段返回错误() {
        let mut pw = default_pw();
        let err = apply_promote(&mut pw, &["title".into()]).unwrap_err();
        assert!(matches!(err, Error::InvalidPromoteField(_)));
    }
}
