//! PageWiki ES mapping JSON helper。
//!
//! 第一版只提供 mapping JSON；真正建索引由业务侧 ops 负责。

use serde_json::{Value, json};

/// 返回 PageWiki 的 ES mapping JSON。
///
/// - 19 顶层字段；`dynamic = "false"` 拒绝未声明字段写入。
/// - `embedding.dims` 由 `embedding_dims` 参数指定。
/// - **不**包含 `settings`：分片 / 副本 / analyzer 走 ops。
pub fn pagewiki_mapping(embedding_dims: usize) -> Value {
    json!({
        "mappings": {
            "dynamic": "false",
            "properties": {
                "id":              { "type": "keyword" },
                "doc_id":          { "type": "keyword" },
                "version":         { "type": "keyword" },
                "scenario":        { "type": "keyword" },
                "idx":             { "type": "integer" },
                "header":          { "type": "text" },
                "content":         { "type": "text" },
                "content_tokens":  { "type": "text", "analyzer": "whitespace" },
                "keywords":        { "type": "keyword" },
                "keyword_tokens":  { "type": "text", "analyzer": "whitespace" },
                "questions":       { "type": "text" },
                "question_tokens": { "type": "text", "analyzer": "whitespace" },
                "tags":            { "type": "keyword" },
                "attributes":      { "type": "object", "enabled": false },
                "graph":           { "type": "object", "enabled": false },
                "metadata":        { "type": "object", "dynamic": true },
                "images":          { "type": "keyword" },
                "spans":           { "type": "object", "enabled": false },
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dims,
                    "index": true,
                    "similarity": "cosine"
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn properties_has_19_fields() {
        let m = pagewiki_mapping(1024);
        let props = m["mappings"]["properties"].as_object().unwrap();
        assert_eq!(props.len(), 19);
    }

    #[test]
    fn embedding_dims_parameterized() {
        assert_eq!(
            pagewiki_mapping(768)["mappings"]["properties"]["embedding"]["dims"],
            json!(768)
        );
    }

    #[test]
    fn no_settings_section() {
        assert!(pagewiki_mapping(1024).get("settings").is_none());
    }

    #[test]
    fn dynamic_is_strict_false() {
        assert_eq!(
            pagewiki_mapping(1024)["mappings"]["dynamic"],
            json!("false")
        );
    }

    #[test]
    fn metadata_dynamic_true_others_disabled() {
        let m = pagewiki_mapping(1024);
        let props = &m["mappings"]["properties"];
        assert_eq!(
            props["metadata"],
            json!({"type": "object", "dynamic": true})
        );
        for f in ["attributes", "graph", "spans"] {
            assert_eq!(props[f], json!({"type": "object", "enabled": false}));
        }
    }
}
