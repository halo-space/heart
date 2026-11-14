use super::*;
use serde_json::json;

fn valid_document() -> Value {
    json!({
        "version": 2,
        "definition_id": "weather",
        "nodes": {
            "search": {
                "type": "function",
                "name": "search",
                "schema": {
                    "input": {"type": "object"},
                    "output": {"type": "object"}
                },
                "attrs": {
                    "input": {
                        "query": {"from": {"node_id": "START", "field": "query"}}
                    }
                }
            }
        },
        "edges": [
            {"from": "START", "to": "search"},
            {"from": "search", "to": "END"}
        ]
    })
}

#[test]
fn parses_native_document_and_defaults_relations_and_retry() {
    let document = Document::parse(valid_document()).expect("valid Native DSL");
    assert_eq!(document.version, CURRENT_VERSION);
    assert_eq!(document.nodes["search"].retry.max_retries, 0);
    assert!(document.join_edges.is_empty());
}

#[test]
fn accepts_page_code_function_without_local_lookup() {
    let mut value = valid_document();
    value["nodes"]["search"]["config"] = json!({
        "language": "python",
        "entry": "reload",
        "source": "def reload(input):\n    return {\"x\": 1}"
    });
    let document = Document::parse(value).expect("code node is static DSL data");
    assert_eq!(document.nodes["search"].config["language"], "python");
}

#[test]
fn rejects_unknown_top_level_field() {
    let mut value = valid_document();
    value["unexpected"] = json!(true);
    let error = Document::parse(value).unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
}

#[test]
fn rejects_missing_code_config_fields() {
    let mut value = valid_document();
    value["nodes"]["search"]["config"] = json!({"language": "python"});
    let error = Document::parse(value).unwrap_err();
    assert!(error.message.contains("config.entry"));
}

#[test]
fn rejects_unknown_source_node() {
    let mut value = valid_document();
    value["nodes"]["search"]["attrs"]["input"]["query"]["from"]["node_id"] = json!("missing");
    let error = Document::parse(value).unwrap_err();
    assert!(error.message.contains("unknown workflow node"));
}

#[test]
fn rejects_duplicate_edges_and_self_edges() {
    let mut value = valid_document();
    value["edges"] = json!([
        {"from": "START", "to": "search"},
        {"from": "START", "to": "search"},
        {"from": "search", "to": "END"}
    ]);
    assert_eq!(
        Document::parse(value).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );

    let mut value = valid_document();
    value["edges"] = json!([
        {"from": "START", "to": "search"},
        {"from": "search", "to": "search"},
        {"from": "search", "to": "END"}
    ]);
    assert_eq!(
        Document::parse(value).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn rejects_join_mixed_with_ordinary_incoming_edge() {
    let mut value = valid_document();
    value["nodes"]["other"] = json!({
        "type": "function",
        "name": "other",
        "schema": {
            "input": {"type": "object"},
            "output": {"type": "object"}
        }
    });
    value["edges"] = json!([
        {"from": "START", "to": "search"},
        {"from": "search", "to": "other"},
        {"from": "other", "to": "END"}
    ]);
    value["join_edges"] = json!([{"from": ["search", "other"], "to": "other"}]);
    assert_eq!(
        Document::parse(value).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn rejects_unreachable_node() {
    let mut value = valid_document();
    value["nodes"]["orphan"] = json!({
        "type": "function",
        "name": "orphan",
        "schema": {
            "input": {"type": "object"},
            "output": {"type": "object"}
        }
    });
    let error = Document::parse(value).unwrap_err();
    assert!(error.message.contains("reachable"));
}
