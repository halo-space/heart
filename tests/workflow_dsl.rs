use halo::agent::workflow::dsl::{CURRENT_VERSION, Document, parse};
use serde_json::json;

fn base() -> serde_json::Value {
    json!({
        "version": CURRENT_VERSION,
        "definition_id": "example",
        "nodes": {
            "reload_001": {
                "type": "function",
                "name": "reload",
                "config": {},
                "schema": {
                    "input": {"type": "object"},
                    "output": {"type": "object"}
                },
                "attrs": {"input": {}}
            }
        },
        "edges": [
            {"from": "START", "to": "reload_001"},
            {"from": "reload_001", "to": "END"}
        ]
    })
}

#[test]
fn public_parser_accepts_page_code_function_source() {
    let mut value = base();
    value["nodes"]["reload_001"]["config"] = json!({
        "language": "python",
        "entry": "reload",
        "source": "def reload(input):\n    return {\"x\": 1}"
    });

    let document = parse(value).expect("page code node should be valid DSL");
    assert_eq!(document.nodes["reload_001"].name, "reload");
    assert_eq!(document.nodes["reload_001"].config["language"], "python");
}

#[test]
fn public_parser_rejects_path_and_secret_fields() {
    let mut value = base();
    value["nodes"]["reload_001"]["config"] = json!({
        "language": "python",
        "entry": "reload",
        "source": "def reload(input): return {}",
        "path": "/tmp/reload.py",
        "api_key": "secret"
    });

    assert_eq!(
        Document::parse(value).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn public_parser_rejects_a_cycle_in_ordinary_edges() {
    let mut value = base();
    value["nodes"]["other"] = json!({
        "type": "function",
        "name": "other",
        "schema": {
            "input": {"type": "object"},
            "output": {"type": "object"}
        }
    });
    value["edges"] = json!([
        {"from": "START", "to": "reload_001"},
        {"from": "reload_001", "to": "other"},
        {"from": "other", "to": "reload_001"},
        {"from": "other", "to": "END"}
    ]);

    // The parser currently reports the graph as structurally invalid even
    // before Runtime compilation; it must never produce a partial document.
    assert_eq!(
        Document::parse(value).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn public_parser_requires_a_schema_for_user_code_nodes() {
    let mut value = base();
    value["nodes"]["reload_001"]
        .as_object_mut()
        .unwrap()
        .remove("schema");
    assert_eq!(
        Document::parse(value).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn public_parser_rejects_a_reachable_branch_without_end() {
    let mut value = base();
    value["nodes"]["orphan"] = json!({
        "type": "function",
        "name": "orphan",
        "schema": {
            "input": {"type": "object"},
            "output": {"type": "object"}
        }
    });
    value["edges"] = json!([
        {"from": "START", "to": "reload_001"},
        {"from": "START", "to": "orphan"},
        {"from": "reload_001", "to": "END"}
    ]);

    let error = Document::parse(value).unwrap_err();
    assert!(error.message.contains("reach END"));
}

#[test]
fn public_parser_rejects_an_invalid_json_schema() {
    let mut value = base();
    value["nodes"]["reload_001"]["schema"]["input"] = json!({"type": "unknown"});

    assert_eq!(
        Document::parse(value).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
}
