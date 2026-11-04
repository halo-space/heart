use std::collections::BTreeMap;

use halo::{
    Cancellation,
    agent::core::sandbox::{Config, Local},
    agent::core::workflow::{
        code::{Code, Object},
        dsl::parse,
    },
    agent::node::{Node, Signal},
    agent::workflow::{WorkflowBuilder, graph::Schema},
};
use serde_json::{Value, json};

fn document() -> serde_json::Value {
    json!({
        "version": 1,
        "definition_id": "code-test",
        "nodes": {
            "add": {
                "type": "function",
                "name": "add",
                "config": {
                    "language": "python",
                    "entry": "add",
                    "source": "def add(value):\n    print('diagnostic output')\n    return {'sum': value['left'] + value['right']}"
                },
                "schema": {
                    "input": {
                        "type": "object",
                        "required": ["left", "right"],
                        "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}}
                    },
                    "output": {
                        "type": "object",
                        "required": ["sum"],
                        "properties": {"sum": {"type": "integer"}}
                    }
                }
            }
        },
        "edges": [
            {"from": "START", "to": "add"},
            {"from": "add", "to": "END"}
        ]
    })
}

#[tokio::test]
async fn executes_page_function_with_real_python_and_local_sandbox() {
    let document = parse(document()).expect("DSL should parse");
    let node = &document.nodes["add"];
    let executor = Code::from_dsl(Local::new(Config::default()).expect("local config"), node)
        .expect("code node should compile");
    let input = Object(BTreeMap::from([
        ("left".into(), Value::from(2)),
        ("right".into(), Value::from(3)),
    ]));

    let result = executor
        .call(input, &Cancellation::new())
        .await
        .expect("python function should execute");
    let Signal::Complete(output) = result else {
        panic!("code node should complete");
    };
    assert_eq!(output.0.get("sum"), Some(&Value::from(5)));
}

#[tokio::test]
async fn input_schema_rejects_before_starting_page_code() {
    let document = parse(document()).expect("DSL should parse");
    let node = &document.nodes["add"];
    let executor = Code::from_dsl(Local::new(Config::default()).expect("local config"), node)
        .expect("code node should compile");
    let input = Object(BTreeMap::from([("left".into(), Value::from(2))]));

    let error = executor
        .call(input, &Cancellation::new())
        .await
        .expect_err("missing required input must fail");
    assert_eq!(error.code, "INVALID_ARGUMENTS");
}

#[tokio::test]
async fn output_schema_rejects_real_page_code_results() {
    let mut value = document();
    value["nodes"]["add"]["config"]["source"] = json!("def add(value): return {'wrong': 0}");
    let document = parse(value).expect("DSL should parse");
    let node = &document.nodes["add"];
    let executor = Code::from_dsl(Local::new(Config::default()).expect("local config"), node)
        .expect("code node should compile");
    let input = Object(BTreeMap::from([
        ("left".into(), Value::from(2)),
        ("right".into(), Value::from(3)),
    ]));

    let error = executor
        .call(input, &Cancellation::new())
        .await
        .expect_err("schema mismatch must fail");
    assert_eq!(error.code, "INVALID_RESPONSE");
}

#[test]
fn code_executor_can_be_bound_as_a_normal_workflow_node() {
    let document = parse(document()).expect("DSL should parse");
    let node = &document.nodes["add"];
    let code = Code::from_dsl(Local::new(Config::default()).expect("local config"), node)
        .expect("code node should compile");
    let schema = Schema {
        input: node.schema.input.clone(),
        output: node.schema.output.clone(),
    };
    let mut builder = WorkflowBuilder::new("code-test");
    builder
        .add_node_with_schema("add", code, schema)
        .expect("code executor should satisfy Node");
    builder
        .add_edge("START", "add")
        .expect("start edge should be valid");
    builder
        .add_edge("add", "END")
        .expect("end edge should be valid");
    let definition = builder.compile().expect("workflow should compile");
    assert_eq!(definition.definition().nodes["add"].node_id, "add");
    assert_eq!(
        definition.definition().nodes["add"].schema.input["required"],
        json!(["left", "right"])
    );
}
