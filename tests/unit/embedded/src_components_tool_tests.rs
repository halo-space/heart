use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use futures_executor::block_on;
use serde_json::json;

use super::*;
use crate::components::message::Role;
use crate::tool::execute;

#[test]
fn schema_checks_patterns_references_and_composition_before_hooks() {
    let called = Arc::new(AtomicBool::new(false));
    let tool = HookTool {
        definition: ToolDefinition {
            parameters: serde_json::from_value(json!({
                "type": "object",
                "$defs": {"city": {"type": "string", "pattern": "^[A-Z][a-z]+$"}},
                "required": ["city"],
                "properties": {"city": {"$ref": "#/$defs/city"}},
                "anyOf": [{"properties": {"city": {"const": "London"}}},
                          {"properties": {"city": {"const": "Paris"}}}]
            }))
            .unwrap(),
            ..definition()
        },
        called: called.clone(),
    };
    for city in ["london", "Beijing"] {
        let call = Call {
            call_id: "call_1".into(),
            name: "weather".into(),
            arguments: Map::from_iter([("city".into(), json!(city))]),
        };
        assert_eq!(
            block_on(execute(&tool, call, &Cancellation::new()))
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
    }
    assert!(!called.load(Ordering::SeqCst));
    assert!(
        validate_call(
            &tool.definition,
            &Call {
                call_id: "call_1".into(),
                name: "weather".into(),
                arguments: Map::from_iter([("city".into(), json!("London"))]),
            }
        )
        .is_ok()
    );
    // London passes initially, but before changes it to Beijing, which
    // violates anyOf. The actual business call must still be excluded.
    assert_eq!(
        block_on(execute(
            &tool,
            Call {
                call_id: "call_2".into(),
                name: "weather".into(),
                arguments: Map::from_iter([("city".into(), json!("London"))]),
            },
            &Cancellation::new()
        ))
        .unwrap_err()
        .code,
        "INVALID_ARGUMENTS"
    );
    assert!(!called.load(Ordering::SeqCst));
}

#[test]
fn closed_empty_schema_rejects_extra_arguments() {
    let definition = ToolDefinition {
        parameters: serde_json::from_value(
            json!({"type": "object", "additionalProperties": false}),
        )
        .unwrap(),
        ..definition()
    };
    let mut call = Call {
        call_id: "call_1".into(),
        name: "weather".into(),
        arguments: Map::new(),
    };
    assert!(validate_call(&definition, &call).is_ok());
    call.arguments.insert("extra".into(), json!(1));
    assert_eq!(
        validate_call(&definition, &call).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn invalid_or_unresolved_schemas_fail_before_model_registration() {
    for schema in [
        json!({"type": "object", "properties": {"unused": {"type": "unknown"}}}),
        json!({"type": "object", "properties": {"unused": {"minimum": "wrong"}}}),
        json!({"type": "object", "properties": {"unused": {"pattern": "["}}}),
        json!({"type": "object", "$ref": "https://example.invalid/schema.json"}),
        json!({"type": "object", "$ref": "file:///etc/passwd"}),
    ] {
        let definition = ToolDefinition {
            parameters: serde_json::from_value(schema).unwrap(),
            ..definition()
        };
        assert_eq!(definition.validate().unwrap_err().code, "INVALID_ARGUMENTS");
    }
}

fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "weather".into(),
        description: "lookup weather".into(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "required": ["city"],
            "properties": {"city": {"type": "string"}}
        }))
        .unwrap(),
    }
}

struct HookTool {
    definition: ToolDefinition,
    called: Arc<AtomicBool>,
}

impl Tool for HookTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn before(&self, call: &mut Call, _cancellation: &Cancellation) -> Result<(), Error> {
        call.arguments.insert("city".into(), json!("Beijing"));
        Ok(())
    }

    async fn call(&self, call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
        self.called.store(true, Ordering::SeqCst);
        Ok(Message {
            role: Role::Tool,
            content: Vec::new(),
            values: Map::from_iter([("city".into(), call.arguments["city"].clone())]),
            metadata: Map::new(),
            usage: None,
        })
    }

    async fn after(
        &self,
        _call: &Call,
        message: &mut Message,
        _cancellation: &Cancellation,
    ) -> Result<(), Error> {
        message.values.insert("after".into(), json!(true));
        Ok(())
    }

    async fn eval(
        &self,
        _call: &Call,
        message: &Message,
        _cancellation: &Cancellation,
    ) -> Result<Evaluation, Error> {
        Ok(Evaluation {
            passed: message.values["after"] == json!(true),
            feedback: None,
        })
    }
}

#[test]
fn execute_runs_hooks_in_order_and_returns_final_message() {
    let called = Arc::new(AtomicBool::new(false));
    let tool = HookTool {
        definition: definition(),
        called: called.clone(),
    };
    let call = Call {
        call_id: "call_1".into(),
        name: "weather".into(),
        arguments: Map::from_iter([("city".into(), json!("London"))]),
    };
    let (message, evaluation) = block_on(execute(&tool, call, &Cancellation::new())).unwrap();
    assert!(called.load(Ordering::SeqCst));
    assert_eq!(message.values["city"], "Beijing");
    assert_eq!(message.values["after"], true);
    assert!(evaluation.passed);
}

struct RejectingTool {
    definition: ToolDefinition,
    called: Arc<AtomicBool>,
}

impl Tool for RejectingTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn before(&self, _call: &mut Call, _cancellation: &Cancellation) -> Result<(), Error> {
        Err(Error::new("INVALID_ARGUMENTS", "rejected by before"))
    }

    async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
        self.called.store(true, Ordering::SeqCst);
        Ok(Message::new(Role::Tool))
    }
}

#[test]
fn before_error_stops_call() {
    let called = Arc::new(AtomicBool::new(false));
    let tool = RejectingTool {
        definition: definition(),
        called: called.clone(),
    };
    let call = Call {
        call_id: "call_1".into(),
        name: "weather".into(),
        arguments: Map::from_iter([("city".into(), json!("Beijing"))]),
    };
    let error = block_on(execute(&tool, call, &Cancellation::new())).unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert!(!called.load(Ordering::SeqCst));
}

#[test]
fn arguments_are_validated_before_lifecycle() {
    let called = Arc::new(AtomicBool::new(false));
    let tool = HookTool {
        definition: definition(),
        called: called.clone(),
    };
    let call = Call {
        call_id: "call_1".into(),
        name: "weather".into(),
        arguments: Map::new(),
    };
    let error = block_on(execute(&tool, call, &Cancellation::new())).unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert!(!called.load(Ordering::SeqCst));
}

#[test]
fn schema_validation_rejects_wrong_types_and_unknown_properties() {
    let called = Arc::new(AtomicBool::new(false));
    let tool = HookTool {
        definition: ToolDefinition {
            name: "weather".into(),
            description: "lookup weather".into(),
            parameters: serde_json::from_value(json!({
                "type": "object",
                "required": ["city"],
                "additionalProperties": false,
                "properties": {"city": {"type": "string"}}
            }))
            .unwrap(),
        },
        called: called.clone(),
    };
    let wrong_type = Call {
        call_id: "call_1".into(),
        name: "weather".into(),
        arguments: Map::from_iter([(String::from("city"), json!(42))]),
    };
    assert_eq!(
        block_on(execute(&tool, wrong_type, &Cancellation::new()))
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    let unknown = Call {
        call_id: "call_2".into(),
        name: "weather".into(),
        arguments: Map::from_iter([
            (String::from("city"), json!("Beijing")),
            (String::from("extra"), json!(true)),
        ]),
    };
    assert_eq!(
        block_on(execute(&tool, unknown, &Cancellation::new()))
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    assert!(!called.load(Ordering::SeqCst));
}

struct IdentityMutatingTool {
    definition: ToolDefinition,
    called: Arc<AtomicBool>,
}

impl Tool for IdentityMutatingTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn before(&self, call: &mut Call, _cancellation: &Cancellation) -> Result<(), Error> {
        call.call_id = "changed".into();
        Ok(())
    }

    async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
        self.called.store(true, Ordering::SeqCst);
        Ok(Message::new(Role::Tool))
    }
}

#[test]
fn before_cannot_change_tool_identity() {
    let called = Arc::new(AtomicBool::new(false));
    let tool = IdentityMutatingTool {
        definition: definition(),
        called: called.clone(),
    };
    let call = Call {
        call_id: "call_1".into(),
        name: "weather".into(),
        arguments: Map::from_iter([(String::from("city"), json!("Beijing"))]),
    };
    let error = block_on(execute(&tool, call, &Cancellation::new())).unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert!(!called.load(Ordering::SeqCst));
}

#[test]
fn schema_validation_enforces_numeric_minimum() {
    let called = Arc::new(AtomicBool::new(false));
    let tool = HookTool {
        definition: ToolDefinition {
            name: "weather".into(),
            description: "lookup weather".into(),
            parameters: serde_json::from_value(json!({
                "type": "object",
                "required": ["days"],
                "properties": {"days": {"type": "integer", "minimum": 1}}
            }))
            .unwrap(),
        },
        called: called.clone(),
    };
    let call = Call {
        call_id: "call_1".into(),
        name: "weather".into(),
        arguments: Map::from_iter([(String::from("days"), json!(0))]),
    };
    let error = block_on(execute(&tool, call, &Cancellation::new())).unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert!(!called.load(Ordering::SeqCst));
}
