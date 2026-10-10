use std::future::Future;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::message::{Message, Messages};
use crate::runtime::cancellation::Cancellation;

pub type Error = crate::Error;

/// The static, model-visible description of one executable Tool.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Map<String, Value>,
}

impl ToolDefinition {
    pub fn validate(&self) -> Result<(), Error> {
        self.validator().map(|_| ())
    }

    fn validator(&self) -> Result<jsonschema::Validator, Error> {
        if self.name.trim().is_empty() || self.description.trim().is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "tool name and description must be non-empty",
            ));
        }
        if self.parameters.get("type") != Some(&Value::String("object".into())) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "tool parameters schema must have object root",
            ));
        }
        compile_schema(&self.parameters)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Call {
    pub call_id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: Map<String, Value>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Evaluation {
    pub passed: bool,
    pub feedback: Option<Messages>,
}

impl Evaluation {
    pub fn passed() -> Self {
        Self {
            passed: true,
            feedback: None,
        }
    }

    pub fn failed(feedback: Messages) -> Self {
        Self {
            passed: false,
            feedback: Some(feedback),
        }
    }
}

/// A Tool owns its optional lifecycle hooks. The surrounding runtime owns
/// retries and execution state; an Agent may place a direct error context
/// message in its next model input, while a Workflow keeps the error direct.
pub trait Tool: Send + Sync {
    fn definition(&self) -> &ToolDefinition;

    fn before(
        &self,
        _call: &mut Call,
        _cancellation: &Cancellation,
    ) -> impl Future<Output = Result<(), Error>> + Send {
        async { Ok(()) }
    }

    fn call(
        &self,
        call: &Call,
        cancellation: &Cancellation,
    ) -> impl Future<Output = Result<Message, Error>> + Send;

    fn after(
        &self,
        _call: &Call,
        _message: &mut Message,
        _cancellation: &Cancellation,
    ) -> impl Future<Output = Result<(), Error>> + Send {
        async { Ok(()) }
    }

    fn eval(
        &self,
        _call: &Call,
        _message: &Message,
        _cancellation: &Cancellation,
    ) -> impl Future<Output = Result<Evaluation, Error>> + Send {
        async { Ok(Evaluation::passed()) }
    }
}

/// Run one Tool call through its fixed lifecycle. The returned tuple contains
/// the final business Message and the optional business evaluation; it is not a
/// second Tool result type.
///
/// This convenience entry does not retain Runtime state or perform retries.
/// Do not retry this whole function after an evaluation error: a caller that
/// needs stage retries must retain the result and retry `Tool::eval` only.
pub async fn execute<T: Tool>(
    tool: &T,
    call: Call,
    cancellation: &Cancellation,
) -> Result<(Message, Evaluation), Error> {
    let (call, message) = execute_call(tool, call, cancellation).await?;
    check_cancelled(cancellation)?;
    let evaluation = tool.eval(&call, &message, cancellation).await?;
    check_cancelled(cancellation)?;
    Ok((message, evaluation))
}

// Runtime uses this private boundary to retain the effective arguments and
// successful result before evaluating. A retry of eval must use this pair,
// not invoke the public whole-lifecycle convenience function again.
pub(crate) async fn execute_call<T: Tool>(
    tool: &T,
    call: Call,
    cancellation: &Cancellation,
) -> Result<(Call, Message), Error> {
    let call = prepare_call(tool, call, cancellation).await?;
    call_prepared(tool, call, cancellation).await
}

// Private split lets Agent authorization inspect the effective arguments
// without repeating before or executing call ahead of that authorization.
pub(crate) async fn prepare_call<T: Tool>(
    tool: &T,
    mut call: Call,
    cancellation: &Cancellation,
) -> Result<Call, Error> {
    check_cancelled(cancellation)?;
    validate_call(tool.definition(), &call)?;
    let call_id = call.call_id.clone();
    let name = call.name.clone();
    tool.before(&mut call, cancellation).await?;
    check_cancelled(cancellation)?;
    ensure_call_identity(&call, &call_id, &name)?;
    validate_call(tool.definition(), &call)?;
    Ok(call)
}

pub(crate) async fn call_prepared<T: Tool>(
    tool: &T,
    call: Call,
    cancellation: &Cancellation,
) -> Result<(Call, Message), Error> {
    check_cancelled(cancellation)?;
    // Only private callers holding the result of prepare_call enter here;
    // that Call has already passed identity and post-before Schema checks.
    let mut message = tool.call(&call, cancellation).await?;
    check_cancelled(cancellation)?;
    tool.after(&call, &mut message, cancellation).await?;
    check_cancelled(cancellation)?;
    Ok((call, message))
}

/// Execute the Tool lifecycle used by an explicit Workflow ToolNode.
///
/// Workflow does not enable the optional business `eval` hook in the current
/// contract. Agent-internal execution retains the call result before eval.
pub(crate) async fn execute_without_eval<T: Tool>(
    tool: &T,
    call: Call,
    cancellation: &Cancellation,
) -> Result<Message, Error> {
    execute_call(tool, call, cancellation)
        .await
        .map(|(_, message)| message)
}

fn ensure_call_identity(call: &Call, call_id: &str, name: &str) -> Result<(), Error> {
    if call.call_id != call_id || call.name != name {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "tool before hook may only modify arguments",
        ));
    }
    Ok(())
}

fn check_cancelled(cancellation: &Cancellation) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        Err(Error::new("CANCELLED", "tool operation cancelled"))
    } else {
        Ok(())
    }
}

pub(crate) fn validate_call(definition: &ToolDefinition, call: &Call) -> Result<(), Error> {
    let validator = definition.validator()?;
    if call.call_id.trim().is_empty() || call.name != definition.name {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "tool call identity does not match its definition",
        ));
    }

    validator
        .validate(&Value::Object(call.arguments.clone()))
        .map_err(|error| {
            Error::new(
                "INVALID_ARGUMENTS",
                "tool arguments violate parameters schema",
            )
            .with_details(error.to_string())
        })
}

fn compile_schema(parameters: &Map<String, Value>) -> Result<jsonschema::Validator, Error> {
    let schema = Value::Object(parameters.clone());
    jsonschema::meta::validate(&schema).map_err(|error| {
        Error::new("INVALID_ARGUMENTS", "tool parameters schema is invalid")
            .with_details(error.to_string())
    })?;
    // External HTTP and file retrieval are disabled in Cargo.toml. Local
    // definitions/references remain available without I/O in a lifecycle hook.
    jsonschema::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| {
            Error::new(
                "INVALID_ARGUMENTS",
                "tool parameters schema cannot be compiled",
            )
            .with_details(error.to_string())
        })
}
#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use futures_executor::block_on;
    use serde_json::json;

    use super::*;
    use crate::components::message::Role;

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

        async fn before(
            &self,
            _call: &mut Call,
            _cancellation: &Cancellation,
        ) -> Result<(), Error> {
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
}
