use std::future::Future;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::message::{Message, Messages};
use crate::{Cancellation, cancellation::check};

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

// Private split lets Agent authorization inspect the effective arguments
// without repeating before or executing call ahead of that authorization.
pub(crate) async fn prepare_call<T: Tool>(
    tool: &T,
    mut call: Call,
    cancellation: &Cancellation,
) -> Result<Call, Error> {
    check(cancellation)?;
    validate_call(tool.definition(), &call)?;
    let call_id = call.call_id.clone();
    let name = call.name.clone();
    tool.before(&mut call, cancellation).await?;
    check(cancellation)?;
    ensure_call_identity(&call, &call_id, &name)?;
    validate_call(tool.definition(), &call)?;
    Ok(call)
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
#[path = "../../tests/unit/embedded/src_components_tool_tests.rs"]
mod tests;
