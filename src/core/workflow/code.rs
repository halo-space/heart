//! Executor for page-authored function nodes in the Native Workflow DSL.
//!
//! A code node is deliberately small: the DSL supplies source, entry and
//! schemas, while a configured `Sandbox` supplies the execution environment.
//! The executor does not discover functions, create a registry, or retain
//! Runtime state.

use std::{collections::BTreeMap, sync::Arc};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{Cancellation, Error};
use crate::{components::node::schema, sandbox::Sandbox};
use crate::{
    node::{Node, Signal},
    sandbox::Request,
};

use super::dsl;

/// JSON object passed between a code node and the Workflow runtime.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Object(pub BTreeMap<String, Value>);

impl Object {
    fn into_value(self) -> Value {
        Value::Object(self.0.into_iter().collect())
    }

    fn from_value(value: Value, message: &str) -> Result<Self, Error> {
        let Value::Object(object) = value else {
            return Err(Error::new("INVALID_RESPONSE", message));
        };
        Ok(Self(object.into_iter().collect()))
    }
}

/// A page-authored function executed through the selected Sandbox.
pub struct Code<S> {
    sandbox: Arc<S>,
    language: String,
    entry: String,
    source: String,
    input: Arc<jsonschema::Validator>,
    output: Arc<jsonschema::Validator>,
}

impl<S> Code<S>
where
    S: Sandbox + 'static,
{
    /// Build an executor from one validated DSL function node.
    pub fn from_dsl(sandbox: S, node: &dsl::Node) -> Result<Self, Error> {
        if node.node_type != "function" {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "code executor requires a function node",
            ));
        }
        let language = config_string(node, "language")?;
        let entry = config_string(node, "entry")?;
        let source = config_string(node, "source")?;
        if !matches!(language.as_str(), "python" | "python3") {
            return Err(Error::new(
                "UNSUPPORTED",
                "code executor currently supports python only",
            ));
        }

        Ok(Self {
            sandbox: Arc::new(sandbox),
            language,
            entry,
            source,
            input: Arc::new(schema::compile(&node.schema.input)?),
            output: Arc::new(schema::compile(&node.schema.output)?),
        })
    }

    fn interpreter(&self) -> &'static str {
        // `python` and `python3` are both accepted DSL spellings; use the
        // explicit executable so a host never silently invokes another shell.
        match self.language.as_str() {
            "python" | "python3" => "python3",
            _ => unreachable!("language checked by from_dsl"),
        }
    }
}

impl<S> Node<Object, Object, Error> for Code<S>
where
    S: Sandbox + 'static,
{
    async fn call(
        &self,
        input: Object,
        cancellation: &Cancellation,
    ) -> Result<Signal<Object>, Error> {
        let input_value = input.into_value();
        schema::validate(
            &self.input,
            &input_value,
            "INVALID_ARGUMENTS",
            "code node input does not match its schema",
        )?;

        let request = Request {
            operation: "command".into(),
            input: json!({
                "args": [
                    self.interpreter(),
                    "-c",
                    PYTHON_RUNNER,
                    self.source,
                    self.entry,
                    input_value.to_string(),
                ]
            }),
            metadata: Map::new(),
        };
        let response = self.sandbox.execute(request, cancellation).await?;
        let stdout = response
            .data
            .get("stdout")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::new("INVALID_RESPONSE", "code sandbox did not return stdout"))?;
        let output_value: Value = serde_json::from_str(stdout.trim()).map_err(|error| {
            Error::new("INVALID_RESPONSE", "code function returned invalid JSON")
                .with_details(error.to_string())
        })?;
        schema::validate(
            &self.output,
            &output_value,
            "INVALID_RESPONSE",
            "code node output does not match its schema",
        )?;
        Ok(Signal::Complete(Object::from_value(
            output_value,
            "code function must return a JSON object",
        )?))
    }
}

fn config_string(node: &dsl::Node, key: &str) -> Result<String, Error> {
    node.config
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", format!("config.{key} is required")))
}

/// Small, deterministic Python bridge. Source and input are argv values, so
/// neither is interpolated into executable code or interpreted by a shell.
const PYTHON_RUNNER: &str = r#"
import contextlib
import io
import json
import sys

source, entry, raw_input = sys.argv[1:4]
namespace = {"__name__": "__heart_code_node__"}
exec(compile(source, "<heart-code>", "exec"), namespace, namespace)
function = namespace.get(entry)
if not callable(function):
    raise TypeError("configured entry is not callable")
value = json.loads(raw_input)
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    result = function(value)
print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
"#;
