use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::future::join_all;
use serde_json::{Map, Value};

use crate::Error;
use crate::components::message::{Message, Messages, Role};
use crate::components::model::content::Part;
use crate::components::node::{Node, Signal};
use crate::components::tool::{Call, Tool, ToolDefinition, execute_without_eval};
use crate::runtime::cancellation::Cancellation;

type ToolFuture = Pin<Box<dyn Future<Output = Result<Message, Error>> + Send + 'static>>;

trait Entry: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, call: Call, cancellation: Cancellation) -> ToolFuture;
}

struct TypedEntry<T> {
    tool: Arc<T>,
}

impl<T> Entry for TypedEntry<T>
where
    T: Tool + 'static,
{
    fn definition(&self) -> ToolDefinition {
        self.tool.definition().clone()
    }

    fn execute(&self, call: Call, cancellation: Cancellation) -> ToolFuture {
        let tool = Arc::clone(&self.tool);
        Box::pin(async move { execute_without_eval(tool.as_ref(), call, &cancellation).await })
    }
}

/// How one Workflow ToolNode schedules the tool calls in one assistant message.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExecutionMode {
    /// Execute calls in their source order and stop at the first error.
    #[default]
    Sequential,
    /// Start all calls together and preserve source order in the output.
    Parallel,
}

/// Explicit Workflow adapter for one or more Tool components.
///
/// It only translates assistant `tool_call` parts into complete tool messages.
/// Execution identity, retry and persistence remain owned by the surrounding
/// runtime::workflow Runtime.
pub struct ToolNode {
    tools: Vec<Box<dyn Entry>>,
    execution_mode: ExecutionMode,
}

impl ToolNode {
    /// Create a node with its first Tool. Additional concrete Tool types can
    /// be added with [`ToolNode::tool`].
    pub fn new<T>(tool: T) -> Self
    where
        T: Tool + 'static,
    {
        Self {
            tools: vec![Box::new(TypedEntry {
                tool: Arc::new(tool),
            })],
            execution_mode: ExecutionMode::default(),
        }
    }

    /// Add a heterogeneous Tool while keeping the graph-facing node type
    /// unchanged. Duplicate names are rejected before any call is executed.
    pub fn tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.tools.push(Box::new(TypedEntry {
            tool: Arc::new(tool),
        }));
        self
    }

    pub fn execution_mode(mut self, execution_mode: ExecutionMode) -> Self {
        self.execution_mode = execution_mode;
        self
    }

    pub fn sequential(self) -> Self {
        self.execution_mode(ExecutionMode::Sequential)
    }

    pub fn parallel(self) -> Self {
        self.execution_mode(ExecutionMode::Parallel)
    }

    fn validate_tools(&self) -> Result<(), Error> {
        let mut names = std::collections::BTreeSet::new();
        for entry in &self.tools {
            let definition = entry.definition();
            definition.validate()?;
            if !names.insert(definition.name.clone()) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "ToolNode contains duplicate tool names",
                )
                .with_details(definition.name));
            }
        }
        if self.tools.is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "ToolNode requires at least one tool",
            ));
        }
        Ok(())
    }

    fn entry(&self, name: &str) -> Result<&dyn Entry, Error> {
        self.tools
            .iter()
            .find(|entry| entry.definition().name == name)
            .map(|entry| entry.as_ref())
            .ok_or_else(|| Error::new("NOT_FOUND", "tool is not registered").with_details(name))
    }

    async fn execute_call(&self, call: Call, cancellation: Cancellation) -> Result<Message, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "tool node cancelled"));
        }
        let entry = self.entry(&call.name)?;
        let message = entry.execute(call.clone(), cancellation).await?;
        Ok(success_message(&call, message))
    }

    fn merge_parallel_errors(errors: Vec<(Call, Error)>) -> Error {
        if errors.len() == 1 {
            return errors.into_iter().next().expect("one error").1;
        }
        let details = errors
            .iter()
            .map(|(call, error)| format!("{}: {} {}", call.call_id, error.code, error.message))
            .collect::<Vec<_>>()
            .join("; ");
        Error::new("TOOL_CALLS_FAILED", "multiple tool calls failed").with_details(details)
    }

    async fn execute_calls(
        &self,
        calls: Vec<Call>,
        cancellation: Cancellation,
    ) -> Result<Messages, Error> {
        match self.execution_mode {
            ExecutionMode::Sequential => {
                let mut output = Messages::default();
                for call in calls {
                    output.push(self.execute_call(call, cancellation.clone()).await?);
                }
                Ok(output)
            }
            ExecutionMode::Parallel => {
                let results = join_all(
                    calls
                        .iter()
                        .cloned()
                        .map(|call| self.execute_call(call, cancellation.clone())),
                )
                .await;
                let mut output = Messages::default();
                let mut errors = Vec::new();
                for (call, result) in calls.into_iter().zip(results) {
                    match result {
                        Ok(message) => output.push(message),
                        Err(error) => errors.push((call, error)),
                    }
                }
                if errors.is_empty() {
                    Ok(output)
                } else {
                    Err(Self::merge_parallel_errors(errors))
                }
            }
        }
    }
}

impl Node<Messages, Messages, Error> for ToolNode {
    async fn call(
        &self,
        input: Messages,
        cancellation: Cancellation,
    ) -> Result<Signal<Messages>, Error> {
        let assistant = input
            .as_slice()
            .iter()
            .rev()
            .find(|message| message.role == Role::Assistant)
            .ok_or_else(|| {
                Error::new(
                    "INVALID_ARGUMENTS",
                    "ToolNode requires an assistant message",
                )
            })?;

        self.validate_tools()?;
        let calls = assistant
            .content
            .iter()
            .filter(|part| part.r#type == "tool_call")
            .map(decode_call)
            .collect::<Result<Vec<_>, _>>()?;
        if calls.is_empty() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "assistant message contains no tool_call",
            ));
        }
        let mut call_ids = std::collections::BTreeSet::new();
        for call in &calls {
            if !call_ids.insert(call.call_id.clone()) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "assistant message contains duplicate tool call_id",
                )
                .with_details(call.call_id.clone()));
            }
            self.entry(&call.name)?;
        }
        let output = self.execute_calls(calls, cancellation).await?;
        Ok(Signal::Complete(output))
    }
}

fn decode_call(part: &Part) -> Result<Call, Error> {
    let call_id = part
        .data
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "tool call_id is missing"))?;
    let name = part
        .data
        .get("name")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "tool name is missing"))?;
    let arguments = match part.data.get("arguments") {
        Some(Value::Object(arguments)) => arguments.clone(),
        Some(Value::String(raw)) => {
            serde_json::from_str::<Map<String, Value>>(raw).map_err(|error| {
                Error::new("INVALID_ARGUMENTS", "tool arguments are not a JSON object")
                    .with_details(format!("raw_arguments={raw}; {error}"))
            })?
        }
        Some(_) => {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "tool arguments must be a JSON object",
            ));
        }
        None => Map::new(),
    };
    Ok(Call {
        call_id: call_id.to_owned(),
        name: name.to_owned(),
        arguments,
    })
}

fn success_message(call: &Call, mut message: Message) -> Message {
    // The Tool's Message is already the result. Keep it intact and use
    // metadata only for correlating it with the originating tool_call.
    message
        .metadata
        .insert("call_id".into(), Value::String(call.call_id.clone()));
    message
        .metadata
        .insert("name".into(), Value::String(call.name.clone()));
    message
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

    struct Weather {
        called: Arc<AtomicBool>,
        reject: bool,
    }

    impl Tool for Weather {
        fn definition(&self) -> &crate::components::tool::ToolDefinition {
            static DEFINITION: std::sync::OnceLock<crate::components::tool::ToolDefinition> =
                std::sync::OnceLock::new();
            DEFINITION.get_or_init(|| crate::components::tool::ToolDefinition {
                name: "weather".into(),
                description: "weather lookup".into(),
                parameters: serde_json::from_value(json!({
                    "type": "object",
                    "required": ["city"],
                    "properties": {"city": {"type": "string"}}
                }))
                .expect("valid test schema"),
            })
        }

        async fn before(
            &self,
            _call: &mut Call,
            _cancellation: &Cancellation,
        ) -> Result<(), Error> {
            if self.reject {
                return Err(Error::new("INVALID_ARGUMENTS", "tool precondition failed"));
            }
            Ok(())
        }

        async fn call(&self, call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
            self.called.store(true, Ordering::SeqCst);
            Ok(Message::function(json!({
                "city": call.arguments["city"],
                "temperature": 22
            })))
        }
    }

    fn request(arguments: Value) -> Messages {
        request_calls([("call_001", "weather", arguments)])
    }

    fn request_calls<const N: usize>(calls: [(&str, &str, Value); N]) -> Messages {
        let mut assistant = Message::new(Role::Assistant);
        assistant
            .content
            .extend(calls.into_iter().map(|(call_id, name, arguments)| Part {
                r#type: "tool_call".into(),
                data: json!({
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }),
            }));
        Messages::new([assistant])
    }

    struct Search {
        definition: crate::components::tool::ToolDefinition,
        called: Arc<AtomicBool>,
    }

    impl Tool for Search {
        fn definition(&self) -> &crate::components::tool::ToolDefinition {
            &self.definition
        }

        async fn call(&self, call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
            self.called.store(true, Ordering::SeqCst);
            Ok(Message::function(json!({
                "query": call.arguments["query"],
                "documents": ["document-1"],
            })))
        }
    }

    fn search(called: Arc<AtomicBool>) -> Search {
        Search {
            definition: crate::components::tool::ToolDefinition {
                name: "search".into(),
                description: "search documents".into(),
                parameters: serde_json::from_value(json!({
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}}
                }))
                .expect("valid test schema"),
            },
            called,
        }
    }

    struct EvalSearch {
        definition: crate::components::tool::ToolDefinition,
        evaluated: Arc<AtomicBool>,
    }

    impl Tool for EvalSearch {
        fn definition(&self) -> &crate::components::tool::ToolDefinition {
            &self.definition
        }

        async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
            Ok(Message::function(json!({"documents": ["document-1"]})))
        }

        async fn eval(
            &self,
            _call: &Call,
            _message: &Message,
            _cancellation: &Cancellation,
        ) -> Result<crate::components::tool::Evaluation, Error> {
            self.evaluated.store(true, Ordering::SeqCst);
            Ok(crate::components::tool::Evaluation::passed())
        }
    }

    fn eval_search(evaluated: Arc<AtomicBool>) -> EvalSearch {
        EvalSearch {
            definition: crate::components::tool::ToolDefinition {
                name: "weather".into(),
                description: "search documents".into(),
                parameters: serde_json::from_value(json!({
                    "type": "object",
                    "properties": {}
                }))
                .expect("valid test schema"),
            },
            evaluated,
        }
    }

    #[test]
    fn tool_node_returns_direct_success_message() {
        let called = Arc::new(AtomicBool::new(false));
        let node = ToolNode::new(Weather {
            called: called.clone(),
            reject: false,
        });
        let result =
            block_on(node.call(request(json!({"city": "Beijing"})), Cancellation::new())).unwrap();
        let Signal::Complete(messages) = result else {
            panic!("ToolNode unexpectedly returned Wait");
        };
        assert!(called.load(Ordering::SeqCst));
        assert_eq!(messages.as_slice()[0].role, Role::Function);
        assert_eq!(messages.as_slice()[0].values["temperature"], 22);
        assert_eq!(messages.as_slice()[0].metadata["call_id"], "call_001");
    }

    #[test]
    fn workflow_tool_node_does_not_run_eval_hook() {
        let evaluated = Arc::new(AtomicBool::new(false));
        let node = ToolNode::new(eval_search(evaluated.clone()));
        let result = block_on(node.call(request(json!({})), Cancellation::new())).unwrap();
        assert!(matches!(result, Signal::Complete(_)));
        assert!(!evaluated.load(Ordering::SeqCst));
    }

    #[test]
    fn malformed_arguments_skip_tool_and_keep_raw_details() {
        let called = Arc::new(AtomicBool::new(false));
        let node = ToolNode::new(Weather {
            called: called.clone(),
            reject: false,
        });
        let error = block_on(node.call(
            request(Value::String("not-json".into())),
            Cancellation::new(),
        ))
        .unwrap_err();
        assert!(!called.load(Ordering::SeqCst));
        assert_eq!(error.code, "INVALID_ARGUMENTS");
        assert!(error.details.as_deref().unwrap().contains("not-json"));
    }

    #[test]
    fn tool_hook_error_is_frozen_without_invoking_call() {
        let called = Arc::new(AtomicBool::new(false));
        let node = ToolNode::new(Weather {
            called: called.clone(),
            reject: true,
        });
        let error = block_on(node.call(request(json!({"city": "Beijing"})), Cancellation::new()))
            .unwrap_err();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn multi_tool_node_supports_heterogeneous_parallel_calls_in_source_order() {
        let weather_called = Arc::new(AtomicBool::new(false));
        let search_called = Arc::new(AtomicBool::new(false));
        let node = ToolNode::new(Weather {
            called: weather_called.clone(),
            reject: false,
        })
        .tool(search(search_called.clone()))
        .parallel();
        let result = block_on(node.call(
            request_calls([
                ("call_weather", "weather", json!({"city": "Beijing"})),
                ("call_search", "search", json!({"query": "rust"})),
            ]),
            Cancellation::new(),
        ))
        .unwrap();
        let Signal::Complete(messages) = result else {
            panic!("ToolNode unexpectedly returned Wait");
        };
        assert!(weather_called.load(Ordering::SeqCst));
        assert!(search_called.load(Ordering::SeqCst));
        assert_eq!(messages.as_slice().len(), 2);
        assert_eq!(messages.as_slice()[0].metadata["call_id"], "call_weather");
        assert_eq!(messages.as_slice()[1].metadata["call_id"], "call_search");
        assert_eq!(messages.as_slice()[1].values["documents"][0], "document-1");
    }

    #[test]
    fn unknown_call_is_rejected_before_any_tool_side_effect() {
        let called = Arc::new(AtomicBool::new(false));
        let node = ToolNode::new(Weather {
            called: called.clone(),
            reject: false,
        });
        let error = block_on(node.call(
            request_calls([
                ("call_weather", "weather", json!({"city": "Beijing"})),
                ("call_missing", "missing", json!({})),
            ]),
            Cancellation::new(),
        ))
        .unwrap_err();
        assert_eq!(error.code, "NOT_FOUND");
        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn duplicate_tool_names_are_rejected_before_execution() {
        let called = Arc::new(AtomicBool::new(false));
        let node = ToolNode::new(Weather {
            called: called.clone(),
            reject: false,
        })
        .tool(Weather {
            called: called.clone(),
            reject: false,
        });
        let error = block_on(node.call(request(json!({"city": "Beijing"})), Cancellation::new()))
            .unwrap_err();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
        assert!(!called.load(Ordering::SeqCst));
    }
}
