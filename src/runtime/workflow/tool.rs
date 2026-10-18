use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::future::join_all;
use serde_json::{Map, Value};

use super::node::{Attempt, AttemptStatus, Step, StepAttempt};
use crate::Cancellation;
use crate::Error;
use crate::components::message::{Message, Messages, Role};
use crate::components::model::content::part::Part;
use crate::components::node::{Node, Signal};
use crate::components::tool::{Call, Tool, ToolDefinition};
use crate::core::lifecycle::Phase;

// Working facts for this invocation only; every acknowledged change is handed
// back to the actual Runtime Attempt through its existing relay.
type Facts = Arc<Mutex<Attempt>>;

type ToolFuture = Pin<Box<dyn Future<Output = Result<Message, Error>> + Send + 'static>>;

/// Private recovery fact used only between the Workflow Runtime and ToolNode.
/// It is not a second public Tool result and is never emitted as a Message.
#[derive(Debug)]
pub(crate) struct RecoveryError {
    pub(crate) completed: Messages,
    pub(crate) steps: Vec<Step>,
    pub(crate) error: Error,
}

trait Entry: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, step: Step, facts: Facts, cancellation: Cancellation) -> ToolFuture;
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

    fn execute(&self, mut step: Step, facts: Facts, cancellation: Cancellation) -> ToolFuture {
        let tool = Arc::clone(&self.tool);
        Box::pin(async move {
            let phase = phase(&step)?;
            let call = step.call.clone().expect("validated Tool Call");
            let result = crate::core::tool::call_from(
                tool.as_ref(),
                phase,
                call,
                step.message.clone(),
                |phase, call, message| {
                    step.stage = Some(
                        match phase {
                            Phase::Call => "call",
                            Phase::After => "after",
                            Phase::Complete => "complete",
                            Phase::Before => unreachable!(),
                        }
                        .into(),
                    );
                    step.call = Some(call.clone());
                    step.message = message.cloned();
                    step.status = if phase == Phase::Complete {
                        "succeeded"
                    } else {
                        "running"
                    }
                    .into();
                    step.error = None;
                    if phase == Phase::Call {
                        step.attempts.push(StepAttempt {
                            status: AttemptStatus::Running,
                            created_time: None,
                            finished_time: None,
                            usage: None,
                            finish_reason: None,
                            error: None,
                        });
                    } else if let Some(attempt) = step.attempts.last_mut() {
                        attempt.status = AttemptStatus::Succeeded;
                        attempt.error = None;
                    }
                    acknowledge(facts.clone(), step.clone())
                },
                &cancellation,
            )
            .await;
            match result {
                Ok((call, message)) => Ok(success_message(&call, message)),
                Err(error) => {
                    step.status = match error.code.as_str() {
                        "WAITING" => "waiting",
                        "CANCELLED" => "cancelled",
                        _ => "failed",
                    }
                    .into();
                    step.error = Some(error.clone());
                    if step.stage.as_deref() == Some("call")
                        && let Some(attempt) = step.attempts.last_mut()
                    {
                        attempt.status = match error.code.as_str() {
                            "WAITING" => AttemptStatus::Waiting,
                            "CANCELLED" => AttemptStatus::Cancelled,
                            _ => AttemptStatus::Failed,
                        };
                        attempt.error = Some(error.clone());
                    }
                    acknowledge(facts, step).await?;
                    Err(error)
                }
            }
        })
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

    async fn execute_call(
        &self,
        call: Call,
        facts: Facts,
        cancellation: Cancellation,
    ) -> Result<Message, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "tool node cancelled"));
        }
        let entry = self.entry(&call.name)?;
        let mut step = facts
            .lock()
            .expect("Tool invocation facts poisoned")
            .steps
            .iter()
            .find(|step| step.call_id.as_deref() == Some(&call.call_id))
            .cloned()
            .unwrap_or_else(|| Step {
                id: call.call_id.clone(),
                kind: "tool".into(),
                status: "running".into(),
                call_id: Some(call.call_id.clone()),
                stage: Some("before".into()),
                call: Some(call),
                ..Default::default()
            });
        step.status = "running".into();
        step.error = None;
        if step.stage.as_deref() == Some("call") {
            if step
                .attempts
                .last()
                .is_some_and(|attempt| attempt.status == AttemptStatus::Failed)
            {
                step.attempts.push(StepAttempt {
                    status: AttemptStatus::Running,
                    created_time: None,
                    finished_time: None,
                    usage: None,
                    finish_reason: None,
                    error: None,
                });
            } else if let Some(attempt) = step.attempts.last_mut() {
                attempt.status = AttemptStatus::Running;
                attempt.error = None;
            }
        }
        acknowledge(facts.clone(), step.clone()).await?;
        entry.execute(step, facts, cancellation).await
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
        if errors.iter().all(|(_, error)| error.code == "WAITING") {
            Error::new("WAITING", "tool calls are waiting").with_details(details)
        } else {
            Error::new("TOOL_CALLS_FAILED", "multiple tool calls failed").with_details(details)
        }
    }

    async fn execute_calls_with_recovery(
        &self,
        calls: Vec<Call>,
        cancellation: Cancellation,
        committed: Attempt,
    ) -> Result<(Messages, Vec<Step>), RecoveryError> {
        let mut committed_by_id = committed_messages(&committed.messages);
        for step in &committed.steps {
            if step.stage.as_deref() == Some("complete") {
                let call = step.call.as_ref().expect("validated complete Call");
                committed_by_id.insert(
                    call.call_id.clone(),
                    success_message(
                        call,
                        step.message.clone().expect("validated complete result"),
                    ),
                );
            }
        }
        let facts = Arc::new(Mutex::new(committed));
        let pending = calls
            .iter()
            .filter(|call| !committed_by_id.contains_key(call.call_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();

        if pending.is_empty() {
            return Ok((
                order_messages(&calls, committed_by_id),
                facts
                    .lock()
                    .expect("Tool invocation facts poisoned")
                    .steps
                    .clone(),
            ));
        }

        match self.execution_mode {
            ExecutionMode::Sequential => {
                let mut output = committed_by_id;
                for call in pending {
                    match self
                        .execute_call(call.clone(), facts.clone(), cancellation.clone())
                        .await
                    {
                        Ok(message) => {
                            if let Some(call_id) = message_call_id(&message) {
                                output.insert(call_id, message);
                            }
                        }
                        Err(error) => {
                            let steps = facts
                                .lock()
                                .expect("Tool invocation facts poisoned")
                                .steps
                                .clone();
                            return Err(RecoveryError {
                                completed: order_messages(&calls, output),
                                steps,
                                error,
                            });
                        }
                    }
                }
                Ok((
                    order_messages(&calls, output),
                    facts
                        .lock()
                        .expect("Tool invocation facts poisoned")
                        .steps
                        .clone(),
                ))
            }
            ExecutionMode::Parallel => {
                let results = join_all(
                    pending
                        .iter()
                        .cloned()
                        .map(|call| self.execute_call(call, facts.clone(), cancellation.clone())),
                )
                .await;
                let mut output = committed_by_id;
                let mut errors = Vec::new();
                for (call, result) in pending.into_iter().zip(results) {
                    match result {
                        Ok(message) => {
                            if let Some(call_id) = message_call_id(&message) {
                                output.insert(call_id, message);
                            }
                        }
                        Err(error) => errors.push((call, error)),
                    }
                }
                if errors.is_empty() {
                    Ok((
                        order_messages(&calls, output),
                        facts
                            .lock()
                            .expect("Tool invocation facts poisoned")
                            .steps
                            .clone(),
                    ))
                } else {
                    Err(RecoveryError {
                        completed: order_messages(&calls, output),
                        steps: facts
                            .lock()
                            .expect("Tool invocation facts poisoned")
                            .steps
                            .clone(),
                        error: Self::merge_parallel_errors(errors),
                    })
                }
            }
        }
    }

    pub(crate) fn validate_recovery(
        &self,
        input: &Messages,
        committed: &Attempt,
    ) -> Result<Vec<Call>, Error> {
        self.validate_tools()?;
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
        let mut ids = std::collections::BTreeSet::new();
        for call in &calls {
            if !ids.insert(&call.call_id) {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "assistant message contains duplicate tool call_id",
                )
                .with_details(&call.call_id));
            }
            self.entry(&call.name)?;
        }
        validate_facts(&calls, committed)?;
        for step in &committed.steps {
            if step.stage.as_deref() != Some("before") {
                let call = step.call.as_ref().expect("validated Call");
                crate::components::tool::validate_call(&self.entry(&call.name)?.definition(), call)
                    .map_err(|_| invalid_facts())?;
            }
        }
        Ok(calls)
    }

    /// Private Runtime entry: successful call messages from a prior Attempt
    /// are reused by call_id, while failed or missing calls are executed.
    pub(crate) async fn call_with_recovery(
        &self,
        input: Messages,
        cancellation: Cancellation,
        committed: Attempt,
    ) -> Result<(Signal<Messages>, Vec<Step>), RecoveryError> {
        let calls = self
            .validate_recovery(&input, &committed)
            .map_err(|error| RecoveryError {
                completed: committed.messages.clone(),
                steps: committed.steps.clone(),
                error,
            })?;
        let (output, steps) = self
            .execute_calls_with_recovery(calls, cancellation, committed)
            .await?;
        Ok((Signal::Complete(output), steps))
    }
}

impl Node<Messages, Messages, Error> for ToolNode {
    async fn call(
        &self,
        input: Messages,
        cancellation: Cancellation,
    ) -> Result<Signal<Messages>, Error> {
        self.call_with_recovery(input, cancellation, Attempt::default())
            .await
            .map(|(signal, _)| signal)
            .map_err(|failure| failure.error)
    }
}

fn invalid_facts() -> Error {
    Error::new("INVALID_STATE", "invalid ToolNode lifecycle facts")
}
fn phase(step: &Step) -> Result<Phase, Error> {
    match step.stage.as_deref() {
        Some("before") => Ok(Phase::Before),
        Some("call") => Ok(Phase::Call),
        Some("after") => Ok(Phase::After),
        Some("complete") => Ok(Phase::Complete),
        _ => Err(invalid_facts()),
    }
}
fn validate_facts(calls: &[Call], attempt: &Attempt) -> Result<(), Error> {
    let mut seen = std::collections::BTreeSet::new();
    for step in &attempt.steps {
        let call = step.call.as_ref().ok_or_else(invalid_facts)?;
        let source = calls
            .iter()
            .find(|source| source.call_id == call.call_id)
            .ok_or_else(invalid_facts)?;
        let phase = phase(step)?;
        if step.kind != "tool"
            || step.id != call.call_id
            || step.call_id.as_ref() != Some(&call.call_id)
            || call.name != source.name
            || !seen.insert(&call.call_id)
            || (phase == Phase::Before && call != source)
            || ((phase >= Phase::After) != step.message.is_some())
            || (phase >= Phase::After
                && step
                    .attempts
                    .last()
                    .is_none_or(|attempt| attempt.status != AttemptStatus::Succeeded))
            || (phase == Phase::Before && !step.attempts.is_empty())
            || (phase == Phase::Complete && (step.status != "succeeded" || step.error.is_some()))
            || !matches!(
                step.status.as_str(),
                "running" | "waiting" | "failed" | "cancelled" | "succeeded"
            )
            || (phase != Phase::Complete && step.status == "succeeded")
        {
            return Err(invalid_facts());
        }
    }
    let mut messages = std::collections::BTreeSet::new();
    for message in attempt.messages.as_slice() {
        let id = message_call_id(message).ok_or_else(invalid_facts)?;
        if !calls.iter().any(|call| call.call_id == id)
            || !messages.insert(id.clone())
            || attempt.steps.iter().any(|step| {
                step.call_id.as_ref() == Some(&id) && step.stage.as_deref() != Some("complete")
            })
        {
            return Err(invalid_facts());
        }
        if let Some(step) = attempt
            .steps
            .iter()
            .find(|step| step.call_id.as_ref() == Some(&id))
            && *message
                != success_message(
                    step.call.as_ref().expect("validated Call"),
                    step.message.clone().expect("complete Message"),
                )
        {
            return Err(invalid_facts());
        }
    }
    Ok(())
}
async fn acknowledge(facts: Facts, step: Step) -> Result<(), Error> {
    let attempt = {
        let mut facts = facts.lock().expect("Tool invocation facts poisoned");
        if step.stage.as_deref() == Some("complete") {
            let call = step.call.as_ref().expect("complete Call");
            facts
                .messages
                .0
                .retain(|message| message_call_id(message).as_deref() != Some(&call.call_id));
            facts.messages.0.push(success_message(
                call,
                step.message.clone().expect("complete Message"),
            ));
        }
        if let Some(saved) = facts.steps.iter_mut().find(|saved| saved.id == step.id) {
            *saved = step;
        } else {
            facts.steps.push(step);
        }
        facts.clone()
    };
    super::events::progress(attempt, None).await
}

fn message_call_id(message: &Message) -> Option<String> {
    message
        .metadata
        .get("call_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn committed_messages(messages: &Messages) -> std::collections::BTreeMap<String, Message> {
    messages
        .as_slice()
        .iter()
        .filter_map(|message| message_call_id(message).map(|id| (id, message.clone())))
        .collect()
}

fn order_messages(
    calls: &[Call],
    messages: std::collections::BTreeMap<String, Message>,
) -> Messages {
    Messages::new(
        calls
            .iter()
            .filter_map(|call| messages.get(&call.call_id).cloned()),
    )
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
                data: crate::components::model::content::part::object(json!({
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                })),
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

    struct FlakySearch {
        definition: crate::components::tool::ToolDefinition,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Tool for FlakySearch {
        fn definition(&self) -> &crate::components::tool::ToolDefinition {
            &self.definition
        }

        async fn call(&self, call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(Error::new("UNAVAILABLE", "search temporarily unavailable"));
            }
            Ok(Message::function(json!({
                "query": call.arguments["query"],
                "documents": ["document-2"],
            })))
        }
    }

    #[test]
    fn private_recovery_reuses_successful_call_ids_only() {
        let weather_called = Arc::new(AtomicBool::new(false));
        let search_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let node = ToolNode::new(Weather {
            called: weather_called.clone(),
            reject: false,
        })
        .tool(FlakySearch {
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
            calls: search_calls.clone(),
        })
        .sequential();
        let input = request_calls([
            ("call_weather", "weather", json!({"city": "Beijing"})),
            ("call_search", "search", json!({"query": "rust"})),
        ]);

        let first = block_on(node.call_with_recovery(
            input.clone(),
            Cancellation::new(),
            Attempt::default(),
        ))
        .expect_err("the second call should fail on the first attempt");
        assert_eq!(first.error.code, "UNAVAILABLE");
        assert_eq!(first.completed.as_slice().len(), 1);
        assert_eq!(
            first.completed.as_slice()[0].metadata["call_id"],
            "call_weather"
        );

        let result = block_on(node.call_with_recovery(
            input,
            Cancellation::new(),
            Attempt {
                messages: first.completed,
                steps: first.steps,
                ..Default::default()
            },
        ))
        .expect("the failed call should succeed on retry");
        let (Signal::Complete(messages), _) = result else {
            panic!("ToolNode unexpectedly returned Wait");
        };
        assert_eq!(messages.as_slice().len(), 2);
        assert!(weather_called.load(Ordering::SeqCst));
        assert_eq!(search_calls.load(Ordering::SeqCst), 2);
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
