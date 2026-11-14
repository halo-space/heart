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

impl crate::components::tool::Tool for Weather {
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

    async fn before(&self, _call: &mut Call, _cancellation: &Cancellation) -> Result<(), Error> {
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

impl crate::components::tool::Tool for Search {
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

impl crate::components::tool::Tool for EvalSearch {
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
        block_on(node.call(request(json!({"city": "Beijing"})), &Cancellation::new())).unwrap();
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
    let result = block_on(node.call(request(json!({})), &Cancellation::new())).unwrap();
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
        &Cancellation::new(),
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
    let error =
        block_on(node.call(request(json!({"city": "Beijing"})), &Cancellation::new())).unwrap_err();
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
        &Cancellation::new(),
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

impl crate::components::tool::Tool for FlakySearch {
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

    let first =
        block_on(node.call_with_recovery(input.clone(), Cancellation::new(), Attempt::default()))
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
        &Cancellation::new(),
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
    let error =
        block_on(node.call(request(json!({"city": "Beijing"})), &Cancellation::new())).unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert!(!called.load(Ordering::SeqCst));
}
