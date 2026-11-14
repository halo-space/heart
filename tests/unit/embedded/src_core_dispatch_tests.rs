use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use futures_executor::block_on;
use serde_json::{Map, json};

use super::*;
use crate::components::message::Role;

struct Weather {
    definition: ToolDefinition,
    called: Arc<AtomicBool>,
}

impl Tool for Weather {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
        self.called.store(true, Ordering::SeqCst);
        Ok(Message {
            role: Role::Tool,
            content: Vec::new(),
            values: Map::from_iter([(String::from("temperature"), json!(22))]),
            metadata: Map::new(),
            usage: None,
        })
    }
}

fn weather(called: Arc<AtomicBool>) -> Weather {
    Weather {
        definition: ToolDefinition {
            name: "weather".into(),
            description: "weather lookup".into(),
            parameters: serde_json::from_value(json!({
                "type": "object",
                "properties": {}
            }))
            .unwrap(),
        },
        called,
    }
}

#[test]
fn dispatch_exports_definitions_and_executes_by_name() {
    let called = Arc::new(AtomicBool::new(false));
    let dispatch = Dispatch::new().tools(weather(called.clone()));
    assert_eq!(dispatch.len(), 1);
    assert_eq!(dispatch.tool_list()[0].name, "weather");
    let (message, evaluation) = block_on(dispatch.execute(
        Call {
            call_id: "call_1".into(),
            name: "weather".into(),
            arguments: Map::new(),
        },
        &Cancellation::new(),
    ))
    .unwrap();
    assert!(called.load(Ordering::SeqCst));
    assert_eq!(message.values["temperature"], 22);
    assert!(evaluation.passed);
}

#[test]
fn dispatch_rejects_unknown_tools_without_invoking_registered_tools() {
    let called = Arc::new(AtomicBool::new(false));
    let dispatch = Dispatch::new().tools(weather(called.clone()));
    let error = block_on(dispatch.execute(
        Call {
            call_id: "call_1".into(),
            name: "unknown".into(),
            arguments: Map::new(),
        },
        &Cancellation::new(),
    ))
    .unwrap_err();
    assert_eq!(error.code, "NOT_FOUND");
    assert!(!called.load(Ordering::SeqCst));
}

#[test]
fn prepared_view_exposes_tools_and_skills_together() {
    let skill = crate::core::skill::Skill::new(
        SkillDefinition {
            name: "research".into(),
            description: Some("Research a topic and cite sources.".into()),
            ..Default::default()
        },
        "Use sources.".into(),
    )
    .unwrap();
    let dispatch = Dispatch::new()
        .tools(weather(Arc::new(AtomicBool::new(false))))
        .skills(vec![skill])
        .unwrap();
    let cancellation = Cancellation::new();
    let prepared = block_on(dispatch.prepared(&cancellation)).unwrap();
    let definitions = prepared.tool_list();
    assert_eq!(definitions.len(), 2);
    assert_eq!(definitions[0].name, "weather");
    assert_eq!(definitions[1].name, load::READER_NAME);
    let prompt = prepared.system_prompt(None).unwrap();
    assert!(prompt.contains("research"));
    assert!(prompt.contains("Research a topic and cite sources."));

    let (call, output) = block_on(prepared.call(
        Call {
            call_id: "call_skill".into(),
            name: load::READER_NAME.into(),
            arguments: Map::from_iter([("skill".into(), json!("research"))]),
        },
        &cancellation,
    ))
    .unwrap();
    assert_eq!(call.name, load::READER_NAME);
    assert_eq!(output.values["instructions"], "Use sources.");

    // Preparing an already prepared view remains idempotent.
    let prepared_again = block_on(prepared.prepared(&cancellation)).unwrap();
    assert_eq!(prepared_again.tool_list().len(), 2);
}

struct RetryingEval {
    definition: ToolDefinition,
    calls: Arc<std::sync::atomic::AtomicUsize>,
    evaluations: Arc<std::sync::atomic::AtomicUsize>,
}

impl Tool for RetryingEval {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    async fn before(&self, call: &mut Call, _cancellation: &Cancellation) -> Result<(), Error> {
        call.arguments.insert("city".into(), json!("Beijing"));
        Ok(())
    }
    async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Message::function(json!({"temperature": 22})))
    }
    async fn eval(
        &self,
        call: &Call,
        message: &Message,
        _cancellation: &Cancellation,
    ) -> Result<Evaluation, Error> {
        assert_eq!(call.arguments["city"], "Beijing");
        assert_eq!(message.values["temperature"], 22);
        if self.evaluations.fetch_add(1, Ordering::SeqCst) == 0 {
            Err(Error::new("TIMEOUT", "evaluation transport failed"))
        } else {
            Ok(Evaluation::failed(crate::Messages::new([
                Message::function(json!({"feedback": "too warm"})),
            ])))
        }
    }
}

#[test]
fn caller_can_retry_only_eval_using_the_retained_call_and_result() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let evaluations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let dispatch = Dispatch::new().tools(RetryingEval {
        definition: weather(Arc::new(AtomicBool::new(false))).definition,
        calls: calls.clone(),
        evaluations: evaluations.clone(),
    });
    let cancellation = Cancellation::new();
    let (call, message) = block_on(dispatch.call(
        Call {
            call_id: "call_1".into(),
            name: "weather".into(),
            arguments: Map::new(),
        },
        &cancellation,
    ))
    .unwrap();
    assert_eq!(
        block_on(dispatch.evaluate(&call, &message, &cancellation))
            .unwrap_err()
            .code,
        "TIMEOUT"
    );
    let result = block_on(dispatch.evaluate(&call, &message, &cancellation)).unwrap();
    assert!(!result.passed);
    assert!(result.feedback.is_some());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(evaluations.load(Ordering::SeqCst), 2);
}
