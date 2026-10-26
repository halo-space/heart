//! Tool cancellation contract tests. They do not synthesize model responses.
//! Raptor's successful end-to-end path requires a real Embed endpoint; that
//! acceptance is tracked separately instead of substituting fake vectors.

use futures_executor::block_on;
use halo::agent::message::Role;
use halo::agent::tool::{self, Call, Evaluation, Tool, ToolDefinition};
use halo::agent::{Cancellation, Error, Message, Messages};
use serde_json::{Map, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct CancelTool {
    definition: ToolDefinition,
    cancel_at: usize,
    calls: Arc<AtomicUsize>,
}

impl CancelTool {
    fn stage(&self, cancellation: &Cancellation) {
        let stage = self.calls.fetch_add(1, Ordering::SeqCst);
        if stage == self.cancel_at {
            cancellation.cancel();
        }
    }
}

impl Tool for CancelTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn before(&self, _call: &mut Call, cancellation: &Cancellation) -> Result<(), Error> {
        self.stage(cancellation);
        Ok(())
    }

    async fn call(&self, _call: &Call, cancellation: &Cancellation) -> Result<Message, Error> {
        self.stage(cancellation);
        Ok(Message::new(Role::Tool))
    }

    async fn after(
        &self,
        _call: &Call,
        _message: &mut Message,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        self.stage(cancellation);
        Ok(())
    }

    async fn eval(
        &self,
        _call: &Call,
        _message: &Message,
        cancellation: &Cancellation,
    ) -> Result<Evaluation, Error> {
        self.stage(cancellation);
        Ok(Evaluation::failed(Messages::default()))
    }
}

fn tool(cancel_at: usize) -> CancelTool {
    CancelTool {
        definition: ToolDefinition {
            name: "test".into(),
            description: "test".into(),
            parameters: serde_json::from_value(json!({"type":"object"})).unwrap(),
        },
        cancel_at,
        calls: Arc::default(),
    }
}

fn call() -> Call {
    Call {
        call_id: "call_1".into(),
        name: "test".into(),
        arguments: Map::new(),
    }
}

#[test]
fn tool_cancel_at_each_stage_excludes_following_hooks() {
    for stage in 0..4 {
        let tool = tool(stage);
        let error = block_on(tool::execute(&tool, call(), &Cancellation::new())).unwrap_err();
        assert_eq!(error.code, "CANCELLED");
        assert_eq!(tool.calls.load(Ordering::SeqCst), stage + 1);
    }
}

#[test]
fn tool_precancel_excludes_all_hooks() {
    let tool = tool(usize::MAX);
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let error = block_on(tool::execute(&tool, call(), &cancellation)).unwrap_err();
    assert_eq!(error.code, "CANCELLED");
    assert_eq!(tool.calls.load(Ordering::SeqCst), 0);
}
