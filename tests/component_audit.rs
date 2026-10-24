//! Frozen component boundaries; no Runtime, real model or backend service.
use futures_executor::block_on;
use futures_util::stream::{self, Iter};
use halo::agent::{Cancellation, Error, Message, Messages};
use halo::{
    agent::document::{Document, raptor::Raptor},
    agent::event::Event,
    agent::message::Role,
    agent::model::{
        chat::{Chat, Request},
        content::part::Part,
        embed::Embed,
    },
    agent::tool::{self, Call, Evaluation, Tool, ToolDefinition},
};
use serde_json::{Map, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Model {
    parts: Vec<Part>,
    calls: Arc<AtomicUsize>,
}
impl Chat for Model {
    type Stream = Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _request: Request) -> Result<Self::Stream, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut message = Message::new(Role::Assistant);
        message.content = self.parts.clone();
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: None,
        })]))
    }
}
fn model(text: &str) -> Model {
    Model {
        parts: vec![Part {
            r#type: "text".into(),
            data: json!({"value":text}).as_object().unwrap().clone(),
        }],
        calls: Arc::default(),
    }
}
struct Embedding {
    calls: Arc<AtomicUsize>,
    cancel: Option<Cancellation>,
}
impl Embed for Embedding {
    async fn embed(&self, _request: halo::agent::model::embed::Request) -> Result<Message, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(cancel) = &self.cancel {
            cancel.cancel();
        }
        halo::agent::model::embed::message(vec![1.0], "test", "test", None)
    }
}
fn embed() -> Embedding {
    Embedding {
        calls: Arc::default(),
        cancel: None,
    }
}
fn document(id: i64) -> Document {
    Document {
        id: Some(id),
        content: format!("source {id}"),
        ..Default::default()
    }
}

#[test]
fn raptor_duplicate_input_ids_fail_before_model_work() {
    let embedding = embed();
    let calls = embedding.calls.clone();
    let raptor = Raptor::new(Map::new(), model("summary"), embedding).unwrap();
    let mut generated = false;
    let error = block_on(raptor.run(
        &[document(1), document(1)],
        &Cancellation::new(),
        &mut || {
            generated = true;
            100
        },
    ))
    .unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(!generated);
}

#[test]
fn raptor_rejects_reused_or_invalid_generated_ids_without_partial_output() {
    let raptor = Raptor::new(
        serde_json::from_value(json!({"cluster_size":1})).unwrap(),
        model("summary"),
        embed(),
    )
    .unwrap();
    for id in [1, 2, 0, -1, 100] {
        // 100 is fresh for the first group, but the second group reuses it.
        let error = block_on(raptor.run(
            &[document(1), document(2)],
            &Cancellation::new(),
            &mut || id,
        ))
        .unwrap_err();
        assert_eq!(error.code, "INVALID_ARGUMENTS");
    }
}

#[test]
fn raptor_reads_all_complete_text_parts_not_think() {
    let mut chat = model("first ");
    chat.parts.push(Part {
        r#type: "think".into(),
        data: json!({"value":"private"}).as_object().unwrap().clone(),
    });
    chat.parts.push(Part {
        r#type: "text".into(),
        data: json!({"value":"second"}).as_object().unwrap().clone(),
    });
    let raptor = Raptor::new(Map::new(), chat, embed()).unwrap();
    let output = block_on(raptor.run(&[document(1)], &Cancellation::new(), &mut || 100)).unwrap();
    assert_eq!(output.documents[0].content, "first second");
    assert_eq!(output.relations[0].from, 1);
    assert_eq!(output.relations[0].to, 100);
}

#[test]
fn empty_raptor_summary_is_not_a_successful_document() {
    for text in ["", " \n"] {
        let raptor = Raptor::new(Map::new(), model(text), embed()).unwrap();
        let mut allocated = false;
        let error = block_on(raptor.run(&[document(1)], &Cancellation::new(), &mut || {
            allocated = true;
            100
        }))
        .unwrap_err();
        assert_eq!(error.code, "INVALID_RESPONSE");
        assert!(!allocated);
    }
}

#[test]
fn synchronous_embed_cancellation_stops_summary_generation() {
    let cancellation = Cancellation::new();
    let chat = model("summary");
    let calls = chat.calls.clone();
    let embedding = Embedding {
        calls: Arc::default(),
        cancel: Some(cancellation.clone()),
    };
    let raptor = Raptor::new(Map::new(), chat, embedding).unwrap();
    let error = block_on(raptor.run(&[document(1)], &cancellation, &mut || {
        panic!("no allocation after cancel")
    }))
    .unwrap_err();
    assert_eq!(error.code, "CANCELLED");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn cancellation_during_id_allocation_does_not_return_output() {
    let cancellation = Cancellation::new();
    let raptor = Raptor::new(Map::new(), model("summary"), embed()).unwrap();
    let error = block_on(raptor.run(&[document(1)], &cancellation, &mut || {
        cancellation.cancel();
        100
    }))
    .unwrap_err();
    assert_eq!(error.code, "CANCELLED");
}

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
