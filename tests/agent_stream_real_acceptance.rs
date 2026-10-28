//! Real-provider acceptance for Agent stream mode behavior.

use futures_util::StreamExt;
use halo::agent::{
    self, Agent, Cancellation, Error, Input, Payload,
    checkpoint::Checkpoint,
    event::Event,
    message::Role,
    middleware::Middleware,
    model::chat::{self, Chat, Request},
    provider::deepseek::DeepSeek,
};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct CountedChat<C> {
    inner: C,
    calls: Arc<AtomicUsize>,
}

impl<C: Chat> Chat for CountedChat<C> {
    type Stream = C::Stream;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.stream(request).await
    }
}

struct WaitAfterModel(Arc<AtomicBool>);

impl Middleware for WaitAfterModel {
    async fn after_model(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        if self.0.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err(Error::new(
                "WAITING",
                "test waits after the real model result",
            ))
        }
    }
}

fn chat() -> impl chat::Chat {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct real DeepSeek provider");
    provider
        .chat(chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct real Chat")
}

fn input(chat_id: i64) -> Input {
    Input {
        metadata: json!({"chat_id":chat_id,"session_id":813_500})
            .as_object()
            .unwrap()
            .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"请用一句简短中文回答：2 加 2 等于几？"}),
            mime_type: None,
        }],
    }
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends live model requests"]
async fn real_model_stream_mode_emits_deltas_and_complete() {
    let agent = agent::chat()
        .model(chat())
        .system_prompt("直接、简短地回答用户问题。")
        .build()
        .expect("build Agent");
    let mut events = agent
        .run(input(813_501), true, Cancellation::new())
        .await
        .expect("start streaming run");
    let mut saw_delta = false;
    let mut final_message = None;
    while let Some(event) = events.next().await {
        match event.expect("real Agent event") {
            Event::Delta { .. } => saw_delta = true,
            Event::Complete { message, .. } if message.role == Role::Assistant => {
                final_message = Some(message);
            }
            Event::Complete { .. } => {}
        }
    }
    assert!(saw_delta, "stream=true should expose real provider deltas");
    let message = final_message.expect("real model should produce an assistant Complete");
    assert!(
        message.content.iter().any(|part| {
            part.r#type == "text"
                && part
                    .data
                    .get("value")
                    .and_then(|value| value.as_str())
                    .is_some_and(|s| !s.is_empty())
        }),
        "real Complete should contain assistant text: {message:?}"
    );
    println!("LIVE_AGENT_STREAM_COMPLETE={message:?}");
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends live model requests"]
async fn real_model_nonstream_mode_emits_complete_without_deltas() {
    let agent = agent::chat()
        .model(chat())
        .system_prompt("直接、简短地回答用户问题。")
        .build()
        .expect("build Agent");
    let mut events = agent
        .run(input(813_502), false, Cancellation::new())
        .await
        .expect("start non-streaming run");
    let mut count = 0;
    let mut final_message = None;
    while let Some(event) = events.next().await {
        match event.expect("real Agent event") {
            Event::Delta { .. } => panic!("stream=false must not emit Delta"),
            Event::Complete { message, .. } if message.role == Role::Assistant => {
                count += 1;
                final_message = Some(message);
            }
            Event::Complete { .. } => {}
        }
    }
    assert_eq!(count, 1, "one real model call should produce one Complete");
    let message = final_message.expect("real model should produce an assistant Complete");
    assert!(message.content.iter().any(|part| part.r#type == "text"));
    println!("LIVE_AGENT_NONSTREAM_COMPLETE={message:?}");
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends live model requests"]
async fn real_react_agent_honors_the_run_stream_mode() {
    for (chat_id, stream) in [(813_504, true), (813_505, false)] {
        let agent = agent::react()
            .model(chat())
            .system_prompt("直接、简短地回答用户问题；不需要工具或额外计划。")
            .with_max_iters(3)
            .build()
            .expect("build ReAct Agent");
        let mut events = agent
            .run(input(chat_id), stream, Cancellation::new())
            .await
            .expect("start real ReAct run");
        let mut deltas = 0;
        let mut assistant_completes = 0;
        let mut final_text = String::new();
        while let Some(event) = events.next().await {
            match event.expect("real ReAct event") {
                Event::Delta { .. } => deltas += 1,
                Event::Complete { message, .. } if message.role == Role::Assistant => {
                    assistant_completes += 1;
                    final_text = message
                        .content
                        .iter()
                        .filter(|part| part.r#type == "text")
                        .filter_map(|part| part.data.get("value").and_then(|value| value.as_str()))
                        .collect();
                }
                Event::Complete { .. } => {}
            }
        }
        if stream {
            assert!(deltas > 0, "real stream=true ReAct call should emit deltas");
        } else {
            assert_eq!(deltas, 0, "real stream=false ReAct call must hide deltas");
        }
        assert!(
            assistant_completes > 0,
            "real ReAct run must complete a model call"
        );
        assert!(
            !final_text.trim().is_empty(),
            "real Complete must contain text"
        );
        println!("LIVE_REACT_STREAM_MODE stream={stream} text={final_text:?}");
    }
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends a live model request"]
async fn real_stream_cancellation_closes_with_the_received_partial_message() {
    let chat_id = 813_503;
    let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
    let agent = agent::chat()
        .model(chat())
        .system_prompt("用简体中文详细说明地球四季形成的原因。")
        .checkpoint(checkpoint.clone())
        .build()
        .expect("build Agent");
    let cancellation = Cancellation::new();
    let mut events = agent
        .run(
            Input {
                metadata: json!({"chat_id":chat_id,"session_id":813_500})
                    .as_object()
                    .unwrap()
                    .clone(),
                payload: vec![Payload {
                    r#type: "text".into(),
                    data: json!({"value":"请分段说明地轴倾斜、公转和日照变化如何形成四季，并说明常见误解。"}),
                    mime_type: None,
                }],
            },
            true,
            cancellation.clone(),
        )
        .await
        .expect("start streaming run");
    let first = events
        .next()
        .await
        .expect("real provider should emit before cancellation")
        .expect("first real delta");
    assert!(matches!(first, Event::Delta { .. }));
    cancellation.cancel();

    let mut partial_complete = None;
    while let Some(event) = events.next().await {
        match event.expect("cancelled stream should close cleanly") {
            Event::Delta { .. } => {}
            Event::Complete {
                message,
                finish_reason,
                ..
            } if message.role == Role::Assistant => {
                partial_complete = Some((message, finish_reason));
            }
            Event::Complete { .. } => {}
        }
    }
    let (message, finish_reason) =
        partial_complete.expect("cancellation must deliver one final partial Assistant Message");
    assert_eq!(finish_reason.as_deref(), Some("cancelled"));
    assert!(
        !message.content.is_empty(),
        "partial Complete must preserve data actually received before cancellation"
    );
    let saved = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("the cancelled real run must have a saved Checkpoint");
    assert_eq!(saved["status"], "stopped");
    let model_step = saved["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["kind"] == "model")
        .expect("the real provider call must have its Model Step");
    assert!(model_step["message"].is_null());
    assert_eq!(model_step["attempts"][0]["status"], "cancelled");
    assert_eq!(
        model_step["attempts"][0]["message"]["content"],
        serde_json::to_value(&message.content).unwrap()
    );
    assert!(model_step["attempts"][0]["usage"].is_object());
    println!("LIVE_CANCELLED_AGENT_PARTIAL={message:?}");
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends a live model request"]
async fn real_model_result_is_not_replayed_when_its_after_hook_resumes() {
    let chat_id = 813_506;
    let calls = Arc::new(AtomicUsize::new(0));
    let allow_after = Arc::new(AtomicBool::new(false));
    let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
    let agent = agent::chat()
        .model(CountedChat {
            inner: chat(),
            calls: Arc::clone(&calls),
        })
        .system_prompt("用简体中文简短回答用户的问题，不需要工具。")
        .middleware(WaitAfterModel(Arc::clone(&allow_after)))
        .checkpoint(checkpoint.clone())
        .build()
        .expect("build Agent with a real Provider");

    let first = agent
        .run(input(chat_id), true, Cancellation::new())
        .await
        .expect("start real model call")
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(first.last(), Some(Err(error)) if error.code == "WAITING"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let waiting = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("save the waiting real-model scene");
    assert_eq!(waiting["status"], "waiting");
    assert_eq!(waiting["steps"][0]["attempts"][0]["status"], "succeeded");
    let original_message = waiting["steps"][0]["message"].clone();
    assert!(!original_message["content"].as_array().unwrap().is_empty());

    allow_after.store(true, Ordering::SeqCst);
    let resumed = agent
        .run(input(chat_id), true, Cancellation::new())
        .await
        .expect("resume the saved hook stage")
        .collect::<Vec<_>>()
        .await;
    assert!(resumed.iter().all(Result::is_ok), "{resumed:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "do not replay the model");
    let final_message = resumed
        .iter()
        .rev()
        .find_map(|event| match event.as_ref().ok()? {
            Event::Complete { message, .. } if message.role == Role::Assistant => Some(message),
            _ => None,
        })
        .expect("the resumed real model result should complete");
    assert!(final_message.content.iter().any(|part| {
        part.r#type == "text"
            && part
                .data
                .get("value")
                .and_then(Value::as_str)
                .is_some_and(|text| !text.trim().is_empty())
    }));
    let complete = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("save completed scene");
    assert_eq!(complete["status"], "succeeded");
    assert_eq!(complete["steps"][0]["message"], original_message);
    println!("LIVE_HOOK_RESUME_MESSAGE={final_message:?}");
}
