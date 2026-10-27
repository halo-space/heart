//! Live DeepSeek-backed Agent with a real Tool and fastrace Reporter.
//!
//! Required environment variable: DEEPSEEK_API_KEY.

use std::sync::{
    Arc, LazyLock,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use halo::agent::{self, Agent, Cancellation, Error, Input, Payload};
use halo::agent::{
    event::Event,
    message::Message,
    model::chat,
    provider::deepseek::DeepSeek,
    tool::{Call, Tool, ToolDefinition},
    trace::fastrace::{Config as TraceConfig, ConsoleReporter, flush, set_reporter},
};
use serde_json::json;

struct CurrentTime(Arc<AtomicUsize>);

impl Tool for CurrentTime {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: LazyLock<ToolDefinition> = LazyLock::new(|| ToolDefinition {
            name: "current_time".into(),
            description: "读取执行机器当前的 Unix 时间戳（秒）".into(),
            parameters: serde_json::from_value(json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }))
            .expect("valid tool schema"),
        });
        &DEFINITION
    }

    async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| Error::new("INTERNAL", error.to_string()))?
            .as_secs();
        self.0.fetch_add(1, Ordering::SeqCst);
        println!("TOOL current_time returned unix_seconds={timestamp}");
        Ok(Message::function(json!({ "unix_seconds": timestamp })))
    }
}

async fn run() -> Result<(), Error> {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(
            std::env::var("DEEPSEEK_API_KEY")
                .map_err(|_| Error::new("AUTHENTICATION", "DEEPSEEK_API_KEY is required"))?,
        )
        .build()?;
    let model = provider.chat(
        chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        )
        .options(serde_json::Map::from_iter([(
            "thinking".into(),
            json!({"type": "disabled"}),
        )])),
    )?;
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = agent::chat()
        .model(model)
        .system_prompt(
            "你必须调用 current_time 工具获取当前 Unix 时间戳，然后只根据工具结果回答。不要猜测时间。",
        )
        .tools(CurrentTime(Arc::clone(&calls)))
        .build()?;
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({ "value": "现在的 Unix 时间戳是多少？请报告工具返回的准确数值。" }),
            mime_type: None,
        }],
        ..Default::default()
    };

    let mut events = agent.run(input, true, Cancellation::new()).await?;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => println!("DELTA {message:?}"),
            Event::Complete {
                message,
                usage,
                finish_reason,
            } => println!("COMPLETE {message:?} usage={usage:?} finish_reason={finish_reason:?}"),
        }
    }
    let call_count = calls.load(Ordering::SeqCst);
    println!("tool_call_count={call_count}");
    if call_count == 0 {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "the live DeepSeek model completed without calling current_time",
        ));
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    set_reporter(ConsoleReporter, TraceConfig::default());
    let result = run().await;
    flush();
    result
}
