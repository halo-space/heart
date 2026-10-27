//! Live VLLM-backed Agent tool call. Reads the current time from the host;
//! no model or tool output is stubbed.
use std::{
    sync::{
        Arc, LazyLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use futures_util::StreamExt;
use halo::{
    agent::{self, Agent, Cancellation, Error, Input, Payload},
    agent::{
        event::Event,
        message::Message,
        model::chat,
        provider::vllm::{Config, Vllm, chat::Chat as VllmChat},
        tool::{Call, Tool, ToolDefinition},
        trace::fastrace::{Config as TraceConfig, ConsoleReporter, flush, set_reporter},
    },
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

fn live_model() -> Result<(Vllm, VllmChat), Error> {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL")
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_VLLM_BASE_URL is required"))?,
        api_key: std::env::var("HEART_VLLM_API_KEY")
            .map_err(|_| Error::new("AUTHENTICATION", "HEART_VLLM_API_KEY is required"))?,
        ..Config::default()
    })?;
    let model_name = std::env::var("HEART_VLLM_MODEL")
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_VLLM_MODEL is required"))?;
    let model = provider.chat(chat::Config::new(model_name).options(
        serde_json::Map::from_iter([(
            "chat_template_kwargs".into(),
            json!({ "enable_thinking": false }),
        )]),
    ))?;
    Ok((provider, model))
}

pub async fn run() -> Result<(), Error> {
    let (_provider, model) = live_model()?;
    let calls = Arc::new(AtomicUsize::new(0));
    let tool = CurrentTime(Arc::clone(&calls));
    let agent = agent::chat()
        .model(model)
        .system_prompt(
            "你必须调用 current_time 工具获取当前 Unix 时间戳，然后根据工具返回值回答。不要猜测时间。",
        )
        .tools(tool)
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
            "the live model completed without calling current_time",
        ));
    }
    Ok(())
}

#[tokio::main]
#[allow(dead_code)]
async fn main() -> Result<(), Error> {
    set_reporter(ConsoleReporter, TraceConfig::default());
    let result = run().await;
    flush();
    result
}
