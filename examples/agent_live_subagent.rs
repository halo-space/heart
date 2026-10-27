//! Live VLLM-backed parent/child Agent run. The child uses the real
//! current_time Tool and the parent must delegate rather than guess.
use std::{
    sync::{
        Arc, LazyLock,
        atomic::{AtomicI64, AtomicUsize, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use futures_util::StreamExt;
use halo::{
    agent::{self, Agent, Cancellation, Error, Input, Payload, SubAgent},
    agent::{
        event::Event,
        message::{Message, Role},
        model::chat,
        provider::vllm::{Config, Vllm, chat::Chat as VllmChat},
        tool::{Call, Tool, ToolDefinition},
        trace::fastrace::{Config as TraceConfig, ConsoleReporter, flush, set_reporter},
    },
};
use serde_json::json;

static NEXT_ID: AtomicI64 = AtomicI64::new(1);

fn next_id() -> i64 {
    NEXT_ID.fetch_add(1, Ordering::SeqCst)
}

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
        println!("CHILD_TOOL current_time returned unix_seconds={timestamp}");
        Ok(Message::function(json!({ "unix_seconds": timestamp })))
    }
}

fn live_model(provider: &Vllm) -> Result<VllmChat, Error> {
    let model_name = std::env::var("HEART_VLLM_MODEL")
        .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_VLLM_MODEL is required"))?;
    provider.chat(
        chat::Config::new(model_name).options(serde_json::Map::from_iter([(
            "chat_template_kwargs".into(),
            json!({ "enable_thinking": false }),
        )])),
    )
}

pub async fn run() -> Result<(), Error> {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL")
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "HEART_VLLM_BASE_URL is required"))?,
        api_key: std::env::var("HEART_VLLM_API_KEY")
            .map_err(|_| Error::new("AUTHENTICATION", "HEART_VLLM_API_KEY is required"))?,
        ..Config::default()
    })?;
    let calls = Arc::new(AtomicUsize::new(0));
    let child = agent::chat()
        .model(live_model(&provider)?)
        .system_prompt(
            "你是时间查询子 Agent。必须调用 current_time 工具，然后只根据工具结果报告 Unix 时间戳。",
        )
        .tools(CurrentTime(Arc::clone(&calls)))
        .build()?;
    let subagent = SubAgent::new("time_reader", child)?
        .description("调用 current_time 工具，返回当前真实 Unix 时间戳");
    let parent = agent::react()
        .model(live_model(&provider)?)
        .system_prompt(
            "你是主 Agent。不要调用 plan，也不需要创建计划。必须直接调用 agent_spawn 委派给 time_reader 子 Agent，且只委派一次。调用参数的 input 必须是完整 agent::Input JSON：{\"metadata\":{},\"payload\":[{\"type\":\"text\",\"data\":{\"value\":\"调用 current_time 工具并返回当前 Unix 时间戳\"},\"mime_type\":null}]}。不要自行猜测或回答。拿到子 Agent 结果后，准确转述时间戳。",
        )
        .subagents([subagent])
        .with_id_fn(next_id)
        .build()?;
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({ "value": "请告诉我当前准确的 Unix 时间戳。必须通过子 Agent 查询。" }),
            mime_type: None,
        }],
        ..Default::default()
    };

    let mut events = parent.run(input, true, Cancellation::new()).await?;
    let mut saw_child_tool_result = false;
    let mut saw_child_model_delta = false;
    while let Some(event) = events.next().await {
        match event? {
            Event::Delta { message } => {
                if message
                    .metadata
                    .get("node_id")
                    .and_then(|id| id.as_str())
                    .is_some()
                    && message
                        .metadata
                        .get("exec_id")
                        .and_then(|id| id.as_i64())
                        .is_some()
                {
                    saw_child_model_delta = true;
                }
                println!("DELTA {message:?}");
            }
            Event::Complete {
                message,
                usage,
                finish_reason,
            } => {
                if message.role == Role::Function
                    && message.metadata.get("name").and_then(|name| name.as_str())
                        == Some("current_time")
                {
                    if message
                        .metadata
                        .get("node_id")
                        .and_then(|id| id.as_str())
                        .is_none()
                        || message
                            .metadata
                            .get("exec_id")
                            .and_then(|id| id.as_i64())
                            .is_none()
                    {
                        return Err(Error::new(
                            "INVALID_RESPONSE",
                            "the live child Tool result is missing its execution source",
                        ));
                    }
                    saw_child_tool_result = true;
                }
                println!("COMPLETE {message:?} usage={usage:?} finish_reason={finish_reason:?}");
            }
        }
    }
    let call_count = calls.load(Ordering::SeqCst);
    println!("child_tool_call_count={call_count}");
    if call_count == 0 {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "the live parent/child run did not execute the child Tool",
        ));
    }
    if !saw_child_tool_result {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "the real child Tool result was not forwarded on the parent Event stream",
        ));
    }
    if !saw_child_model_delta {
        return Err(Error::new(
            "INVALID_RESPONSE",
            "the real child model did not forward a sourced Delta on the parent Event stream",
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
