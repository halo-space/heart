//! Opt-in live acceptance tests. These call the configured VLLM endpoint and
//! do not replace the provider or model output with test doubles.

use futures_util::StreamExt;
use halo::agent::{
    self, Agent, Cancellation, Error, Input, Message, Payload,
    event::Event,
    mcp::Definition,
    middleware::Middleware,
    model::chat,
    permission::{Behavior, Decision, Request as PermissionRequest},
    provider::deepseek::DeepSeek,
    provider::vllm::{Config, Vllm},
    tool::{Call, Evaluation, Tool, ToolDefinition},
};
use serde_json::{Value, json};
use std::sync::{
    Arc, LazyLock,
    atomic::{AtomicUsize, Ordering},
};
use tokio::process::Command;

struct CountFinalExecution(Arc<AtomicUsize>);

impl Middleware for CountFinalExecution {
    async fn after_execute(&self, _: &mut Value, _: &Cancellation) -> Result<(), Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[path = "../examples/agent_live_tool.rs"]
mod agent_live_tool;

#[path = "../examples/agent_live_subagent.rs"]
mod agent_live_subagent;

struct LiveClock {
    calls: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

fn live_clock(calls: Arc<AtomicUsize>) -> LiveClock {
    LiveClock {
        calls,
        active: Arc::new(AtomicUsize::new(0)),
        peak: Arc::new(AtomicUsize::new(0)),
    }
}

fn next_permission_reply_id() -> i64 {
    static NEXT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(8_139_000);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn live_vllm_chat() -> impl chat::Chat + Clone {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL").expect("set HEART_VLLM_BASE_URL"),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("set HEART_VLLM_API_KEY"),
        ..Config::default()
    })
    .expect("construct VLLM provider");
    provider
        .chat(
            chat::Config::new(std::env::var("HEART_VLLM_MODEL").expect("set HEART_VLLM_MODEL"))
                .options(serde_json::Map::from_iter([(
                    "chat_template_kwargs".into(),
                    json!({"enable_thinking":false}),
                )])),
        )
        .expect("construct real Chat model")
}

fn live_ask_agent(
    checkpoint: Arc<halo::agent::core::checkpoint::InMemory>,
    calls: Arc<AtomicUsize>,
    authorizations: Arc<std::sync::Mutex<Vec<PermissionRequest>>>,
) -> impl Agent {
    let policy_authorizations = Arc::clone(&authorizations);
    let permission = halo::agent::core::permission::Permission::new(
        move |request: PermissionRequest, _cancellation| {
            let authorizations = Arc::clone(&policy_authorizations);
            async move {
                authorizations.lock().unwrap().push(request);
                Ok(Decision {
                    behavior: Behavior::Ask,
                    reason: Some("approval required before reading the clock".into()),
                })
            }
        },
    );
    agent::react()
        .model(live_vllm_chat())
        .permission(permission)
        .checkpoint(checkpoint)
        .with_id_fn(next_permission_reply_id)
        .system_prompt(
            "必须调用 current_time 获取真实 Unix 时间戳。权限批准后才会执行工具；最终回答只能使用工具返回的时间戳。",
        )
        .tools(live_clock(calls))
        .build()
        .expect("build ReAct Agent with Ask Permission")
}

impl Tool for LiveClock {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: LazyLock<ToolDefinition> = LazyLock::new(|| ToolDefinition {
            name: "current_time".into(),
            description: "读取当前系统 Unix 时间戳（秒）".into(),
            parameters: serde_json::from_value(json!({
                "type":"object",
                "properties":{},
                "additionalProperties":false
            }))
            .expect("valid clock tool schema"),
        });
        &DEFINITION
    }

    async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| Error::new("INTERNAL", error.to_string()))?
            .as_secs();
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(Message::function(json!({"unix_seconds":timestamp})))
    }
}

struct LiveEvalWeather {
    calls: Arc<AtomicUsize>,
    evals: Arc<AtomicUsize>,
}

struct LiveChangingLookup {
    calls: Arc<AtomicUsize>,
}

impl Tool for LiveChangingLookup {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: LazyLock<ToolDefinition> = LazyLock::new(|| ToolDefinition {
            name: "lookup_city".into(),
            description: "查询指定城市的实时测试信息".into(),
            parameters: serde_json::from_value(json!({
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"],
                "additionalProperties":false
            }))
            .expect("valid lookup schema"),
        });
        &DEFINITION
    }

    async fn before(&self, call: &mut Call, _: &Cancellation) -> Result<(), Error> {
        call.arguments.insert("city".into(), json!("Paris"));
        Ok(())
    }

    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Message::function(
            json!({"city":call.arguments["city"],"source":"live tool"}),
        ))
    }
}

impl Tool for LiveEvalWeather {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: LazyLock<ToolDefinition> = LazyLock::new(|| ToolDefinition {
            name: "weather".into(),
            description: "读取北京当前天气和温度".into(),
            parameters: serde_json::from_value(json!({
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"],
                "additionalProperties":false
            }))
            .expect("valid weather tool schema"),
        });
        &DEFINITION
    }

    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Message::function(json!({
            "city":call.arguments.get("city"),
            "temperature_celsius":22,
            "condition":"sunny"
        })))
    }

    async fn eval(
        &self,
        _: &Call,
        message: &Message,
        _: &Cancellation,
    ) -> Result<Evaluation, Error> {
        assert_eq!(message.values["temperature_celsius"], 22);
        if self.evals.fetch_add(1, Ordering::SeqCst) == 0 {
            Err(Error::new(
                "WAITING",
                "weather evaluation temporarily waiting",
            ))
        } else {
            Ok(Evaluation::passed())
        }
    }
}

#[derive(Default)]
struct RejectWaitingCheckpoint(halo::agent::core::checkpoint::InMemory);

impl halo::agent::checkpoint::Checkpoint for RejectWaitingCheckpoint {
    async fn load(&self, id: i64, cancellation: &Cancellation) -> Result<Value, Error> {
        self.0.load(id, cancellation).await
    }

    async fn save(&self, id: i64, state: Value, cancellation: &Cancellation) -> Result<(), Error> {
        if state["status"] == "waiting" {
            return Err(Error::new(
                "INTERNAL",
                "injected waiting Checkpoint persistence failure",
            ));
        }
        self.0.save(id, state, cancellation).await
    }

    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        self.0.delete(id, cancellation).await
    }
}

struct LiveWorkingDirectory {
    calls: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

impl Tool for LiveWorkingDirectory {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: std::sync::LazyLock<ToolDefinition> =
            std::sync::LazyLock::new(|| ToolDefinition {
                name: "current_directory".into(),
                description: "读取当前进程真实工作目录".into(),
                parameters: serde_json::from_value(json!({
                    "type":"object",
                    "properties":{},
                    "additionalProperties":false
                }))
                .expect("valid working-directory tool schema"),
            });
        &DEFINITION
    }

    async fn call(&self, _call: &Call, _cancellation: &Cancellation) -> Result<Message, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        // Keep the two real side effects observable as overlapping work when
        // the Provider emits both calls in one Assistant message.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        let path =
            std::env::current_dir().map_err(|error| Error::new("INTERNAL", error.to_string()))?;
        Ok(Message::function(json!({"path":path})))
    }
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_react_agent_uses_a_live_tool_result() {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL").expect("set HEART_VLLM_BASE_URL"),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("set HEART_VLLM_API_KEY"),
        ..Config::default()
    })
    .expect("construct VLLM provider");
    let model_name = std::env::var("HEART_VLLM_MODEL").expect("set HEART_VLLM_MODEL");
    let model = provider
        .chat(
            chat::Config::new(model_name).options(serde_json::Map::from_iter([(
                "chat_template_kwargs".into(),
                json!({"enable_thinking":false}),
            )])),
        )
        .expect("construct ReAct Chat model");
    let calls = Arc::new(AtomicUsize::new(0));
    let authorizations = Arc::new(std::sync::Mutex::new(Vec::<PermissionRequest>::new()));
    let policy_authorizations = Arc::clone(&authorizations);
    let permission = halo::agent::core::permission::Permission::new(
        move |request: PermissionRequest, _cancellation| {
            let authorizations = Arc::clone(&policy_authorizations);
            async move {
                authorizations.lock().unwrap().push(request);
                Ok(Decision {
                    behavior: Behavior::Allow,
                    reason: None,
                })
            }
        },
    );
    let clock = live_clock(Arc::clone(&calls));
    let finalized = Arc::new(AtomicUsize::new(0));
    let agent = agent::react()
        .model(model)
        .permission(permission)
        .middleware(CountFinalExecution(Arc::clone(&finalized)))
        .system_prompt(
            "必须调用 current_time 获取真实 Unix 时间戳，然后只根据工具结果作答，不得猜测。",
        )
        .tools(clock)
        .build()
        .expect("build ReAct Agent");
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"现在的 Unix 时间戳是多少？请查询工具并准确回答。"}),
            mime_type: None,
        }],
        ..Default::default()
    };
    let mut events = agent
        .run(input, true, Cancellation::new())
        .await
        .expect("start real ReAct run");
    let mut final_text = String::new();
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event.expect("ReAct event")
            && message.role == halo::agent::message::Role::Assistant
        {
            final_text = message
                .content
                .iter()
                .filter(|part| part.r#type == "text")
                .filter_map(|part| part.data.get("value").and_then(Value::as_str))
                .collect();
        }
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "real ReAct must call the live clock Tool"
    );
    let authorizations = authorizations.lock().unwrap();
    assert_eq!(
        authorizations.len(),
        1,
        "the real tool call must be checked"
    );
    assert_eq!(authorizations[0].action, "call");
    assert_eq!(authorizations[0].resource, "current_time");
    assert_eq!(authorizations[0].metadata["call"]["name"], "current_time");
    assert_eq!(
        finalized.load(Ordering::SeqCst),
        1,
        "the live tool-selection model response is not the final Agent execution"
    );
    let timestamp = final_text
        .split(|character: char| !character.is_ascii_digit())
        .find(|candidate| candidate.len() == 10)
        .and_then(|candidate| candidate.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("ReAct did not return the real Tool timestamp: {final_text:?}"));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_secs();
    assert!(timestamp.abs_diff(now) <= 10);
    println!("LIVE_REACT_TOOL_TIMESTAMP={timestamp}");
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_model_emits_and_executes_a_multi_tool_batch() {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL").expect("set HEART_VLLM_BASE_URL"),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("set HEART_VLLM_API_KEY"),
        ..Config::default()
    })
    .expect("construct VLLM provider");
    let model_name = std::env::var("HEART_VLLM_MODEL").expect("set HEART_VLLM_MODEL");
    let model = provider
        .chat(
            chat::Config::new(model_name).options(serde_json::Map::from_iter([(
                "chat_template_kwargs".into(),
                json!({"enable_thinking":false}),
            )])),
        )
        .expect("construct ReAct Chat model");

    let clock_calls = Arc::new(AtomicUsize::new(0));
    let directory_calls = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let agent = agent::react()
        .model(model)
        .system_prompt(
            "必须在同一轮中分别调用 current_time 和 current_directory，拿到两个真实结果后再回答。不得自行推测或省略任一工具调用。",
        )
        .tools(LiveClock {
            calls: Arc::clone(&clock_calls),
            active: Arc::clone(&active),
            peak: Arc::clone(&peak),
        })
        .tools(LiveWorkingDirectory {
            calls: Arc::clone(&directory_calls),
            active: Arc::clone(&active),
            peak: Arc::clone(&peak),
        })
        .build()
        .expect("build ReAct Agent");
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"请同时查询当前 Unix 时间戳和当前工作目录，然后简要报告这两个真实值。"}),
            mime_type: None,
        }],
        ..Default::default()
    };
    let mut events = agent
        .run(input, true, Cancellation::new())
        .await
        .expect("start live multi-tool run");
    let mut model_tool_batches = Vec::new();
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event.expect("ReAct event")
            && message.role == halo::agent::message::Role::Assistant
        {
            let tool_calls = message
                .content
                .iter()
                .filter(|part| part.r#type == "tool_call")
                .count();
            if tool_calls > 0 {
                model_tool_batches.push(tool_calls);
            }
        }
    }

    assert!(
        model_tool_batches.contains(&2),
        "the real Provider must emit both tool calls in one Assistant message; batches: {model_tool_batches:?}"
    );
    assert_eq!(clock_calls.load(Ordering::SeqCst), 1);
    assert_eq!(directory_calls.load(Ordering::SeqCst), 1);
    assert_eq!(active.load(Ordering::SeqCst), 0);
    assert_eq!(peak.load(Ordering::SeqCst), 2, "tool calls should overlap");
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_react_agent_uses_its_builtin_plan_for_independent_live_tasks() {
    use halo::agent::checkpoint::Checkpoint as _;

    let chat_id = 8_139_101;
    let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
    let clock_calls = Arc::new(AtomicUsize::new(0));
    let directory_calls = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let cwd = std::env::current_dir().expect("read actual process directory");
    let agent = agent::react()
        .model(live_vllm_chat())
        .system_prompt("需要实时事实时使用已配置的工具，不要猜测工具返回的数据。")
        .tools(live_clock(Arc::clone(&clock_calls)))
        .tools(LiveWorkingDirectory {
            calls: Arc::clone(&directory_calls),
            active: Arc::clone(&active),
            peak: Arc::clone(&peak),
        })
        .with_id_fn(next_permission_reply_id)
        .with_max_concurrency(2)
        .checkpoint(Arc::clone(&checkpoint))
        .build()
        .expect("build real planned ReAct Agent");
    let input: Input = serde_json::from_value(json!({
        "metadata":{"chat_id":chat_id,"session_id":8_139_102},
            "payload":[{"type":"text","data":{"value":"请计划如何读取当前 Unix 时间和工作目录，然后报告这两个事实。"}}]
    }))
    .expect("valid planned Agent input");

    let events = agent
        .run(input, true, Cancellation::new())
        .await
        .expect("start real planned Agent")
        .collect::<Vec<_>>()
        .await;
    let errors = events
        .iter()
        .filter_map(|event| event.as_ref().err())
        .collect::<Vec<_>>();
    assert!(errors.is_empty(), "{errors:?}");
    assert!(
        clock_calls.load(Ordering::SeqCst) >= 1,
        "read the live clock"
    );
    assert!(
        directory_calls.load(Ordering::SeqCst) >= 1,
        "read the live working directory"
    );
    let saved = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("complete Agent Checkpoint");
    assert_eq!(saved["status"], "succeeded");
    let plan = &saved["graph"]["plan"];
    let planned_nodes = plan["nodes"].as_object().expect("static plan graph");
    let nodes = saved["nodes"].as_object().expect("executed plan state");
    assert!(planned_nodes.len() >= 2);
    assert_eq!(planned_nodes.len(), nodes.len());
    let incoming = plan["edges"]
        .as_array()
        .expect("plan edges")
        .iter()
        .filter_map(|edge| edge["to"].as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let roots = planned_nodes
        .iter()
        .filter(|(node_id, _)| !incoming.contains(node_id.as_str()))
        .map(|(_, node)| node)
        .collect::<Vec<_>>();
    let objectives = roots
        .iter()
        .map(|node| node["objective"].as_str().expect("planned objective"))
        .collect::<Vec<_>>();
    assert!(
        objectives
            .iter()
            .any(|objective| objective.contains("Unix"))
    );
    assert!(
        objectives
            .iter()
            .any(|objective| objective.contains("工作目录"))
    );
    assert!(nodes.values().all(|node| node["status"] == "succeeded"));
    let result = events
        .iter()
        .filter_map(|event| match event.as_ref().ok()? {
            Event::Complete { message, .. }
                if message.role == halo::agent::message::Role::Assistant =>
            {
                Some(message)
            }
            _ => None,
        })
        .next_back()
        .expect("real final Assistant message");
    let rendered = serde_json::to_string(result).expect("serialize final response");
    assert!(rendered.contains(&cwd.to_string_lossy().to_string()));
    println!("LIVE_REACT_PLAN_FINAL={result:?}");
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_model_tool_selection_is_stopped_by_live_permission_before_side_effect() {
    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL").expect("set HEART_VLLM_BASE_URL"),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("set HEART_VLLM_API_KEY"),
        ..Config::default()
    })
    .expect("construct VLLM provider");
    let model = provider
        .chat(
            chat::Config::new(std::env::var("HEART_VLLM_MODEL").expect("set HEART_VLLM_MODEL"))
                .options(serde_json::Map::from_iter([(
                    "chat_template_kwargs".into(),
                    json!({"enable_thinking":false}),
                )])),
        )
        .expect("construct real Chat");
    let calls = Arc::new(AtomicUsize::new(0));
    let authorizations = Arc::new(std::sync::Mutex::new(Vec::<PermissionRequest>::new()));
    let policy_authorizations = Arc::clone(&authorizations);
    let permission = halo::agent::core::permission::Permission::new(
        move |request: PermissionRequest, _cancellation| {
            let authorizations = Arc::clone(&policy_authorizations);
            async move {
                authorizations.lock().unwrap().push(request);
                Ok(Decision {
                    behavior: Behavior::Deny,
                    reason: Some("acceptance test denies the live operation".into()),
                })
            }
        },
    );
    let agent = agent::react()
        .model(model)
        .permission(permission)
        .system_prompt(
            "你必须调用 current_time 获取真实 Unix 时间戳；即使调用被拒绝，也不能声称调用成功或编造时间。",
        )
        .tools(live_clock(Arc::clone(&calls)))
        .build()
        .expect("build ReAct Agent with denying Permission");
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"查询当前 Unix 时间戳。"}),
            mime_type: None,
        }],
        ..Default::default()
    };

    let events = agent
        .run(input, true, Cancellation::new())
        .await
        .expect("start real ReAct run")
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "Permission denial must prevent the Tool side effect"
    );
    let authorizations = authorizations.lock().unwrap();
    assert!(
        !authorizations.is_empty(),
        "the real model must select the protected Tool"
    );
    assert_eq!(authorizations[0].action, "call");
    assert_eq!(authorizations[0].resource, "current_time");
    println!("LIVE_DENIED_TOOL_PERMISSION={:?}", authorizations[0]);
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_tool_permission_ask_waits_for_reply_before_the_live_side_effect() {
    use halo::agent::checkpoint::Checkpoint as _;

    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL").expect("set HEART_VLLM_BASE_URL"),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("set HEART_VLLM_API_KEY"),
        ..Config::default()
    })
    .expect("construct VLLM provider");
    let model = provider
        .chat(
            chat::Config::new(std::env::var("HEART_VLLM_MODEL").expect("set HEART_VLLM_MODEL"))
                .options(serde_json::Map::from_iter([(
                    "chat_template_kwargs".into(),
                    json!({"enable_thinking":false}),
                )])),
        )
        .expect("construct real Chat");
    let calls = Arc::new(AtomicUsize::new(0));
    let authorizations = Arc::new(std::sync::Mutex::new(Vec::<PermissionRequest>::new()));
    let policy_authorizations = Arc::clone(&authorizations);
    let permission = halo::agent::core::permission::Permission::new(
        move |request: PermissionRequest, _cancellation| {
            let authorizations = Arc::clone(&policy_authorizations);
            async move {
                authorizations.lock().unwrap().push(request);
                Ok(Decision {
                    behavior: Behavior::Ask,
                    reason: Some("approval required before reading the clock".into()),
                })
            }
        },
    );
    let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
    let chat_id = 8_139_001;
    let session_id = 8_139_002;
    let agent = agent::react()
        .model(model)
        .permission(permission)
        .checkpoint(checkpoint.clone())
        .with_id_fn(next_permission_reply_id)
        .system_prompt(
            "必须调用 current_time 获取真实 Unix 时间戳。权限批准后才会执行工具；最终回答只能使用工具返回的时间戳。",
        )
        .tools(live_clock(Arc::clone(&calls)))
        .build()
        .expect("build ReAct Agent with Ask Permission");
    let input = Input {
        metadata: serde_json::from_value(json!({"chat_id":chat_id,"session_id":session_id}))
            .expect("valid run identity"),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"查询当前 Unix 时间戳。请先申请工具权限。"}),
            mime_type: None,
        }],
    };
    let cancellation = Cancellation::new();
    let mut events = agent
        .run(input, true, cancellation.clone())
        .await
        .expect("start real permission run");
    let mut reply_id = None;
    while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(60), events.next())
        .await
        .expect("real model must reach the permission question")
    {
        let event = event.expect("permission event");
        if let Event::Complete { message, .. } = event
            && let Some(id) = message.metadata.get("reply_id").and_then(Value::as_i64)
        {
            reply_id = Some(id);
            break;
        }
    }
    let reply_id = reply_id.expect("Permission Ask must publish a reply_id");
    assert_eq!(calls.load(Ordering::SeqCst), 0, "Ask must gate the Tool");
    let waiting = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("saved waiting scene");
    assert_eq!(waiting["status"], "waiting");
    let tool_step = waiting["steps"]
        .as_array()
        .expect("root steps")
        .iter()
        .find(|step| step["kind"] == "tool")
        .expect("waiting Tool Step");
    assert_eq!(tool_step["status"], "waiting");
    assert_eq!(tool_step["waits"][reply_id.to_string()]["kind"], "approval");
    assert_eq!(
        tool_step["waits"][reply_id.to_string()]["status"],
        "waiting"
    );
    assert_eq!(authorizations.lock().unwrap().len(), 1);

    let reply = Input {
        metadata: serde_json::from_value(json!({
            "chat_id":chat_id,"session_id":session_id,"reply_id":reply_id
        }))
        .expect("valid reply identity"),
        payload: vec![Payload {
            r#type: "json".into(),
            data: json!({"value":{"approved":true}}),
            mime_type: None,
        }],
    };
    agent
        .reply(reply, cancellation)
        .await
        .expect("deliver explicit approval");
    let tail = tokio::time::timeout(std::time::Duration::from_secs(60), async {
        events.collect::<Vec<_>>().await
    })
    .await
    .expect("Agent should finish after approval");
    assert!(tail.iter().all(Result::is_ok), "{tail:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "approved Tool runs once");
    let final_message = tail
        .iter()
        .filter_map(|event| match event.as_ref().unwrap() {
            Event::Complete { message, .. }
                if message.role == halo::agent::message::Role::Assistant
                    && !message.metadata.contains_key("reply_id") =>
            {
                Some(message)
            }
            _ => None,
        })
        .next_back()
        .expect("model must answer after the live Tool result");
    let final_text: String = final_message
        .content
        .iter()
        .filter(|part| part.r#type == "text")
        .filter_map(|part| part.data.get("value").and_then(Value::as_str))
        .collect();
    let timestamp = final_text
        .split(|character: char| !character.is_ascii_digit())
        .find(|candidate| candidate.len() == 10)
        .and_then(|candidate| candidate.parse::<u64>().ok())
        .expect("final answer must report the live Tool timestamp");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_secs();
    assert!(timestamp.abs_diff(now) <= 10);
    println!("LIVE_PERMISSION_ASK_TIMESTAMP={timestamp}");
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_model_requires_new_approval_after_tool_before_changes_call_arguments() {
    let calls = Arc::new(AtomicUsize::new(0));
    let authorizations = Arc::new(std::sync::Mutex::new(Vec::<PermissionRequest>::new()));
    let policy_authorizations = Arc::clone(&authorizations);
    let permission = halo::agent::core::permission::Permission::new(
        move |request: PermissionRequest, _cancellation| {
            let authorizations = Arc::clone(&policy_authorizations);
            async move {
                authorizations.lock().unwrap().push(request);
                Ok(Decision {
                    behavior: Behavior::Ask,
                    reason: Some("approve this exact city lookup".into()),
                })
            }
        },
    );
    let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
    let chat_id = 8_139_051;
    let session_id = 8_139_052;
    let agent = agent::react()
        .model(live_vllm_chat())
        .permission(permission)
        .checkpoint(checkpoint)
        .with_id_fn(next_permission_reply_id)
        .system_prompt(
            "必须调用 lookup_city，参数 city 必须是 Beijing。若用户拒绝后续的权限申请，停止调用工具并简短说明无法继续。",
        )
        .tools(LiveChangingLookup {
            calls: Arc::clone(&calls),
        })
        .build()
        .expect("build ReAct Agent with argument-changing Tool hook");
    let input = Input {
        metadata: serde_json::from_value(json!({"chat_id":chat_id,"session_id":session_id}))
            .expect("valid run identity"),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"查询 Beijing 的信息。"}),
            mime_type: None,
        }],
    };
    let mut events = agent
        .run(input, true, Cancellation::new())
        .await
        .expect("start real model run");
    let mut approvals = 0;
    while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(90), events.next())
        .await
        .expect("real model should finish the approval flow")
    {
        let event = event.expect("Agent event");
        if let Event::Complete { message, .. } = event
            && let Some(reply_id) = message.metadata.get("reply_id").and_then(Value::as_i64)
        {
            approvals += 1;
            assert!(approvals <= 2, "denial must not trigger an approval loop");
            agent
                .reply(
                    Input {
                        metadata: serde_json::from_value(json!({
                            "chat_id":chat_id,
                            "session_id":session_id,
                            "reply_id":reply_id
                        }))
                        .expect("valid approval identity"),
                        payload: vec![Payload {
                            r#type: "json".into(),
                            data: json!({"value":{"approved":approvals == 1}}),
                            mime_type: None,
                        }],
                    },
                    Cancellation::new(),
                )
                .await
                .expect("submit explicit decision for this exact operation");
        }
    }

    let authorizations = authorizations.lock().unwrap();
    assert_eq!(approvals, 2, "changed arguments require fresh approval");
    assert_eq!(authorizations.len(), 2);
    assert_eq!(
        authorizations[0].metadata["call"]["arguments"]["city"],
        "Beijing"
    );
    assert_eq!(
        authorizations[1].metadata["call"]["arguments"]["city"],
        "Paris"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "second operation was denied"
    );
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_chat_agent_waits_for_tool_approval_then_executes_the_live_tool() {
    use halo::agent::checkpoint::Checkpoint as _;

    let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let authorizations = Arc::new(std::sync::Mutex::new(Vec::<PermissionRequest>::new()));
    let policy_authorizations = Arc::clone(&authorizations);
    let permission = halo::agent::core::permission::Permission::new(
        move |request: PermissionRequest, _cancellation| {
            let authorizations = Arc::clone(&policy_authorizations);
            async move {
                authorizations.lock().unwrap().push(request);
                Ok(Decision {
                    behavior: Behavior::Ask,
                    reason: Some("approval required before reading the clock".into()),
                })
            }
        },
    );
    let agent = agent::chat()
        .model(live_vllm_chat())
        .system_prompt(
            "必须调用 current_time 获取真实 Unix 时间戳。请先等待工具权限批准；工具返回后再用结果回答。",
        )
        .tools(live_clock(Arc::clone(&calls)))
        .permission(permission)
        .with_id_fn(next_permission_reply_id)
        .checkpoint(checkpoint.clone())
        .build()
        .expect("build real Chat Agent with Ask Permission");
    let chat_id = 8_139_061;
    let session_id = 8_139_062;
    let mut events = Box::pin(
        agent
            .run(
                Input {
                    metadata: serde_json::from_value(
                        json!({"chat_id":chat_id,"session_id":session_id}),
                    )
                    .expect("valid run identity"),
                    payload: vec![Payload {
                        r#type: "text".into(),
                        data: json!({"value":"查询当前 Unix 时间戳。"}),
                        mime_type: None,
                    }],
                },
                true,
                Cancellation::new(),
            )
            .await
            .expect("start real Chat Agent run"),
    );
    let reply_id = loop {
        let event = tokio::time::timeout(std::time::Duration::from_secs(90), events.next())
            .await
            .expect("real Chat Agent should reach its approval wait")
            .expect("Chat Agent should emit an event")
            .expect("real Chat Agent event should succeed");
        if let Event::Complete { message, .. } = event
            && let Some(reply_id) = message.metadata.get("reply_id").and_then(Value::as_i64)
        {
            break reply_id;
        }
    };
    assert_eq!(calls.load(Ordering::SeqCst), 0, "Ask gates the real Tool");
    let waiting = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("Chat approval Checkpoint exists");
    assert_eq!(waiting["status"], "waiting");

    agent
        .reply(
            Input {
                metadata: serde_json::from_value(json!({
                    "chat_id":chat_id,
                    "session_id":session_id,
                    "reply_id":reply_id
                }))
                .expect("valid approval identity"),
                payload: vec![Payload {
                    r#type: "json".into(),
                    data: json!({"value":{"approved":true}}),
                    mime_type: None,
                }],
            },
            Cancellation::new(),
        )
        .await
        .expect("approve the exact live Tool call");
    while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(90), events.next())
        .await
        .expect("Chat Agent should finish after approval")
    {
        event.expect("real Chat Agent event after approval");
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(authorizations.lock().unwrap().len(), 1);
    let completed = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("Chat Agent completion Checkpoint exists");
    assert_eq!(completed["status"], "succeeded");
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_agent_restores_a_saved_tool_approval_with_a_fresh_instance() {
    use halo::agent::checkpoint::Checkpoint as _;

    let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let authorizations = Arc::new(std::sync::Mutex::new(Vec::<PermissionRequest>::new()));
    let chat_id = 8_139_101;
    let session_id = 8_139_102;
    let initial_input = Input {
        metadata: serde_json::from_value(json!({"chat_id":chat_id,"session_id":session_id}))
            .expect("valid run identity"),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"查询当前 Unix 时间戳。请先申请工具权限。"}),
            mime_type: None,
        }],
    };

    let initial = live_ask_agent(
        checkpoint.clone(),
        Arc::clone(&calls),
        Arc::clone(&authorizations),
    );
    let mut events = Box::pin(
        initial
            .run(initial_input, true, Cancellation::new())
            .await
            .expect("start real model run"),
    );
    let reply_id = loop {
        let event = tokio::time::timeout(std::time::Duration::from_secs(60), events.next())
            .await
            .expect("real model must reach its approval question")
            .expect("Agent emits the approval question")
            .expect("model event succeeds");
        if let Event::Complete { message, .. } = event
            && let Some(reply_id) = message.metadata.get("reply_id").and_then(Value::as_i64)
        {
            break reply_id;
        }
    };
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let waiting = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("approval Checkpoint exists before interruption");
    assert_eq!(waiting["status"], "waiting");
    drop(events);

    let restored = live_ask_agent(checkpoint.clone(), Arc::clone(&calls), authorizations);
    let reply = Input {
        metadata: serde_json::from_value(json!({
            "chat_id":chat_id,"session_id":session_id,"reply_id":reply_id
        }))
        .expect("valid reply identity"),
        payload: vec![Payload {
            r#type: "json".into(),
            data: json!({"value":{"approved":true}}),
            mime_type: None,
        }],
    };
    let mut resumed = Box::pin(
        restored
            .run(reply, true, Cancellation::new())
            .await
            .expect("fresh Agent resumes from the saved approval and Checkpoint"),
    );
    let mut final_text = String::new();
    while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(60), resumed.next())
        .await
        .expect("restored real Agent should finish")
    {
        if let Event::Complete { message, .. } = event.expect("restored Agent event")
            && message.role == halo::agent::message::Role::Assistant
            && !message.metadata.contains_key("reply_id")
        {
            final_text = message
                .content
                .iter()
                .filter(|part| part.r#type == "text")
                .filter_map(|part| part.data.get("value").and_then(Value::as_str))
                .collect();
        }
    }

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "restored Tool executes once"
    );
    assert!(
        !final_text.is_empty(),
        "the real model answers after the Tool"
    );
    let final_state = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("restored completion Checkpoint exists");
    assert_eq!(final_state["status"], "succeeded");
    assert_eq!(
        final_state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|step| step["kind"] == "tool")
            .count(),
        1
    );
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_model_approval_is_not_published_when_waiting_checkpoint_save_fails() {
    use halo::agent::checkpoint::Checkpoint as _;

    let checkpoint = Arc::new(RejectWaitingCheckpoint::default());
    let calls = Arc::new(AtomicUsize::new(0));
    let authorizations = Arc::new(std::sync::Mutex::new(Vec::<PermissionRequest>::new()));
    let policy_authorizations = Arc::clone(&authorizations);
    let permission = halo::agent::core::permission::Permission::new(
        move |request: PermissionRequest, _cancellation| {
            let authorizations = Arc::clone(&policy_authorizations);
            async move {
                authorizations.lock().unwrap().push(request);
                Ok(Decision {
                    behavior: Behavior::Ask,
                    reason: Some("approval required before reading the clock".into()),
                })
            }
        },
    );
    let chat_id = 8_139_201;
    let session_id = 8_139_202;
    let agent = agent::react()
        .model(live_vllm_chat())
        .permission(permission)
        .checkpoint(checkpoint.clone())
        .with_id_fn(next_permission_reply_id)
        .system_prompt(
            "必须调用 current_time 获取真实 Unix 时间戳。权限批准后才会执行工具；最终回答只能使用工具返回的时间戳。",
        )
        .tools(live_clock(Arc::clone(&calls)))
        .build()
        .expect("build ReAct Agent with real model and failing Checkpoint");
    let input = Input {
        metadata: serde_json::from_value(json!({"chat_id":chat_id,"session_id":session_id}))
            .expect("valid run identity"),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"查询当前 Unix 时间戳。请先申请工具权限。"}),
            mime_type: None,
        }],
    };

    let mut events = agent
        .run(input, true, Cancellation::new())
        .await
        .expect("start real model run");
    let mut saw_persistence_error = false;
    while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(60), events.next())
        .await
        .expect("real model should reach the approval checkpoint")
    {
        match event {
            Err(error)
                if error
                    .message
                    .contains("waiting Checkpoint persistence failure") =>
            {
                saw_persistence_error = true;
            }
            Err(error) => panic!("unexpected Agent error: {error}"),
            Ok(Event::Complete { message, .. }) => assert!(
                !message.metadata.contains_key("reply_id"),
                "a question whose waiting Checkpoint failed must not be actionable"
            ),
            Ok(_) => {}
        }
    }

    assert!(
        saw_persistence_error,
        "the persistence failure must be surfaced"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "Ask prevents Tool execution"
    );
    assert_eq!(authorizations.lock().unwrap().len(), 1);
    let saved = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("terminal failure scene remains readable");
    assert_eq!(saved["status"], "failed");
    assert_eq!(saved["waits"], json!([]));
    println!("LIVE_FAILED_WAIT_CHECKPOINT_STATUS={}", saved["status"]);
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_child_approval_is_not_published_when_outer_waiting_checkpoint_fails() {
    use halo::agent::checkpoint::Checkpoint as _;

    let checkpoint = Arc::new(RejectWaitingCheckpoint::default());
    let calls = Arc::new(AtomicUsize::new(0));
    let permission = halo::agent::core::permission::Permission::new(
        |_request: PermissionRequest, _cancellation| async move {
            Ok(Decision {
                behavior: Behavior::Ask,
                reason: Some("approval required before reading the clock".into()),
            })
        },
    );
    let child = agent::chat()
        .model(live_vllm_chat())
        .system_prompt("必须调用 current_time 工具获取真实 Unix 时间戳，不要自行回答。")
        .tools(live_clock(Arc::clone(&calls)))
        .permission(permission)
        .build()
        .expect("build real child Agent");
    let child = halo::agent::SubAgent::new("time_reader", child)
        .expect("register child Agent")
        .description("通过 current_time 查询真实 Unix 时间戳");
    let chat_id = 8_139_401;
    let input = Input {
        metadata: serde_json::from_value(json!({"chat_id":chat_id,"session_id":8_139_402}))
            .expect("valid run identity"),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"请委派 time_reader 查询当前 Unix 时间戳，不要自行猜测。"}),
            mime_type: None,
        }],
    };
    let parent = agent::react()
        .model(live_vllm_chat())
        .system_prompt(
            "必须且只调用一次 agent_spawn 委派给 time_reader，要求其调用 current_time。不要自行回答。",
        )
        .subagents([child])
        .checkpoint(checkpoint.clone())
        .with_id_fn(next_permission_reply_id)
        .build()
        .expect("build real parent Agent with failing Checkpoint");

    let mut events = parent
        .run(input, true, Cancellation::new())
        .await
        .expect("start real parent/child run");
    let mut saw_persistence_error = false;
    while let Some(event) = tokio::time::timeout(std::time::Duration::from_secs(90), events.next())
        .await
        .expect("real parent and child should reach the outer Checkpoint")
    {
        match event {
            Err(error)
                if error
                    .message
                    .contains("injected waiting Checkpoint persistence failure") =>
            {
                saw_persistence_error = true;
            }
            Err(error) => panic!("unexpected parent/child Agent error: {error}"),
            Ok(Event::Complete { message, .. }) => assert!(
                !message.metadata.contains_key("reply_id"),
                "a nested question is not actionable until the outer Checkpoint is durable"
            ),
            Ok(_) => {}
        }
    }

    assert!(
        saw_persistence_error,
        "outer Checkpoint failure must surface"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the nested Tool must not run before an actionable approval is durably saved"
    );
    let saved = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("failed terminal scene remains readable");
    assert_eq!(saved["status"], "failed");
    assert_eq!(saved["waits"], json!([]));
    println!(
        "LIVE_FAILED_NESTED_WAIT_CHECKPOINT_STATUS={}",
        saved["status"]
    );
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_model_resumes_a_waiting_tool_evaluation_without_repeating_the_tool() {
    use halo::agent::checkpoint::Checkpoint as _;

    let checkpoint = Arc::new(halo::agent::core::checkpoint::InMemory::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let evals = Arc::new(AtomicUsize::new(0));
    let chat_id = 8_139_301;
    let session_id = 8_139_302;
    let input = Input {
        metadata: serde_json::from_value(json!({"chat_id":chat_id,"session_id":session_id}))
            .expect("valid run identity"),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"查询北京现在的天气和气温，请使用 weather 工具。"}),
            mime_type: None,
        }],
    };
    let build = || {
        agent::react()
            .model(live_vllm_chat())
            .checkpoint(checkpoint.clone())
            .system_prompt(
                "查询天气时必须调用 weather 工具。若恢复后已有 weather 工具结果，不得再次调用工具，直接依据该结果用一句中文回答。",
            )
            .tools(LiveEvalWeather {
                calls: Arc::clone(&calls),
                evals: Arc::clone(&evals),
            })
            .build()
            .expect("build ReAct Agent with real model and Tool evaluation")
    };

    let first = build();
    let events = first
        .run(input.clone(), true, Cancellation::new())
        .await
        .expect("start real Tool evaluation run")
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events.last(), Some(Err(error)) if error.code == "WAITING"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(evals.load(Ordering::SeqCst), 1);
    let waiting = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("Tool evaluation waiting Checkpoint");
    assert_eq!(waiting["status"], "waiting");
    let tool = waiting["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["kind"] == "tool")
        .expect("real Tool Step");
    assert_eq!(tool["stage"], "eval");
    assert_eq!(tool["status"], "waiting");
    assert_eq!(tool["message"]["values"]["temperature_celsius"], 22);
    assert!(tool.get("eval").is_none());
    drop(first);

    let restored = build();
    let resumed = restored
        .run(input, true, Cancellation::new())
        .await
        .expect("restore waiting Tool evaluation")
        .collect::<Vec<_>>()
        .await;
    assert!(resumed.iter().all(Result::is_ok), "{resumed:?}");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "Tool side effect is not replayed"
    );
    assert_eq!(evals.load(Ordering::SeqCst), 2, "eval resumes exactly once");
    let final_answer = resumed
        .iter()
        .filter_map(|event| match event.as_ref().unwrap() {
            Event::Complete { message, .. }
                if message.role == halo::agent::message::Role::Assistant =>
            {
                Some(message)
            }
            _ => None,
        })
        .next_back()
        .expect("real model final answer after evaluation resumes");
    assert!(
        final_answer
            .content
            .iter()
            .any(|part| part.r#type == "text"),
        "real model should turn the evaluated tool result into a final answer"
    );
    let completed = checkpoint
        .load(chat_id, &Cancellation::new())
        .await
        .expect("completed Tool evaluation Checkpoint");
    let tool = completed["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|step| step["kind"] == "tool")
        .expect("committed Tool Step");
    assert_eq!(tool["stage"], "committed");
    assert_eq!(tool["eval"]["passed"], true);
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_model_calls_a_tool_and_answers_from_its_result() {
    agent_live_tool::run()
        .await
        .expect("live model tool-call flow should complete");
}

#[tokio::test]
#[ignore = "requires real HEART_VLLM_BASE_URL, HEART_VLLM_API_KEY and HEART_VLLM_MODEL"]
async fn real_parent_agent_delegates_to_a_subagent_that_calls_a_tool() {
    agent_live_subagent::run()
        .await
        .expect("live parent/subagent flow should complete");
}

#[tokio::test]
#[ignore = "requires real VLLM credentials and the local MCP acceptance server binary"]
async fn real_agent_calls_a_live_mcp_stdio_service() {
    let cancellation = Cancellation::new();
    let mcp = halo::agent::core::mcp::Server::stdio(
        Definition {
            name: "local-clock".into(),
            ..Definition::default()
        },
        Command::new(env!("CARGO_BIN_EXE_heart_mcp_acceptance_server")),
        &cancellation,
    )
    .await
    .expect("start the local MCP service and discover its tool");

    let provider = Vllm::new(Config {
        base_url: std::env::var("HEART_VLLM_BASE_URL").expect("set HEART_VLLM_BASE_URL"),
        api_key: std::env::var("HEART_VLLM_API_KEY").expect("set HEART_VLLM_API_KEY"),
        ..Config::default()
    })
    .expect("construct live VLLM provider");
    let model_name = std::env::var("HEART_VLLM_MODEL").expect("set HEART_VLLM_MODEL");
    let model = provider
        .chat(
            chat::Config::new(model_name).options(serde_json::Map::from_iter([(
                "chat_template_kwargs".into(),
                json!({"enable_thinking":false}),
            )])),
        )
        .expect("construct live Chat model");
    let agent = agent::chat()
        .model(model)
        .system_prompt(
            "你必须调用 current_time 工具。最终回答只能输出工具返回的 Unix 时间戳整数，不要猜测或添加其他文字。",
        )
        .mcps([mcp])
        .build()
        .expect("build Agent with the connected MCP service");
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"请查询当前 Unix 时间戳，并只返回这个整数。"}),
            mime_type: None,
        }],
        ..Default::default()
    };

    let mut events = agent
        .run(input, true, cancellation)
        .await
        .expect("run Agent against the live provider and MCP service");
    let mut final_text = None;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event.expect("Agent event")
            && message.role == halo::agent::message::Role::Assistant
        {
            let text = message
                .content
                .iter()
                .filter_map(|part| part.data.get("value").and_then(|value| value.as_str()))
                .collect::<String>();
            if !text.is_empty() {
                final_text = Some(text);
            }
        }
    }
    let final_text = final_text.expect("live Agent should return assistant text");
    let timestamp = final_text
        .trim()
        .parse::<u64>()
        .unwrap_or_else(|_| panic!("Agent did not return only the MCP timestamp: {final_text:?}"));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_secs();
    assert!(
        timestamp.abs_diff(now) <= 10,
        "not a current MCP timestamp: {timestamp} vs {now}"
    );
    println!("LIVE_AGENT_MCP_TIMESTAMP={timestamp}");
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and the local MCP acceptance server binary"]
async fn real_deepseek_agent_calls_a_live_mcp_stdio_service() {
    let cancellation = Cancellation::new();
    let mcp = halo::agent::core::mcp::Server::stdio(
        Definition {
            name: "local-clock".into(),
            ..Definition::default()
        },
        Command::new(env!("CARGO_BIN_EXE_heart_mcp_acceptance_server")),
        &cancellation,
    )
    .await
    .expect("start the local MCP service and discover its tool");
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct real DeepSeek provider");
    let model = provider
        .chat(
            chat::Config::new(
                std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
            )
            .options(serde_json::Map::from_iter([(
                "thinking".into(),
                json!({"type":"enabled"}),
            )])),
        )
        .expect("construct real DeepSeek Chat");
    let agent = agent::chat()
        .model(model)
        .system_prompt(
            "你必须调用 current_time 工具。最终回答只能输出工具返回的 Unix 时间戳整数，不要猜测或添加其他文字。",
        )
        .mcps([mcp])
        .build()
        .expect("build Agent with the connected MCP service");
    let input = Input {
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"请查询当前 Unix 时间戳，并只返回这个整数。"}),
            mime_type: None,
        }],
        ..Default::default()
    };
    let mut events = agent
        .run(input, true, cancellation)
        .await
        .expect("run Agent against DeepSeek and the MCP service");
    let mut final_text = None;
    while let Some(event) = events.next().await {
        if let Event::Complete { message, .. } = event.expect("Agent event")
            && message.role == halo::agent::message::Role::Assistant
        {
            let text = message
                .content
                .iter()
                .filter(|part| part.r#type == "text")
                .filter_map(|part| part.data.get("value").and_then(|value| value.as_str()))
                .collect::<String>();
            if !text.is_empty() {
                final_text = Some(text);
            }
        }
    }
    let final_text = final_text.expect("DeepSeek Agent should return assistant text");
    let timestamp = final_text
        .split(|character: char| !character.is_ascii_digit())
        .find(|candidate| candidate.len() == 10)
        .and_then(|candidate| candidate.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("DeepSeek did not return the MCP timestamp: {final_text:?}"));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_secs();
    assert!(
        timestamp.abs_diff(now) <= 10,
        "not a current MCP timestamp: {timestamp} vs {now}"
    );
    println!("LIVE_DEEPSEEK_AGENT_MCP_TIMESTAMP={timestamp}");
}
