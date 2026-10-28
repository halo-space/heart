//! Live DeepSeek coverage for the Agent input projection and actual Tool result
//! projection. These assertions run against model-generated responses rather
//! than a scripted Chat implementation.
use std::{
    future::Future,
    sync::{Arc, Mutex},
};

use futures_util::StreamExt;
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::checkpoint::Checkpoint,
    agent::core,
    agent::event::Event,
    agent::message::Role,
    agent::model::{
        Error as ModelError,
        chat::{Chat, Request},
    },
    agent::provider::deepseek::DeepSeek,
    agent::tool::{Call, Tool, ToolDefinition},
    agent::{self, Agent, Input},
};
use serde_json::{Value, json};

struct ObservedChat<C> {
    chat: C,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl<C: Chat> Chat for ObservedChat<C> {
    type Stream = C::Stream;

    fn stream(&self, request: Request) -> impl Future<Output = Result<Self::Stream, ModelError>> {
        self.requests.lock().unwrap().push(request.clone());
        self.chat.stream(request)
    }
}

struct Lookup;

impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: std::sync::LazyLock<ToolDefinition> =
            std::sync::LazyLock::new(|| ToolDefinition {
                name: "current_time".into(),
                description: "Return the current Unix timestamp in seconds".into(),
                parameters: json!({"type":"object", "properties":{}, "additionalProperties":false})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        &DEFINITION
    }

    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        let unix_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| Error::new("INTERNAL", error.to_string()))?
            .as_secs();
        let mut message = Message::function(json!({"unix_seconds": unix_seconds}));
        message.metadata = json!({
            "chat_id":999, "node_id":"untrusted", "exec_id":999,
            "reply_id":5002, "label":"business metadata"
        })
        .as_object()
        .unwrap()
        .clone();
        Ok(message)
    }
}

fn deepseek() -> impl Chat {
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct live DeepSeek provider");
    provider
        .chat(halo::agent::model::chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct live DeepSeek Chat")
}

async fn run(react: bool, with_tool: bool) -> (Vec<Request>, Vec<Event>, Value) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let input: Input = serde_json::from_value(json!({
        "metadata": {
            "chat_id":1001, "session_id":2001, "reply_id":5001,
            "language":"zh-CN", "business":{"reply_id":77}
        },
        "payload":[{"type":"text", "data":{"value": if with_tool {
            "请调用 current_time 工具获取当前 Unix 时间戳，然后简短回答。"
        } else {
            "请用一句话回答：北京是中国的首都吗？"
        }}}]
    }))
    .unwrap();
    let prompt = if with_tool {
        "你必须调用 current_time 工具获取当前 Unix 时间戳，然后根据工具结果回答；不要自行编造工具返回值。"
    } else {
        "请直接回答用户问题，不要调用任何工具。"
    };
    let chat = ObservedChat {
        chat: deepseek(),
        requests: requests.clone(),
    };
    let events = if react {
        let mut builder = agent::react()
            .model(chat)
            .system_prompt(prompt)
            .checkpoint(checkpoint.clone());
        if with_tool {
            builder = builder.tools([Lookup]);
        }
        builder
            .build()
            .unwrap()
            .run(input, true, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    } else {
        let mut builder = agent::chat()
            .model(chat)
            .system_prompt(prompt)
            .checkpoint(checkpoint.clone());
        if with_tool {
            builder = builder.tools([Lookup]);
        }
        builder
            .build()
            .unwrap()
            .run(input, true, Cancellation::new())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    };
    let events = events.into_iter().collect::<Result<Vec<_>, _>>().unwrap();
    let state = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    let requests = requests.lock().unwrap().clone();
    (requests, events, state)
}

fn user_message(request: &Request) -> &Message {
    request
        .messages
        .0
        .iter()
        .find(|message| message.role == Role::User)
        .expect("the actual model request contains the user input")
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends live model requests"]
async fn live_chat_and_react_exclude_reply_identity_from_model_input() {
    for react in [false, true] {
        let (requests, events, state) = run(react, false).await;
        assert!(!requests.is_empty(), "the real model must be called");
        let user = user_message(&requests[0]);
        for key in ["chat_id", "node_id", "exec_id", "reply_id"] {
            assert!(!user.metadata.contains_key(key), "model received {key}");
        }
        assert_eq!(user.metadata["language"], "zh-CN");
        assert_eq!(user.metadata["business"]["reply_id"], 77);
        assert!(user.content[0].data["value"].as_str().is_some());
        assert_eq!(state["query"]["metadata"]["reply_id"], 5001);

        let complete = events
            .iter()
            .filter_map(|event| match event {
                Event::Complete { message, .. } => Some(message),
                Event::Delta { .. } => None,
            })
            .next_back()
            .expect("the live Agent must emit a final Complete");
        assert_eq!(complete.metadata["chat_id"], 1001);
        assert!(!complete.metadata.contains_key("reply_id"));
    }
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and sends live model/tool requests"]
async fn live_chat_and_react_exclude_reply_identity_from_actual_tool_results() {
    for react in [false, true] {
        let (requests, events, state) = run(react, true).await;
        assert!(
            requests.len() >= 2,
            "the live model must call the Tool and continue"
        );
        let tool_result = requests
            .iter()
            .flat_map(|request| request.messages.0.iter())
            .find(|message| {
                message.role == Role::Function && message.metadata.contains_key("label")
            })
            .expect("the actual Tool result must reach a subsequent model request");
        assert!(!tool_result.metadata.contains_key("reply_id"));
        assert_eq!(tool_result.metadata["label"], "business metadata");
        assert!(tool_result.values["unix_seconds"].as_u64().unwrap() > 0);

        let tool_step = state["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|step| step["kind"] == "tool")
            .expect("the executed Tool must be recorded in Agent state");
        assert!(
            !tool_step["message"]["metadata"]
                .as_object()
                .unwrap()
                .contains_key("reply_id")
        );
        let completes = events
            .iter()
            .filter(|event| matches!(event, Event::Complete { .. }))
            .count();
        assert!(
            completes >= 2,
            "expect model-call Complete plus final Complete"
        );
    }
}
