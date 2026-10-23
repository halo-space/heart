//! Question identity stays outside model context. These tests exercise the
//! shared projection boundary, not the still-pending live reply protocol.
use std::sync::{Arc, LazyLock, Mutex};

use futures_util::{StreamExt, stream};
use halo::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, toolkit::Toolkit},
    checkpoint::Checkpoint,
    core,
    event::Event,
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
    },
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::json;

#[derive(Clone)]
struct Model {
    requests: Arc<Mutex<Vec<Request>>>,
    with_tool: bool,
}

impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let first = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request);
            requests.len() == 1
        };
        let mut message = Message::new(Role::Assistant);
        if self.with_tool && first {
            message.content.push(Part {
                r#type: "tool_call".into(),
                data: json!({"call_id":"call_1", "name":"lookup", "arguments":{}})
                    .as_object()
                    .unwrap()
                    .clone(),
            });
        }
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

struct Lookup;

impl Tool for Lookup {
    fn definition(&self) -> &ToolDefinition {
        static DEFINITION: LazyLock<ToolDefinition> = LazyLock::new(|| ToolDefinition {
            name: "lookup".into(),
            description: "Return a business result with observational metadata".into(),
            parameters: json!({"type":"object", "properties":{}})
                .as_object()
                .unwrap()
                .clone(),
        });
        &DEFINITION
    }

    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        let mut message = Message::function(json!({"reply_id": 99, "found": true}));
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

async fn run(react: bool, with_tool: bool) -> (Vec<Request>, Vec<Event>, serde_json::Value) {
    let model = Model {
        requests: Default::default(),
        with_tool,
    };
    let requests = model.requests.clone();
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    let input: Input = serde_json::from_value(json!({
        "metadata": {
            "chat_id":1001, "session_id":2001, "reply_id":5001,
            "language":"zh-CN", "business":{"reply_id":77}
        },
        "payload":[{"type":"json", "data":{"value":{"reply_id":88,"text":"question"}}}]
    }))
    .unwrap();
    let capabilities = Toolkit::new().tools(vec![Lookup]);
    let events = if react {
        let mut builder = agent::react().model(model).checkpoint(checkpoint.clone());
        if with_tool {
            builder = builder.toolkit(capabilities);
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
        let mut builder = agent::chat().model(model).checkpoint(checkpoint.clone());
        if with_tool {
            builder = builder.toolkit(capabilities);
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

#[tokio::test]
async fn both_agents_exclude_reply_identity_from_input_projection() {
    for react in [false, true] {
        let (requests, events, state) = run(react, false).await;
        assert_eq!(requests.len(), 1);
        let user = requests[0].messages.0.last().unwrap();
        assert_eq!(user.role, Role::User);
        for key in ["chat_id", "node_id", "exec_id", "reply_id"] {
            assert!(!user.metadata.contains_key(key), "model received {key}");
        }
        assert_eq!(user.metadata["language"], "zh-CN");
        assert_eq!(user.metadata["business"]["reply_id"], 77);
        assert_eq!(user.content[0].data["value"]["reply_id"], 88);
        // Projection must not erase the authoritative original input.
        assert_eq!(state["query"]["metadata"]["reply_id"], 5001);
        assert_eq!(events.len(), 1);
        let Event::Complete { message, .. } = &events[0] else {
            panic!("missing complete");
        };
        assert_eq!(message.metadata["chat_id"], 1001);
        assert!(!message.metadata.contains_key("reply_id"));
    }
}

#[tokio::test]
async fn both_agents_exclude_reply_identity_from_reused_tool_results() {
    for react in [false, true] {
        let (requests, events, state) = run(react, true).await;
        assert_eq!(requests.len(), 2);
        let tool = requests[1]
            .messages
            .0
            .iter()
            .find(|message| message.metadata.get("call_id") == Some(&json!("call_1")))
            .expect("tool result reaches the next model call");
        assert!(!tool.metadata.contains_key("reply_id"));
        assert_eq!(tool.metadata["label"], "business metadata");
        assert_eq!(tool.values["reply_id"], 99);
        assert!(
            state["steps"]
                .as_array()
                .unwrap()
                .iter()
                .find(|step| step["kind"] == "tool")
                .expect("actual Tool Step")["message"]["metadata"]
                .get("reply_id")
                .is_none()
        );
        assert_eq!(events.len(), 3, "retain every actual call Complete");
        for event in events {
            let Event::Complete { message, .. } = event else {
                panic!("unexpected event");
            };
            assert!(!message.metadata.contains_key("reply_id"));
        }
    }
}
