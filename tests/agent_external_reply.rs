//! Ambiguous external feedback must be rejected before model dispatch.
//! The Agent is bound to the real DeepSeek provider; a counting pass-through
//! wrapper verifies that this validation path does not send a model request.
use std::future::Future;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use halo::agent::{Cancellation, Message, Messages};
use halo::{
    agent::checkpoint::Checkpoint as _,
    agent::core,
    agent::message::Role,
    agent::model::Error as ModelError,
    agent::model::chat::{Chat, Request},
    agent::plan::Plan,
    agent::provider::deepseek::DeepSeek,
    agent::{self, Agent, Input},
    runtime::agent::{
        State,
        graph::{Graph, State as PlanState},
    },
};
use serde_json::json;

struct ObservedChat<C> {
    chat: C,
    calls: Arc<AtomicUsize>,
}

impl<C: Chat> Chat for ObservedChat<C> {
    type Stream = C::Stream;

    fn stream(&self, request: Request) -> impl Future<Output = Result<Self::Stream, ModelError>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.chat.stream(request)
    }
}

#[tokio::test]
async fn one_external_reply_must_not_be_broadcast_to_multiple_waiting_nodes() {
    let original: Input = serde_json::from_value(json!({
        "metadata":{"chat_id":501,"session_id":601},
        "payload":[{"type":"text","data":{"value":"original task"}}]
    }))
    .unwrap();
    let plan: Plan = serde_json::from_value(json!({
        "version":1,
        "nodes":{
            "a":{"name":"research","objective":"first approval"},
            "b":{"name":"research","objective":"different approval"}
        },
        "edges":[]
    }))
    .unwrap();
    let mut graph = Graph::new(plan).unwrap();
    graph.start("a", 7001).unwrap();
    graph.wait("a").unwrap();
    graph.start("b", 7002).unwrap();
    graph.wait("b").unwrap();
    let messages = Messages::new([Message::new(Role::User)]);
    let mut snapshot = serde_json::to_value(State {
        chat_id: Some(501),
        r#type: "agent".into(),
        status: "waiting".into(),
        graph: Some(PlanState {
            plan: graph.plan.clone(),
        }),
        nodes: graph.nodes,
        messages: messages.clone(),
        metadata: original.metadata.clone().into_iter().collect(),
        ..Default::default()
    })
    .unwrap();
    snapshot["query"] = serde_json::to_value(&original).unwrap();
    snapshot["task_base"] = serde_json::to_value(messages).unwrap();
    snapshot["iteration"] = json!(1);
    snapshot["plan_execution"] = json!(true);
    snapshot["plan_finalizing"] = json!(false);
    snapshot["tool_results"] = json!({});
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    checkpoint
        .save(501, snapshot.clone(), &Cancellation::new())
        .await
        .unwrap();
    let provider = DeepSeek::builder()
        .base_url(
            std::env::var("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|_| "https://api.deepseek.com".into()),
        )
        .api_key(std::env::var("DEEPSEEK_API_KEY").expect("set DEEPSEEK_API_KEY"))
        .build()
        .expect("construct real DeepSeek provider");
    let model = provider
        .chat(halo::agent::model::chat::Config::new(
            std::env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-flash".into()),
        ))
        .expect("construct real DeepSeek Chat");
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = agent::react()
        .model(ObservedChat {
            chat: model,
            calls: calls.clone(),
        })
        .checkpoint(checkpoint.clone())
        .with_id_fn(|| panic!("restoration must not allocate new identities"))
        .build()
        .unwrap();
    for reply_id in [None, Some(5001)] {
        let mut reply: Input = serde_json::from_value(json!({
            "metadata":original.metadata,
            "payload":[{"type":"json","data":{"value":{"approved":true}}}]
        }))
        .unwrap();
        if let Some(reply_id) = reply_id {
            reply.metadata.insert("reply_id".into(), json!(reply_id));
        }
        let error = match agent.run(reply, true, Cancellation::new()).await {
            Ok(_) => panic!("ambiguous feedback must not start a model stream"),
            Err(error) => error,
        };
        assert_eq!(error.code, "CONFLICT");
        assert_eq!(
            error.message,
            "feedback has multiple possible waiting consumers"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            checkpoint.load(501, &Cancellation::new()).await.unwrap(),
            snapshot,
            "rejecting an ambiguous reply must retain its saved scene"
        );
    }
}
