//! Ambiguous external feedback must be rejected before model dispatch.
//! This guard is not the still-pending targeted live reply implementation.
use std::sync::{Arc, Mutex};

use futures_util::stream;
use halo_agents::{
    Cancellation, Error, Message, Messages,
    agent::{self, Agent, Input},
    checkpoint::Checkpoint as _,
    core,
    event::Event,
    message::Role,
    model::chat::{Chat, Request},
    plan::Plan,
    runtime::agent::{GraphState, State, graph::Graph},
};
use serde_json::json;

struct Model(Arc<Mutex<Vec<Request>>>);
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.0.lock().unwrap().push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: Message::new(Role::Assistant),
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
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
        graph: Some(GraphState {
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
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = agent::react()
        .model(Model(requests.clone()))
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
        assert!(requests.lock().unwrap().is_empty());
        assert_eq!(
            checkpoint.load(501, &Cancellation::new()).await.unwrap(),
            snapshot,
            "rejecting an ambiguous reply must retain its saved scene"
        );
    }
}
