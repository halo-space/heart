//! Known external-reply defect; not a completed recovery capability.
use std::sync::{Arc, Mutex};

use futures_util::{StreamExt, stream};
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
#[ignore = "ISSUE-021: external reply currently reaches both waiting plan nodes; waiting consumer contract needs confirmation"]
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
    let reply: Input = serde_json::from_value(json!({
        "metadata":original.metadata,
        "payload":[{"type":"json","data":{"value":{"approved":true}}}]
    }))
    .unwrap();
    let events = agent
        .run(reply, Cancellation::new())
        .await
        .expect("valid waiting fixture")
        .collect::<Vec<_>>()
        .await;
    assert!(
        events.iter().all(Result::is_ok),
        "fixture failed before reply consumption: {events:?}"
    );
    let recipients = requests
        .lock()
        .unwrap()
        .iter()
        .filter(|request| {
            request
                .messages
                .0
                .iter()
                .any(|message| message.values.contains_key("plan_node"))
                && request.messages.0.iter().any(|message| {
                    message
                        .content
                        .iter()
                        .any(|part| part.data.get("value") == Some(&json!({"approved":true})))
                })
        })
        .count();
    assert!(
        recipients <= 1,
        "one external reply reached {recipients} distinct waiting nodes"
    );
    if recipients == 0 {
        assert_eq!(
            checkpoint.load(501, &Cancellation::new()).await.unwrap(),
            snapshot,
            "rejecting an ambiguous reply must retain its saved scene"
        );
    }
}
