use std::sync::{Arc, Mutex};

use futures_executor::block_on;
use futures_util::{StreamExt, stream};
use halo_agents::agent::{self, Agent, Input, Payload};
use halo_agents::event::Event;
use halo_agents::message::Role;
use halo_agents::model::chat::{Chat, Request};
use halo_agents::runtime::agent::{graph::Graph, node::Status};
use halo_agents::{Cancellation, Error, Message};
use serde_json::{Map, json};

struct Model {
    requests: Arc<Mutex<Vec<Request>>>,
}

impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.requests.lock().unwrap().push(request);
        let mut message = Message::new(Role::Assistant);
        message.values.insert("answer".into(), json!("done"));
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

async fn check_runs<A: Agent>(agent: &A, requests: &Mutex<Vec<Request>>) {
    for chat_id in [101_i64, 102] {
        let input = Input {
            metadata: Map::from_iter([("chat_id".into(), json!(chat_id))]),
            payload: vec![Payload {
                r#type: "text".into(),
                data: json!({"value":"question"}),
                mime_type: None,
            }],
        };
        let events = agent.run(input, true, Cancellation::new()).await.unwrap();
        futures_util::pin_mut!(events);
        let Some(Ok(Event::Complete { message, .. })) = events.next().await else {
            panic!("Agent must forward the shared Complete event");
        };
        assert_eq!(message.values["answer"], "done");
        assert_eq!(message.metadata["chat_id"], chat_id);
        assert!(events.next().await.is_none());
    }
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    for request in requests.iter() {
        assert_eq!(request.messages.0.len(), 1);
        let user = &request.messages.0[0];
        assert_eq!(user.role, Role::User);
        assert!(!user.metadata.contains_key("chat_id"));
        assert_eq!(user.content[0].data["value"], "question");
    }
}

#[test]
fn chat_is_constructed_and_run_from_agent_without_a_runtime_owner() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = agent::chat()
        .model(Model {
            requests: Arc::clone(&requests),
        })
        .build()
        .unwrap();
    block_on(check_runs(&agent, &requests));
}

#[test]
fn react_and_toolkit_are_agent_assembly_not_runtime_kinds() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent: halo_agents::core::agent::react::Agent<Model> = agent::react()
        .model(Model {
            requests: Arc::clone(&requests),
        })
        .toolkit(agent::toolkit::Toolkit::new())
        .build()
        .unwrap();
    block_on(check_runs(&agent, &requests));
}

#[test]
fn runtime_graph_keeps_execution_state_and_its_existing_json_shape() {
    use halo_agents::plan::{Plan, PlanNode};
    let mut plan = Plan::new(1).unwrap();
    plan.add(
        PlanNode {
            node_id: "a".into(),
            name: "search".into(),
            objective: "find information".into(),
        },
        vec![],
        vec![],
    )
    .unwrap();
    let graph = Graph::new(plan).unwrap();
    assert_eq!(graph.state("a").unwrap().status, Status::Ready);
    let json = serde_json::to_value(&graph).unwrap();
    assert_eq!(json["plan"]["nodes"]["a"]["name"], "search");
    assert_eq!(json["nodes"]["a"]["status"], "ready");
    assert_eq!(json["nodes"]["a"]["execution"], serde_json::Value::Null);
    assert_eq!(serde_json::from_value::<Graph>(json).unwrap(), graph);
}
