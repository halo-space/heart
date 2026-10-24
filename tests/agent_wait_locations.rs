//! Native waiting-position validation, not a completed live reply consumer.
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::checkpoint::Checkpoint,
    agent::event::Event,
    agent::message::Role,
    agent::model::chat::{Chat, Request},
    agent::{self, Agent, Input},
    core,
    runtime::agent::{
        State,
        wait::{Node, Wait},
    },
};
use serde_json::{Value, json};

struct Model(Arc<AtomicUsize>);
impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;
    async fn stream(&self, _: Request) -> Result<Self::Stream, Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: Message::new(Role::Assistant),
            usage: None,
            finish_reason: Some("stop".into()),
        })]))
    }
}

fn input() -> Input {
    serde_json::from_value(json!({"metadata":{"chat_id":1001},"payload":[{"type":"text","data":{"value":"question"}}]})).unwrap()
}

fn step(id: &str, reply_id: i64) -> Value {
    serde_json::to_value(halo::runtime::agent::node::Step {
        id: id.into(),
        kind: "human".into(),
        status: "waiting".into(),
        waits: [(
            reply_id,
            halo::runtime::agent::node::Wait {
                count: 1,
                kind: "approval".into(),
                status: "waiting".into(),
                question: Message::new(Role::Assistant),
                feedback: Default::default(),
            },
        )]
        .into(),
        ..Default::default()
    })
    .unwrap()
}

fn attempt(steps: Vec<Value>, nodes: Value, agent: Value) -> Value {
    json!({"idx":0,"status":"waiting","created_time":1,"finished_time":null,
        "usage":null,"finish_reason":null,"steps":steps,"messages":[],"nodes":nodes,
        "agent":agent,"error":null})
}

fn execution(exec_id: i64, attempt: Value) -> Value {
    json!({"exec_id":exec_id,"attempts":[attempt],"messages":[],"usage":null,"error":null})
}

fn node(exec_id: i64, steps: Vec<Value>, nodes: Value, agent: Value) -> Value {
    json!({"status":"waiting","messages":null,"usage":null,
        "execution":execution(exec_id,attempt(steps,nodes,agent))})
}

fn coordinate(id: &str, exec_id: i64) -> Value {
    json!({"node_id":id,"exec_id":exec_id,"idx":0})
}

fn scene(steps: Vec<Value>) -> Value {
    serde_json::to_value(State {
        chat_id: Some(1001),
        r#type: "agent".into(),
        status: "waiting".into(),
        steps: steps
            .into_iter()
            .map(|v| serde_json::from_value(v).unwrap())
            .collect(),
        ..Default::default()
    })
    .unwrap()
}

fn positioned(mut base: Value, nodes: Vec<Value>) -> Value {
    base["status"] = json!("waiting");
    base["final_message"] = Value::Null;
    base["final_usage"] = Value::Null;
    base["finish_reason"] = Value::Null;
    base["error"] = Value::Null;
    base["waits"] = json!([{"nodes":nodes,"step":"review"}]);
    base
}

async fn seed<A: Agent>(agent: &A, checkpoint: &core::checkpoint::InMemory) -> Value {
    let events = agent
        .run(input(), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    let base = checkpoint.load(1001, &Cancellation::new()).await.unwrap();
    assert_eq!(base["waits"], json!([]));
    base
}

async fn reject<A: Agent>(
    agent: &A,
    checkpoint: &core::checkpoint::InMemory,
    calls: &AtomicUsize,
    state: Value,
    code: &str,
    message: &str,
) {
    // Every fixture remains representable by the canonical State, while the
    // same saved JSON retains each Agent's own additional recovery fields.
    let _: State = serde_json::from_value(state.clone()).unwrap();
    checkpoint
        .save(1001, state.clone(), &Cancellation::new())
        .await
        .unwrap();
    let before = calls.load(Ordering::SeqCst);
    let error = match agent.run(input(), false, Cancellation::new()).await {
        Ok(_) => panic!("waiting position must not fall through to inference"),
        Err(error) => error,
    };
    assert_eq!(error.code, code);
    assert_eq!(error.message, message);
    assert_eq!(calls.load(Ordering::SeqCst), before);
    assert_eq!(
        checkpoint.load(1001, &Cancellation::new()).await.unwrap(),
        state
    );
}

async fn cases<A: Agent>(
    agent: A,
    checkpoint: Arc<core::checkpoint::InMemory>,
    calls: Arc<AtomicUsize>,
) {
    let base = seed(&agent, &checkpoint).await;
    let mut root = positioned(base.clone(), vec![]);
    root["steps"] = json!([step("review", 5001)]);
    reject(
        &agent,
        &checkpoint,
        &calls,
        root.clone(),
        "WAITING",
        "Agent is waiting for targeted external feedback",
    )
    .await;

    for count in [0, 2] {
        let mut invalid_count = root.clone();
        invalid_count["steps"][0]["waits"]["5001"]["count"] = json!(count);
        reject(
            &agent,
            &checkpoint,
            &calls,
            invalid_count,
            "INVALID_STATE",
            "invalid Agent Step waiting record",
        )
        .await;
    }

    let mut planned = positioned(base.clone(), vec![coordinate("research", 7001)]);
    planned["graph"] = json!({"plan":{"version":1,"nodes":{"research":{"name":"research","objective":"research"}},"edges":[]}});
    planned["nodes"] =
        json!({"research":node(7001,vec![step("review",5001)],json!({}),Value::Null)});
    reject(
        &agent,
        &checkpoint,
        &calls,
        planned.clone(),
        "WAITING",
        "Agent is waiting for targeted external feedback",
    )
    .await;

    let mut unrelated = planned.clone();
    let mut finished = attempt(vec![], json!({}), Value::Null);
    finished["status"] = json!("succeeded");
    unrelated["execution"] = execution(6000, finished);
    reject(
        &agent,
        &checkpoint,
        &calls,
        unrelated,
        "WAITING",
        "Agent is waiting for targeted external feedback",
    )
    .await;

    let child = node(9102, vec![], json!({}), scene(vec![step("review", 5002)]));
    let mut direct = positioned(base.clone(), vec![coordinate("9101", 9102)]);
    direct["execution"] = execution(6000, attempt(vec![], json!({"9101":child}), Value::Null));
    reject(
        &agent,
        &checkpoint,
        &calls,
        direct.clone(),
        "WAITING",
        "Agent is waiting for targeted external feedback",
    )
    .await;

    let mut nested = positioned(
        planned.clone(),
        vec![coordinate("research", 7001), coordinate("9101", 9102)],
    );
    nested["nodes"]["research"]["execution"]["attempts"][0]["steps"] = json!([]);
    nested["nodes"]["research"]["execution"]["attempts"][0]["nodes"] = json!({"9101":child});
    reject(
        &agent,
        &checkpoint,
        &calls,
        nested.clone(),
        "WAITING",
        "Agent is waiting for targeted external feedback",
    )
    .await;

    // A child's own Plan remains inside its Attempt.agent scene; repeated
    // Step IDs in other owners must not change this exact containment route.
    let mut deep = nested.clone();
    let inner = &mut deep["nodes"]["research"]["execution"]["attempts"][0]["nodes"]["9101"]["execution"]
        ["attempts"][0]["agent"];
    inner["steps"] = json!([]);
    inner["nodes"] = json!({"inner":node(9200,vec![step("review",5003)],json!({}),Value::Null)});
    inner["graph"] = json!({"plan":{"version":1,"nodes":{"inner":{"name":"inner","objective":"inner"}},"edges":[]}});
    deep["waits"][0]["nodes"]
        .as_array_mut()
        .unwrap()
        .push(coordinate("inner", 9200));
    reject(
        &agent,
        &checkpoint,
        &calls,
        deep.clone(),
        "WAITING",
        "Agent is waiting for targeted external feedback",
    )
    .await;

    let mut bad = deep;
    bad["nodes"]["research"]["execution"]["attempts"][0]["nodes"]["9101"]["execution"]["attempts"]
        [0]["agent"]["status"] = json!("succeeded");
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "waiting locator Agent scene is not active",
    )
    .await;

    let mut bad = planned.clone();
    bad["waits"][0]["nodes"][0]["exec_id"] = json!(6999);
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "waiting locator execution identity is stale",
    )
    .await;
    let mut bad = planned.clone();
    let mut current = bad["nodes"]["research"]["execution"]["attempts"][0].clone();
    current["idx"] = json!(1);
    bad["nodes"]["research"]["execution"]["attempts"]
        .as_array_mut()
        .unwrap()
        .push(current);
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "waiting locator attempt is not current",
    )
    .await;
    let mut bad = root.clone();
    bad["waits"] = json!([]);
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "Agent waiting question has no active locator",
    )
    .await;
    let mut bad = nested.clone();
    bad["waits"] = json!([]);
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "Agent waiting question has no active locator",
    )
    .await;
    let mut bad = root.clone();
    bad["steps"]
        .as_array_mut()
        .unwrap()
        .push(step("other", 5002));
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "Agent waiting question has no active locator",
    )
    .await;

    let mut bad = root.clone();
    bad["steps"]
        .as_array_mut()
        .unwrap()
        .push(step("review", 5002));
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "waiting locator Step identity is ambiguous",
    )
    .await;
    let mut bad = root.clone();
    bad["steps"][0]["waits"]["5001"]["status"] = json!("resolved");
    bad["steps"][0]["waits"]["5001"]["feedback"] = json!([Message::new(Role::User)]);
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "waiting locator has no unresolved question",
    )
    .await;
    let mut bad = root.clone();
    bad["waits"]
        .as_array_mut()
        .unwrap()
        .push(root["waits"][0].clone());
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "duplicate Agent waiting locator",
    )
    .await;
    let mut bad = root.clone();
    bad["steps"]
        .as_array_mut()
        .unwrap()
        .push(step("other", 5001));
    bad["waits"]
        .as_array_mut()
        .unwrap()
        .push(json!({"nodes":[],"step":"other"}));
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "Agent waiting reply identity is ambiguous",
    )
    .await;
    let mut bad = planned.clone();
    bad["execution"] = execution(
        6000,
        attempt(
            vec![],
            json!({"research":node(7001,vec![step("review",5001)],json!({}),Value::Null)}),
            Value::Null,
        ),
    );
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "waiting locator node ownership is ambiguous",
    )
    .await;
    let mut bad = planned;
    bad["nodes"]["research"]["status"] = json!("succeeded");
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "waiting locator references an inactive execution",
    )
    .await;
    let mut bad = nested;
    bad["nodes"]["research"]["execution"]["attempts"][0]["nodes"]["9101"]["execution"]["attempts"]
        [0]["agent"]["status"] = json!("succeeded");
    reject(
        &agent,
        &checkpoint,
        &calls,
        bad,
        "INVALID_STATE",
        "waiting locator Agent scene is not active",
    )
    .await;
}

#[tokio::test]
async fn restored_locations_follow_current_containment_without_dispatch_or_rewriting() {
    let calls = Arc::new(AtomicUsize::new(0));
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    cases(
        agent::chat()
            .model(Model(calls.clone()))
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap(),
        checkpoint,
        calls,
    )
    .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    cases(
        agent::react()
            .model(Model(calls.clone()))
            .checkpoint(checkpoint.clone())
            .build()
            .unwrap(),
        checkpoint,
        calls,
    )
    .await;
}

#[test]
fn native_wait_array_preserves_fifo_and_exact_execution_coordinates() {
    let waits = vec![
        Wait {
            nodes: vec![Node {
                node_id: "z".into(),
                exec_id: 9_007_199_254_740_993,
                idx: 2,
            }],
            step: "tool_9".into(),
        },
        Wait {
            nodes: vec![],
            step: "tool_0".into(),
        },
    ];
    let state = State {
        waits: waits.clone(),
        ..Default::default()
    };
    let restored: State = serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
    assert_eq!(restored.waits, waits);
    assert_eq!(
        serde_json::to_value(State::default()).unwrap()["waits"],
        json!([])
    );
}
