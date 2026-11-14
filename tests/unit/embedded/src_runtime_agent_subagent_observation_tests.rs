use super::*;
use crate::runtime::agent::node::status::Status;
use crate::runtime::agent::node::{Attempt, Execution, State};
use serde_json::json;
use std::collections::BTreeMap;

fn running_node(exec_id: i64) -> State {
    State {
        status: Status::Running,
        execution: Some(Execution {
            exec_id,
            attempts: vec![Attempt {
                idx: 0,
                status: Status::Running,
                created_time: Some(1),
                finished_time: None,
                usage: None,
                finish_reason: None,
                steps: Vec::new(),
                messages: crate::Messages::default(),
                completion: None,
                nodes: BTreeMap::new(),
                agent: None,
                error: None,
            }],
            messages: crate::Messages::default(),
            usage: None,
            error: None,
        }),
        ..Default::default()
    }
}

fn observation(
    event: Option<Event>,
    scene: Option<Value>,
    message: Option<Message>,
) -> Observation {
    Observation {
        event,
        feedback: false,
        source: ("reviewer".into(), 77),
        scene: scene.map(Arc::new),
        usage: None,
        message,
    }
}

#[test]
fn intermediate_child_complete_does_not_complete_its_parent_node() {
    let mut nodes = BTreeMap::from([("reviewer".into(), running_node(77))]);
    let mut event_message = Message::new(Role::Assistant);
    event_message
        .values
        .insert("progress".into(), json!("partial"));
    let mut result = Message::function(json!({"progress":"partial"}));
    result
        .metadata
        .insert("call_id".into(), json!("child_call"));
    let observation = observation(
        Some(Event::Complete {
            message: event_message,
            usage: None,
            finish_reason: Some("tool".into()),
        }),
        None,
        Some(result),
    );

    observe(&mut nodes, &observation).unwrap();

    let child = &nodes["reviewer"];
    assert_eq!(child.status, Status::Running);
    assert!(child.messages.is_none());
    assert_eq!(
        child.execution.as_ref().unwrap().attempts[0].status,
        Status::Running
    );
    assert!(child.execution.as_ref().unwrap().messages.is_empty());
}

#[test]
fn only_a_terminal_child_scene_commits_its_result_message() {
    let mut nodes = BTreeMap::from([("reviewer".into(), running_node(77))]);
    let message = Message::function(json!({"unix_seconds":1234}));
    let observation = observation(
        None,
        Some(json!({
            "type":"agent",
            "status":"succeeded",
            "usage":{"input":4,"output":2,"total":6,"details":{}}
        })),
        Some(message.clone()),
    );

    observe(&mut nodes, &observation).unwrap();

    let child = &nodes["reviewer"];
    assert_eq!(child.status, Status::Succeeded);
    assert_eq!(
        child.messages.as_ref().unwrap().as_slice(),
        std::slice::from_ref(&message)
    );
    let execution = child.execution.as_ref().unwrap();
    assert_eq!(execution.messages.as_slice(), &[message]);
    assert_eq!(
        execution.attempts[0].agent.as_ref().unwrap()["status"],
        "succeeded"
    );
    assert_eq!(execution.usage.as_ref().unwrap().total, Some(6));
    assert!(execution.attempts[0].finished_time.is_some());
}

#[test]
fn nested_child_wait_locators_are_prefixed_once_in_the_saved_order() {
    let nodes = BTreeMap::from([("reviewer".into(), running_node(77))]);
    let scene = json!({
        "type":"agent",
        "status":"waiting",
        "waits":[
            {"nodes":[],"step":"first"},
            {"nodes":[{"node_id":"inner","exec_id":88,"idx":0}],"step":"second"}
        ]
    });
    let observation = observation(None, Some(scene), None);
    let mut waits = Vec::new();

    sync_waits(&nodes, &mut waits, &observation).unwrap();
    sync_waits(&nodes, &mut waits, &observation).unwrap();

    assert_eq!(waits.len(), 2);
    assert_eq!(waits[0].step, "first");
    assert_eq!(waits[0].nodes[0].node_id, "reviewer");
    assert_eq!(waits[0].nodes[0].exec_id, 77);
    assert_eq!(waits[0].nodes[0].idx, 0);
    assert_eq!(waits[1].step, "second");
    assert_eq!(waits[1].nodes[0].node_id, "reviewer");
    assert_eq!(waits[1].nodes[1].node_id, "inner");
}
