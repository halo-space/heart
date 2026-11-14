use super::*;
use crate::{Message, message::Role, runtime::agent::node};
use serde_json::json;
use std::collections::BTreeMap;

fn nested_wait_scene() -> (Value, i64) {
    let reply_id = 701;
    let mut question = Message::new(Role::Assistant);
    question.values.insert("question".into(), json!("请确认"));
    let mut record = node::Wait::new("input", question, 3).unwrap();
    let mut feedback = Message::new(Role::User);
    feedback.values.insert("answer".into(), json!("确认"));
    record.reply(feedback).unwrap();

    let child = crate::runtime::agent::State {
        chat_id: Some(1001),
        r#type: "agent".into(),
        status: crate::runtime::agent::status::Status::Waiting,
        steps: vec![node::Step {
            id: "collect".into(),
            kind: "tool".into(),
            status: crate::runtime::agent::node::status::step::Status::Waiting,
            waits: BTreeMap::from([(reply_id, record)]),
            ..Default::default()
        }],
        ..Default::default()
    };
    let attempt = node::Attempt {
        idx: 0,
        status: node::status::Status::Waiting,
        created_time: None,
        finished_time: None,
        usage: None,
        finish_reason: None,
        steps: Vec::new(),
        messages: Default::default(),
        completion: None,
        nodes: BTreeMap::new(),
        agent: Some(serde_json::to_value(child).unwrap()),
        error: None,
    };
    let reviewer = node::State {
        status: node::status::Status::Waiting,
        execution: Some(node::Execution {
            exec_id: 77,
            attempts: vec![attempt],
            messages: Default::default(),
            usage: None,
            error: None,
        }),
        ..Default::default()
    };
    let root = crate::runtime::agent::State {
        chat_id: Some(1001),
        r#type: "agent".into(),
        status: crate::runtime::agent::status::Status::Waiting,
        nodes: BTreeMap::from([("reviewer".into(), reviewer)]),
        waits: vec![Wait {
            nodes: vec![Node {
                node_id: "reviewer".into(),
                exec_id: 77,
                idx: 0,
            }],
            step: "collect".into(),
        }],
        ..Default::default()
    };
    (serde_json::to_value(root).unwrap(), reply_id)
}

fn nested_sibling_waits() -> (Value, [i64; 2]) {
    let (mut scene, first_id) = nested_wait_scene();
    let second_id = first_id + 1;
    let mut second_node = scene["nodes"]["reviewer"].clone();
    second_node["execution"]["exec_id"] = json!(78);
    let second_wait = second_node["execution"]["attempts"][0]["agent"]["steps"][0]["waits"]
        [first_id.to_string()]
    .clone();
    second_node["execution"]["attempts"][0]["agent"]["steps"][0]["waits"] =
        json!({second_id.to_string(): second_wait});
    second_node["execution"]["attempts"][0]["agent"]["steps"][0]["waits"][&second_id.to_string()]
        ["question"]["metadata"]["reply_id"] = json!(second_id);
    scene["nodes"]["reviewer_b"] = second_node;
    scene["waits"].as_array_mut().unwrap().push(json!({
        "nodes":[{"node_id":"reviewer_b","exec_id":78,"idx":0}],
        "step":"collect"
    }));
    (scene, [first_id, second_id])
}

#[test]
fn nested_child_feedback_is_resolved_through_its_native_owner_and_locator() {
    let (scene, reply_id) = nested_wait_scene();

    assert!(validate(&scene).unwrap());
    assert!(question(&scene, reply_id).unwrap());
    assert_eq!(input_progress(&scene).get(&reply_id.to_string()), Some(&1));
    assert!(has_input(&scene));
    assert!(has_waiting_input(&scene));
    assert_eq!(scene["waits"][0]["nodes"][0]["node_id"], "reviewer");
}

#[test]
fn nested_feedback_rejects_stale_execution_and_unlocated_waits() {
    let (scene, _) = nested_wait_scene();
    let mut stale = scene.clone();
    stale["waits"][0]["nodes"][0]["exec_id"] = json!(78);
    assert_eq!(validate(&stale).unwrap_err().code, "INVALID_STATE");

    let mut unlocated = scene;
    unlocated["waits"] = json!([]);
    assert_eq!(validate(&unlocated).unwrap_err().code, "INVALID_STATE");
}

#[test]
fn sibling_child_waits_keep_their_native_owners_and_saved_fifo_order() {
    let (scene, [first_id, second_id]) = nested_sibling_waits();
    let checkpoint: Value = serde_json::from_slice(
        &serde_json::to_vec(&scene).expect("serialize complete Runtime State"),
    )
    .expect("restore complete Runtime State");

    assert!(validate(&checkpoint).unwrap());
    assert!(question(&checkpoint, first_id).unwrap());
    assert!(question(&checkpoint, second_id).unwrap());
    assert_eq!(
        input_progress(&checkpoint),
        std::collections::BTreeMap::from([(first_id.to_string(), 1), (second_id.to_string(), 1),])
    );
    assert_eq!(checkpoint["waits"][0]["nodes"][0]["node_id"], "reviewer");
    assert_eq!(checkpoint["waits"][1]["nodes"][0]["node_id"], "reviewer_b");

    let mut duplicate = checkpoint;
    duplicate["waits"][1] = duplicate["waits"][0].clone();
    assert_eq!(validate(&duplicate).unwrap_err().code, "INVALID_STATE");
}
