use super::*;
use crate::{Message, components::agent::Payload, message::Role, runtime::agent::node};
use serde_json::json;
use std::collections::BTreeMap;

const CHAT_ID: i64 = 41;
const REPLY_ID: i64 = 701;

fn question(reply_id: i64) -> Message {
    let mut message = Message::new(Role::Assistant);
    message.metadata.insert("chat_id".into(), CHAT_ID.into());
    message.metadata.insert("reply_id".into(), reply_id.into());
    message
        .values
        .insert("question".into(), json!("请批准操作"));
    message
}

fn feedback(approved: bool) -> Message {
    let mut message = Message::new(Role::User);
    message
        .content
        .push(crate::components::model::content::part::Part {
            r#type: "json".into(),
            data: json!({"value":{"approved":approved}})
                .as_object()
                .unwrap()
                .clone(),
        });
    message
}

fn approval_scene(reply_id: i64, resolved: Option<bool>) -> Value {
    let mut record = node::Wait::new("approval", question(reply_id), 1).unwrap();
    if let Some(approved) = resolved {
        record.reply(feedback(approved)).unwrap();
    }
    let call = Call {
        call_id: "call-1".into(),
        name: "publish".into(),
        arguments: serde_json::Map::new(),
    };
    let step = node::Step {
        id: "tool-1".into(),
        kind: "tool".into(),
        status: if resolved.is_some() {
            crate::runtime::agent::node::status::step::Status::Running
        } else {
            crate::runtime::agent::node::status::step::Status::Waiting
        },
        stage: Some("before".into()),
        call_id: Some(call.call_id.clone()),
        call: Some(call),
        waits: BTreeMap::from([(reply_id, record)]),
        ..Default::default()
    };
    json!({
        "type":"agent",
        "chat_id":CHAT_ID,
        "status":if resolved.is_some() { "running" } else { "waiting" },
        "steps":[step],
        "waits":if resolved.is_some() { json!([]) } else { json!([{"nodes":[],"step":"tool-1"}]) },
        "nodes":{}
    })
}

fn submitted(reply_id: i64, approved: bool) -> Input {
    Input {
        metadata: serde_json::Map::from_iter([
            ("chat_id".into(), CHAT_ID.into()),
            ("reply_id".into(), reply_id.into()),
        ]),
        payload: vec![Payload {
            r#type: "json".into(),
            data: json!({"value":{"approved":approved}}),
            mime_type: None,
        }],
    }
}

fn sibling_child_scene() -> Value {
    let mut nodes = BTreeMap::new();
    let mut waits = Vec::new();
    for (node_id, exec_id, reply_id) in [("left", 77, REPLY_ID), ("right", 78, REPLY_ID + 1)] {
        let record = node::Wait::new("approval", question(reply_id), 1).unwrap();
        let call = Call {
            call_id: "shared-call".into(),
            name: "publish".into(),
            arguments: serde_json::Map::new(),
        };
        let step = node::Step {
            id: "tool-1".into(),
            kind: "tool".into(),
            status: crate::runtime::agent::node::status::step::Status::Waiting,
            stage: Some("before".into()),
            call_id: Some(call.call_id.clone()),
            call: Some(call),
            waits: BTreeMap::from([(reply_id, record.clone())]),
            ..Default::default()
        };
        let child = crate::runtime::agent::State {
            chat_id: Some(CHAT_ID),
            r#type: "agent".into(),
            status: crate::runtime::agent::status::Status::Waiting,
            steps: vec![step],
            waits: vec![wait::Wait {
                nodes: Vec::new(),
                step: "tool-1".into(),
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
        nodes.insert(
            node_id.to_owned(),
            node::State {
                status: node::status::Status::Waiting,
                execution: Some(node::Execution {
                    exec_id,
                    attempts: vec![attempt],
                    messages: Default::default(),
                    usage: None,
                    error: None,
                }),
                ..Default::default()
            },
        );
        waits.push(wait::Wait {
            nodes: vec![wait::Node {
                node_id: node_id.to_owned(),
                exec_id,
                idx: 0,
            }],
            step: "tool-1".into(),
        });
    }
    let root = crate::runtime::agent::State {
        chat_id: Some(CHAT_ID),
        r#type: "agent".into(),
        status: crate::runtime::agent::status::Status::Waiting,
        nodes,
        waits,
        ..Default::default()
    };
    serde_json::to_value(root).unwrap()
}

fn planned_sibling_child_scene() -> Value {
    let mut scene = sibling_child_scene();
    for (node_id, task_exec_id) in [("left", 88), ("right", 89)] {
        let node = &mut scene["nodes"][node_id];
        let attempt = &mut node["execution"]["attempts"][0];
        let child = std::mem::replace(&mut attempt["agent"], Value::Null);
        attempt["nodes"] = json!({"task-1": {
            "status":"waiting",
            "execution": {
                "exec_id":task_exec_id,
                "attempts":[{
                    "idx":0,
                    "status":"waiting",
                    "steps":[],
                    "nodes":{},
                    "agent":child
                }]
            }
        }});
        scene["waits"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|wait| wait["nodes"][0]["node_id"] == node_id)
            .unwrap()["nodes"]
            .as_array_mut()
            .unwrap()
            .push(json!({"node_id":"task-1","exec_id":task_exec_id,"idx":0}));
    }
    scene
}

#[test]
fn restore_parses_the_actual_wait_step_and_preserves_reply_identity() {
    let scene = approval_scene(REPLY_ID, None);
    let restore = prepare(&scene, &submitted(REPLY_ID, true))
        .unwrap()
        .expect("active approval is restored");

    assert_eq!(restore.reply_order, [REPLY_ID]);
    let question = restore.questions.get("call-1").unwrap();
    assert_eq!(question.id, REPLY_ID);
    assert_eq!(question.call.name, "publish");
    assert_eq!(
        question.input.as_ref().unwrap().metadata["reply_id"],
        REPLY_ID
    );
}

#[test]
fn checkpoint_json_round_trip_restores_the_same_pending_approval() {
    let saved = approval_scene(REPLY_ID, None);
    let bytes = serde_json::to_vec(&saved).expect("serialize saved Runtime State");
    let restored: Value = serde_json::from_slice(&bytes).expect("deserialize Runtime State");
    let prepared = prepare(&restored, &submitted(REPLY_ID, true))
        .unwrap()
        .expect("saved pending approval can be rebuilt from persisted State");

    assert_eq!(prepared.reply_order, [REPLY_ID]);
    let question = prepared.questions.get("call-1").unwrap();
    assert_eq!(question.id, REPLY_ID);
    assert_eq!(question.call.name, "publish");
    assert!(question.call.arguments.is_empty());
}

#[test]
fn restore_rejects_missing_or_unknown_reply_identity_without_dispatching() {
    let scene = approval_scene(REPLY_ID, None);
    let mut untargeted = submitted(REPLY_ID, true);
    untargeted.metadata.remove("reply_id");
    assert_eq!(prepare(&scene, &untargeted).err().unwrap().code, "WAITING");

    assert_eq!(
        prepare(&scene, &submitted(REPLY_ID + 1, true))
            .err()
            .unwrap()
            .code,
        "NOT_FOUND"
    );
}

#[test]
fn restore_requires_the_saved_fifo_head_before_accepting_feedback() {
    let mut scene = approval_scene(REPLY_ID, None);
    let second_id = REPLY_ID + 1;
    let mut second = scene["steps"][0].clone();
    second["id"] = json!("tool-2");
    second["call_id"] = json!("call-2");
    second["call"]["call_id"] = json!("call-2");
    second["waits"] = json!({second_id.to_string(): second["waits"][REPLY_ID.to_string()].clone()});
    second["waits"][second_id.to_string()]["question"]["metadata"]["reply_id"] = json!(second_id);
    scene["steps"].as_array_mut().unwrap().push(second);
    scene["waits"]
        .as_array_mut()
        .unwrap()
        .push(json!({"nodes":[],"step":"tool-2"}));

    let error = prepare(&scene, &submitted(second_id, true)).err().unwrap();
    assert_eq!(error.code, "CONFLICT", "{}", error.message);
}

#[test]
fn restored_approval_is_idempotent_only_for_the_same_saved_feedback() {
    let scene = approval_scene(REPLY_ID, Some(true));
    assert!(duplicate(&scene, &submitted(REPLY_ID, true)).unwrap());
    assert_eq!(
        duplicate(&scene, &submitted(REPLY_ID, false))
            .unwrap_err()
            .code,
        "CONFLICT"
    );
}

#[test]
fn restore_does_not_turn_a_denial_into_an_approval() {
    let scene = approval_scene(REPLY_ID, None);
    let restore = prepare(&scene, &submitted(REPLY_ID, false))
        .unwrap()
        .expect("denial is still a valid reply to restore");

    assert_eq!(restore.reply_order, [REPLY_ID]);
    assert_eq!(
        restore
            .questions
            .get("call-1")
            .unwrap()
            .input
            .as_ref()
            .unwrap()
            .payload[0]
            .data["value"]["approved"],
        false
    );
}

#[tokio::test]
async fn recovered_approval_cannot_authorize_a_changed_tool_call() {
    let scene = approval_scene(REPLY_ID, None);
    let restore = prepare(&scene, &submitted(REPLY_ID, true))
        .unwrap()
        .expect("pending approval is restorable");
    let hub = Hub::default();
    let run = hub
        .start(serde_json::from_value(json!({"chat_id":CHAT_ID})).unwrap())
        .unwrap();
    run.restore_approvals(restore);

    let mut step: Step = serde_json::from_value(scene["steps"][0].clone()).unwrap();
    step.call
        .as_mut()
        .unwrap()
        .arguments
        .insert("version".into(), json!(99));
    assert_eq!(
        run.resume_approval(&step, &crate::Cancellation::new())
            .await
            .unwrap_err()
            .code,
        "WAITING",
        "a saved reply is bound to the exact serialized Call"
    );
}

#[tokio::test]
async fn restored_reply_ids_remain_reserved_against_a_reused_generator() {
    fn reused_id() -> i64 {
        REPLY_ID
    }

    let mut scene = approval_scene(REPLY_ID, Some(true));
    let pending = approval_scene(REPLY_ID + 1, None);
    let mut pending_step = pending["steps"][0].clone();
    pending_step["id"] = json!("tool-2");
    pending_step["call_id"] = json!("call-2");
    pending_step["call"]["call_id"] = json!("call-2");
    scene["steps"].as_array_mut().unwrap().push(pending_step);
    scene["waits"] = pending["waits"].clone();
    scene["waits"][0]["step"] = json!("tool-2");
    let restore = prepare(&scene, &submitted(REPLY_ID, true))
        .unwrap()
        .expect("old reply is an idempotent duplicate while another question is pending");
    let hub = Hub::default();
    let run = hub
        .start(serde_json::from_value(json!({"chat_id":CHAT_ID})).unwrap())
        .unwrap();
    run.restore_approvals(restore);
    let step = Step {
        id: "new-tool".into(),
        kind: "tool".into(),
        status: crate::runtime::agent::node::status::step::Status::Running,
        call_id: Some("new-call".into()),
        stage: Some("before".into()),
        call: Some(Call {
            call_id: "new-call".into(),
            name: "publish".into(),
            arguments: Default::default(),
        }),
        ..Default::default()
    };

    assert_eq!(
        run.ask(
            step.call.as_ref().unwrap(),
            "approve",
            reused_id,
            &crate::Cancellation::new(),
        )
        .await
        .unwrap_err()
        .code,
        "CONFLICT"
    );
}

#[test]
fn restore_routes_reused_call_ids_to_their_native_child_owners() {
    let scene = sibling_child_scene();
    let located = wait::scene_step(
        &scene,
        &[wait::Node {
            node_id: "left".into(),
            exec_id: 77,
            idx: 0,
        }],
        "tool-1",
    )
    .unwrap();
    assert!(std::ptr::eq(
        located,
        &scene["nodes"]["left"]["execution"]["attempts"][0]["agent"]["steps"][0]
    ));
    let located_right = wait::scene_step(
        &scene,
        &[wait::Node {
            node_id: "right".into(),
            exec_id: 78,
            idx: 0,
        }],
        "tool-1",
    )
    .unwrap();
    assert!(std::ptr::eq(
        located_right,
        &scene["nodes"]["right"]["execution"]["attempts"][0]["agent"]["steps"][0]
    ));
    let restore = prepare(&scene, &submitted(REPLY_ID, true))
        .unwrap()
        .expect("child approval is restored");
    let left = wait::Node {
        node_id: "left".into(),
        exec_id: 77,
        idx: 0,
    };
    let right = wait::Node {
        node_id: "right".into(),
        exec_id: 78,
        idx: 0,
    };

    assert_eq!(restore.reply_order, [REPLY_ID, REPLY_ID + 1]);
    let left_question = restore.tasks[&left].questions.get("shared-call").unwrap();
    let right_question = restore.tasks[&right].questions.get("shared-call").unwrap();
    assert_eq!(left_question.id, REPLY_ID);
    assert_eq!(right_question.id, REPLY_ID + 1);
    assert_eq!(
        left_question.input.as_ref().unwrap().metadata["reply_id"],
        REPLY_ID
    );
    assert!(right_question.input.is_none());
}

#[test]
fn restore_keeps_planned_child_questions_under_both_native_coordinates() {
    let scene = planned_sibling_child_scene();
    let restore = prepare(&scene, &submitted(REPLY_ID, true))
        .unwrap()
        .expect("planned child approval is restored");
    let parent = wait::Node {
        node_id: "left".into(),
        exec_id: 77,
        idx: 0,
    };
    let task = wait::Node {
        node_id: "task-1".into(),
        exec_id: 88,
        idx: 0,
    };
    let child = &restore.tasks[&parent].children[&task];
    let question = child.questions.get("shared-call").unwrap();

    assert_eq!(question.id, REPLY_ID);
    assert_eq!(
        question.input.as_ref().unwrap().metadata["reply_id"],
        REPLY_ID
    );
    assert_eq!(restore.tasks[&parent].children.len(), 1);
}
