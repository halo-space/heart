//! Frozen Step-owned question data; this does not claim live reply routing.
use std::collections::BTreeMap;

use halo_agents::{
    Message, Messages,
    message::Role,
    runtime::agent::{
        State,
        node::{Step, Wait},
    },
};
use serde_json::json;

#[test]
fn native_step_roundtrip_preserves_questions_feedback_and_business_result_separately() {
    let question = Message::new(Role::Assistant);
    let first = Message::new(Role::User);
    let second = Message::new(Role::User);
    let business = Message::function(json!({"result": "unchanged"}));
    let step = Step {
        id: "tool_0".into(),
        kind: "tool".into(),
        status: "waiting".into(),
        stage: Some("eval".into()),
        message: Some(business.clone()),
        waits: BTreeMap::from([
            (
                5001,
                Wait {
                    kind: "approval".into(),
                    status: "resolved".into(),
                    question: question.clone(),
                    feedback: Messages::new([first.clone()]),
                },
            ),
            (
                5002,
                Wait {
                    kind: "ask".into(),
                    status: "waiting".into(),
                    question: question.clone(),
                    feedback: Messages::new([first, second]),
                },
            ),
        ]),
        ..Default::default()
    };
    let state = State {
        chat_id: Some(1001),
        r#type: "agent".into(),
        status: "waiting".into(),
        steps: vec![step],
        ..Default::default()
    };
    let value = serde_json::to_value(&state).unwrap();
    let wait = &value["steps"][0]["waits"]["5002"];
    assert_eq!(wait.as_object().unwrap().len(), 4);
    assert_eq!(wait["kind"], "ask");
    assert_eq!(wait["status"], "waiting");
    assert_eq!(wait["question"], serde_json::to_value(question).unwrap());
    assert_eq!(wait["feedback"].as_array().unwrap().len(), 2);
    let restored: State = serde_json::from_value(value).unwrap();
    assert_eq!(restored, state);
    assert_eq!(restored.steps[0].waits[&5001].status, "resolved");
    assert_eq!(
        restored.steps[0].status, "waiting",
        "resolved is not node success"
    );
    assert_eq!(restored.steps[0].message.as_ref(), Some(&business));
    assert_eq!(restored.steps[0].stage.as_deref(), Some("eval"));
}

#[test]
fn reply_identity_remains_exact_i64_despite_json_object_keys() {
    let reply_id = 9_007_199_254_740_993_i64;
    let step = Step {
        waits: BTreeMap::from([(
            reply_id,
            Wait {
                kind: "ask".into(),
                status: "waiting".into(),
                question: Message::new(Role::Assistant),
                feedback: Messages::default(),
            },
        )]),
        ..Default::default()
    };
    let encoded = serde_json::to_string(&step).unwrap();
    assert!(encoded.contains("\"9007199254740993\""));
    let restored: Step = serde_json::from_str(&encoded).unwrap();
    assert!(restored.waits.contains_key(&reply_id));
    assert_eq!(restored, step);
}
