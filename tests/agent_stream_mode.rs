//! Corrupt checkpoint protocol is tested from canonical State, not from a
//! scripted Chat implementation that invents a successful model response.
use std::collections::BTreeMap;
use std::sync::Arc;

use halo::agent::{
    self, Agent, Cancellation, Input, Message,
    checkpoint::Checkpoint as _,
    message::Role,
    model::{chat, content::part::Part},
};
use halo::{
    core,
    runtime::agent::{
        State,
        node::{Step, Wait},
        wait,
    },
};
use serde_json::{Value, json};

fn configured_chat() -> halo::agent::provider::vllm::chat::Chat {
    let provider = halo::agent::provider::vllm::Vllm::builder()
        .api_key("compile-only")
        .build()
        .expect("construct Provider without dispatching a request");
    provider
        .chat(chat::Config::new("compile-only"))
        .expect("construct real Provider Chat")
}

fn input() -> Input {
    serde_json::from_value(json!({
        "metadata":{"chat_id":1001},
        "payload":[{"type":"text","data":{"value":"resume"}}]
    }))
    .expect("valid user input")
}

fn valid_waiting_state() -> Value {
    let mut question = Message::new(Role::Assistant);
    question.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":"Please provide feedback"})
            .as_object()
            .unwrap()
            .clone(),
    });
    question.metadata.insert("chat_id".into(), json!(1001));
    question.metadata.insert("reply_id".into(), json!(5001));
    let wait = Wait::new("input", question, 1).expect("valid wait fact");
    serde_json::to_value(State {
        chat_id: Some(1001),
        r#type: "agent".into(),
        status: "waiting".into(),
        stream: true,
        steps: vec![Step {
            id: "model_0".into(),
            kind: "model".into(),
            status: "waiting".into(),
            waits: BTreeMap::from([(5001, wait)]),
            ..Default::default()
        }],
        waits: vec![wait::Wait {
            nodes: Vec::new(),
            step: "model_0".into(),
        }],
        ..Default::default()
    })
    .expect("serialize canonical Agent State")
}

async fn rejects_corrupt_wait<A: Agent>(agent: A, checkpoint: &core::checkpoint::InMemory) {
    let cancellation = Cancellation::new();
    let mut scene = valid_waiting_state();
    scene["steps"][0]["waits"]["5001"]["status"] = json!("received");
    checkpoint
        .save(1001, scene.clone(), &cancellation)
        .await
        .expect("save the deliberately corrupt checkpoint");

    let error = match agent.run(input(), true, cancellation.clone()).await {
        Err(error) => error,
        Ok(_) => panic!("invalid saved wait must be rejected before model dispatch"),
    };
    assert_eq!(error.code, "INVALID_STATE");
    assert_eq!(error.message, "invalid Agent Step waiting record");
    assert_eq!(
        checkpoint
            .load(1001, &cancellation)
            .await
            .expect("read saved state"),
        scene,
        "failed validation must not overwrite the original checkpoint"
    );
}

#[tokio::test]
async fn chat_and_react_reject_corrupt_wait_from_canonical_checkpoint_state() {
    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    rejects_corrupt_wait(
        agent::chat()
            .model(configured_chat())
            .checkpoint(checkpoint.clone())
            .build()
            .expect("build Chat Agent"),
        &checkpoint,
    )
    .await;

    let checkpoint = Arc::new(core::checkpoint::InMemory::new());
    rejects_corrupt_wait(
        agent::react()
            .model(configured_chat())
            .checkpoint(checkpoint.clone())
            .build()
            .expect("build ReAct Agent"),
        &checkpoint,
    )
    .await;
}
