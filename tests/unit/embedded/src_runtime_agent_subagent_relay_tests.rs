use super::*;
use crate::{Message, message::Role, runtime::agent::wait::Node};
use futures_util::future::poll_fn;
use serde_json::json;

#[tokio::test]
async fn child_event_waits_for_parent_ack_and_keeps_its_execution_source() {
    let relay = Relay::default();
    let observer = relay.observer(
        Some(501),
        Node {
            node_id: "reviewer".into(),
            exec_id: 77,
            idx: 0,
        },
    );
    let sender = tokio::spawn(async move {
        let mut message = Message::new(Role::Assistant);
        message.values.insert("phase".into(), json!("child"));
        observer
            .emit(Some(Event::Delta { message }), None, None, None, false)
            .await;
    });

    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), async {
            let _ = sender.is_finished();
            while !sender.is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .is_err(),
        "child producer stays backpressured until its event is consumed"
    );
    let observation = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        poll_fn(|context| relay.poll_event(context)),
    )
    .await
    .expect("parent must receive the queued child event");
    let Some(Event::Delta { message }) = observation.event else {
        panic!("expected the original child Delta")
    };
    assert_eq!(message.metadata["chat_id"], 501);
    assert_eq!(message.metadata["node_id"], "reviewer");
    assert_eq!(message.metadata["exec_id"], 77);
    assert_eq!(message.values["phase"], "child");
    tokio::time::timeout(std::time::Duration::from_secs(1), sender)
        .await
        .expect("consumption acknowledges the event")
        .expect("child emitter task succeeds");
}
