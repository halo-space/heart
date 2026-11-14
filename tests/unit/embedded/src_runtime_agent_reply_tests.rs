use super::*;
use crate::components::agent::Payload;
use serde_json::json;
fn metadata() -> Map<String, Value> {
    Map::from_iter([
        ("chat_id".into(), json!(101)),
        ("session_id".into(), json!(202)),
        ("agent_id".into(), json!(303)),
        ("user_id".into(), json!(404)),
    ])
}

fn reply(reply_id: i64, approved: bool) -> Input {
    Input {
        metadata: Map::from_iter([
            ("chat_id".into(), json!(101)),
            ("session_id".into(), json!(202)),
            ("agent_id".into(), json!(303)),
            ("user_id".into(), json!(404)),
            ("reply_id".into(), json!(reply_id)),
        ]),
        payload: vec![Payload {
            r#type: "json".into(),
            data: json!({"value":{"approved":approved}}),
            mime_type: None,
        }],
    }
}

fn register(run: &Handle, id: i64) -> oneshot::Receiver<Input> {
    let (sender, receiver) = oneshot::channel();
    run.0.questions.lock().unwrap().insert(
        id,
        Question {
            active: true,
            owner: Arc::downgrade(&run.0),
            sender: Some(sender),
            submitted: None,
            feedback: None,
        },
    );
    run.0.order.lock().unwrap().push_back(id);
    receiver
}

fn call() -> Call {
    Call {
        call_id: "call-1".into(),
        name: "deploy".into(),
        arguments: Map::from_iter([("version".into(), json!(2))]),
    }
}

fn tool_step(call: &Call) -> Step {
    Step {
        id: "tool-0".into(),
        kind: "tool".into(),
        status: crate::runtime::agent::node::status::step::Status::Running,
        call_id: Some(call.call_id.clone()),
        stage: Some("before".into()),
        call: Some(call.clone()),
        ..Step::default()
    }
}

fn approval_question(reply_id: i64) -> Message {
    let mut question = Message::new(Role::Assistant);
    question.metadata.insert("chat_id".into(), json!(101));
    question.metadata.insert("reply_id".into(), json!(reply_id));
    question
}

#[test]
fn live_reply_submission_enforces_identity_fifo_and_same_reply_idempotency() {
    let hub = Hub::default();
    let run = hub.start(metadata()).unwrap();
    let mut first = register(&run, 501);
    let _second = register(&run, 502);

    let out_of_order = hub.submit(reply(502, true), &Cancellation::new());
    assert_eq!(out_of_order.unwrap_err().code, "CONFLICT");
    assert!(first.try_recv().is_err());

    let mut wrong_owner = reply(501, true);
    wrong_owner.metadata.insert("user_id".into(), json!(999));
    assert_eq!(
        hub.submit(wrong_owner, &Cancellation::new())
            .unwrap_err()
            .code,
        "CONFLICT"
    );

    let accepted = reply(501, true);
    hub.submit(accepted.clone(), &Cancellation::new()).unwrap();
    assert_eq!(first.try_recv().unwrap(), accepted);
    hub.submit(accepted, &Cancellation::new()).unwrap();
    assert_eq!(
        hub.submit(reply(501, false), &Cancellation::new())
            .unwrap_err()
            .code,
        "CONFLICT"
    );
}

#[test]
fn invalid_approval_does_not_consume_the_presented_question() {
    let hub = Hub::default();
    let run = hub.start(metadata()).unwrap();
    let mut receiver = register(&run, 501);
    let mut invalid = reply(501, true);
    invalid.payload[0].data = json!({"value":{"approved":"yes"}});

    assert_eq!(
        hub.submit(invalid, &Cancellation::new()).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
    assert!(receiver.try_recv().is_err());
    assert_eq!(run.0.order.lock().unwrap().front(), Some(&501));
}

#[test]
fn closed_live_run_rejects_late_replies() {
    let hub = Hub::default();
    let run = hub.start(metadata()).unwrap();
    let _receiver = register(&run, 501);
    run.close();

    assert_eq!(
        hub.submit(reply(501, true), &Cancellation::new())
            .unwrap_err()
            .code,
        "NOT_FOUND"
    );
    assert!(hub.ensure_idle(&metadata()).is_ok());
}

#[test]
fn one_chat_id_cannot_own_two_live_runs() {
    let hub = Hub::default();
    let first = hub.start(metadata()).unwrap();
    assert_eq!(hub.ensure_idle(&metadata()).unwrap_err().code, "CONFLICT");
    assert_eq!(hub.start(metadata()).err().unwrap().code, "CONFLICT");

    first.close();
    assert!(hub.ensure_idle(&metadata()).is_ok());
    assert!(hub.start(metadata()).is_ok());
}

#[tokio::test]
async fn a_run_rejects_reusing_a_reply_id_for_a_second_question() {
    fn fixed_id() -> i64 {
        501
    }

    let hub = Hub::default();
    let run = hub.start(metadata()).unwrap();
    let call = call();
    let cancellation = Cancellation::new();
    let mut first = Box::pin(run.ask(&call, "approve", fixed_id, &cancellation));
    assert!(futures_util::poll!(first.as_mut()).is_pending());

    assert_eq!(
        run.ask(
            &call,
            "approve another operation",
            fixed_id,
            &Cancellation::new()
        )
        .await
        .unwrap_err()
        .code,
        "CONFLICT"
    );

    let mut context = Context::from_waker(futures_util::task::noop_waker_ref());
    let Poll::Ready(Change::Present {
        id,
        question: presented,
        ack,
        ..
    }) = run.poll_change(&mut context)
    else {
        panic!("the first approval question should be queued for its owner");
    };
    assert_eq!(id, 501);
    assert_eq!(presented.metadata["reply_id"], json!(501));
    drop(ack);
    drop(first);
}

#[tokio::test]
async fn closing_the_root_releases_a_nested_question_consumer() {
    fn id() -> i64 {
        701
    }

    let hub = Hub::default();
    let root = hub.start(metadata()).unwrap();
    let child = root.child();
    let call = call();
    let cancellation = Cancellation::new();
    let mut waiting = Box::pin(child.ask(&call, "approve", id, &cancellation));
    assert!(futures_util::poll!(waiting.as_mut()).is_pending());

    let mut context = Context::from_waker(futures_util::task::noop_waker_ref());
    let Poll::Ready(Change::Present { ack, .. }) = child.poll_change(&mut context) else {
        panic!("the nested question should be queued to its owner");
    };
    ack.send(Ok(())).unwrap();
    assert!(futures_util::poll!(waiting.as_mut()).is_pending());

    root.close();
    assert_eq!(waiting.await.unwrap_err().code, "CANCELLED");
}

#[tokio::test]
async fn closing_one_child_does_not_cancel_its_sibling_question() {
    fn left_id() -> i64 {
        711
    }
    fn right_id() -> i64 {
        712
    }

    let hub = Hub::default();
    let root = hub.start(metadata()).unwrap();
    let left = root.child();
    let left_nested = left.child();
    let right = root.child();
    let call = call();
    let cancellation = Cancellation::new();
    let mut left_wait =
        Box::pin(left_nested.ask(&call, "nested left operation", left_id, &cancellation));
    let mut right_wait = Box::pin(right.ask(&call, "right operation", right_id, &cancellation));
    assert!(futures_util::poll!(left_wait.as_mut()).is_pending());
    assert!(futures_util::poll!(right_wait.as_mut()).is_pending());

    let mut context = Context::from_waker(futures_util::task::noop_waker_ref());
    let Poll::Ready(left_change) = left_nested.poll_change(&mut context) else {
        panic!("left child question should be queued");
    };
    let Poll::Ready(right_change) = right.poll_change(&mut context) else {
        panic!("right child question should be queued");
    };
    let mut left_steps = vec![tool_step(&call)];
    let mut right_steps = vec![tool_step(&call)];
    let mut left_waits = Vec::new();
    let mut right_waits = Vec::new();
    apply(left_change, &mut left_steps, &mut left_waits, &left_nested).unwrap();
    apply(right_change, &mut right_steps, &mut right_waits, &right).unwrap();
    assert!(futures_util::poll!(left_wait.as_mut()).is_pending());
    assert!(futures_util::poll!(right_wait.as_mut()).is_pending());

    left.close();
    assert_eq!(left_wait.await.unwrap_err().code, "CANCELLED");
    hub.submit(reply(712, true), &cancellation).unwrap();
    assert!(futures_util::poll!(right_wait.as_mut()).is_pending());
    let Poll::Ready(Change::Answered { ack, .. }) = right.poll_change(&mut context) else {
        panic!("right child's own reply should reach its owner");
    };
    ack.send(Ok(())).unwrap();
    assert!(right_wait.await.is_ok());
}

#[test]
fn closing_a_child_preserves_duplicate_semantics_for_an_accepted_reply() {
    let hub = Hub::default();
    let root = hub.start(metadata()).unwrap();
    let child = root.child();
    let mut receiver = register(&child, 713);
    let accepted = reply(713, true);

    hub.submit(accepted.clone(), &Cancellation::new()).unwrap();
    assert_eq!(receiver.try_recv().unwrap(), accepted);
    child.close();

    hub.submit(reply(713, true), &Cancellation::new())
        .expect("the same accepted reply remains idempotent after child close");
    assert_eq!(
        hub.submit(reply(713, false), &Cancellation::new())
            .unwrap_err()
            .code,
        "CONFLICT"
    );
    root.close();
}

#[tokio::test]
async fn interleaved_nested_questions_share_fifo_without_crossing_owners() {
    fn first_id() -> i64 {
        721
    }
    fn second_id() -> i64 {
        722
    }
    fn third_id() -> i64 {
        723
    }

    let hub = Hub::default();
    let root = hub.start(metadata()).unwrap();
    let left = root.child();
    let left_nested = left.child();
    let right = root.child();
    let cancellation = Cancellation::new();
    let call = call();
    let mut first = Box::pin(left.ask(&call, "left", first_id, &cancellation));
    let mut second = Box::pin(right.ask(&call, "right", second_id, &cancellation));
    let mut third = Box::pin(left_nested.ask(&call, "nested left", third_id, &cancellation));
    assert!(futures_util::poll!(first.as_mut()).is_pending());
    assert!(futures_util::poll!(second.as_mut()).is_pending());
    assert!(futures_util::poll!(third.as_mut()).is_pending());

    let mut left_steps = vec![tool_step(&call)];
    let mut right_steps = vec![tool_step(&call)];
    let mut nested_steps = vec![tool_step(&call)];
    let mut left_waits = Vec::new();
    let mut right_waits = Vec::new();
    let mut nested_waits = Vec::new();
    let mut context = Context::from_waker(futures_util::task::noop_waker_ref());
    for (owner, steps, waits) in [
        (&left, &mut left_steps, &mut left_waits),
        (&right, &mut right_steps, &mut right_waits),
        (&left_nested, &mut nested_steps, &mut nested_waits),
    ] {
        let Poll::Ready(change) = owner.poll_change(&mut context) else {
            panic!("each nested question should reach its native owner");
        };
        let event = apply(change, steps, waits, owner).unwrap();
        assert!(
            matches!(event, Some(Event::Complete { message, .. }) if message.metadata.contains_key("reply_id"))
        );
    }
    assert_eq!(
        *root.0.order.lock().unwrap(),
        VecDeque::from([721, 722, 723])
    );

    assert_eq!(
        hub.submit(reply(722, true), &cancellation)
            .unwrap_err()
            .code,
        "CONFLICT",
        "a later sibling cannot skip the first question in the shared FIFO"
    );
    hub.submit(reply(721, true), &cancellation).unwrap();
    assert!(futures_util::poll!(first.as_mut()).is_pending());
    let Poll::Ready(answered) = left.poll_change(&mut context) else {
        panic!("the first reply should be delivered only to its left owner");
    };
    assert!(matches!(answered, Change::Answered { id: 721, .. }));
    apply(answered, &mut left_steps, &mut left_waits, &left).unwrap();
    assert!(first.await.is_ok());
    assert!(futures_util::poll!(second.as_mut()).is_pending());
    assert!(futures_util::poll!(third.as_mut()).is_pending());

    hub.submit(reply(722, true), &cancellation).unwrap();
    assert!(futures_util::poll!(second.as_mut()).is_pending());
    let Poll::Ready(answered) = right.poll_change(&mut context) else {
        panic!("the second reply should be delivered only to its right owner");
    };
    assert!(matches!(answered, Change::Answered { id: 722, .. }));
    apply(answered, &mut right_steps, &mut right_waits, &right).unwrap();
    assert!(second.await.is_ok());
    hub.submit(reply(723, true), &cancellation).unwrap();
    assert!(futures_util::poll!(third.as_mut()).is_pending());
    let Poll::Ready(answered) = left_nested.poll_change(&mut context) else {
        panic!("the third reply should be delivered only to its nested left owner");
    };
    assert!(matches!(answered, Change::Answered { id: 723, .. }));
    apply(answered, &mut nested_steps, &mut nested_waits, &left_nested).unwrap();
    assert!(third.await.is_ok());
    assert!(root.0.order.lock().unwrap().is_empty());
}

#[tokio::test]
async fn planned_task_slot_is_released_while_waiting_and_reacquired_before_continuation() {
    let permits = Arc::new(tokio::sync::Semaphore::new(1));
    let changed = Arc::new(AtomicWaker::new());
    let slot = Slot::new(Arc::clone(&permits), Arc::clone(&changed)).unwrap();
    assert!(slot.held());
    assert_eq!(permits.available_permits(), 0);

    feedback::with_slot(slot.clone(), async {
        feedback::release_slot();
        assert!(!slot.held());
        assert_eq!(permits.available_permits(), 1);

        slot.acquire(&Cancellation::new()).await.unwrap();
        assert!(slot.held());
        assert_eq!(permits.available_permits(), 0);
    })
    .await;
}

#[tokio::test]
async fn cancelled_capacity_reacquisition_does_not_fabricate_a_held_slot() {
    let permits = Arc::new(tokio::sync::Semaphore::new(1));
    let changed = Arc::new(AtomicWaker::new());
    let slot = Slot::new(Arc::clone(&permits), changed).unwrap();
    slot.release();
    let _other_owner = permits.clone().acquire_owned().await.unwrap();
    let cancellation = Cancellation::new();
    cancellation.cancel();

    assert_eq!(
        slot.acquire(&cancellation).await.unwrap_err().code,
        "CANCELLED"
    );
    assert!(!slot.held());
    assert_eq!(permits.available_permits(), 0);
}

#[test]
fn presenting_an_approval_commits_its_actual_tool_step_and_fifo_locator() {
    let hub = Hub::default();
    let run = hub.start(metadata()).unwrap();
    let _receiver = register(&run, 501);
    let call = call();
    let question = approval_question(501);
    let (ack, mut acknowledged) = oneshot::channel();
    let change = Change::Present {
        call: call.clone(),
        id: 501,
        question: Box::new(question.clone()),
        ack,
    };
    let mut steps = vec![tool_step(&call)];
    let mut waits = Vec::new();

    let event = apply(change, &mut steps, &mut waits, &run)
        .unwrap()
        .expect("the saved approval is presented as an event");

    assert!(matches!(event, Event::Complete { message, .. } if message == question));
    assert_eq!(acknowledged.try_recv().unwrap().unwrap(), ());
    assert_eq!(
        steps[0].status,
        crate::runtime::agent::node::status::step::Status::Waiting
    );
    assert_eq!(steps[0].call.as_ref(), Some(&call));
    assert_eq!(steps[0].waits[&501].kind, "approval");
    assert_eq!(waits.len(), 1);
    assert_eq!(waits[0].step, "tool-0");
    assert_eq!(
        run.0
            .order
            .lock()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [501]
    );
}

#[test]
fn answer_transition_rejects_non_fifo_approval_without_mutating_runtime_facts() {
    let hub = Hub::default();
    let run = hub.start(metadata()).unwrap();
    let _first = register(&run, 501);
    let _second = register(&run, 502);
    let call = call();
    let mut step = tool_step(&call);
    for id in [501, 502] {
        step.waits
            .insert(id, Wait::new("approval", approval_question(id), 1).unwrap());
    }
    let mut steps = vec![step];
    let mut waits = vec![crate::runtime::agent::wait::Wait {
        nodes: Vec::new(),
        step: "tool-0".into(),
    }];
    let (ack, mut acknowledged) = oneshot::channel();

    let result = apply(
        Change::Answered {
            call,
            id: 502,
            input: reply(502, true),
            ack,
        },
        &mut steps,
        &mut waits,
        &run,
    );

    assert_eq!(result.unwrap_err().code, "CONFLICT");
    assert_eq!(
        acknowledged.try_recv().unwrap().unwrap_err().code,
        "CONFLICT"
    );
    assert_eq!(
        steps[0].status,
        crate::runtime::agent::node::status::step::Status::Running
    );
    assert_eq!(
        steps[0].waits[&502].status,
        crate::runtime::agent::node::status::wait::Status::Waiting
    );
    assert_eq!(waits.len(), 1);
    assert_eq!(
        run.0
            .order
            .lock()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [501, 502]
    );
}

#[test]
fn answer_transition_consumes_only_the_matching_fifo_head_and_records_feedback() {
    let hub = Hub::default();
    let run = hub.start(metadata()).unwrap();
    let _receiver = register(&run, 501);
    let call = call();
    let mut step = tool_step(&call);
    step.status = crate::runtime::agent::node::status::step::Status::Waiting;
    step.waits.insert(
        501,
        Wait::new("approval", approval_question(501), 1).unwrap(),
    );
    let mut steps = vec![step];
    let mut waits = vec![crate::runtime::agent::wait::Wait {
        nodes: Vec::new(),
        step: "tool-0".into(),
    }];
    let (ack, mut acknowledged) = oneshot::channel();

    assert!(
        apply(
            Change::Answered {
                call,
                id: 501,
                input: reply(501, true),
                ack,
            },
            &mut steps,
            &mut waits,
            &run,
        )
        .unwrap()
        .is_none()
    );

    assert_eq!(acknowledged.try_recv().unwrap().unwrap(), ());
    assert_eq!(
        steps[0].status,
        crate::runtime::agent::node::status::step::Status::Running
    );
    assert_eq!(
        steps[0].waits[&501].status,
        crate::runtime::agent::node::status::wait::Status::Resolved
    );
    assert_eq!(steps[0].waits[&501].feedback.as_slice().len(), 1);
    assert!(waits.is_empty());
    assert!(run.0.order.lock().unwrap().is_empty());
}
