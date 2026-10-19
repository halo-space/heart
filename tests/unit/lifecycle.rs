//! Native test scenes stand in for distinct owners, not a recovery store.
use super::*;
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Debug, PartialEq)]
struct Scene {
    phase: Phase,
    input: i32,
    output: Option<String>,
}
impl Default for Scene {
    fn default() -> Self {
        Self {
            phase: Phase::Before,
            input: 1,
            output: None,
        }
    }
}
type Owner = Rc<RefCell<Scene>>;
type Calls = Rc<RefCell<[usize; 3]>>;

async fn run(
    owner: Owner,
    calls: Calls,
    stop: Option<Phase>,
    until: Phase,
    c: &Cancellation,
) -> Result<(i32, Option<String>), Error> {
    let scene = owner.borrow().clone();
    advance(
        scene.phase,
        until,
        scene.input,
        scene.output,
        Methods {
            before: {
                let calls = calls.clone();
                move |mut input| {
                    calls.borrow_mut()[0] += 1;
                    async move {
                        input += 1;
                        if stop == Some(Phase::Before) {
                            return Err(Error::new("WAITING", "before"));
                        }
                        Ok(input)
                    }
                }
            },
            call: {
                let calls = calls.clone();
                move |input| {
                    calls.borrow_mut()[1] += 1;
                    async move {
                        if stop == Some(Phase::Call) {
                            return Err(Error::new("WAITING", "call"));
                        }
                        Ok(format!("raw:{input}"))
                    }
                }
            },
            after: move |_, mut output: String| {
                calls.borrow_mut()[2] += 1;
                async move {
                    output.push_str(":processed");
                    if stop == Some(Phase::After) {
                        return Err(Error::new("WAITING", "after"));
                    }
                    Ok(output)
                }
            },
            commit: move |phase, input: &i32, output: Option<&String>| {
                *owner.borrow_mut() = Scene {
                    phase,
                    input: *input,
                    output: output.cloned(),
                };
                async { Ok(()) }
            },
        },
        c,
    )
    .await
}

#[test]
fn completed_facts_stay_in_the_owner_and_only_unfinished_methods_repeat() {
    futures_executor::block_on(async {
        for phase in [Phase::Before, Phase::Call, Phase::After] {
            let owner = Owner::default();
            let calls = Calls::default();
            assert_eq!(
                run(
                    owner.clone(),
                    calls.clone(),
                    Some(phase),
                    Phase::Complete,
                    &Cancellation::new()
                )
                .await
                .unwrap_err()
                .code,
                "WAITING"
            );
            let saved = owner.borrow().clone();
            assert_eq!(saved.phase, phase);
            assert_eq!(saved.input, if phase == Phase::Before { 1 } else { 2 });
            assert_eq!(
                saved.output,
                (phase == Phase::After).then(|| "raw:2".into())
            );
            // Restore from only the owner's data, discarding all working copies.
            let restored = Rc::new(RefCell::new(saved));
            assert_eq!(
                run(
                    restored.clone(),
                    calls.clone(),
                    None,
                    Phase::Complete,
                    &Cancellation::new()
                )
                .await
                .unwrap(),
                (2, Some("raw:2:processed".into()))
            );
            assert_eq!(
                *calls.borrow(),
                match phase {
                    Phase::Before => [2, 1, 1],
                    Phase::Call => [1, 2, 1],
                    _ => [1, 1, 2],
                }
            );
            let counts = *calls.borrow();
            run(
                restored,
                calls.clone(),
                None,
                Phase::Complete,
                &Cancellation::new(),
            )
            .await
            .unwrap();
            assert_eq!(*calls.borrow(), counts, "completed owners invoke no method");
        }
    });
}

#[test]
fn admission_can_split_before_from_call_without_a_second_driver() {
    futures_executor::block_on(async {
        let owner = Owner::default();
        let calls = Calls::default();
        run(
            owner.clone(),
            calls.clone(),
            None,
            Phase::Call,
            &Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(*calls.borrow(), [1, 0, 0]);
        run(
            owner.clone(),
            calls.clone(),
            None,
            Phase::Complete,
            &Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(*calls.borrow(), [1, 1, 1]);
        assert_eq!(owner.borrow().phase, Phase::Complete);
    });
}

#[test]
fn shared_methods_do_not_share_progress_between_executions() {
    futures_executor::block_on(async {
        let a = Owner::default();
        let b = Owner::default();
        let calls = Calls::default();
        run(
            a.clone(),
            calls.clone(),
            Some(Phase::After),
            Phase::Complete,
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
        let saved = a.borrow().clone();
        run(
            b.clone(),
            calls.clone(),
            None,
            Phase::Complete,
            &Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(*a.borrow(), saved);
        run(
            a,
            calls.clone(),
            None,
            Phase::Complete,
            &Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(*calls.borrow(), [2, 2, 3]);
        let another_execution = Owner::default();
        run(
            another_execution,
            calls.clone(),
            None,
            Phase::Complete,
            &Cancellation::new(),
        )
        .await
        .unwrap();
        assert_eq!(*calls.borrow(), [3, 3, 4]);
    });
}

#[test]
fn default_hooks_preserve_business_values_and_native_types() {
    #[derive(Debug, PartialEq)]
    struct Native(usize); // Deliberately not Clone: a no-op commit copies nothing.
    futures_executor::block_on(async {
        let result: Result<_, Error> = advance(
            Phase::Before,
            Phase::Complete,
            vec![1, 2],
            None,
            Methods {
                before: |input| async move { Ok(input) },
                call: |input: Vec<i32>| async move { Ok(Native(input.len())) },
                after: |_, output| async move { Ok(output) },
                commit: |_: Phase, _: &Vec<i32>, _: Option<&Native>| async { Ok(()) },
            },
            &Cancellation::new(),
        )
        .await;
        assert_eq!(result.unwrap(), (vec![1, 2], Some(Native(2))));
    });
}

#[test]
fn invalid_phase_facts_do_not_invoke_any_method_or_commit() {
    futures_executor::block_on(async {
        for (from, until, output) in [
            (Phase::Before, Phase::Complete, Some("unexpected".into())),
            (Phase::After, Phase::Complete, None),
            (Phase::Complete, Phase::Before, Some("raw".into())),
        ] {
            let result: Result<_, Error> = advance(
                from,
                until,
                1,
                output,
                Methods {
                    before: |_| async {
                        panic!("invalid scene entered before");
                    },
                    call: |_| async {
                        panic!("invalid scene entered call");
                    },
                    after: |_, _: String| async {
                        panic!("invalid scene entered after");
                    },
                    commit: |_: Phase, _: &i32, _: Option<&String>| async {
                        panic!("invalid scene committed");
                    },
                },
                &Cancellation::new(),
            )
            .await;
            assert_eq!(result.unwrap_err().code, "INVALID_STATE");
        }
    });
}

#[test]
fn failed_acknowledgment_prevents_the_next_business_method() {
    futures_executor::block_on(async {
        let result: Result<_, Error> = advance(
            Phase::Before,
            Phase::Complete,
            1,
            None,
            Methods {
                before: |input| async move { Ok(input + 1) },
                call: |_| async {
                    panic!("unacknowledged before entered call");
                },
                after: |_, output: String| async move { Ok(output) },
                commit: |_: Phase, _: &i32, _: Option<&String>| async {
                    Err(Error::new("BACKEND", "owner rejected facts"))
                },
            },
            &Cancellation::new(),
        )
        .await;
        assert_eq!(
            result.unwrap_err(),
            Error::new("BACKEND", "owner rejected facts")
        );
    });
}

#[test]
fn cancellation_wakes_a_noncooperative_hook_without_committing_its_mutations() {
    futures_executor::block_on(async {
        let c = Cancellation::new();
        let result: Result<_, Error> = advance(
            Phase::Before,
            Phase::Complete,
            1,
            None,
            Methods {
                before: |_: i32| {
                    let c = c.clone();
                    async move {
                        c.cancel();
                        std::future::pending().await
                    }
                },
                call: |_| async {
                    panic!("cancelled before entered call");
                },
                after: |_, output: String| async move { Ok(output) },
                commit: |_: Phase, _: &i32, _: Option<&String>| async {
                    panic!("cancelled hook committed");
                },
            },
            &c,
        )
        .await;
        assert_eq!(result.unwrap_err().code, "CANCELLED");
    });
}

#[test]
fn a_large_business_future_does_not_enlarge_the_shared_driver_frame() {
    async fn business() -> Result<u8, Error> {
        let mut data = [0_u8; 64 * 1024];
        data[0] = 1;
        std::hint::black_box(&data);
        std::future::pending::<()>().await;
        std::hint::black_box(&data);
        Ok(data[0])
    }
    assert!(std::mem::size_of_val(&business()) >= 64 * 1024);
    let c = Cancellation::new();
    let operation = advance(
        Phase::Before,
        Phase::Complete,
        1,
        None,
        Methods {
            before: |input| async move { Ok::<_, Error>(input) },
            call: |_| business(),
            after: |_, value| async move { Ok(value) },
            commit: |_: Phase, _: &i32, _: Option<&u8>| async { Ok(()) },
        },
        &c,
    );
    assert!(
        std::mem::size_of_val(&operation) < 8192,
        "driver must not embed the 64 KiB user future"
    );
    // Inspecting the frame must not execute the pending business operation.
    drop(operation);
}
