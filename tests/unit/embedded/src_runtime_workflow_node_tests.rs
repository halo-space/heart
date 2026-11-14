use super::*;

#[test]
fn preserves_attempt_history_and_rejects_gaps() {
    let mut state = State::new(status::Status::Active);
    state
        .insert(Execution {
            exec_id: 1,
            ..Execution::default()
        })
        .unwrap();
    state
        .append_attempt(
            1,
            Attempt {
                idx: 0,
                status: status::attempt::Status::Failed,
                ..Attempt::default()
            },
        )
        .unwrap();
    assert_eq!(state.executions[&1].attempts.len(), 1);
    assert_eq!(
        state
            .append_attempt(
                1,
                Attempt {
                    idx: 2,
                    status: status::attempt::Status::Running,
                    ..Attempt::default()
                },
            )
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
}

#[test]
fn rejects_duplicate_and_non_positive_execution_ids() {
    let mut state = State::new(status::Status::Idle);
    assert_eq!(
        state
            .insert(Execution {
                exec_id: 0,
                ..Execution::default()
            })
            .unwrap_err()
            .code,
        "INVALID_ARGUMENTS"
    );
    state
        .insert(Execution {
            exec_id: 2,
            ..Execution::default()
        })
        .unwrap();
    assert_eq!(
        state
            .insert(Execution {
                exec_id: 2,
                ..Execution::default()
            })
            .unwrap_err()
            .code,
        "CONFLICT"
    );
}
