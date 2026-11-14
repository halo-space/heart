use super::Cancellation;
use std::task::Waker;

impl Cancellation {
    fn waiter_count(&self) -> usize {
        self.state
            .waiters
            .lock()
            .expect("cancel lock poisoned")
            .len()
    }
}

#[test]
fn cancellation_is_shared_within_a_run_but_isolated_between_runs() {
    let first_run = Cancellation::new();
    let first_run_child = first_run.clone();
    let second_run = Cancellation::new();

    first_run.cancel();

    assert!(first_run_child.is_cancelled());
    assert!(!second_run.is_cancelled());
}

#[test]
fn a_finished_waiter_is_removed_from_the_cancellation_scope() {
    let cancellation = Cancellation::new();
    let waker = Waker::noop();
    let waiter = cancellation.register(waker);
    assert_eq!(cancellation.waiter_count(), 1);
    drop(waiter);
    assert_eq!(cancellation.waiter_count(), 0);
}
