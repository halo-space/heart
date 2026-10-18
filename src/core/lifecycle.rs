//! Shared, stateless advancement of an owner's native lifecycle facts.
//!
//! Inputs below are working copies read from the actual execution. `commit`
//! hands successful facts back to that same owner before the next method is
//! entered. This module owns no execution identities, storage, retries or
//! global progress. Ranges permit admission/authorization between phases.
use std::future::Future;

use super::operation::cancellable as interruptible;
use crate::{Cancellation, Error};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Phase {
    Before,
    Call,
    After,
    Complete,
}

pub(crate) struct Methods<B, C, A, K> {
    pub before: B,
    pub call: C,
    pub after: A,
    pub commit: K,
}

/// Advance only the unfinished methods in `[from, until)`. Business input and
/// output types remain native; omitted Hooks are ordinary identity methods.
/// Commit borrows successful values; the owner copies only what its native
/// scene needs. A stateless caller need not clone large outputs for a no-op ack.
pub(crate) async fn advance<I, O, E, B, BF, C, CF, A, AF, K, KF>(
    from: Phase,
    until: Phase,
    mut input: I,
    mut output: Option<O>,
    mut methods: Methods<B, C, A, K>,
    cancellation: &Cancellation,
) -> Result<(I, Option<O>), E>
where
    I: Clone,
    E: From<Error>,
    B: FnMut(I) -> BF,
    BF: Future<Output = Result<I, E>>,
    C: FnMut(I) -> CF,
    CF: Future<Output = Result<O, E>>,
    A: FnMut(I, O) -> AF,
    AF: Future<Output = Result<O, E>>,
    K: FnMut(Phase, &I, Option<&O>) -> KF,
    KF: Future<Output = Result<(), E>>,
{
    // Reject corrupt/inconsistent owner facts before even invoking a Hook.
    if from > until || (from >= Phase::After) != output.is_some() {
        return Err(Error::new("INVALID_STATE", "invalid lifecycle phase or result").into());
    }
    check(cancellation)?;
    let mut phase = from;
    while phase < until {
        check(cancellation)?;
        // User futures can own large request/Step scenes. Keep unfinished
        // methods heap-backed so nested Agent/Workflow frames do not multiply
        // their inline size on the default thread stack. This spawns no task.
        match phase {
            Phase::Before => {
                input = interruptible(Box::pin((methods.before)(input)), cancellation).await?;
                phase = Phase::Call;
            }
            Phase::Call => {
                output = Some(
                    interruptible(Box::pin((methods.call)(input.clone())), cancellation).await?,
                );
                phase = Phase::After;
            }
            Phase::After => {
                output = Some(
                    interruptible(
                        Box::pin((methods.after)(
                            input.clone(),
                            output.take().expect("validated result"),
                        )),
                        cancellation,
                    )
                    .await?,
                );
                phase = Phase::Complete;
            }
            Phase::Complete => unreachable!("complete is the last phase"),
        }
        interruptible(
            Box::pin((methods.commit)(phase, &input, output.as_ref())),
            cancellation,
        )
        .await?;
    }
    Ok((input, output))
}

/// Stop at the authorization/admission boundary while using the same driver.
/// A restored Call phase skips before and does not send another completion ack.
pub(crate) async fn prepare<I, E, B, BF, K, KF>(
    from: Phase,
    input: I,
    before: B,
    mut commit: K,
    cancellation: &Cancellation,
) -> Result<I, E>
where
    I: Clone,
    E: From<Error>,
    B: FnMut(I) -> BF,
    BF: Future<Output = Result<I, E>>,
    K: FnMut(I) -> KF,
    KF: Future<Output = Result<(), E>>,
{
    advance(
        from,
        Phase::Call,
        input,
        None::<()>,
        Methods {
            before,
            call: |_| async { Ok(()) },
            after: |_, ()| async { Ok(()) },
            commit: move |_: Phase, input: &I, _: Option<&()>| commit(input.clone()),
        },
        cancellation,
    )
    .await
    .map(|(input, _)| input)
}

fn check<E: From<Error>>(cancellation: &Cancellation) -> Result<(), E> {
    super::operation::check(cancellation).map_err(Into::into)
}

#[cfg(test)]
#[path = "../../tests/unit/lifecycle.rs"]
mod tests;
