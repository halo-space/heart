//! Native Tool binding for the shared lifecycle driver. No Runtime storage.
use super::lifecycle::{Methods, Phase, advance};
use crate::components::tool::{Call, Evaluation, Tool, prepare_call};
use crate::{Cancellation, Error, Message};

/// Run one Tool's before/call/after/eval lifecycle without retaining Runtime
/// state or retrying. After a successful call, a caller requiring eval retries
/// must retain that result and retry eval alone, not this convenience entry.
pub async fn execute<T: Tool>(
    tool: &T,
    call: Call,
    cancellation: &Cancellation,
) -> Result<(Message, Evaluation), Error> {
    let (call, message) = call_once(tool, call, cancellation).await?;
    let evaluation =
        super::operation::cancellable(tool.eval(&call, &message, cancellation), cancellation)
            .await?;
    Ok((message, evaluation))
}

/// Workflow binds this same operation without enabling Agent business eval.
/// Recovery facts for complete calls remain owned by its actual Attempt.
pub(crate) async fn call_once<T: Tool>(
    tool: &T,
    call: Call,
    cancellation: &Cancellation,
) -> Result<(Call, Message), Error> {
    call_from(
        tool,
        Phase::Before,
        call,
        None,
        |_, _, _| async { Ok(()) },
        cancellation,
    )
    .await
}

/// The same Tool binding with an owner's acknowledged phase/result. No
/// storage or retry policy is introduced by this internal entry.
pub(crate) async fn call_from<T, K, KF>(
    tool: &T,
    phase: Phase,
    call: Call,
    message: Option<Message>,
    commit: K,
    cancellation: &Cancellation,
) -> Result<(Call, Message), Error>
where
    T: Tool,
    K: FnMut(Phase, &Call, Option<&Message>) -> KF,
    KF: std::future::Future<Output = Result<(), Error>>,
{
    let (call, message) = advance(
        phase,
        Phase::Complete,
        call,
        message,
        Methods {
            before: |call| prepare_call(tool, call, cancellation),
            call: |call: Call| async move { tool.call(&call, cancellation).await },
            after: |call: Call, mut message: Message| async move {
                tool.after(&call, &mut message, cancellation).await?;
                Ok(message)
            },
            commit,
        },
        cancellation,
    )
    .await?;
    Ok((call, message.expect("completed Tool call has a result")))
}
