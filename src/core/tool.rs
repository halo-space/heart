//! Standalone execution of one configured Tool.
//!
//! Runtime owns persisted Tool lifecycle facts and retry decisions. Core only
//! provides this stateless convenience call for callers that need one direct
//! Tool invocation without Runtime state.
use crate::cancellation::cancellable;
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
    let call = prepare_call(tool, call, cancellation).await?;
    let mut message = cancellable(tool.call(&call, cancellation), cancellation).await?;
    cancellable(tool.after(&call, &mut message, cancellation), cancellation).await?;
    let evaluation = cancellable(tool.eval(&call, &message, cancellation), cancellation).await?;
    Ok((message, evaluation))
}
