//! Binding of run preparation to this Agent's native scene. No persistence.
use super::hooks::{Hooks, Stage as Hook};
use crate::runtime::agent::history::{History, Run};
use crate::runtime::agent::preparation::{Stage, State};
use crate::{Cancellation, Error, Messages, agent::Input};

/// Valid preparation failures close the same original Chat. Invalid identity
/// rewrites are rejected before creating it. Restores never create another
/// Chat or interpret a continuation payload as model input here.
pub(crate) async fn advance(
    state: &mut State,
    query: &Input,
    messages: &mut Messages,
    run: &mut Run,
    history: &History,
    hooks: &Hooks,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    let starting = state.stage == Stage::Start;
    if !starting {
        *run = crate::runtime::cancellation::cancellable(
            Box::pin(history.resume(query.clone(), cancellation)),
            cancellation,
        )
        .await?;
    }
    if matches!(state.stage, Stage::Start | Stage::Execute) {
        let mut invalid_identity = false;
        let result = hooks
            .apply_from(
                Hook::Execute,
                true,
                state.input.clone(),
                state.idx,
                |idx, input| {
                    if let Err(error) = history.check_identity(query, &input) {
                        invalid_identity = true;
                        return Err(error);
                    }
                    state.input = input;
                    state.idx = idx;
                    Ok(())
                },
                cancellation,
            )
            .await;
        if invalid_identity {
            return result.map(|_| ());
        }
        if starting {
            // A cancelled Hook still has a real stopped run to persist. This
            // is bookkeeping, not permission to continue business execution.
            let close_cancellation = Cancellation::new();
            let record_cancellation = if result
                .as_ref()
                .is_err_and(|error| error.code == "CANCELLED")
            {
                &close_cancellation
            } else {
                cancellation
            };
            *run = crate::runtime::cancellation::cancellable(
                Box::pin(history.begin(query.clone(), record_cancellation)),
                record_cancellation,
            )
            .await?;
            state.stage = Stage::Execute;
        }
        result?;
        state.next(Stage::BeforePrompt);
    }
    history.check_identity(query, &state.input)?;
    for (stage, before, next) in [
        (Stage::BeforePrompt, true, Stage::AfterPrompt),
        (Stage::AfterPrompt, false, Stage::History),
    ] {
        if state.stage == stage {
            hooks
                .apply_from(
                    Hook::SystemPrompt,
                    before,
                    state.prompt.clone(),
                    state.idx,
                    |idx, prompt| {
                        state.prompt = prompt;
                        state.idx = idx;
                        Ok(())
                    },
                    cancellation,
                )
                .await?;
            state.next(next);
        }
    }
    if state.stage == Stage::History {
        let recalled = history
            .load(state.input.metadata.clone(), hooks.clone(), cancellation)
            .await?;
        *messages =
            super::chat::initial_messages(state.input.clone(), state.prompt.as_deref(), recalled);
        state.next(Stage::Complete);
    }
    Ok(())
}
