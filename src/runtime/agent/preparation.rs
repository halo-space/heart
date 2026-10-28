//! Native preparation facts of this Agent execution, never component-global.
use serde::{Deserialize, Serialize};

use crate::runtime::agent::history::{History, Run};
use crate::runtime::agent::hooks::{Hooks, Stage as Hook};
use crate::{Cancellation, Error, Messages, agent::Input};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// New invocation; no Memory Chat has been acknowledged yet.
    Start,
    Execute,
    BeforePrompt,
    AfterPrompt,
    History,
    Complete,
}

/// Successful working input and prompt are distinct from the original query.
/// The cursor acknowledges completed middleware methods in the current stage.
/// Embedded Agents keep this inside their actual parent's Attempt.agent.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct State {
    pub stage: Stage,
    pub idx: usize,
    pub input: Input,
    pub prompt: Option<String>,
}

impl State {
    pub(crate) fn validate(&self, middleware_count: usize, query: &Input) -> Result<(), Error> {
        if self.idx > middleware_count
            || (!matches!(
                self.stage,
                Stage::Start | Stage::Execute | Stage::BeforePrompt | Stage::AfterPrompt
            ) && self.idx != 0)
            || ["tenant_id", "user_id", "agent_id", "session_id", "chat_id"]
                .into_iter()
                .any(|key| self.input.metadata.get(key) != query.metadata.get(key))
        {
            return Err(Error::new(
                "INVALID_STATE",
                "Agent preparation does not match its execution",
            ));
        }
        Ok(())
    }

    pub(crate) fn new(input: Input, prompt: Option<String>) -> Self {
        Self {
            stage: Stage::Start,
            idx: 0,
            input,
            prompt,
        }
    }

    pub(crate) fn restore(snapshot: &serde_json::Value) -> Result<Option<Self>, Error> {
        snapshot
            .get("preparation")
            .filter(|value| !value.is_null())
            .map(|value| {
                serde_json::from_value(value.clone())
                    .map_err(|_| Error::new("INVALID_STATE", "invalid Agent preparation"))
            })
            .transpose()
    }

    pub(crate) fn next(&mut self, stage: Stage) {
        self.stage = stage;
        self.idx = 0;
    }
}

/// Bind preparation middleware and history to this Agent's native scene.
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
