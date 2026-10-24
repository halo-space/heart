//! Private background Memory scene in Session metadata, not Agent Checkpoint.
//! Only a successful conditional write advances a durable stage. No async Drop,
//! lease, polling loop, independent task store or second in-process truth.
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::components::memory::short::session::Store as _;
use crate::{
    Cancellation, Error,
    memory::short::{self, session},
    model::token::Usage,
};

use super::extraction::{self, Job};

pub(super) const KEY: &str = "memory";

#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(tag = "stage", rename_all = "lowercase", deny_unknown_fields)]
pub(super) enum State {
    Summary {
        base: Box<Base>,
        summary: String,
        usage: Option<Box<Usage>>,
        from: i64,
        to: i64,
        extraction: Option<Job>,
    },
    Extraction {
        job: Job,
    },
}

#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Base {
    summary: Option<String>,
    usage: Option<Usage>,
    from: Option<Value>,
    to: Option<Value>,
}

impl Base {
    pub(super) fn new(value: &session::Session) -> Self {
        Self {
            summary: value.summary.clone(),
            usage: value.usage.clone(),
            from: value.metadata.get("from").cloned(),
            to: value.metadata.get("to").cloned(),
        }
    }
    pub(super) fn matches(&self, value: &session::Session) -> bool {
        self == &Self::new(value)
    }
}

pub(super) fn read(value: &session::Session) -> Result<Option<State>, Error> {
    value
        .metadata
        .get(KEY)
        .map(|encoded| {
            let state: State = serde_json::from_value(encoded.clone()).map_err(|_| invalid())?;
            match &state {
                State::Summary {
                    summary,
                    from,
                    to,
                    extraction,
                    ..
                } => {
                    if summary.trim().is_empty() || *from <= 0 || to < from {
                        return Err(invalid());
                    }
                    if let Some(job) = extraction {
                        job.validate(value.user_id)?;
                        if !job.covers(*from, *to) {
                            return Err(invalid());
                        }
                    }
                }
                State::Extraction { job } => job.validate(value.user_id)?,
            }
            Ok(state)
        })
        .transpose()
}

pub(super) fn encode(value: &mut session::Session, state: Option<&State>) -> Result<(), Error> {
    if let Some(state) = state {
        value.metadata.insert(
            KEY.into(),
            serde_json::to_value(state).map_err(|_| invalid())?,
        );
    } else {
        value.metadata.remove(KEY);
    }
    Ok(())
}

pub(super) struct Journal<S> {
    store: Arc<S>,
    pub(super) current: session::Session,
}

impl<S: short::Store> Journal<S> {
    pub(super) fn new(store: Arc<S>, current: session::Session) -> Self {
        Self { store, current }
    }
    pub(super) async fn update(
        &mut self,
        value: session::Session,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        let committed = self
            .store
            .session()
            .update(value.clone(), cancellation)
            .await?;
        let mut acknowledged = committed.clone();
        acknowledged.updated_time = value.updated_time;
        if acknowledged != value || committed.updated_time <= value.updated_time {
            // A malformed or lost acknowledgement stops this worker. The next
            // request reads the actual committed scene, never a guessed stage.
            return Err(Error::new(
                "INVALID_RESPONSE",
                "Session did not acknowledge Memory progress",
            ));
        }
        self.current = committed;
        Ok(())
    }
    pub(super) async fn save(
        &mut self,
        state: Option<&State>,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        let mut updated = self.current.clone();
        encode(&mut updated, state)?;
        self.update(updated, cancellation).await
    }
}

impl<S: short::Store> extraction::Progress for Journal<S> {
    async fn save(&mut self, job: &Job, cancellation: &Cancellation) -> Result<(), Error> {
        Journal::save(
            self,
            Some(&State::Extraction { job: job.clone() }),
            cancellation,
        )
        .await
    }
}

fn invalid() -> Error {
    Error::new("INVALID_STATE", "invalid Session background Memory scene")
}
