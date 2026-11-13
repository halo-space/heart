//! Static Team composition around the ReAct Agent runtime.
//!
//! Team is a Leader plus a fixed set of Agent members. The Leader remains a
//! normal ReAct Agent; this module supplies the member roster and the static
//! `team_assign` delegation / `agent_handoff` transfer relations.

use std::pin::Pin;

use futures_core::Stream;

use crate::Cancellation;
use crate::components::agent::{Error, Input};
use crate::components::event::Event;
use crate::components::model::chat::Chat;
use crate::runtime::agent::react;
use crate::runtime::agent::subagent::SubAgent;

/// Start building a static Team with a ReAct Leader.
pub fn builder() -> Builder {
    Builder {
        leader: None,
        members: Vec::new(),
        handoffs: Vec::new(),
        max_concurrency: None,
        id_fn: None,
    }
}

/// Team builder. The Leader must be an unbuilt ReAct builder so the runtime
/// can install the fixed member roster before the model request contract is
/// compiled.
pub struct Builder<M = ()> {
    leader: Option<react::Builder<M>>,
    members: Vec<SubAgent>,
    handoffs: Vec<(String, String)>,
    max_concurrency: Option<usize>,
    id_fn: Option<fn() -> i64>,
}

impl<M> Builder<M> {
    /// Set the ReAct builder used by the Team Leader.
    pub fn leader<N>(self, leader: react::Builder<N>) -> Builder<N> {
        Builder {
            leader: Some(leader),
            members: self.members,
            handoffs: self.handoffs,
            max_concurrency: self.max_concurrency,
            id_fn: self.id_fn,
        }
    }

    /// Add one fixed member. Member names must be unique and are validated
    /// when the Team is built.
    pub fn member(mut self, member: SubAgent) -> Self {
        self.members.push(member);
        self
    }

    /// Declare a static control transfer from the Team Leader to one member.
    ///
    /// Handoff is deliberately separate from `team_assign`: after the target
    /// completes, the Leader does not resume and the target's final result
    /// closes the current Team run. The relation is validated during `build`.
    pub fn handoff(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.handoffs.push((from.into(), to.into()));
        self
    }

    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = Some(max_concurrency);
        self
    }

    pub fn with_id_fn(mut self, id_fn: fn() -> i64) -> Self {
        self.id_fn = Some(id_fn);
        self
    }
}

impl<M> Builder<M>
where
    M: Chat + 'static,
{
    /// Build a Team whose execution is owned by the existing ReAct runtime.
    pub fn build(self) -> Result<Agent<M>, Error> {
        let mut leader = self
            .leader
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "team requires a ReAct leader"))?;
        if let Some(max_concurrency) = self.max_concurrency {
            leader = leader.with_max_concurrency(max_concurrency);
        }
        if let Some(id_fn) = self.id_fn {
            leader = leader.with_id_fn(id_fn);
        }
        let targets = validate_handoffs(&self.members, &self.handoffs)?;
        let leader = leader
            .with_team_members(self.members)
            .with_team_handoffs(targets)
            .build()?;
        Ok(Agent { inner: leader })
    }
}

fn validate_handoffs(
    members: &[SubAgent],
    handoffs: &[(String, String)],
) -> Result<Vec<String>, Error> {
    let names = members
        .iter()
        .map(|member| member.name().to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    if names.contains("leader") {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "Team member name leader is reserved",
        ));
    }
    let mut targets = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for (from, to) in handoffs {
        if from != "leader" {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "Team Handoff source must be leader",
            ));
        }
        if to.trim().is_empty() || !names.contains(to) {
            return Err(Error::new(
                "NOT_FOUND",
                format!("Team Handoff target is not registered: {to}"),
            ));
        }
        if !seen.insert(to) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "Team Handoff target must be unique",
            ));
        }
        targets.push(to.clone());
    }
    Ok(targets)
}

/// A static Team exposed through the same Agent contract as Chat and ReAct.
pub struct Agent<M> {
    inner: react::Agent<M>,
}

impl<M> crate::components::agent::Agent for Agent<M>
where
    M: Chat + 'static,
{
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send + 'static>>;

    async fn run(
        &self,
        input: Input,
        stream: bool,
        cancellation: Cancellation,
    ) -> Result<Self::Stream, Error> {
        self.inner.run(input, stream, cancellation).await
    }

    async fn reply(&self, input: Input, cancellation: Cancellation) -> Result<(), Error> {
        self.inner.reply(input, cancellation).await
    }
}
