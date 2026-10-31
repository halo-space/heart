//! Static Team composition around the ReAct Agent runtime.
//!
//! Team is a Leader plus a fixed set of Agent members. The Leader remains a
//! normal ReAct Agent; this module only supplies the member roster and makes
//! the runtime use the private `team_assign` delegation operation.

use std::pin::Pin;

use futures_core::Stream;

use crate::Cancellation;
use crate::components::agent::{Agent as AgentContract, Error, Input};
use crate::components::event::Event;
use crate::components::model::chat::Chat;
use crate::runtime::agent::react;
use crate::runtime::agent::subagent::SubAgent;

/// Start building a static Team with a ReAct Leader.
pub fn builder() -> Builder {
    Builder {
        leader: None,
        members: Vec::new(),
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
    max_concurrency: Option<usize>,
    id_fn: Option<fn() -> i64>,
}

impl<M> Builder<M> {
    /// Set the ReAct builder used by the Team Leader.
    pub fn leader<N>(self, leader: react::Builder<N>) -> Builder<N> {
        Builder {
            leader: Some(leader),
            members: self.members,
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
        let leader = leader.with_team_members(self.members).build()?;
        Ok(Agent { inner: leader })
    }
}

/// A static Team exposed through the same Agent contract as Chat and ReAct.
pub struct Agent<M> {
    inner: react::Agent<M>,
}

impl<M> AgentContract for Agent<M>
where
    M: Chat + 'static,
{
    type Stream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + 'static>>;

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
