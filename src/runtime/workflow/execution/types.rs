//! Private invocation facts shared by the Workflow execution phases.
//!
//! Keeping these result variants together prevents the execution loop,
//! retry driver and nested graph adapter from each defining a second outcome
//! vocabulary. They are runtime-internal and never part of the public
//! Workflow contract.

use std::sync::Arc;

use serde_json::Value;

use crate::Error;
use crate::runtime::workflow::CompiledWorkflow;
use crate::runtime::workflow::graph;
use crate::runtime::workflow::node::Attempt;

#[derive(Debug)]
pub(crate) struct Invocation {
    pub attempts: Vec<Attempt>,
    pub outcome: Outcome,
    pub graph: Option<Box<graph::State>>,
}

#[derive(Debug)]
pub(crate) enum Outcome {
    Complete(Value),
    Wait(String),
    Waiting,
    Failed(Error),
    Cancelled,
}

pub(crate) enum Target {
    Node(Arc<dyn graph::ErasedNode + 'static>),
    Agent(Arc<dyn graph::ErasedAgent + 'static>, Option<Value>),
    Graph {
        workflow: Arc<CompiledWorkflow>,
        state: Option<Box<graph::State>>,
        history: Option<Box<graph::State>>,
    },
}
