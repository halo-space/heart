#![allow(unused_imports)]

use futures_executor::block_on;
use futures_util::stream;
use halo::agent::checkpoint::Checkpoint;
use halo::agent::memory;
use halo::agent::message::Role;
use halo::agent::model;
use halo::agent::node::{Node, Signal, View, Wait as NodeWait};
use halo::agent::tool::{Call, Tool, ToolDefinition};
use halo::agent::workflow::graph::{END, START};
use halo::agent::workflow::*;
use halo::agent::{Agent, Cancellation, Error, Input as AgentInput, Message, Messages};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

mod unit {
    use super::*;
    include!("unit/workflow_runtime.rs");
}
