#![allow(unused_imports)]
use futures_executor::block_on;
use halo::agent::core::provider;
use halo::agent::react::execution;
use halo::agent::react::*;
use halo::agent::subagent::cancel_ready;
use halo::agent::*;
use std::future::Future;
use std::task::{Context, Poll};
mod unit {
    use super::*;
    include!("unit/agent_progress.rs");
}
