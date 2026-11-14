#![allow(unused_imports)]
use crate::runtime::agent::subagent::cancel_ready;
use futures_executor::block_on;
use halo::agent::core::provider;
use halo::agent::react::execution;
use halo::agent::react::*;
use halo::agent::*;
use std::future::Future;
use std::task::{Context, Poll};

mod unit {
    use super::*;
    include!("../agent_progress.rs");
}
