//! A2A task projection contracts.
//!
//! The task store is separate from Runtime State and Checkpoint persistence.

use std::future::Future;

use crate::Error;
use crate::components::protocol::a2a::{Context, Message, StreamResponse, Task};

pub mod memory;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageState {
    Accepted,
    Duplicate,
}

/// Storage contract for the externally visible A2A Task projection.
///
/// Runtime State and Checkpoint data stay outside this contract. The default
/// `memory::Memory` implementation is enough for local use; applications can provide
/// a database-backed implementation with the same operations.
pub trait Store: Clone + Send + Sync + 'static {
    fn create(
        &self,
        session_id: i64,
        chat_id: i64,
    ) -> impl Future<Output = Result<Context, Error>> + Send;
    fn task(&self, task_id: &str) -> impl Future<Output = Result<Task, Error>> + Send;
    fn context(&self, task_id: &str) -> impl Future<Output = Result<Context, Error>> + Send;
    fn message(
        &self,
        task_id: &str,
        message: &Message,
    ) -> impl Future<Output = Result<MessageState, Error>> + Send;
    fn record(
        &self,
        task_id: &str,
        response: StreamResponse,
    ) -> impl Future<Output = Result<(), Error>> + Send;
}
