use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::components::memory::short::Error;
use crate::runtime::cancellation::Cancellation;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Trace {
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub session_id: Option<i64>,
    pub agent_id: Option<i64>,
    pub execution: Value,
    pub created_time: i64,
    pub updated_time: i64,
}

pub trait Store: Send + Sync {
    async fn create(&self, value: Trace, cancellation: &Cancellation) -> Result<Trace, Error>;
    async fn read(&self, chat_id: i64, cancellation: &Cancellation) -> Result<Trace, Error>;
    async fn update(&self, value: Trace, cancellation: &Cancellation) -> Result<Trace, Error>;
    async fn delete(&self, chat_id: i64, cancellation: &Cancellation) -> Result<(), Error>;
}
