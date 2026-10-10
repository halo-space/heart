use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::memory::short::Error;
use crate::components::memory::short::Type;
use crate::components::model::token::Usage;
use crate::runtime::cancellation::Cancellation;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Active,
    Archived,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Session {
    pub id: i64,
    pub tenant_id: i64,
    pub user_id: i64,
    pub r#type: Type,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub usage: Option<Usage>,
    pub status: Status,
    /// Summary coverage markers (`from`/`to`) and application extensions.
    pub metadata: Map<String, Value>,
    pub created_time: i64,
    pub updated_time: i64,
}

pub trait Store: Send + Sync {
    async fn create(&self, value: Session, cancellation: &Cancellation) -> Result<Session, Error>;
    async fn read(&self, id: i64, cancellation: &Cancellation) -> Result<Session, Error>;
    async fn update(&self, value: Session, cancellation: &Cancellation) -> Result<Session, Error>;
    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error>;
}
