use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Cancellation;
use crate::components::memory::short::Error;
use crate::components::memory::short::Type;
use crate::components::message::Message;
use crate::components::model::token::Usage;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Payload {
    #[serde(rename = "type")]
    pub r#type: String,
    pub data: Value,
    pub mime_type: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Request {
    pub metadata: Map<String, Value>,
    pub payload: Vec<Payload>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Running,
    Completed,
    Failed,
    Stopped,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Chat {
    pub id: i64,
    pub tenant_id: i64,
    pub user_id: i64,
    pub agent_id: i64,
    pub session_id: i64,
    pub r#type: Type,
    pub status: Status,
    pub input: Option<Request>,
    pub message: Option<Message>,
    pub usage: Option<Usage>,
    pub graph: Option<Value>,
    pub metadata: Map<String, Value>,
    pub started_time: Option<i64>,
    pub completed_time: Option<i64>,
    pub created_time: i64,
    pub updated_time: i64,
}

pub trait Store: Send + Sync {
    fn create(
        &self,
        value: Chat,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Chat, Error>>;
    fn read(
        &self,
        id: i64,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Chat, Error>>;
    fn update(
        &self,
        value: Chat,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Chat, Error>>;
    fn delete(
        &self,
        id: i64,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<(), Error>>;

    /// Standard history read. Options contain explicit ownership filter,
    /// created_time ordering and pagination; no business-specific list name.
    fn list(
        &self,
        options: Map<String, Value>,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Chat>, Error>> {
        async move {
            let _ = (options, cancellation);
            Err(Error::new(
                "UNSUPPORTED",
                "chat history listing is not supported",
            ))
        }
    }
}
