use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Cancellation;

pub type Error = crate::Error;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Type {
    Summary,
    Fact,
    Preference,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Request {
    pub id: i64,
    pub r#type: Type,
    pub user_id: i64,
    pub agent_id: Option<i64>,
    pub session_id: Option<i64>,
    pub from: i64,
    pub to: i64,
    pub content: String,
    pub metadata: Option<Map<String, Value>>,
    pub vector: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Item {
    pub id: i64,
    pub r#type: Type,
    pub user_id: i64,
    pub agent_id: Option<i64>,
    pub session_id: Option<i64>,
    pub from: i64,
    pub to: i64,
    pub content: String,
    pub version: u64,
    pub created_time: i64,
    pub updated_time: i64,
    pub metadata: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Filter {
    pub user_id: i64,
    pub agent_id: Option<i64>,
    pub session_id: Option<i64>,
    pub r#type: Type,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct KeywordRequest {
    pub query: String,
    #[serde(default)]
    pub options: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VectorRequest {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub options: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HybridRequest {
    pub query: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub options: Map<String, Value>,
}

pub mod keyword {
    pub use super::KeywordRequest as Request;
}

pub mod vector {
    pub use super::VectorRequest as Request;
}

pub mod hybrid {
    pub use super::HybridRequest as Request;
}

pub mod extraction;
pub mod short;

pub trait Memory: Send + Sync {
    fn write(
        &self,
        request: Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Item, Error>> + Send;

    fn update(
        &self,
        request: Request,
        version: u64,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Item, Error>> + Send;

    fn delete(
        &self,
        id: i64,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<(), Error>> + Send;

    fn get(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        r#type: Type,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Option<Item>, Error>> + Send;

    fn list(
        &self,
        user_id: i64,
        agent_id: Option<i64>,
        session_id: Option<i64>,
        r#type: Type,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Item>, Error>> + Send;

    fn keyword(
        &self,
        request: keyword::Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Item>, Error>> + Send;

    fn vector(
        &self,
        request: vector::Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Item>, Error>> + Send;

    fn hybrid(
        &self,
        request: hybrid::Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Item>, Error>> + Send;
}
