use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Cancellation;

pub type Error = crate::Error;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Definition {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// Instructions and resource information for an Agent, not an executable
/// function. Reading a Skill never grants access to tools or owns Runtime state.
pub trait Skill: Send + Sync {
    fn definition(&self) -> &Definition;

    fn read(
        &self,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<String, Error>> + Send;
}

/// A Workspace-owned Skill source. Local workspaces may discover directories;
/// server workspaces can implement the same contract using database records.
/// Names returned by `list` identify the exact instructions and resources
/// subsequently read during this Agent run.
pub trait Source: Send + Sync {
    fn list(
        &self,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Definition>, Error>> + Send;

    fn read(
        &self,
        name: &str,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<String, Error>> + Send;

    fn read_resource(
        &self,
        name: &str,
        path: &str,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<u8>, Error>> + Send;
}
