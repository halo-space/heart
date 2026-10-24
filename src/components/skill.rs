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

/// A Skill loader owned by a Workspace or another application backend. Local
/// loaders may discover directories; server loaders can implement the same
/// contract using database records. Names returned by `list` identify the
/// exact instructions and resources subsequently read during this Agent run.
pub trait Loader: Send + Sync {
    /// Return a stable source version or fingerprint for the current Skill
    /// catalog. Workspace-backed loaders use this value to reuse a parsed
    /// snapshot across executions and refresh only after the source changes.
    /// A default value derived from the definitions keeps existing loaders
    /// source-compatible; database-backed loaders should override it with a
    /// cheap revision or content fingerprint that also covers Skill bodies
    /// and resources.
    fn version(
        &self,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<String, Error>> + Send {
        async move {
            let definitions = self.list(cancellation).await?;
            serde_json::to_string(&definitions)
                .map_err(|_| Error::new("INTERNAL", "Skill definitions cannot be serialized"))
        }
    }

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
