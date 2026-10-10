use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::runtime::cancellation::Cancellation;

pub type Error = crate::Error;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Local,
    Server,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Definition {
    pub mode: Mode,
    /// Required only by local implementations. Server implementations keep
    /// resources behind their service and leave this unset.
    pub path: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// Workspace persistence boundary. The framework does not prescribe a local
/// filesystem layout or a server database schema.
#[allow(async_fn_in_trait)]
pub trait Workspace: Send + Sync {
    fn definition(&self) -> &Definition;

    async fn read(&self, path: &str, cancellation: &Cancellation) -> Result<Vec<u8>, Error>;

    async fn write(
        &self,
        path: &str,
        data: Vec<u8>,
        cancellation: &Cancellation,
    ) -> Result<(), Error>;

    async fn delete(&self, path: &str, cancellation: &Cancellation) -> Result<(), Error>;
}
