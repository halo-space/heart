use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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

/// Common workspace identity and assembly boundary. The selected concrete
/// implementation owns file or database operations; bound tools expose their
/// native argument schemas through the existing Tool contract. A database
/// workspace does not need to emulate file paths or byte-oriented methods.
pub trait Workspace: Send + Sync {
    fn definition(&self) -> &Definition;
}
