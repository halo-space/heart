use serde::{Deserialize, Serialize};

/// Lifecycle of one complete Agent run.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    #[default]
    Running,
    Waiting,
    Succeeded,
    Failed,
    Cancelled,
}
