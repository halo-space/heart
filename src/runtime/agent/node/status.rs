use serde::{Deserialize, Serialize};

/// Lifecycle of one logical Agent plan node.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    #[default]
    Pending,
    Ready,
    Running,
    Waiting,
    Succeeded,
    Failed,
    Cancelled,
}

/// Lifecycle of one internal Agent operation.
pub mod step {
    use serde::{Deserialize, Serialize};

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

    impl Status {
        pub(crate) fn as_str(self) -> &'static str {
            match self {
                Self::Running => "running",
                Self::Waiting => "waiting",
                Self::Succeeded => "succeeded",
                Self::Failed => "failed",
                Self::Cancelled => "cancelled",
            }
        }
    }
}

/// Lifecycle of one Agent question and its feedback collection.
pub mod wait {
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Status {
        #[default]
        Waiting,
        Resolved,
    }

    impl Status {
        pub(crate) fn as_str(self) -> &'static str {
            match self {
                Self::Waiting => "waiting",
                Self::Resolved => "resolved",
            }
        }
    }
}
