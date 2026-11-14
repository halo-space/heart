use serde::{Deserialize, Serialize};

/// Lifecycle of one Workflow node.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    #[default]
    Idle,
    Active,
    Succeeded,
    Failed,
    Cancelled,
}

/// Lifecycle of one Workflow execution.
pub mod execution {
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Status {
        #[default]
        Pending,
        Running,
        Waiting,
        Succeeded,
        Failed,
        Cancelled,
    }
}

/// Lifecycle of one Workflow attempt.
pub mod attempt {
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Status {
        #[default]
        Running,
        Waiting,
        Succeeded,
        Failed,
        Cancelled,
    }
}

/// Lifecycle of one Workflow node step.
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

/// Lifecycle of one Workflow external wait.
pub mod wait {
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Status {
        #[default]
        Waiting,
        Resolved,
    }
}
