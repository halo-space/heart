//! Backend 错误类型与公共类型。

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("document not found: index={index} id={id}")]
    NotFound { index: String, id: String },

    #[error("index not found: {0}")]
    IndexNotFound(String),

    #[error("conflict: index={index} id={id} reason={reason}")]
    Conflict {
        index: String,
        id: String,
        reason: String,
    },

    #[error("serialize error: {0}")]
    Serialize(#[from] serde_json::Error),

    #[error("transport error: {0}")]
    Transport(String),

    #[error("es non-2xx: status={status} body={body}")]
    Es {
        status: u16,
        body: serde_json::Value,
    },

    #[error("bulk partial failure: {} item(s) failed", failures.len())]
    BulkPartialFailure { failures: Vec<BulkItemFailure> },
}

#[derive(Debug, Clone)]
pub struct BulkItemFailure {
    pub action: BulkAction,
    pub id: String,
    pub status: u16,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BulkAction {
    Create,
    Update,
    Delete,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_not_found() {
        let e = Error::NotFound {
            index: "i".into(),
            id: "x".into(),
        };
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn display_conflict() {
        let e = Error::Conflict {
            index: "i".into(),
            id: "x".into(),
            reason: "r".into(),
        };
        assert!(e.to_string().contains("conflict"));
    }

    #[test]
    fn display_transport() {
        let e = Error::Transport("boom".into());
        assert!(e.to_string().contains("transport"));
    }

    #[test]
    fn display_bulk_partial() {
        let e = Error::BulkPartialFailure {
            failures: vec![BulkItemFailure {
                action: BulkAction::Create,
                id: "a".into(),
                status: 409,
                reason: "dup".into(),
            }],
        };
        assert!(e.to_string().contains("bulk partial failure"));
    }

    #[test]
    fn serde_json_error_converts_into_serialize_variant() {
        let parse_err = serde_json::from_str::<i32>("not int").unwrap_err();
        let e: Error = parse_err.into();
        assert!(matches!(e, Error::Serialize(_)));
    }
}
