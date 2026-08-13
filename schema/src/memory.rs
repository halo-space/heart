use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Error {
    InvalidInput,
    NotFound,
    Storage,
    Model,
    InvalidOutput,
    Conflict,
    Timeout,
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_serialize_as_stable_categories() {
        assert_eq!(
            serde_json::to_value(Error::InvalidOutput).expect("error must serialize"),
            "invalid_output"
        );
    }
}
