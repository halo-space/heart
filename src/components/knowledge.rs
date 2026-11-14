use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Cancellation;
use crate::components::document::Document;

pub type Error = crate::Error;

pub mod keyword {
    use super::*;

    #[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
    pub struct Request {
        pub query: String,
        #[serde(default)]
        pub options: Map<String, Value>,
    }

    impl Request {
        pub fn validate(&self) -> Result<(), Error> {
            if self.query.trim().is_empty() {
                return Err(Error::new("INVALID_ARGUMENTS", "keyword query is empty"));
            }
            Ok(())
        }
    }
}

pub mod vector {
    use super::*;

    #[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
    pub struct Request {
        pub vector: Vec<f32>,
        #[serde(default)]
        pub options: Map<String, Value>,
    }

    impl Request {
        pub fn validate(&self) -> Result<(), Error> {
            if self.vector.is_empty() || self.vector.iter().any(|value| !value.is_finite()) {
                return Err(Error::new("INVALID_ARGUMENTS", "search vector is invalid"));
            }
            Ok(())
        }
    }
}

pub mod hybrid {
    use super::*;

    #[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
    pub struct Request {
        pub query: String,
        pub vector: Vec<f32>,
        #[serde(default)]
        pub options: Map<String, Value>,
    }

    impl Request {
        pub fn validate(&self) -> Result<(), Error> {
            if self.query.trim().is_empty() {
                return Err(Error::new("INVALID_ARGUMENTS", "hybrid query is empty"));
            }
            if self.vector.is_empty() || self.vector.iter().any(|value| !value.is_finite()) {
                return Err(Error::new("INVALID_ARGUMENTS", "hybrid vector is invalid"));
            }
            Ok(())
        }
    }
}

/// Search is a backend capability. It does not own an Embed instance and does
/// not choose storage; callers provide vectors before invoking vector/hybrid.
pub trait Search: Send + Sync {
    fn keyword(
        &self,
        request: keyword::Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Document>, Error>> + Send;

    fn vector(
        &self,
        request: vector::Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Document>, Error>> + Send;

    fn hybrid(
        &self,
        request: hybrid::Request,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<Vec<Document>, Error>> + Send;
}

#[cfg(test)]
#[path = "../../tests/unit/embedded/src_components_knowledge_tests.rs"]
mod tests;
