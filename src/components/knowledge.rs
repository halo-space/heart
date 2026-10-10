use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::components::document::Document;
use crate::runtime::cancellation::Cancellation;

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
#[allow(async_fn_in_trait)]
pub trait Search: Send + Sync {
    async fn keyword(
        &self,
        request: keyword::Request,
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error>;

    async fn vector(
        &self,
        request: vector::Request,
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error>;

    async fn hybrid(
        &self,
        request: hybrid::Request,
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error>;
}

#[cfg(test)]
mod tests {
    use super::{hybrid, keyword, vector};

    #[test]
    fn search_modes_validate_their_own_query_shape() {
        assert!(
            keyword::Request {
                query: "北京".into(),
                ..Default::default()
            }
            .validate()
            .is_ok()
        );
        assert!(keyword::Request::default().validate().is_err());
        assert!(
            vector::Request {
                vector: vec![0.1, 0.2],
                ..Default::default()
            }
            .validate()
            .is_ok()
        );
        assert!(vector::Request::default().validate().is_err());
        assert!(
            hybrid::Request {
                query: "北京".into(),
                vector: vec![0.1],
                ..Default::default()
            }
            .validate()
            .is_ok()
        );
        assert!(hybrid::Request::default().validate().is_err());
    }
}
