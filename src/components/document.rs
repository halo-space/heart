use serde::{Deserialize, Serialize};
use serde_json::Map;

use crate::runtime::cancellation::Cancellation;

pub type Error = crate::Error;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Source {
    pub uri: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Document {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_id: Option<i64>,
    pub content: String,
    #[serde(default)]
    pub metadata: Map<String, serde_json::Value>,
}

/// A derived relation between two Documents. This is distinct from the
/// entity relation returned by `document::graph::Graph`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Relation {
    pub from: i64,
    pub to: i64,
    pub kind: String,
}

pub mod location {
    use serde::{Deserialize, Serialize};

    use super::Error;

    #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
    pub struct Range<T> {
        pub from: T,
        pub to: T,
    }

    #[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
    pub struct Location {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub page: Option<Range<u64>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub paragraph: Option<Range<u64>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub r#char: Option<Range<u64>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub x: Option<Range<f64>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub y: Option<Range<f64>>,
    }

    impl Location {
        pub fn validate(&self) -> Result<(), Error> {
            if let Some(page) = &self.page {
                validate_positive(page.from, "location.page.from")?;
                validate_positive(page.to, "location.page.to")?;
                if page.from > page.to {
                    return Err(invalid("location.page", "range is reversed"));
                }
            }
            if let Some(paragraph) = &self.paragraph {
                validate_positive(paragraph.from, "location.paragraph.from")?;
                validate_positive(paragraph.to, "location.paragraph.to")?;
                if self.page.as_ref().is_none_or(|page| page.from == page.to)
                    && paragraph.from > paragraph.to
                {
                    return Err(invalid("location.paragraph", "range is reversed"));
                }
            }
            if let Some(character) = &self.r#char {
                if self.paragraph.is_none() {
                    return Err(invalid(
                        "location.char",
                        "char range requires paragraph range",
                    ));
                }
                validate_positive(character.from, "location.char.from")?;
                validate_positive(character.to, "location.char.to")?;
                if character.from > character.to {
                    return Err(invalid("location.char", "range is reversed"));
                }
            }
            match (&self.x, &self.y) {
                (Some(x), Some(y)) => {
                    validate_ratio(x, "location.x")?;
                    validate_ratio(y, "location.y")?;
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(invalid(
                        "location",
                        "x and y ranges must be provided together",
                    ));
                }
                (None, None) => {}
            }
            Ok(())
        }
    }

    fn validate_positive(value: u64, field: &str) -> Result<(), Error> {
        if value == 0 {
            Err(invalid(field, "must be greater than zero"))
        } else {
            Ok(())
        }
    }

    fn validate_ratio(range: &Range<f64>, field: &str) -> Result<(), Error> {
        if !range.from.is_finite() || !(0.0..=1.0).contains(&range.from) {
            return Err(invalid(
                &format!("{field}.from"),
                "coordinates must be finite values in [0, 1]",
            ));
        }
        if !range.to.is_finite() || !(0.0..=1.0).contains(&range.to) {
            return Err(invalid(
                &format!("{field}.to"),
                "coordinates must be finite values in [0, 1]",
            ));
        }
        if range.from > range.to {
            return Err(invalid(field, "range is reversed"));
        }
        Ok(())
    }

    fn invalid(field: &str, reason: &str) -> Error {
        Error::new("INVALID_ARGUMENTS", reason).with_details(field)
    }
}

pub mod graph_types {
    use serde::{Deserialize, Serialize};

    use super::{Error, location::Location};

    #[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Method {
        Text,
        Ocr,
    }

    #[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
    pub struct Evidence {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub doc_id: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub method: Option<Method>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub location: Option<Location>,
    }

    impl Evidence {
        pub fn validate(&self) -> Result<(), Error> {
            if self.method.is_some()
                && self
                    .content
                    .as_ref()
                    .is_none_or(|content| content.trim().is_empty())
            {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "evidence method requires non-empty content",
                )
                .with_details("evidence.content"));
            }
            if self
                .content
                .as_ref()
                .is_some_and(|content| content.trim().is_empty())
            {
                return Err(
                    Error::new("INVALID_ARGUMENTS", "evidence content must not be empty")
                        .with_details("evidence.content"),
                );
            }
            if let Some(location) = &self.location {
                location.validate().map_err(|error| {
                    Error::new(error.code, error.message)
                        .with_details(error.details.unwrap_or_else(|| "location".into()))
                })?;
            }
            Ok(())
        }
    }
}

#[allow(async_fn_in_trait)]
pub trait Loader: Send + Sync {
    async fn load(&self, source: &Source, cancellation: &Cancellation) -> Result<Vec<u8>, Error>;
}

#[allow(async_fn_in_trait)]
pub trait Parser: Send + Sync {
    async fn parse(&self, data: &[u8], cancellation: &Cancellation)
    -> Result<Vec<Document>, Error>;
}

#[allow(async_fn_in_trait)]
pub trait Transformer: Send + Sync {
    async fn transform(
        &self,
        documents: Vec<Document>,
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error>;
}

#[allow(async_fn_in_trait)]
pub trait Chunker: Send + Sync {
    /// Split owned Documents into a complete chunk collection. The application
    /// supplies IDs; the component does not initialize its own generator.
    async fn chunk(
        &self,
        documents: Vec<Document>,
        cancellation: &Cancellation,
        next_id: &mut dyn FnMut() -> i64,
    ) -> Result<Vec<Document>, Error>;
}
