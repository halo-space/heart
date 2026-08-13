use serde::{Deserialize, Serialize};

use crate::Metadata;

use super::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scenario {
    #[default]
    General,
    Qa,
    Manual,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Wiki {
    pub id: u64,
    pub doc_id: u64,
    pub version: Option<String>,
    pub scenario: Option<Scenario>,
    pub idx: Option<usize>,
    pub header: String,
    pub content: String,
    pub keywords: Vec<String>,
    pub questions: Vec<String>,
    pub tags: Vec<String>,
    pub attributes: Metadata,
    pub spans: Vec<Span>,
    pub graph: Graph,
    pub content_tokens: String,
    pub keyword_tokens: String,
    pub question_tokens: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    pub metadata: Metadata,
    pub images: Vec<String>,
}

impl Wiki {
    pub fn validate(&self) -> Result<(), Error> {
        if self.id == 0 || self.doc_id == 0 || self.content.trim().is_empty() {
            return Err(Error::InvalidInput);
        }

        Ok(())
    }

    pub fn validate_vector(&self) -> Result<(), Error> {
        self.validate()?;

        if self.embedding.as_ref().is_none_or(|embedding| {
            embedding.is_empty() || embedding.iter().any(|value| !value.is_finite())
        }) {
            return Err(Error::InvalidInput);
        }

        Ok(())
    }
}

impl Default for Wiki {
    fn default() -> Self {
        Self {
            id: 0,
            doc_id: 0,
            version: None,
            scenario: None,
            idx: None,
            header: String::new(),
            content: String::new(),
            keywords: Vec::new(),
            questions: Vec::new(),
            tags: Vec::new(),
            attributes: Metadata::new(),
            spans: Vec::new(),
            graph: Graph::default(),
            content_tokens: String::new(),
            keyword_tokens: String::new(),
            question_tokens: String::new(),
            embedding: None,
            metadata: Metadata::new(),
            images: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub original_text: String,
    pub extra: Metadata,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub node_type: String,
    pub neighbors: Vec<String>,
    pub properties: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wiki_uses_numeric_doc_id() {
        let wiki = Wiki {
            id: 7,
            doc_id: 42,
            content: "Knowledge chunk".into(),
            ..Wiki::default()
        };

        let value = serde_json::to_value(&wiki).expect("wiki must serialize");

        assert_eq!(value["doc_id"], 42);
        assert_eq!(value["id"], 7);
        assert!(value.get("embedding").is_none());
        assert_eq!(wiki.validate(), Ok(()));
    }

    #[test]
    fn wiki_serializes_embedding_when_present() {
        let wiki = Wiki {
            id: 7,
            doc_id: 42,
            content: "Knowledge chunk".into(),
            embedding: Some(vec![0.1, 0.2]),
            ..Wiki::default()
        };

        let value = serde_json::to_value(&wiki).expect("wiki must serialize");

        let embedding = value["embedding"]
            .as_array()
            .expect("embedding must serialize as an array");
        assert_eq!(embedding.len(), 2);
        assert!((embedding[0].as_f64().expect("embedding item") - 0.1).abs() < 1e-6);
        assert!((embedding[1].as_f64().expect("embedding item") - 0.2).abs() < 1e-6);
        assert_eq!(wiki.validate_vector(), Ok(()));
    }

    #[test]
    fn wiki_requires_identity_and_content() {
        let mut wiki = Wiki::default();
        assert_eq!(wiki.validate(), Err(Error::InvalidInput));

        wiki.id = 1;
        wiki.doc_id = 2;
        wiki.content = "Knowledge chunk".into();
        assert_eq!(wiki.validate(), Ok(()));
        assert_eq!(wiki.validate_vector(), Err(Error::InvalidInput));
    }

    #[test]
    fn vector_rejects_non_finite_values() {
        let wiki = Wiki {
            id: 1,
            doc_id: 2,
            content: "Knowledge chunk".into(),
            embedding: Some(vec![f32::NAN]),
            ..Wiki::default()
        };

        assert_eq!(wiki.validate_vector(), Err(Error::InvalidInput));
    }
}
