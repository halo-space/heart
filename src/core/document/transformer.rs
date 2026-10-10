use std::collections::{HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::check_cancelled;
use crate::{
    Cancellation,
    document::{self, Document, Error},
};

type MapFunction = dyn Fn(Document) -> Result<Document, Error> + Send + Sync;
type FilterFunction = dyn Fn(&Document) -> bool + Send + Sync;

enum Operation {
    RemoveEmpty,
    Deduplicate,
    Metadata(Map<String, Value>),
    Map(Box<MapFunction>),
    Filter(Box<FilterFunction>),
}

/// Ordered collection transformations, not a second public chain protocol.
#[derive(Default)]
pub struct Transformer {
    operations: Vec<Operation>,
}

pub fn new() -> Transformer {
    Transformer::default()
}

impl Transformer {
    pub fn remove_empty(mut self) -> Self {
        self.operations.push(Operation::RemoveEmpty);
        self
    }
    /// Remove only exact duplicates, including IDs and metadata. Equal text from
    /// different sources is not evidence that the sources are interchangeable.
    pub fn deduplicate(mut self) -> Self {
        self.operations.push(Operation::Deduplicate);
        self
    }
    pub fn metadata(mut self, metadata: Map<String, Value>) -> Self {
        self.operations.push(Operation::Metadata(metadata));
        self
    }
    pub fn map<F: Fn(Document) -> Result<Document, Error> + Send + Sync + 'static>(
        mut self,
        function: F,
    ) -> Self {
        self.operations.push(Operation::Map(Box::new(function)));
        self
    }
    pub fn filter<F: Fn(&Document) -> bool + Send + Sync + 'static>(mut self, function: F) -> Self {
        self.operations.push(Operation::Filter(Box::new(function)));
        self
    }
}

impl document::Transformer for Transformer {
    async fn transform(
        &self,
        mut documents: Vec<Document>,
        cancellation: &Cancellation,
    ) -> Result<Vec<Document>, Error> {
        check_cancelled(cancellation)?;
        for operation in &self.operations {
            let mut result: Vec<Document> = Vec::with_capacity(documents.len());
            let mut seen: HashMap<u64, Vec<usize>> = HashMap::new();
            for mut document in documents {
                check_cancelled(cancellation)?;
                match operation {
                    Operation::RemoveEmpty if document.content.trim().is_empty() => continue,
                    Operation::Filter(function) if !function(&document) => continue,
                    Operation::Deduplicate => {
                        // Hash borrowed fields without copying all unique text
                        // into a second corpus. Verify equality on collisions.
                        let mut hasher = DefaultHasher::new();
                        document.id.hash(&mut hasher);
                        document.doc_id.hash(&mut hasher);
                        document.content.hash(&mut hasher);
                        document.metadata.hash(&mut hasher);
                        let matches = seen.entry(hasher.finish()).or_default();
                        if matches.iter().any(|index| result[*index] == document) {
                            continue;
                        }
                        matches.push(result.len());
                    }
                    Operation::Metadata(metadata) => document.metadata.extend(metadata.clone()),
                    Operation::Map(function) => {
                        // Snapshot only content-dependent facts and a private
                        // digest, not another owned copy of the complete text.
                        let digest = Sha256::digest(document.content.as_bytes());
                        let facts: Vec<_> = ["location", "vector", "method"]
                            .into_iter()
                            .filter_map(|key| {
                                document.metadata.get(key).map(|value| (key, value.clone()))
                            })
                            .collect();
                        document = function(document)?;
                        if digest != Sha256::digest(document.content.as_bytes()) {
                            for (key, previous) in facts {
                                if document.metadata.get(key) == Some(&previous) {
                                    document.metadata.remove(key);
                                }
                            }
                        }
                    }
                    _ => {}
                }
                result.push(document);
            }
            documents = result;
            check_cancelled(cancellation)?;
        }
        Ok(documents)
    }
}
