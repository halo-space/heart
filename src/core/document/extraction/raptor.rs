//! RAPTOR clustering and summary generation over already parsed Documents.

use super::{Document, Error, await_model, complete_message, message_text};
use crate::Cancellation;
use crate::components::document::Relation;
use crate::components::message::{Message, Messages};
use crate::components::model::{chat::Chat, embed::Embed};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashSet;

pub type Options = Map<String, Value>;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Output {
    pub documents: Vec<Document>,
    pub relations: Vec<Relation>,
}

pub struct Raptor<C, E> {
    options: Options,
    chat: C,
    embed: E,
}

impl<C, E> Raptor<C, E>
where
    C: Chat,
    E: Embed,
{
    pub fn new(options: Options, chat: C, embed: E) -> Result<Self, Error> {
        let cluster_size = positive_option(&options, "cluster_size", 4)?;
        let levels = positive_option(&options, "levels", 1)?;
        let threshold = match options.get("threshold") {
            None => 0.72,
            Some(value) => value.as_f64().ok_or_else(|| {
                Error::new("INVALID_ARGUMENTS", "raptor threshold must be a number")
            })?,
        };
        if cluster_size == 0
            || levels == 0
            || !threshold.is_finite()
            || !(0.0..=1.0).contains(&threshold)
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "raptor cluster_size and levels must be positive and threshold must be in [0, 1]",
            ));
        }
        Ok(Self {
            options,
            chat,
            embed,
        })
    }

    pub async fn run(
        &self,
        documents: &[Document],
        cancellation: &Cancellation,
        next_id: &mut (dyn FnMut() -> i64 + Send),
    ) -> Result<Output, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "raptor cancelled"));
        }
        let cluster_size = self
            .options
            .get("cluster_size")
            .and_then(Value::as_u64)
            .unwrap_or(4) as usize;
        let levels = self
            .options
            .get("levels")
            .and_then(Value::as_u64)
            .unwrap_or(1) as usize;
        let threshold = self
            .options
            .get("threshold")
            .and_then(Value::as_f64)
            .unwrap_or(0.72);
        if documents.is_empty() {
            return Ok(Output::default());
        }
        if documents.iter().any(|document| document.id.is_none()) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "raptor input documents require id",
            ));
        }
        let mut ids = HashSet::new();
        if documents
            .iter()
            .any(|document| !ids.insert(document.id.expect("validated above")))
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "raptor input document IDs must be unique",
            ));
        }
        let mut current = documents.to_vec();
        let mut output = Output::default();
        let mut dimension = None;
        for level in 1..=levels {
            let mut next = Vec::new();
            let mut embedded = Vec::with_capacity(current.len());
            for document in &current {
                if cancellation.is_cancelled() {
                    return Err(Error::new("CANCELLED", "raptor cancelled"));
                }
                let message = await_model(
                    self.embed.embed(crate::components::model::embed::Request {
                        input: document.content.clone(),
                        dimensions: None,
                        options: Map::new(),
                    }),
                    cancellation,
                )
                .await?;
                let vector = message
                    .values
                    .get("vector")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        Error::new("INVALID_RESPONSE", "embed response does not contain vector")
                    })?
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .and_then(|value| value.is_finite().then_some(value as f32))
                    })
                    .collect::<Option<Vec<_>>>()
                    .filter(|vector| {
                        !vector.is_empty() && vector.iter().all(|value| value.is_finite())
                    })
                    .ok_or_else(|| {
                        Error::new("INVALID_RESPONSE", "embed response vector is invalid")
                    })?;
                if dimension.is_some_and(|dimension| dimension != vector.len()) {
                    return Err(Error::new(
                        "INVALID_RESPONSE",
                        "embed response dimensions changed",
                    ));
                }
                dimension = Some(vector.len());
                embedded.push((document.clone(), vector));
            }
            for group in cluster_documents(&embedded, cluster_size, threshold) {
                if cancellation.is_cancelled() {
                    return Err(Error::new("CANCELLED", "raptor cancelled"));
                }
                let prompt = serde_json::to_string(
                    &group
                        .iter()
                        .map(|(document, _)| document.content.as_str())
                        .collect::<Vec<_>>(),
                )
                .map_err(|error| Error::new("INVALID_ARGUMENTS", error.to_string()))?;
                let mut input = Message::new(crate::components::message::Role::User);
                input
                        .content
                        .push(crate::components::model::content::part::Part {
                            r#type: "text".into(),
                            data: crate::components::model::content::part::object(serde_json::json!({"value": format!("Summarize these documents: {prompt}")})),
                        });
                let response =
                    complete_message(&self.chat, Messages::new([input]), cancellation).await?;
                let summary = message_text(&response, "summary")
                    .filter(|text| !text.trim().is_empty())
                    .ok_or_else(|| Error::new("INVALID_RESPONSE", "summary is missing"))?;
                let id = next_id();
                if id <= 0 || !ids.insert(id) {
                    return Err(Error::new(
                        "INVALID_ARGUMENTS",
                        "raptor next_id must return a fresh positive ID",
                    ));
                }
                if cancellation.is_cancelled() {
                    return Err(Error::new("CANCELLED", "raptor cancelled"));
                }
                let summary_document = Document {
                    id: Some(id),
                    doc_id: None,
                    content: summary,
                    metadata: Map::from_iter([(
                        "raptor".into(),
                        serde_json::json!({ "level": level, "method": "raptor" }),
                    )]),
                };
                for (source, _) in &group {
                    output.relations.push(Relation {
                        from: source.id.expect("validated above"),
                        to: id,
                        kind: "summary".into(),
                    });
                }
                next.push(summary_document.clone());
                output.documents.push(summary_document);
            }
            current = next;
        }
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "raptor cancelled"));
        }
        Ok(output)
    }
}

fn positive_option(options: &Options, name: &str, default: u64) -> Result<u64, Error> {
    match options.get(name) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .filter(|value| *value > 0 && usize::try_from(*value).is_ok())
            .ok_or_else(|| {
                Error::new(
                    "INVALID_ARGUMENTS",
                    "raptor option must be a positive integer",
                )
                .with_details(name)
            }),
    }
}

fn cluster_documents(
    documents: &[(Document, Vec<f32>)],
    cluster_size: usize,
    threshold: f64,
) -> Vec<Vec<(Document, Vec<f32>)>> {
    let mut used = vec![false; documents.len()];
    let mut clusters = Vec::new();
    for seed in 0..documents.len() {
        if used[seed] {
            continue;
        }
        used[seed] = true;
        let mut cluster = vec![documents[seed].clone()];
        for candidate in (seed + 1)..documents.len() {
            if used[candidate] || cluster.len() >= cluster_size {
                continue;
            }
            if cosine(&documents[seed].1, &documents[candidate].1) >= threshold {
                used[candidate] = true;
                cluster.push(documents[candidate].clone());
            }
        }
        clusters.push(cluster);
    }
    clusters
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}
