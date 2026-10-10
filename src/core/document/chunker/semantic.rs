//! Contiguous semantic splitting with an explicit Embed instance. Length
//! calculation is internal by default, with an optional user function.

use std::{collections::HashSet, ops::Range};

use futures_util::{StreamExt, stream};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    Cancellation, Error,
    core::operation,
    document::{self, Document},
    model::embed::{Embed, Request},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    max_tokens: usize,
    min_tokens: usize,
    threshold: f64,
    #[serde(default = "one")]
    max_concurrency: usize,
}

fn one() -> usize {
    1
}

/// Defaults to the coarse UTF-8 estimate. The application supplies the Embed
/// instance and may replace length calculation without adding a counter
/// component or modifying the Embed contract. Length is not billing Usage.
pub fn new<E: Embed>(options: Value, embed: E) -> Result<Chunker<E>, Error> {
    let options: Options =
        serde_json::from_value(options).map_err(|_| invalid("invalid semantic chunker options"))?;
    if options.max_tokens == 0
        || options.min_tokens == 0
        || options.min_tokens > options.max_tokens
        || !options.threshold.is_finite()
        || !(-1.0..=1.0).contains(&options.threshold)
        || options.max_concurrency == 0
    {
        return Err(invalid(
            "require 0 < min_tokens <= max_tokens, threshold in [-1, 1], and positive max_concurrency",
        ));
    }
    Ok(Chunker {
        options,
        embed,
        length: estimate,
    })
}

pub struct Chunker<E, L = fn(&str) -> Result<usize, Error>> {
    options: Options,
    embed: E,
    length: L,
}

impl<E, L> Chunker<E, L> {
    /// Optional pure, deterministic text-length rule. The default requires
    /// no additional configuration. A captured tokenizer may be used here.
    pub fn with_length<F>(self, length: F) -> Chunker<E, F>
    where
        F: Fn(&str) -> Result<usize, Error> + Send + Sync,
    {
        Chunker {
            options: self.options,
            embed: self.embed,
            length,
        }
    }
}

impl<E: Embed, L> document::Chunker for Chunker<E, L>
where
    L: Fn(&str) -> Result<usize, Error> + Send + Sync,
{
    async fn chunk(
        &self,
        documents: Vec<Document>,
        cancellation: &Cancellation,
        next_id: &mut dyn FnMut() -> i64,
    ) -> Result<Vec<Document>, Error> {
        operation::check(cancellation)?;
        let mut used: HashSet<_> = documents
            .iter()
            .filter_map(|document| document.id)
            .collect();
        let mut chunks = Vec::new();
        for document in documents {
            operation::check(cancellation)?;
            if document.content.is_empty() {
                continue;
            }
            let mut offsets: Vec<_> = document.content.char_indices().map(|(i, _)| i).collect();
            offsets.push(document.content.len());
            let ranges = self.candidates(&document.content, &offsets, cancellation)?;
            // Buffer futures without spawning: this works with native async
            // Embed traits and never creates a thread pool or runtime.
            let mut results = stream::iter(ranges.iter().enumerate().map(|(index, range)| {
                let text = &document.content[offsets[range.start]..offsets[range.end]];
                async move {
                    let result = if text.trim().is_empty() {
                        Ok(None)
                    } else {
                        operation::cancellable(
                            self.embed.embed(Request {
                                input: text.to_owned(),
                                dimensions: None,
                                options: Default::default(),
                            }),
                            cancellation,
                        )
                        .await
                        .and_then(|message| {
                            let vector: Vec<f64> = serde_json::from_value(
                                message.values.get("vector").cloned().ok_or_else(|| {
                                    Error::new("INVALID_RESPONSE", "Embed returned no vector")
                                })?,
                            )
                            .map_err(|_| Error::new("INVALID_RESPONSE", "invalid Embed vector"))?;
                            if vector.is_empty()
                                || vector.iter().any(|v| !v.is_finite())
                                || vector.iter().all(|v| *v == 0.0)
                            {
                                return Err(Error::new(
                                    "INVALID_RESPONSE",
                                    "Embed vector must be finite and nonzero",
                                ));
                            }
                            Ok(Some(vector))
                        })
                    };
                    (index, result)
                }
            }))
            .buffer_unordered(self.options.max_concurrency)
            .collect::<Vec<_>>()
            .await;
            operation::check(cancellation)?;
            results.sort_by_key(|(index, _)| *index);
            let mut vectors = Vec::new();
            let mut failure = None;
            let mut dimension = None;
            for (index, result) in results {
                match result {
                    Ok(vector) => {
                        if let Some(vector) = &vector {
                            if dimension.is_some_and(|size| size != vector.len()) {
                                return Err(Error::new(
                                    "INVALID_RESPONSE",
                                    "inconsistent Embed vector dimensions",
                                ));
                            }
                            dimension = Some(vector.len());
                        }
                        vectors.push(vector);
                    }
                    Err(error) => {
                        // Candidate errors are diagnostics, not Attempts or
                        // a second persistent execution/progress structure.
                        eprintln!("semantic candidate {index} failed: {}", error.code);
                        if failure.is_none() {
                            failure = Some(error);
                        }
                        vectors.push(None);
                    }
                }
            }
            if let Some(error) = failure {
                return Err(error);
            }
            let length = offsets.len() - 1;
            let mut start = 0;
            for index in 1..=ranges.len() {
                operation::check(cancellation)?;
                let end = ranges[index - 1].end;
                let count = self.count(&document.content[offsets[start]..offsets[end]])?;
                let should_cut = if index == ranges.len() {
                    true
                } else {
                    let combined = &document.content[offsets[start]..offsets[ranges[index].end]];
                    self.count(combined)? > self.options.max_tokens
                        || (count >= self.options.min_tokens
                            && similarity(vectors[index - 1].as_deref(), vectors[index].as_deref())
                                .is_some_and(|value| value < self.options.threshold))
                };
                if should_cut {
                    if count > self.options.max_tokens {
                        return Err(Error::new(
                            "INVALID_RESPONSE",
                            "length rule changed during splitting",
                        ));
                    }
                    let sliced = start != 0 || end != length;
                    let mut metadata = document
                        .metadata
                        .iter()
                        .filter(|(key, _)| {
                            !sliced || !matches!(key.as_str(), "vector" | "location")
                        })
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect::<serde_json::Map<_, _>>();
                    if sliced
                        && let Some(location) = document.metadata.get("location")
                        && let Some(location) =
                            super::slice_location(location.clone(), &(start..end), length)
                    {
                        metadata.insert("location".into(), location);
                    }
                    chunks.push(Document {
                        id: None,
                        doc_id: document.doc_id,
                        content: document.content[offsets[start]..offsets[end]].to_owned(),
                        metadata,
                    });
                    start = end;
                }
            }
        }
        for chunk in &mut chunks {
            operation::check(cancellation)?;
            let id = next_id();
            if !used.insert(id) {
                return Err(invalid(
                    "next_id must return a fresh ID for every new chunk",
                ));
            }
            chunk.id = Some(id);
        }
        operation::check(cancellation)?;
        Ok(chunks)
    }
}

impl<E, L> Chunker<E, L>
where
    L: Fn(&str) -> Result<usize, Error> + Send + Sync,
{
    fn count(&self, text: &str) -> Result<usize, Error> {
        let count = (self.length)(text)?;
        if !text.is_empty() && count == 0 {
            return Err(Error::new(
                "INVALID_RESPONSE",
                "length rule returned zero for non-empty text",
            ));
        }
        Ok(count)
    }

    fn candidates(
        &self,
        text: &str,
        offsets: &[usize],
        cancellation: &Cancellation,
    ) -> Result<Vec<Range<usize>>, Error> {
        let mut boundaries = Vec::new();
        for (index, character) in text.chars().enumerate() {
            if index % 1024 == 0 {
                operation::check(cancellation)?;
            }
            if matches!(character, '\n' | '.' | '?' | '!' | '。' | '？' | '！') {
                boundaries.push(index + 1);
            }
        }
        let length = offsets.len() - 1;
        if boundaries.last() != Some(&length) {
            boundaries.push(length);
        }
        let mut ranges = Vec::new();
        let mut start = 0;
        for end in boundaries {
            while start < end {
                operation::check(cancellation)?;
                if self.count(&text[offsets[start]..offsets[end]])? <= self.options.max_tokens {
                    ranges.push(start..end);
                    start = end;
                } else {
                    // Find a fitting contiguous prefix. Only accepted slices
                    // are required to fit; tokenizer counts need not be monotonic.
                    let mut fitting = start + 1;
                    if self.count(&text[offsets[start]..offsets[fitting]])?
                        > self.options.max_tokens
                    {
                        return Err(invalid(
                            "max_tokens cannot fit one Unicode scalar with the selected length rule",
                        ));
                    }
                    let mut upper = end;
                    while upper - fitting > 1 {
                        operation::check(cancellation)?;
                        let middle = fitting + (upper - fitting) / 2;
                        if self.count(&text[offsets[start]..offsets[middle]])?
                            <= self.options.max_tokens
                        {
                            fitting = middle;
                        } else {
                            upper = middle;
                        }
                    }
                    ranges.push(start..fitting);
                    start = fitting;
                }
            }
        }
        Ok(ranges)
    }
}

fn similarity(left: Option<&[f64]>, right: Option<&[f64]>) -> Option<f64> {
    let (left, right) = (left?, right?);
    // Scale before squaring to avoid overflow with custom finite vectors.
    let a = left.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let b = right.iter().fold(0.0_f64, |b, v| b.max(v.abs()));
    let dot: f64 = left.iter().zip(right).map(|(x, y)| (x / a) * (y / b)).sum();
    let norm_a: f64 = left.iter().map(|x| (x / a).powi(2)).sum();
    let norm_b: f64 = right.iter().map(|x| (x / b).powi(2)).sum();
    Some((dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0))
}

fn invalid(message: &str) -> Error {
    Error::new("INVALID_ARGUMENTS", message)
}

fn estimate(text: &str) -> Result<usize, Error> {
    Ok(text.len().div_ceil(4))
}
