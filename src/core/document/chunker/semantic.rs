//! Contiguous semantic splitting with an explicit Embed instance. Length
//! calculation is internal by default, with an optional user function.

use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    sync::{Arc, Mutex},
};

use futures_util::{StreamExt, stream};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

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

type VectorCache = Arc<Mutex<HashMap<String, Vec<Option<Vec<f64>>>>>>;

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
        cache: Arc::new(Mutex::new(HashMap::new())),
    })
}

pub struct Chunker<E, L = fn(&str) -> Result<usize, Error>> {
    options: Options,
    embed: E,
    length: L,
    // Successful candidate vectors are a private, in-process retry cache.
    // It is deliberately not a second Runtime/Checkpoint state structure.
    cache: VectorCache,
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
            cache: self.cache,
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
        next_id: &mut (dyn FnMut() -> i64 + Send),
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
            let range_count = ranges.len();
            let ends = ranges.iter().map(|range| range.end).collect::<Vec<_>>();
            let cache_key = candidate_cache_key(&document.content, &ranges);
            let cached = self
                .cache
                .lock()
                .ok()
                .and_then(|cache| cache.get(&cache_key).cloned())
                .filter(|vectors| vectors.len() == range_count)
                .unwrap_or_else(|| vec![None; range_count]);
            // Buffer futures without spawning: this works with native async
            // Embed traits and never creates a thread pool or runtime.
            let mut results = stream::iter(ranges.into_iter().enumerate().map(|(index, range)| {
                // Own each candidate before creating the buffered future so
                // the future is independent of the iterator's borrow.
                let text = document.content[offsets[range.start]..offsets[range.end]].to_owned();
                let cached_vector = cached[index].clone();
                async move {
                    if let Some(vector) = cached_vector {
                        return (index, Ok(Some(vector)));
                    }
                    let result = if text.trim().is_empty() {
                        Ok(None)
                    } else {
                        operation::cancellable(
                            self.embed.embed(Request {
                                input: text,
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
                        if let Ok(mut cache) = self.cache.lock() {
                            if cache.get(&cache_key).is_none() {
                                cache.insert(cache_key.clone(), vec![None; range_count]);
                            }
                            if let Some(entry) = cache.get_mut(&cache_key) {
                                entry[index] = vector.clone();
                            }
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
            for index in 1..=range_count {
                operation::check(cancellation)?;
                let end = ends[index - 1];
                let count = self.count(&document.content[offsets[start]..offsets[end]])?;
                let should_cut = if index == range_count {
                    true
                } else {
                    let combined = &document.content[offsets[start]..offsets[ends[index]]];
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

fn candidate_cache_key(content: &str, ranges: &[Range<usize>]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    for range in ranges {
        hasher.update(range.start.to_le_bytes());
        hasher.update(range.end.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
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

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use futures_executor::block_on;
    use serde_json::json;

    use super::*;
    use crate::{
        document::{Chunker as _, Document},
        message::{Message, Role},
    };

    struct FailFirst {
        calls: Arc<AtomicUsize>,
    }

    impl Embed for FailFirst {
        fn embed(
            &self,
            _request: Request,
        ) -> impl std::future::Future<Output = Result<Message, Error>> + Send {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            std::future::ready(if call == 0 {
                Err(Error::new("TIMEOUT", "temporary embed failure"))
            } else {
                let mut message = Message::new(Role::Assistant);
                message.values.insert("vector".into(), json!([1.0, 1.0]));
                Ok(message)
            })
        }
    }

    #[test]
    fn retries_only_failed_candidates_from_the_private_cache() {
        let calls = Arc::new(AtomicUsize::new(0));
        let chunker = new(
            json!({
                "max_tokens": 8,
                "min_tokens": 1,
                "threshold": 0.5,
                "max_concurrency": 1
            }),
            FailFirst {
                calls: calls.clone(),
            },
        )
        .unwrap();
        let documents = vec![Document {
            content: "first. second. third.".into(),
            ..Document::default()
        }];
        let cancellation = Cancellation::new();
        let first = block_on(chunker.chunk(documents.clone(), &cancellation, &mut || 1));
        assert!(first.is_err());
        let second = block_on(chunker.chunk(documents, &cancellation, &mut || 2));
        assert!(second.is_ok());
        // The first batch embeds all three candidates; the second call only
        // retries the one that failed in the first batch.
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }
}
