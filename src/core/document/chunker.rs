use std::collections::HashSet;
use std::ops::Range;

use serde::Deserialize;
use serde_json::Value;

use super::check_cancelled;
use crate::{
    Cancellation,
    document::{self, Document, Error, location::Location},
};

pub mod semantic;

enum Method {
    Fixed,
    Sliding { overlap: usize },
    Delimiter(String),
}

/// Model-independent, Unicode-safe contiguous text splitting.
/// Sizes count Unicode scalar values, not bytes or model tokens.
pub struct Chunker {
    method: Method,
    max_chars: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    max_chars: usize,
    overlap: Option<usize>,
    delimiter: Option<String>,
}

/// Construct fixed, sliding or delimiter splitting without a model dependency.
/// Semantic uses `semantic::new(options, embed)` so its model dependency is
/// explicit; this model-independent factory never looks up an Embed instance.
pub fn new(method: &str, options: Value) -> Result<Chunker, Error> {
    if !matches!(method, "fixed" | "sliding" | "delimiter") {
        return Err(
            Error::new("UNSUPPORTED", "chunking method is not implemented").with_details(method),
        );
    }
    if ["overlap", "delimiter"]
        .iter()
        .any(|key| options.get(key).is_some_and(Value::is_null))
    {
        return Err(invalid("method-specific options must not be null"));
    }
    let options: Options = serde_json::from_value(options).map_err(|_| {
        invalid("chunker options must contain max_chars and valid method-specific fields")
    })?;
    if options.max_chars == 0 {
        return Err(invalid("max_chars must be greater than zero"));
    }
    let method = match method {
        "fixed" if options.overlap.is_none() && options.delimiter.is_none() => Method::Fixed,
        "sliding" if options.delimiter.is_none() => {
            let overlap = options.overlap.unwrap_or(0);
            if overlap >= options.max_chars {
                return Err(invalid("overlap must be smaller than max_chars"));
            }
            Method::Sliding { overlap }
        }
        "delimiter" if options.overlap.is_none() => {
            let delimiter = options
                .delimiter
                .ok_or_else(|| invalid("delimiter is required"))?;
            if delimiter.is_empty() {
                return Err(invalid("delimiter must not be empty"));
            }
            Method::Delimiter(delimiter)
        }
        _ => {
            return Err(invalid(
                "options do not apply to the selected chunking method",
            ));
        }
    };
    Ok(Chunker {
        method,
        max_chars: options.max_chars,
    })
}

impl document::Chunker for Chunker {
    async fn chunk(
        &self,
        documents: Vec<Document>,
        cancellation: &Cancellation,
        next_id: &mut dyn FnMut() -> i64,
    ) -> Result<Vec<Document>, Error> {
        check_cancelled(cancellation)?;
        // IDs are assigned only after all documents have been split. A failure
        // never exposes partial chunks. The application owns the ID allocator;
        // IDs already consumed from it cannot be rolled back by this component.
        let mut used: HashSet<i64> = documents
            .iter()
            .filter_map(|document| document.id)
            .collect();
        let mut chunks = Vec::new();
        for document in documents {
            check_cancelled(cancellation)?;
            let mut offsets = Vec::new();
            for (index, (offset, _)) in document.content.char_indices().enumerate() {
                if index % 1024 == 0 {
                    check_cancelled(cancellation)?;
                }
                offsets.push(offset);
            }
            offsets.push(document.content.len());
            let length = offsets.len() - 1;
            let ranges = match &self.method {
                Method::Fixed => windows(length, self.max_chars, 0, cancellation)?,
                Method::Sliding { overlap } => {
                    windows(length, self.max_chars, *overlap, cancellation)?
                }
                Method::Delimiter(delimiter) => {
                    self.delimited(&document.content, &offsets, delimiter, cancellation)?
                }
            };
            for range in ranges {
                check_cancelled(cancellation)?;
                let content = document.content[offsets[range.start]..offsets[range.end]].to_owned();
                let sliced = range.start != 0 || range.end != length;
                // Do not clone a potentially large whole-document vector for
                // every small chunk only to discard it immediately afterwards.
                let mut metadata = document
                    .metadata
                    .iter()
                    .filter(|(key, _)| !sliced || !matches!(key.as_str(), "vector" | "location"))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<serde_json::Map<String, Value>>();
                if sliced
                    && let Some(value) = document.metadata.get("location")
                    && let Some(location) = slice_location(value.clone(), &range, length)
                {
                    metadata.insert("location".into(), location);
                }
                chunks.push(Document {
                    id: None,
                    doc_id: document.doc_id,
                    content,
                    metadata,
                });
            }
        }
        for chunk in &mut chunks {
            check_cancelled(cancellation)?;
            let id = next_id();
            if !used.insert(id) {
                return Err(invalid(
                    "next_id must return a fresh ID for every new chunk",
                ));
            }
            chunk.id = Some(id);
        }
        check_cancelled(cancellation)?;
        Ok(chunks)
    }
}

impl Chunker {
    fn delimited(
        &self,
        content: &str,
        offsets: &[usize],
        delimiter: &str,
        cancellation: &Cancellation,
    ) -> Result<Vec<Range<usize>>, Error> {
        let length = offsets.len() - 1;
        let mut boundaries = Vec::new();
        for (index, _) in content.match_indices(delimiter) {
            check_cancelled(cancellation)?;
            // Include delimiters in the preceding unit: no text is dropped or
            // rewritten, including repeated separators and CRLF sequences.
            let boundary = offsets
                .binary_search(&(index + delimiter.len()))
                .expect("string matches end at a Unicode scalar boundary");
            boundaries.push(boundary);
        }
        if boundaries.last().copied() != Some(length) {
            boundaries.push(length);
        }
        let mut ranges = Vec::new();
        let mut start = 0;
        let mut end = 0;
        for boundary in boundaries {
            check_cancelled(cancellation)?;
            if boundary - start > self.max_chars {
                if end > start {
                    ranges.push(start..end);
                    start = end;
                }
                while boundary - start > self.max_chars {
                    check_cancelled(cancellation)?;
                    let next = start + self.max_chars;
                    ranges.push(start..next);
                    start = next;
                }
            }
            end = boundary;
        }
        if end > start {
            ranges.push(start..end);
        }
        Ok(ranges)
    }
}

fn windows(
    length: usize,
    size: usize,
    overlap: usize,
    cancellation: &Cancellation,
) -> Result<Vec<Range<usize>>, Error> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < length {
        check_cancelled(cancellation)?;
        let end = start.saturating_add(size).min(length);
        ranges.push(start..end);
        if end == length {
            break;
        }
        start += size - overlap;
    }
    Ok(ranges)
}

pub(super) fn slice_location(value: Value, range: &Range<usize>, length: usize) -> Option<Value> {
    let mut location: Location = serde_json::from_value(value).ok()?;
    location.validate().ok()?;
    // No page/paragraph endpoint interpolation: without an actual text-to-page
    // mapping a sliced multi-page range cannot identify a chunk's position.
    if location
        .page
        .as_ref()
        .is_some_and(|page| page.from != page.to)
    {
        return None;
    }
    let single_paragraph = location.paragraph.as_ref().is_some_and(|p| p.from == p.to);
    if single_paragraph {
        if let Some(characters) = &mut location.r#char {
            if characters.to.checked_sub(characters.from)?.checked_add(1)?
                == u64::try_from(length).ok()?
            {
                let from = characters
                    .from
                    .checked_add(u64::try_from(range.start).ok()?)?;
                characters.to = from.checked_add(u64::try_from(range.len() - 1).ok()?)?;
                characters.from = from;
            } else {
                location.r#char = None;
            }
        }
    } else {
        location.paragraph = None;
        location.r#char = None;
    }
    location.x = None;
    location.y = None;
    if location.page.is_none() && location.paragraph.is_none() {
        return None;
    }
    serde_json::to_value(location).ok()
}

fn invalid(message: &str) -> Error {
    Error::new("INVALID_ARGUMENTS", message)
}
