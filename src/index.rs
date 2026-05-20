//! `index` — 构建层。
//!
//! # Pipeline (7 steps)
//!
//! ```text
//! source::Item
//!   -> normalize_text(item.text)
//!   -> pagewiki::Base::cut(normalized)
//!   -> sort by spans.start
//!   -> assemble_system_fields (id, doc_id, version, scenario, idx)
//!   -> apply metadata + promote
//!   -> tokenize (content_tokens / keyword_tokens / question_tokens)
//!   -> embed (optional)
//!   -> Vec<pagewiki::PageWiki>
//! ```
//!
//! # Minimal usage
//!
//! ```rust,no_run
//! use std::collections::HashMap;
//! use rag::index::{Builder, NoopTokenizer};
//! use rag::index::source::Scenario;
//! use rag::index::pagewiki;
//!
//! # async fn example() -> Result<(), rag::index::Error> {
//! let mut pw_map: HashMap<Scenario, Box<dyn pagewiki::Base>> = HashMap::new();
//! pw_map.insert(Scenario::General, Box::new(pagewiki::Fixed::new(200)));
//!
//! let builder = Builder::new(pw_map, vec![], Box::new(NoopTokenizer), None)?;
//! let items = vec![]; // source::Items with doc_id in metadata
//! let pages = builder.build(items).await?;
//! # Ok(())
//! # }
//! ```

pub mod builder;
pub mod pagewiki;
pub mod source;

pub use builder::{Builder, Error, Tokenizer, NoopTokenizer, Embedder, NoopEmbedder};
