//! Concrete processing implementations. Model use, persistence and graph
//! orchestration remain explicit caller choices.
pub mod chunker;
mod extraction;
pub use extraction::{LocalLoader, graph, raptor};
pub mod parser;
pub mod transformer;
