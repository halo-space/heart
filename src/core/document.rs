//! Concrete processing implementations. Model use, persistence and graph
//! orchestration remain explicit caller choices.
pub mod chunker;
mod extraction;
pub use extraction::{LocalLoader, graph, raptor};
pub mod parser;
pub mod transformer;

use crate::{Cancellation, document::Error};

pub(super) fn check_cancelled(cancellation: &Cancellation) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        Err(Error::new("CANCELLED", "document processing cancelled"))
    } else {
        Ok(())
    }
}
