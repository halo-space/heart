//! Local file loading only; parsing and downstream document processing live in sibling modules.

use crate::Cancellation;
use crate::cancellation::cancellable;
use crate::components::document::{Error, Loader, Source};

/// Built-in loader for the framework's default local-file source.
#[derive(Clone, Debug, Default)]
pub struct LocalLoader;

impl Loader for LocalLoader {
    async fn load(&self, source: &Source, cancellation: &Cancellation) -> Result<Vec<u8>, Error> {
        cancellable(
            async {
                tokio::fs::read(&source.uri)
                    .await
                    .map_err(|error| Error::new("IO_ERROR", error.to_string()))
            },
            cancellation,
        )
        .await
    }
}
