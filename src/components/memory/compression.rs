//! Private dispatch for a configured component, not another public contract.
use std::{future::Future, pin::Pin, sync::Arc};

use crate::{Cancellation, Error, Messages, compression::Compressor};

type FutureResult = Pin<Box<dyn Future<Output = Result<Messages, Error>> + Send + 'static>>;

trait Compression: Send + Sync {
    fn size(&self) -> Option<u64>;
    fn limit(&self) -> Option<u64>;
    fn retain(&self) -> Option<u64>;
    fn count(&self, input: &Messages) -> u64;
    fn compress(&self, input: Messages, cancellation: Cancellation) -> FutureResult;
}

mod adapter {
    use std::sync::Arc;

    pub(super) struct Compression<P>(pub(super) Arc<P>);
}

impl<P: Compressor + 'static> Compression for adapter::Compression<P> {
    fn size(&self) -> Option<u64> {
        self.0.size()
    }

    fn limit(&self) -> Option<u64> {
        self.0.limit()
    }

    fn retain(&self) -> Option<u64> {
        self.0.retain()
    }

    fn count(&self, input: &Messages) -> u64 {
        self.0.count(input)
    }

    fn compress(&self, input: Messages, cancellation: Cancellation) -> FutureResult {
        let component = self.0.clone();
        // The owned component and arguments form an owned background task.
        Box::pin(async move { component.compress(input, &cancellation).await })
    }
}

#[derive(Clone)]
pub(crate) struct Configured(Arc<dyn Compression>);

impl Configured {
    pub(crate) fn new<P: Compressor + 'static>(component: P) -> Self {
        Self(Arc::new(adapter::Compression(Arc::new(component))))
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        self.size().map(|_| ())
    }

    pub(crate) fn size(&self) -> Result<u64, Error> {
        self.0.size().filter(|size| *size > 0).ok_or_else(|| {
            Error::new(
                "INVALID_ARGUMENTS",
                "automatic compression requires a positive size",
            )
        })
    }

    pub(crate) fn limit(&self) -> Option<u64> {
        self.0.limit()
    }

    pub(crate) fn retain(&self) -> Option<u64> {
        self.0.retain()
    }

    pub(crate) fn count(&self, input: &Messages) -> u64 {
        self.0.count(input)
    }

    pub(crate) async fn compress(
        &self,
        input: Messages,
        cancellation: &Cancellation,
    ) -> Result<Messages, Error> {
        self.0.compress(input, cancellation.clone()).await
    }
}
