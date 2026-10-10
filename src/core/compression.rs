use crate::{
    Cancellation, Messages,
    compression::{self, Error},
    model::chat::{Chat, Request},
};

/// A pure Chat-backed compression operation. Its prompt is caller-supplied;
/// this type does not own compression scheduling or Memory persistence.
pub struct Compressor<C> {
    chat: C,
    request: Request,
    size: Option<u64>,
}

impl<C: Chat> Compressor<C> {
    /// Fixed instruction messages precede each call's explicit input. All other
    /// fields retain their normal chat::Request meanings, including options.
    pub fn new(chat: C, request: Request) -> Result<Self, Error> {
        request.validate()?;
        Ok(Self {
            chat,
            request,
            size: None,
        })
    }

    /// Configure the automatic trigger threshold without changing the model
    /// Request. The automatic owner validates that it is present and positive;
    /// standalone compress never schedules or persists anything.
    pub fn with_size(mut self, size: u64) -> Self {
        self.size = Some(size);
        self
    }
}

impl<C: Chat> compression::Compressor for Compressor<C> {
    fn size(&self) -> Option<u64> {
        self.size
    }

    async fn compress(
        &self,
        input: Messages,
        cancellation: &Cancellation,
    ) -> Result<Messages, Error> {
        super::message_transform::run(&self.chat, &self.request, input, cancellation).await
    }
}
