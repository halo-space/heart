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
    limit: Option<u64>,
    retain: Option<u64>,
    count: fn(&Messages) -> u64,
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
            limit: None,
            retain: None,
            count: default_count,
        })
    }

    /// Configure the automatic trigger threshold without changing the model
    /// Request. The automatic owner validates that it is present and positive;
    /// standalone compress never schedules or persists anything.
    pub fn with_size(mut self, size: u64) -> Self {
        self.size = Some(size);
        self
    }

    /// Limit the historical input sent to the compression model.
    pub fn with_limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Keep the newest uncovered historical input for a later request.
    pub fn with_retain(mut self, retain: u64) -> Self {
        self.retain = Some(retain);
        self
    }

    /// Use an application-provided estimate for temporary compression input.
    pub fn with_count(mut self, count: fn(&Messages) -> u64) -> Self {
        self.count = count;
        self
    }
}

fn default_count(input: &Messages) -> u64 {
    serde_json::to_vec(input)
        .map(|bytes| u64::try_from(bytes.len().saturating_add(3) / 4).unwrap_or(u64::MAX))
        .unwrap_or(u64::MAX)
}

impl<C: Chat> compression::Compressor for Compressor<C> {
    fn size(&self) -> Option<u64> {
        self.size
    }

    fn limit(&self) -> Option<u64> {
        self.limit
    }

    fn retain(&self) -> Option<u64> {
        self.retain
    }

    fn count(&self, input: &Messages) -> u64 {
        (self.count)(input)
    }

    async fn compress(
        &self,
        input: Messages,
        cancellation: &Cancellation,
    ) -> Result<Messages, Error> {
        super::message_transform::run(&self.chat, &self.request, input, cancellation).await
    }
}
