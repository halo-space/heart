use crate::{
    Cancellation, Messages,
    model::chat::{Chat, Request},
    reason::{self, Error},
};

/// Independent reasoning through one explicitly supplied Chat. It does not
/// create plans, call tools, or own Agent execution state.
pub struct Reasoner<C> {
    chat: C,
    request: Request,
}

impl<C: Chat> Reasoner<C> {
    pub fn new(chat: C, request: Request) -> Result<Self, Error> {
        request.validate()?;
        Ok(Self { chat, request })
    }
}

impl<C: Chat> reason::Reasoner for Reasoner<C> {
    async fn reason(
        &self,
        input: Messages,
        cancellation: &Cancellation,
    ) -> Result<Messages, Error> {
        super::message_transform::run(&self.chat, &self.request, input, cancellation).await
    }
}
