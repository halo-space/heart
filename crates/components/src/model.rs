use futures_util::{future::BoxFuture, stream::BoxStream};
use schema::{
    Metadata,
    event::Event,
    message::Message,
    model::{self, Error},
};

use crate::Cancellation;

pub trait Model: Send + Sync {
    fn after<'a>(
        &'a self,
        event: Event,
        _metadata: &'a Metadata,
        _cancellation: &'a Cancellation,
    ) -> BoxFuture<'a, Result<Event, Error>> {
        Box::pin(async move { Ok(event) })
    }
}

pub trait Chat: Model {
    fn before<'a>(
        &'a self,
        _messages: &'a mut Vec<Message>,
        _options: &'a mut model::chat::Options,
        _metadata: &'a Metadata,
        _cancellation: &'a Cancellation,
    ) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }

    fn validate(&self, options: &model::chat::Options) -> Result<(), Error> {
        options.validate()
    }

    fn execute<'a>(
        &'a self,
        messages: Vec<Message>,
        options: model::chat::Options,
        metadata: &'a Metadata,
        cancellation: Cancellation,
    ) -> BoxStream<'a, Result<Event, Error>>;
}

pub trait Speech: Model {
    fn before<'a>(
        &'a self,
        _messages: &'a mut Vec<Message>,
        _options: &'a mut model::audio::speech::Options,
        _metadata: &'a Metadata,
        _cancellation: &'a Cancellation,
    ) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }

    fn validate(&self, options: &model::audio::speech::Options) -> Result<(), Error> {
        options.validate()
    }

    fn execute<'a>(
        &'a self,
        messages: Vec<Message>,
        options: model::audio::speech::Options,
        metadata: &'a Metadata,
        cancellation: Cancellation,
    ) -> BoxStream<'a, Result<Event, Error>>;
}

pub trait Transcription: Model {
    fn before<'a>(
        &'a self,
        _messages: &'a mut Vec<Message>,
        _options: &'a mut model::audio::transcription::Options,
        _metadata: &'a Metadata,
        _cancellation: &'a Cancellation,
    ) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }

    fn validate(&self, options: &model::audio::transcription::Options) -> Result<(), Error> {
        options.validate()
    }

    fn execute<'a>(
        &'a self,
        messages: Vec<Message>,
        options: model::audio::transcription::Options,
        metadata: &'a Metadata,
        cancellation: Cancellation,
    ) -> BoxStream<'a, Result<Event, Error>>;
}

pub trait Embedding: Model {
    fn before<'a>(
        &'a self,
        _messages: &'a mut Vec<Message>,
        _options: &'a mut model::embedding::Options,
        _metadata: &'a Metadata,
        _cancellation: &'a Cancellation,
    ) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }

    fn validate(&self, options: &model::embedding::Options) -> Result<(), Error> {
        options.validate()
    }

    fn execute<'a>(
        &'a self,
        messages: Vec<Message>,
        options: model::embedding::Options,
        metadata: &'a Metadata,
        cancellation: Cancellation,
    ) -> BoxStream<'a, Result<Event, Error>>;
}

#[cfg(test)]
mod tests {
    use futures_util::stream;
    use schema::model::{ErrorKind, chat};

    use super::*;

    struct TestChat;

    impl Model for TestChat {}

    impl Chat for TestChat {
        fn execute<'a>(
            &'a self,
            _messages: Vec<Message>,
            _options: chat::Options,
            _metadata: &'a Metadata,
            _cancellation: Cancellation,
        ) -> BoxStream<'a, Result<Event, Error>> {
            Box::pin(stream::empty())
        }
    }

    #[test]
    fn chat_validation_uses_chat_options_directly() {
        let model = TestChat;
        let options = chat::Options {
            feats: Vec::new(),
            ..chat::Options::default()
        };

        assert_eq!(
            model.validate(&options).unwrap_err().kind,
            ErrorKind::InvalidInput
        );
    }
}
