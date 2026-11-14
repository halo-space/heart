pub mod graph;
mod loader;
pub mod raptor;
pub use loader::LocalLoader;

use std::future::Future;

use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::Value;

use crate::components::document::{Document, Error};
use crate::components::event::Event;
use crate::components::message::{Message, Messages};
use crate::components::model;
use crate::{Cancellation, cancellation::cancellable};

async fn complete_message<C: model::chat::Chat>(
    chat: &C,
    messages: Messages,
    cancellation: &Cancellation,
) -> Result<Message, Error> {
    complete_request(
        chat,
        model::chat::Request {
            messages,
            stream: true,
            ..model::chat::Request::default()
        },
        cancellation,
    )
    .await
}

async fn await_model<F: Future<Output = Result<T, model::Error>>, T>(
    future: F,
    cancellation: &Cancellation,
) -> Result<T, Error> {
    cancellable(
        async {
            future.await.map_err(|error| {
                Error::new("MODEL_ERROR", error.message)
                    .with_details(error.details.unwrap_or(error.code))
            })
        },
        cancellation,
    )
    .await
}

async fn complete_request<C: model::chat::Chat>(
    chat: &C,
    request: model::chat::Request,
    cancellation: &Cancellation,
) -> Result<Message, Error> {
    let request = crate::components::model::chat::fit_request(request, chat.context_limit())
        .map_err(|error| {
            let mut mapped = Error::new(error.code, error.message);
            if let Some(details) = error.details {
                mapped = mapped.with_details(details);
            }
            mapped
        })?;
    let stream = cancellable(
        async {
            chat.stream(request).await.map_err(|error| {
                Error::new("MODEL_ERROR", error.message)
                    .with_details(error.details.unwrap_or(error.code))
            })
        },
        cancellation,
    )
    .await?;
    complete_stream(stream, cancellation).await
}

async fn complete_stream<S>(stream: S, cancellation: &Cancellation) -> Result<Message, Error>
where
    S: Stream<Item = Result<Event, model::Error>>,
{
    let mut stream = Box::pin(stream);
    loop {
        let item = cancellable(async { Ok::<_, Error>(stream.next().await) }, cancellation).await?;
        let item =
            item.ok_or_else(|| Error::new("INVALID_RESPONSE", "model did not return Complete"))?;
        match item.map_err(|error| {
            Error::new("MODEL_ERROR", error.message)
                .with_details(error.details.unwrap_or(error.code))
        })? {
            Event::Delta { .. } => {}
            Event::Complete {
                mut message, usage, ..
            } => {
                if let Some(usage) = usage {
                    message.usage = Some(usage);
                }
                return Ok(message);
            }
        }
    }
}

fn message_text(message: &Message, key: &str) -> Option<String> {
    message
        .values
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| {
            let parts: Vec<_> = message
                .content
                .iter()
                .filter(|part| part.r#type == "text")
                .filter_map(|part| part.data.get("value").and_then(Value::as_str))
                .collect();
            (!parts.is_empty()).then(|| parts.concat())
        })
}

#[cfg(test)]
#[path = "../../../tests/unit/embedded/src_core_document_extraction_tests.rs"]
mod tests;
