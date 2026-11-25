//! OpenAI-compatible multimodal requests backed by Chat Completions.
//!
//! The first V3 adapter supports image input and text output. It reuses the
//! existing Chat wire encoder and Event decoder so multimodal calls do not
//! create a second streaming or tool-call protocol.

use crate::model::{self, multimodal::Request};

use super::{chat, transport::Connection};

type Error = model::Error;

#[derive(Clone)]
pub struct Multimodal {
    pub(super) connection: Connection,
    pub(super) config: model::multimodal::Config,
}

impl model::multimodal::Multimodal for Multimodal {
    type Stream = chat::Events;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        let request = into_chat(request)?;
        let chat = chat::Chat {
            connection: self.connection.clone(),
            config: model::chat::Config {
                model_name: self.config.model_name.clone(),
                context_limit: self.config.context_limit,
                options: self.config.options.clone(),
            },
        };
        model::chat::Chat::stream(&chat, request).await
    }
}

pub(super) fn into_chat(request: Request) -> Result<model::chat::Request, Error> {
    request.validate()?;
    if request.modalities.iter().any(|modality| modality != "text") {
        return Err(Error::new(
            "UNSUPPORTED",
            "OpenAI-compatible multimodal output currently supports text only",
        ));
    }
    Ok(model::chat::Request {
        messages: request.messages,
        tools: request.tools,
        tool_choice: request.tool_choice.map(into_tool_choice),
        stream: request.stream,
        temperature: request.temperature,
        max_tokens: request.max_tokens,
        top_p: request.top_p,
        stop: request.stop,
        response_format: request.response_format.map(into_response_format),
        options: request.options,
    })
}

fn into_tool_choice(value: model::multimodal::ToolChoice) -> model::chat::ToolChoice {
    match value {
        model::multimodal::ToolChoice::None => model::chat::ToolChoice::None,
        model::multimodal::ToolChoice::Auto => model::chat::ToolChoice::Auto,
        model::multimodal::ToolChoice::Required => model::chat::ToolChoice::Required,
        model::multimodal::ToolChoice::Named { name } => model::chat::ToolChoice::Named { name },
    }
}

fn into_response_format(value: model::multimodal::ResponseFormat) -> model::chat::ResponseFormat {
    match value {
        model::multimodal::ResponseFormat::JsonObject => model::chat::ResponseFormat::JsonObject,
        model::multimodal::ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => model::chat::ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        },
    }
}
