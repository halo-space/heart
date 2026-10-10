//! Harness dispatch of configured component lifecycle hooks.
use std::{future::Future, pin::Pin, sync::Arc, task::Poll};

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{Cancellation, Error, middleware::Middleware};

type HookFuture = Pin<Box<dyn Future<Output = Result<Value, Error>>>>;

#[derive(Clone, Copy)]
enum Stage {
    Execute,
    Model,
    SystemPrompt,
    Permission,
    Compression,
}

trait Entry: Send + Sync {
    fn apply(
        &self,
        stage: Stage,
        before: bool,
        value: Value,
        cancellation: Cancellation,
    ) -> HookFuture;
}

struct Typed<M>(Arc<M>);
impl<M: Middleware + 'static> Entry for Typed<M> {
    fn apply(
        &self,
        stage: Stage,
        before: bool,
        mut value: Value,
        cancellation: Cancellation,
    ) -> HookFuture {
        let middleware = self.0.clone();
        Box::pin(async move {
            match (stage, before) {
                (Stage::Execute, true) => {
                    middleware.before_execute(&mut value, &cancellation).await?
                }
                (Stage::Execute, false) => {
                    middleware.after_execute(&mut value, &cancellation).await?
                }
                (Stage::Model, true) => middleware.before_model(&mut value, &cancellation).await?,
                (Stage::Model, false) => middleware.after_model(&mut value, &cancellation).await?,
                (Stage::SystemPrompt, true) => {
                    middleware
                        .before_system_prompt(&mut value, &cancellation)
                        .await?
                }
                (Stage::SystemPrompt, false) => {
                    middleware
                        .after_system_prompt(&mut value, &cancellation)
                        .await?
                }
                (Stage::Permission, true) => {
                    middleware
                        .before_permission(&mut value, &cancellation)
                        .await?
                }
                (Stage::Permission, false) => {
                    middleware
                        .after_permission(&mut value, &cancellation)
                        .await?
                }
                (Stage::Compression, true) => {
                    middleware
                        .before_compress_context(&mut value, &cancellation)
                        .await?
                }
                (Stage::Compression, false) => {
                    middleware
                        .after_compress_context(&mut value, &cancellation)
                        .await?
                }
            }
            Ok(value)
        })
    }
}

#[derive(Clone, Default)]
pub(crate) struct Hooks(Vec<Arc<dyn Entry>>);

impl Hooks {
    pub(crate) fn add<M: Middleware + 'static>(&mut self, middleware: M) {
        self.0.push(Arc::new(Typed(Arc::new(middleware))));
    }

    async fn apply<T: Serialize + DeserializeOwned>(
        &self,
        stage: Stage,
        before: bool,
        value: T,
        cancellation: &Cancellation,
    ) -> Result<T, Error> {
        if self.0.is_empty() {
            return Ok(value);
        }
        let mut value = serde_json::to_value(value)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "cannot encode middleware input"))?;
        // Normal stages unwind in reverse; system prompt is a sequential transform.
        let reverse = !before && !matches!(stage, Stage::SystemPrompt);
        for offset in 0..self.0.len() {
            let index = if reverse {
                self.0.len() - 1 - offset
            } else {
                offset
            };
            value = cancellable(
                self.0[index].apply(stage, before, value, cancellation.clone()),
                cancellation,
            )
            .await?;
        }
        serde_json::from_value(value).map_err(|_| {
            Error::new(
                "INVALID_ARGUMENTS",
                "middleware returned an invalid stage value",
            )
        })
    }

    pub(crate) async fn before_execute(
        &self,
        input: crate::agent::Input,
        cancellation: &Cancellation,
    ) -> Result<crate::agent::Input, Error> {
        self.apply(Stage::Execute, true, input, cancellation).await
    }

    pub(crate) async fn before_permission(
        &self,
        request: crate::permission::Request,
        cancellation: &Cancellation,
    ) -> Result<crate::permission::Request, Error> {
        let original = request.clone();
        let request = self
            .apply(Stage::Permission, true, request, cancellation)
            .await?;
        check_permission_identity(&original, &request)?;
        Ok(request)
    }

    pub(crate) async fn after_permission(
        &self,
        request: crate::permission::Request,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        let original = request.clone();
        let request = self
            .apply(Stage::Permission, false, request, cancellation)
            .await?;
        check_permission_identity(&original, &request)
    }

    pub(crate) async fn prompt(
        &self,
        prompt: Option<String>,
        cancellation: &Cancellation,
    ) -> Result<Option<String>, Error> {
        let prompt = self
            .apply(Stage::SystemPrompt, true, prompt, cancellation)
            .await?;
        self.apply(Stage::SystemPrompt, false, prompt, cancellation)
            .await
    }

    pub(crate) async fn before_model(
        &self,
        request: crate::model::chat::Request,
        cancellation: &Cancellation,
    ) -> Result<crate::model::chat::Request, Error> {
        let request = self
            .apply(Stage::Model, true, request, cancellation)
            .await?;
        request.validate()?;
        Ok(request)
    }

    pub(crate) async fn after_model(
        &self,
        message: crate::Message,
        cancellation: &Cancellation,
    ) -> Result<crate::Message, Error> {
        let usage = message.usage.clone();
        let mut message: crate::Message = self
            .apply(Stage::Model, false, message, cancellation)
            .await?;
        // Provider accounting is a fact, not a mutable business output.
        message.usage = usage;
        Ok(message)
    }

    pub(crate) async fn after_execute(
        &self,
        message: crate::Message,
        cancellation: &Cancellation,
    ) -> Result<crate::Message, Error> {
        let usage = message.usage.clone();
        let mut message: crate::Message = self
            .apply(Stage::Execute, false, message, cancellation)
            .await?;
        message.usage = usage;
        Ok(message)
    }

    pub(crate) async fn before_compress(
        &self,
        messages: crate::Messages,
        cancellation: &Cancellation,
    ) -> Result<crate::Messages, Error> {
        self.apply(Stage::Compression, true, messages, cancellation)
            .await
    }

    pub(crate) async fn after_compress(
        &self,
        messages: crate::Messages,
        cancellation: &Cancellation,
    ) -> Result<crate::Messages, Error> {
        // Accounting is captured before this business transform by its owner.
        self.apply(Stage::Compression, false, messages, cancellation)
            .await
    }
}

fn check_permission_identity(
    original: &crate::permission::Request,
    request: &crate::permission::Request,
) -> Result<(), Error> {
    if request.action != original.action
        || request.resource != original.resource
        || request.metadata.get("call") != original.metadata.get("call")
    {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "permission middleware cannot change the checked operation or call",
        ));
    }
    Ok(())
}

pub(crate) async fn cancellable<T>(
    future: impl Future<Output = Result<T, Error>>,
    cancellation: &Cancellation,
) -> Result<T, Error> {
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|cx| {
        cancellation.register(cx.waker());
        if cancellation.is_cancelled() {
            return Poll::Ready(Err(Error::new("CANCELLED", "Agent stage cancelled")));
        }
        let result = future.as_mut().poll(cx);
        if cancellation.is_cancelled() {
            return Poll::Ready(Err(Error::new("CANCELLED", "Agent stage cancelled")));
        }
        result
    })
    .await
}
