//! Core Agent dispatch of configured component lifecycle hooks.
use std::{future::Future, pin::Pin, sync::Arc};

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{Cancellation, Error, middleware::Middleware};

type HookFuture = Pin<Box<dyn Future<Output = Result<Value, Error>> + Send + 'static>>;

#[derive(Clone, Copy)]
pub(crate) enum Stage {
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
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn add<M: Middleware + 'static>(&mut self, middleware: M) {
        self.0.push(Arc::new(Typed(Arc::new(middleware))));
    }

    async fn apply<T: Clone + Serialize + DeserializeOwned>(
        &self,
        stage: Stage,
        before: bool,
        value: T,
        cancellation: &Cancellation,
    ) -> Result<T, Error> {
        Box::pin(self.apply_from(stage, before, value, 0, |_, _| Ok(()), cancellation)).await
    }

    /// The owner acknowledges only successful, typed transformations. An
    /// unfinished method's local mutation never overwrites the saved input.
    pub(crate) async fn apply_from<T: Clone + Serialize + DeserializeOwned>(
        &self,
        stage: Stage,
        before: bool,
        value: T,
        idx: usize,
        mut commit: impl FnMut(usize, T) -> Result<(), Error>,
        cancellation: &Cancellation,
    ) -> Result<T, Error> {
        self.apply_from_async(
            stage,
            before,
            value,
            idx,
            |idx, value| std::future::ready(commit(idx, value)),
            cancellation,
        )
        .await
    }

    /// Same driver, with an asynchronous native-owner acknowledgement. This
    /// permits parallel Tool work to yield facts before the next method runs.
    pub(crate) async fn apply_from_async<T, K, KF>(
        &self,
        stage: Stage,
        before: bool,
        mut value: T,
        idx: usize,
        mut commit: K,
        cancellation: &Cancellation,
    ) -> Result<T, Error>
    where
        T: Clone + Serialize + DeserializeOwned,
        K: FnMut(usize, T) -> KF,
        KF: Future<Output = Result<(), Error>>,
    {
        if idx > self.0.len() {
            return Err(Error::new("INVALID_STATE", "invalid middleware cursor"));
        }
        // Normal stages unwind in reverse; system prompt is a sequential transform.
        let reverse = !before && !matches!(stage, Stage::SystemPrompt);
        for offset in idx..self.0.len() {
            let index = if reverse {
                self.0.len() - 1 - offset
            } else {
                offset
            };
            value = Box::pin(crate::core::lifecycle::prepare(
                crate::core::lifecycle::Phase::Before,
                value,
                |value| async {
                    let value = serde_json::to_value(value).map_err(|_| {
                        Error::new("INVALID_ARGUMENTS", "cannot encode middleware input")
                    })?;
                    let usage = (!before && matches!(stage, Stage::Model | Stage::Execute))
                        .then(|| value.get("usage").cloned())
                        .flatten();
                    let mut result = self.0[index]
                        .apply(stage, before, value, cancellation.clone())
                        .await?;
                    // Consumption is an invocation fact, not business output.
                    // Normalize before the next method can observe this value.
                    if let Some(usage) = usage
                        && let Some(object) = result.as_object_mut()
                    {
                        object.insert("usage".into(), usage);
                    }
                    serde_json::from_value(result).map_err(|_| {
                        Error::new(
                            "INVALID_ARGUMENTS",
                            "middleware returned an invalid stage value",
                        )
                    })
                },
                |value| commit(offset + 1, value),
                cancellation,
            ))
            .await?;
        }
        Ok(value)
    }

    pub(crate) async fn after_from(
        &self,
        stage: Stage,
        message: crate::Message,
        idx: usize,
        mut commit: impl FnMut(usize, crate::Message) -> Result<(), Error>,
        cancellation: &Cancellation,
    ) -> Result<crate::Message, Error> {
        let usage = message.usage.clone();
        let mut result = Box::pin(self.apply_from(
            stage,
            false,
            message,
            idx,
            |idx, mut message: crate::Message| {
                message.usage = usage.clone();
                commit(idx, message)
            },
            cancellation,
        ))
        .await?;
        result.usage = usage;
        Ok(result)
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

pub(crate) async fn cancellable<T>(
    future: impl Future<Output = Result<T, Error>>,
    cancellation: &Cancellation,
) -> Result<T, Error> {
    crate::runtime::cancellation::cancellable(future, cancellation).await
}

#[cfg(test)]
mod tests {
    use super::{Hooks, Stage};
    use crate::{Cancellation, Error, middleware::Middleware};
    use serde_json::{Value, json};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    struct Transform {
        key: &'static str,
        calls: Arc<AtomicUsize>,
        fail: Arc<AtomicBool>,
    }

    impl Transform {
        fn apply(&self, value: &mut Value) -> Result<(), Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            value["values"][self.key] = json!(true);
            if self.fail.load(Ordering::SeqCst) {
                return Err(Error::new("HOOK_FAILED", "controlled hook failure"));
            }
            Ok(())
        }
    }

    impl Middleware for Transform {
        async fn after_system_prompt(
            &self,
            value: &mut Value,
            _: &Cancellation,
        ) -> Result<(), Error> {
            self.apply(value)
        }
    }

    #[tokio::test]
    async fn only_acknowledged_hook_transformations_advance_the_cursor() {
        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));
        let fail_second = Arc::new(AtomicBool::new(true));
        let mut hooks = Hooks::default();
        hooks.add(Transform {
            key: "first",
            calls: Arc::clone(&first_calls),
            fail: Arc::new(AtomicBool::new(false)),
        });
        hooks.add(Transform {
            key: "second",
            calls: Arc::clone(&second_calls),
            fail: Arc::clone(&fail_second),
        });

        let mut acknowledged = Vec::new();
        let error = hooks
            .apply_from(
                Stage::SystemPrompt,
                false,
                json!({"values": {}, "usage": {"total": 15}}),
                0,
                |idx, value| {
                    acknowledged.push((idx, value));
                    Ok(())
                },
                &Cancellation::new(),
            )
            .await
            .expect_err("second hook is configured to fail");
        assert_eq!(error.code, "HOOK_FAILED");
        assert_eq!(acknowledged.len(), 1);
        assert_eq!(acknowledged[0].0, 1);
        assert_eq!(acknowledged[0].1["values"]["first"], true);
        assert!(acknowledged[0].1["values"].get("second").is_none());
        assert_eq!(acknowledged[0].1["usage"]["total"], 15);
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);

        fail_second.store(false, Ordering::SeqCst);
        let (cursor, saved_value) = acknowledged.pop().unwrap();
        let mut restored_cursor = 0;
        let mut restored_value = Value::Null;
        let final_value = hooks
            .apply_from(
                Stage::SystemPrompt,
                false,
                saved_value,
                cursor,
                |idx, value| {
                    restored_cursor = idx;
                    restored_value = value;
                    Ok(())
                },
                &Cancellation::new(),
            )
            .await
            .expect("resume from the acknowledged cursor");
        assert_eq!(restored_cursor, 2);
        assert_eq!(restored_value, final_value);
        assert_eq!(final_value["values"]["first"], true);
        assert_eq!(final_value["values"]["second"], true);
        assert_eq!(final_value["usage"]["total"], 15);
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 2);
    }
}
