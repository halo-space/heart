use futures_util::future::BoxFuture;
use schema::{Metadata, event::Event, tool};
use serde_json::Value;

use crate::Cancellation;

pub trait Tool: Send + Sync {
    fn definition(&self) -> &tool::Definition;

    fn before<'a>(
        &'a self,
        _execute_id: &'a str,
        _arguments: &'a mut Value,
        _metadata: &'a Metadata,
        _cancellation: &'a Cancellation,
    ) -> BoxFuture<'a, Result<(), tool::Error>> {
        Box::pin(async { Ok(()) })
    }

    fn execute<'a>(
        &'a self,
        execute_id: String,
        arguments: Value,
        metadata: &'a Metadata,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Event, tool::Error>>;

    fn after<'a>(
        &'a self,
        event: Event,
        _metadata: &'a Metadata,
        _cancellation: &'a Cancellation,
    ) -> BoxFuture<'a, Result<Event, tool::Error>> {
        Box::pin(async move { Ok(event) })
    }
}
