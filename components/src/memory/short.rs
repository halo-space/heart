use futures_util::future::BoxFuture;
use schema::{
    Metadata,
    memory::Error,
    message::Message,
    model::Usage,
    session::{Idx, Session, chat::Chat, chat::Status},
    trace::{Status as TraceStatus, Trace},
};
use serde_json::Value;

use crate::{Cancellation, model};

pub trait Short: Send + Sync {
    fn chats(&self) -> &dyn Chats;

    fn sessions(&self) -> &dyn Sessions;

    fn traces(&self) -> &dyn Traces;
}

pub trait Chats: Send + Sync {
    fn create<'a>(
        &'a self,
        chat: Chat,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Chat, Error>>;

    fn read<'a>(
        &'a self,
        tenant_id: u64,
        app_id: u64,
        agent_id: u64,
        user_id: u64,
        session_id: u64,
        idx: Idx,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Vec<Chat>, Error>>;

    fn update<'a>(
        &'a self,
        chat_id: u64,
        message: Option<Message>,
        usage: Option<Usage>,
        status: Status,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Chat, Error>>;

    fn delete(&self, chat_id: u64, cancellation: Cancellation) -> BoxFuture<'_, Result<(), Error>>;
}

pub trait Sessions: Send + Sync {
    fn create<'a>(
        &'a self,
        session: Session,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Session, Error>>;

    fn read(
        &self,
        tenant_id: u64,
        app_id: u64,
        agent_id: u64,
        user_id: u64,
        session_id: u64,
        cancellation: Cancellation,
    ) -> BoxFuture<'_, Result<Session, Error>>;

    fn update<'a>(
        &'a self,
        session: Session,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Session, Error>>;

    fn delete(
        &self,
        session_id: u64,
        cancellation: Cancellation,
    ) -> BoxFuture<'_, Result<(), Error>>;

    fn compress<'a>(
        &'a self,
        tenant_id: u64,
        app_id: u64,
        agent_id: u64,
        user_id: u64,
        session_id: u64,
        model: &'a dyn model::Chat,
        metadata: &'a Metadata,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Session, Error>>;
}

pub trait Traces: Send + Sync {
    fn create<'a>(
        &'a self,
        trace: Trace,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Trace, Error>>;

    fn read(
        &self,
        tenant_id: u64,
        agent_id: u64,
        user_id: u64,
        session_id: u64,
        chat_id: u64,
        cancellation: Cancellation,
    ) -> BoxFuture<'_, Result<Vec<Trace>, Error>>;

    fn update<'a>(
        &'a self,
        trace_id: u64,
        message: Option<Message>,
        error: Option<Value>,
        duration: u64,
        status: TraceStatus,
        cancellation: Cancellation,
    ) -> BoxFuture<'a, Result<Trace, Error>>;

    fn delete(&self, trace_id: u64, cancellation: Cancellation)
    -> BoxFuture<'_, Result<(), Error>>;
}
