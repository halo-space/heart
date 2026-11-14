use std::sync::Arc;

use elasticsearch::Elasticsearch;
use serde_json::{Map, Value, json};

use super::{Backend, encode_error, now_ms, parse_chat_options, validate_chat, validate_session};
use crate::Cancellation;
use crate::cancellation::check;
use crate::components::memory::Error;

/// Index names used by the two short-memory records.
///
/// Keeping these names together lets an application construct one
/// short-memory backend without exposing how the stores are split
/// internally.
pub struct Config {
    pub session_index: String,
    pub chat_index: String,
}

impl Config {
    pub fn new(session_index: impl Into<String>, chat_index: impl Into<String>) -> Self {
        Self {
            session_index: session_index.into(),
            chat_index: chat_index.into(),
        }
    }
}

pub struct Session {
    backend: Arc<Backend>,
}

pub struct Chat {
    backend: Arc<Backend>,
}

/// One short-memory backend containing Session and Chat stores.
///
/// The two stores keep their independent component contracts, while the
/// application only constructs this backend once.
pub struct Memory {
    session: Session,
    chat: Chat,
}

impl Memory {
    pub fn new(client: Elasticsearch, config: Config) -> Result<Self, Error> {
        let session = Session::new(client.clone(), config.session_index)?;
        let chat = Chat::new(client, config.chat_index)?;
        Ok(Self { session, chat })
    }
}

impl crate::components::memory::short::Store for Memory {
    type Session = Session;
    type Chat = Chat;

    fn session(&self) -> &Self::Session {
        &self.session
    }

    fn chat(&self) -> &Self::Chat {
        &self.chat
    }
}

impl Session {
    pub fn new(client: Elasticsearch, index: impl Into<String>) -> Result<Self, Error> {
        Ok(Self {
            backend: Arc::new(Backend::new(client, index)?),
        })
    }
}

impl Chat {
    pub fn new(client: Elasticsearch, index: impl Into<String>) -> Result<Self, Error> {
        Ok(Self {
            backend: Arc::new(Backend::new(client, index)?),
        })
    }
}

impl crate::components::memory::short::session::Store for Session {
    async fn create(
        &self,
        mut value: crate::components::memory::short::session::Session,
        cancellation: &Cancellation,
    ) -> Result<crate::components::memory::short::session::Session, Error> {
        check(cancellation)?;
        validate_session(&value)?;
        self.backend.ensure_index().await?;
        value.created_time = now_ms();
        value.updated_time = value.created_time;
        self.backend
            .create(
                value.id,
                serde_json::to_value(&value).map_err(encode_error)?,
            )
            .await?;
        Ok(value)
    }

    async fn read(
        &self,
        id: i64,
        cancellation: &Cancellation,
    ) -> Result<crate::components::memory::short::session::Session, Error> {
        check(cancellation)?;
        self.backend
            .get(id)
            .await?
            .map(|raw| raw.source)
            .ok_or_else(|| Error::new("NOT_FOUND", "session not found"))
    }

    async fn update(
        &self,
        mut value: crate::components::memory::short::session::Session,
        cancellation: &Cancellation,
    ) -> Result<crate::components::memory::short::session::Session, Error> {
        check(cancellation)?;
        validate_session(&value)?;
        let current = self
            .backend
            .get::<crate::components::memory::short::session::Session>(value.id)
            .await?
            .ok_or_else(|| Error::new("NOT_FOUND", "session not found"))?;
        if current.source.tenant_id != value.tenant_id
            || current.source.user_id != value.user_id
            || current.source.r#type != value.r#type
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "session ownership and type cannot change",
            ));
        }
        if current.source.updated_time != value.updated_time {
            return Err(Error::new("CONFLICT", "session snapshot is stale"));
        }
        value.created_time = current.source.created_time;
        value.updated_time = now_ms();
        self.backend
            .replace(
                value.id,
                serde_json::to_value(&value).map_err(encode_error)?,
                Some((current.seq_no, current.primary_term)),
            )
            .await?;
        Ok(value)
    }

    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        check(cancellation)?;
        self.backend.delete(id).await
    }
}

impl crate::components::memory::short::chat::Store for Chat {
    async fn create(
        &self,
        mut value: crate::components::memory::short::chat::Chat,
        cancellation: &Cancellation,
    ) -> Result<crate::components::memory::short::chat::Chat, Error> {
        check(cancellation)?;
        validate_chat(&value)?;
        self.backend.ensure_index().await?;
        value.created_time = now_ms();
        value.updated_time = value.created_time;
        self.backend
            .create(
                value.id,
                serde_json::to_value(&value).map_err(encode_error)?,
            )
            .await?;
        Ok(value)
    }

    async fn read(
        &self,
        id: i64,
        cancellation: &Cancellation,
    ) -> Result<crate::components::memory::short::chat::Chat, Error> {
        check(cancellation)?;
        self.backend
            .get(id)
            .await?
            .map(|raw| raw.source)
            .ok_or_else(|| Error::new("NOT_FOUND", "chat not found"))
    }

    async fn update(
        &self,
        mut value: crate::components::memory::short::chat::Chat,
        cancellation: &Cancellation,
    ) -> Result<crate::components::memory::short::chat::Chat, Error> {
        check(cancellation)?;
        validate_chat(&value)?;
        let current = self
            .backend
            .get::<crate::components::memory::short::chat::Chat>(value.id)
            .await?
            .ok_or_else(|| Error::new("NOT_FOUND", "chat not found"))?;
        if current.source.tenant_id != value.tenant_id
            || current.source.user_id != value.user_id
            || current.source.agent_id != value.agent_id
            || current.source.session_id != value.session_id
            || current.source.r#type != value.r#type
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "chat ownership and type cannot change",
            ));
        }
        if current.source.input != value.input {
            return Err(Error::new("INVALID_ARGUMENTS", "chat input cannot change"));
        }
        value.created_time = current.source.created_time;
        value.updated_time = now_ms();
        self.backend
            .replace(
                value.id,
                serde_json::to_value(&value).map_err(encode_error)?,
                Some((current.seq_no, current.primary_term)),
            )
            .await?;
        Ok(value)
    }

    async fn delete(&self, id: i64, cancellation: &Cancellation) -> Result<(), Error> {
        check(cancellation)?;
        self.backend.delete(id).await
    }

    async fn list(
        &self,
        options: Map<String, Value>,
        cancellation: &Cancellation,
    ) -> Result<Vec<crate::components::memory::short::chat::Chat>, Error> {
        check(cancellation)?;
        let (filters, descending, offset, limit) = parse_chat_options(&options)?;
        let order = if descending { "desc" } else { "asc" };
        let body = json!({
            "from": offset,
            "size": limit,
            "query": {"bool": {"filter": filters}},
            "sort": [
                {"created_time": {"order": order}},
                {"id": {"order": order}}
            ]
        });
        Ok(self
            .backend
            .search::<crate::components::memory::short::chat::Chat>(body)
            .await?
            .into_iter()
            .map(|record| record.source)
            .collect())
    }
}
