//! Harness history initialization and Chat closure over Memory components.
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Map, Value, json};

use crate::{
    Cancellation, Error, Message, Messages,
    memory::{
        self, Memory,
        short::{self, chat, session, trace},
    },
    message::Role,
    model::content::Part,
};

use super::hooks::{Hooks, cancellable};
use crate::agent::{Input, Payload};
use crate::runtime::agent::usage::Ledger;

mod compression;
mod extraction;

type LoadFuture = Pin<Box<dyn Future<Output = Result<Vec<Message>, Error>>>>;
type CreateFuture = Pin<Box<dyn Future<Output = Result<chat::Chat, Error>>>>;
type SaveFuture = Pin<Box<dyn Future<Output = Result<(), Error>>>>;

trait Entry: Send + Sync {
    fn validate(&self) -> Result<(), Error>;
    fn load(
        &self,
        metadata: Map<String, Value>,
        hooks: Hooks,
        cancellation: Cancellation,
    ) -> LoadFuture;
    fn create(&self, input: Input, cancellation: Cancellation) -> CreateFuture;
    fn save(&self, value: chat::Chat, execution: Option<Value>) -> SaveFuture;
}

struct Typed<S, C, T, L>(Arc<memory::Config<S, C, T, L>>, Arc<extraction::Jobs>);

impl<S, C, T, L> Entry for Typed<S, C, T, L>
where
    S: session::Store + 'static,
    C: chat::Store + 'static,
    T: trace::Store + 'static,
    L: Memory + 'static,
{
    fn validate(&self) -> Result<(), Error> {
        self.0.validate()
    }

    fn load(
        &self,
        metadata: Map<String, Value>,
        hooks: Hooks,
        cancellation: Cancellation,
    ) -> LoadFuture {
        let config = self.0.clone();
        let jobs = self.1.clone();
        Box::pin(async move { load(config, jobs, metadata, hooks, &cancellation).await })
    }

    fn create(&self, input: Input, cancellation: Cancellation) -> CreateFuture {
        let config = self.0.clone();
        Box::pin(async move {
            let id = Identity::parse(&input.metadata)?;
            let now = now_ms();
            let metadata = input.metadata.clone();
            let request = chat::Chat {
                id: id.chat_id,
                tenant_id: id.tenant_id,
                user_id: id.user_id,
                agent_id: id.agent_id,
                session_id: id.session_id,
                r#type: short::Type::Agent,
                status: chat::Status::Running,
                input: Some(chat::Request {
                    metadata: input.metadata,
                    payload: input
                        .payload
                        .into_iter()
                        .map(|p| chat::Payload {
                            r#type: p.r#type,
                            data: p.data,
                            mime_type: p.mime_type,
                        })
                        .collect(),
                }),
                message: None,
                usage: None,
                graph: None,
                metadata,
                started_time: Some(now),
                completed_time: None,
                created_time: now,
                updated_time: now,
            };
            let record = config.chat.create(request.clone(), &cancellation).await?;
            if record.id != request.id
                || !id.matches(&record)
                || record.input != request.input
                || record.status != chat::Status::Running
            {
                return Err(Error::new(
                    "INVALID_RESPONSE",
                    "Chat create returned a different request or identity",
                ));
            }
            Ok(record)
        })
    }

    fn save(&self, value: chat::Chat, execution: Option<Value>) -> SaveFuture {
        let config = self.0.clone();
        Box::pin(async move {
            // Cancellation of execution must not cancel its final Stopped write.
            // Backend I/O deadlines remain the backend's responsibility.
            config
                .chat
                .update(value.clone(), &Cancellation::new())
                .await?;
            if let Some(execution) = execution {
                config
                    .trace
                    .create(
                        trace::Trace {
                            chat_id: value.id,
                            user_id: Some(value.user_id),
                            session_id: Some(value.session_id),
                            agent_id: Some(value.agent_id),
                            execution,
                            created_time: value.created_time,
                            updated_time: value.updated_time,
                        },
                        &Cancellation::new(),
                    )
                    .await?;
            }
            Ok(())
        })
    }
}

#[derive(Clone, Default)]
pub(crate) struct History(Option<Arc<dyn Entry>>);

impl History {
    pub(crate) fn check_identity(&self, original: &Input, modified: &Input) -> Result<(), Error> {
        if self.0.is_some() {
            for field in ["tenant_id", "user_id", "agent_id", "session_id", "chat_id"] {
                if modified.metadata.get(field) != original.metadata.get(field) {
                    return Err(Error::new(
                        "INVALID_ARGUMENTS",
                        "middleware cannot change run identity",
                    ));
                }
            }
        }
        Ok(())
    }
    pub(crate) fn set<S, C, T, L>(&mut self, config: memory::Config<S, C, T, L>)
    where
        S: session::Store + 'static,
        C: chat::Store + 'static,
        T: trace::Store + 'static,
        L: Memory + 'static,
    {
        self.0 = Some(Arc::new(Typed(
            Arc::new(config),
            Arc::new(extraction::Jobs::default()),
        )));
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        match &self.0 {
            Some(entry) => entry.validate(),
            None => Ok(()),
        }
    }

    pub(crate) async fn load(
        &self,
        metadata: Map<String, Value>,
        hooks: Hooks,
        cancellation: &Cancellation,
    ) -> Result<Vec<Message>, Error> {
        match &self.0 {
            Some(entry) => {
                cancellable(
                    entry.load(metadata, hooks, cancellation.clone()),
                    cancellation,
                )
                .await
            }
            None => Ok(Vec::new()),
        }
    }

    pub(crate) async fn begin(
        &self,
        input: Input,
        cancellation: &Cancellation,
    ) -> Result<Run, Error> {
        let record = match &self.0 {
            Some(entry) => Some(entry.create(input, cancellation.clone()).await?),
            None => None,
        };
        Ok(Run {
            history: self.clone(),
            record,
            usage: Ledger::default(),
        })
    }
}

/// Harness persistence closure for one Chat. Model accounting is maintained
/// by runtime::agent; this type owns only the configured writeback operation.
#[derive(Clone, Default)]
pub(crate) struct Run {
    history: History,
    record: Option<chat::Chat>,
    usage: Ledger,
}

impl Run {
    pub(crate) async fn fail(&self, error: Error) -> Error {
        let status = if error.code == "CANCELLED" {
            chat::Status::Stopped
        } else {
            chat::Status::Failed
        };
        self.finish(status, None).await.err().unwrap_or(error)
    }
    pub(crate) fn settle(&self, usage: Option<crate::model::token::Usage>) {
        self.usage.settle(usage);
    }

    pub(crate) fn refine_last(&self, usage: Option<crate::model::token::Usage>) {
        self.usage.refine_last(usage);
    }

    pub(crate) fn usage(&self) -> Option<crate::model::token::Usage> {
        self.usage.total()
    }

    pub(crate) async fn finish(
        &self,
        status: chat::Status,
        message: Option<Message>,
    ) -> Result<(), Error> {
        if let (Some(entry), Some(mut record)) = (&self.history.0, self.record.clone()) {
            record.status = status;
            record.message = message.filter(|m| m.role == Role::Assistant);
            record.usage = self.usage.total();
            record.completed_time = Some(now_ms());
            entry.save(record, None).await?;
        }
        Ok(())
    }

    pub(crate) async fn close(
        &self,
        result: Result<crate::event::Event, Error>,
    ) -> Result<crate::event::Event, Error> {
        self.close_with_state(result, None, None).await
    }

    pub(crate) async fn close_with_state(
        &self,
        result: Result<crate::event::Event, Error>,
        graph: Option<Value>,
        execution: Option<Value>,
    ) -> Result<crate::event::Event, Error> {
        let result = match &result {
            Ok(crate::event::Event::Complete { message, .. })
                if self.record.is_some() && message.role != Role::Assistant =>
            {
                Err(Error::new(
                    "INVALID_RESPONSE",
                    "final Agent message must have assistant role",
                ))
            }
            _ => result,
        };
        let (status, message) = match &result {
            Ok(crate::event::Event::Complete {
                message,
                finish_reason,
                ..
            }) => (
                if finish_reason.as_deref() == Some("cancelled") {
                    chat::Status::Stopped
                } else {
                    chat::Status::Completed
                },
                Some(message.clone()),
            ),
            Err(error) => (
                if error.code == "CANCELLED" {
                    chat::Status::Stopped
                } else {
                    chat::Status::Failed
                },
                None,
            ),
            _ => {
                return Err(Error::new(
                    "INVALID_STATE",
                    "run closure requires a terminal result",
                ));
            }
        };
        if let (Some(entry), Some(mut record)) = (&self.history.0, self.record.clone()) {
            record.graph = graph;
            record.status = status;
            record.message = message.filter(|m| m.role == Role::Assistant);
            record.usage = self.usage.total();
            record.completed_time = Some(now_ms());
            entry.save(record, execution).await?;
        }
        result
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|v| i64::try_from(v.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

struct Identity {
    tenant_id: i64,
    user_id: i64,
    agent_id: i64,
    session_id: i64,
    chat_id: i64,
}

impl Identity {
    fn parse(metadata: &Map<String, Value>) -> Result<Self, Error> {
        let id = |field: &str| {
            metadata
                .get(field)
                .and_then(Value::as_i64)
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    Error::new("INVALID_ARGUMENTS", "memory requires an i64 identity")
                        .with_details(field)
                })
        };
        Ok(Self {
            tenant_id: id("tenant_id")?,
            user_id: id("user_id")?,
            agent_id: id("agent_id")?,
            session_id: id("session_id")?,
            chat_id: id("chat_id")?,
        })
    }

    fn matches(&self, value: &chat::Chat) -> bool {
        value.tenant_id == self.tenant_id
            && value.user_id == self.user_id
            && value.agent_id == self.agent_id
            && value.session_id == self.session_id
            && value.r#type == short::Type::Agent
    }
}

async fn load<S, C, T, L>(
    config: Arc<memory::Config<S, C, T, L>>,
    jobs: Arc<extraction::Jobs>,
    metadata: Map<String, Value>,
    hooks: Hooks,
    cancellation: &Cancellation,
) -> Result<Vec<Message>, Error>
where
    S: session::Store + 'static,
    C: chat::Store + 'static,
    T: trace::Store + 'static,
    L: Memory + 'static,
{
    let identity = Identity::parse(&metadata)?;
    let session = cancellable(
        config.session.read(identity.session_id, cancellation),
        cancellation,
    )
    .await?;
    if session.id != identity.session_id
        || session.tenant_id != identity.tenant_id
        || session.user_id != identity.user_id
        || session.r#type != short::Type::Agent
    {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "memory Session ownership does not match input",
        ));
    }
    let options = json!({"filter": {
        "tenant_id":identity.tenant_id,"user_id":identity.user_id,"agent_id":identity.agent_id,
        "session_id":identity.session_id,"type":"agent"
    },"sort":{"field":"created_time","order":"asc"}})
    .as_object()
    .unwrap()
    .clone();
    let mut chats = cancellable(config.chat.list(options, cancellation), cancellation).await?;
    let mut ids = std::collections::BTreeSet::new();
    for value in &chats {
        if value.id <= 0 || !identity.matches(value) || !ids.insert(value.id) {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "invalid memory Chat history ownership or duplicate id",
            ));
        }
    }
    // Snapshot-only initialization; subsequent writes are not merged into run.
    chats.retain(|value| value.id != identity.chat_id);
    chats.sort_by_key(|value| (value.created_time, value.id));
    let mut messages = Vec::new();
    let after = coverage(&session, &chats)?;
    if let Some(summary) = &session.summary {
        messages.push(context("Session summary", summary, Map::new()));
    }
    let mut compression_input = messages.clone();
    let mut extraction_input = Vec::new();
    let mut total = if session.summary.is_some() {
        compression::tokens(session.usage.as_ref())
    } else {
        0
    };
    let mut endpoints = None;
    let mut blocked = false;
    for value in chats.into_iter().skip(after) {
        // A running/incomplete earlier Chat may complete later. A closed range
        // must not swallow it just because a later Chat completed first.
        blocked |= value.status == chat::Status::Running;
        if value.status != chat::Status::Completed {
            continue;
        }
        let (Some(input), Some(message)) = (value.input, value.message) else {
            blocked = true;
            continue;
        };
        if message.role != Role::Assistant {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "Chat final message must have assistant role",
            ));
        }
        let user = crate::components::agent::chat::user_message(Input {
            metadata: input.metadata,
            payload: input
                .payload
                .into_iter()
                .map(|payload| Payload {
                    r#type: payload.r#type,
                    data: payload.data,
                    mime_type: payload.mime_type,
                })
                .collect(),
        });
        if !blocked {
            extraction_input.push(user.clone());
            extraction_input.push(message.clone());
            compression_input.push(user.clone());
            compression_input.push(message.clone());
            total = total.saturating_add(compression::tokens(value.usage.as_ref()));
            endpoints = Some((endpoints.map_or(value.id, |(from, _)| from), value.id));
        }
        messages.push(user);
        messages.push(message);
    }
    for scope in &config.scopes {
        let agent_id = (scope == "agent").then_some(identity.agent_id);
        for kind in [
            memory::Type::Summary,
            memory::Type::Fact,
            memory::Type::Preference,
        ] {
            let item = cancellable(
                config
                    .long
                    .get(identity.user_id, agent_id, None, kind.clone(), cancellation),
                cancellation,
            )
            .await?;
            if let Some(item) = item {
                if item.user_id != identity.user_id
                    || item.agent_id != agent_id
                    || item.session_id.is_some()
                    || item.r#type != kind
                    || item.content.trim().is_empty()
                {
                    return Err(Error::new(
                        "INVALID_ARGUMENTS",
                        "long memory returned a different scope or invalid content",
                    ));
                }
                messages.push(context(
                    "Long-term memory",
                    &item.content,
                    item.metadata.unwrap_or_default(),
                ));
            }
        }
    }
    if !cancellation.is_cancelled() {
        compression::start(
            config,
            session,
            compression_input,
            endpoints,
            total,
            hooks,
            extraction::Work {
                jobs,
                agent_id: identity.agent_id,
                input: Messages::new(extraction_input),
            },
        );
    }
    Ok(messages)
}

fn coverage(session: &session::Session, chats: &[chat::Chat]) -> Result<usize, Error> {
    let from = session
        .metadata
        .get("from")
        .filter(|value| !value.is_null());
    let to = session.metadata.get("to").filter(|value| !value.is_null());
    let invalid = || Error::new("INVALID_ARGUMENTS", "invalid Session summary coverage");
    match (&session.summary, from, to) {
        (None, None, None) => Ok(0),
        (Some(summary), Some(from), Some(to)) if !summary.trim().is_empty() => {
            // Business endpoint IDs, resolved against original retained history.
            // They are not timestamps, idx values or an id-arithmetic interval.
            let from = from.as_i64().ok_or_else(invalid)?;
            let to = to.as_i64().ok_or_else(invalid)?;
            let start = chats
                .iter()
                .position(|value| value.id == from)
                .ok_or_else(invalid)?;
            let end = chats
                .iter()
                .position(|value| value.id == to)
                .ok_or_else(invalid)?;
            if start > end {
                return Err(invalid());
            }
            Ok(end + 1)
        }
        _ => Err(invalid()),
    }
}

fn context(label: &str, content: &str, metadata: Map<String, Value>) -> Message {
    // Recalled user-derived data is context, not higher-priority instructions.
    let mut message = Message::new(Role::User);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":format!("[{label}]\n{content}")}),
    });
    message.metadata = metadata;
    message
}
