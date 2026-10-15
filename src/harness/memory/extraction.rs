//! Stage-preserving long Memory processing. Production progress is saved in
//! Session metadata before external writes and after acknowledged completion.
use crate::harness::hooks::cancellable;
use crate::{
    Cancellation, Error, Messages,
    memory::{self, Item, Memory, Request},
    message::Role,
};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[cfg(test)]
#[derive(Default)]
pub(super) struct Jobs(Mutex<BTreeMap<i64, Job>>);

pub(super) struct Work {
    pub(super) agent_id: i64,
    pub(super) input: Messages,
}

#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Job {
    user_id: i64,
    agent_id: i64,
    from: i64,
    to: i64,
    input: Messages,
    tasks: Option<Vec<Task>>,
}
#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Task {
    agent_id: Option<i64>,
    kind: memory::Type,
    content: String,
    prepared: Option<Prepared>,
    done: bool,
}
#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Prepared {
    previous: Option<Item>,
    request: Request,
}

#[cfg(test)]
impl Jobs {
    pub(super) fn contains(&self, session_id: i64) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&session_id)
    }
    pub(super) fn insert(&self, session_id: i64, job: Job) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(session_id)
            .or_insert(job);
    }
}
impl Job {
    pub(super) fn new(
        user_id: i64,
        agent_id: i64,
        from: i64,
        to: i64,
        input: Messages,
    ) -> Result<Self, Error> {
        // Session coverage endpoints are inclusive IDs; long-memory provenance
        // is the established half-open interval. Do not generate a new ID here.
        let to = to
            .checked_add(1)
            .filter(|to| *to > from)
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "invalid long memory source range"))?;
        let input = Messages::new(input.0.into_iter().filter_map(|mut message| {
            if !matches!(message.role, Role::User | Role::Assistant) {
                return None;
            }
            message.content.retain(|part| part.r#type == "text");
            message.values.clear();
            message.metadata.clear();
            message.usage = None;
            (!message.content.is_empty()).then_some(message)
        }));
        Ok(Self {
            user_id,
            agent_id,
            from,
            to,
            input,
            tasks: None,
        })
    }
    pub(super) fn covers(&self, from: i64, to: i64) -> bool {
        self.from == from && to.checked_add(1) == Some(self.to)
    }
    pub(super) fn validate(&self, user_id: i64) -> Result<(), Error> {
        if self.user_id != user_id
            || user_id <= 0
            || self.agent_id <= 0
            || self.from <= 0
            || self.to <= self.from
        {
            return Err(invalid());
        }
        if let Some(tasks) = &self.tasks {
            if !self.input.0.is_empty() || tasks.len() > 6 {
                return Err(invalid());
            }
            let mut seen = std::collections::BTreeSet::new();
            for task in tasks {
                if task.agent_id.is_some_and(|id| id != self.agent_id)
                    || task.content.trim().is_empty()
                    || !seen.insert((task.agent_id, format!("{:?}", task.kind)))
                    || (task.done && task.prepared.is_none())
                {
                    return Err(invalid());
                }
                if let Some(prepared) = &task.prepared {
                    let request = &prepared.request;
                    if request.id <= 0
                        || request.user_id != user_id
                        || request.agent_id != task.agent_id
                        || request.session_id.is_some()
                        || request.r#type != task.kind
                        || request.content.trim().is_empty()
                        || request.from > self.from
                        || request.from < 0
                        || request.to < self.to
                        || request.vector.is_some()
                    {
                        return Err(invalid());
                    }
                    if let Some(previous) = &prepared.previous
                        && (previous.id != request.id
                            || previous.version == 0
                            || previous.user_id != user_id
                            || previous.agent_id != task.agent_id
                            || previous.session_id.is_some()
                            || previous.r#type != task.kind
                            || previous.content.trim().is_empty()
                            || previous.from < 0
                            || previous.to <= previous.from)
                    {
                        return Err(invalid());
                    }
                }
            }
        } else if self.input.0.iter().any(|message| {
            !matches!(message.role, Role::User | Role::Assistant)
                || !message.values.is_empty()
                || !message.metadata.is_empty()
                || message.usage.is_some()
                || message.content.iter().any(|part| part.r#type != "text")
        }) {
            return Err(invalid());
        }
        Ok(())
    }
}

pub(super) trait Progress {
    fn save(
        &mut self,
        job: &Job,
        cancellation: &Cancellation,
    ) -> impl std::future::Future<Output = Result<(), Error>>;
}

/// Drop requeues failures/panics; no mutex is held while a component awaits.
#[cfg(test)]
struct Ticket {
    jobs: Arc<Jobs>,
    session_id: i64,
    job: Option<Job>,
}
#[cfg(test)]
impl Drop for Ticket {
    fn drop(&mut self) {
        if let Some(job) = self.job.take() {
            self.jobs.insert(self.session_id, job);
        }
    }
}

#[cfg(test)]
async fn resume<L: Memory>(
    jobs: Arc<Jobs>,
    session_id: i64,
    component: &memory::extraction::Configured,
    store: &L,
    next_id: fn() -> i64,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    let job = jobs
        .0
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&session_id);
    let Some(job) = job else {
        return Ok(());
    };
    let mut ticket = Ticket {
        jobs,
        session_id,
        job: Some(job),
    };
    struct Local;
    impl Progress for Local {
        async fn save(&mut self, _: &Job, _: &Cancellation) -> Result<(), Error> {
            Ok(())
        }
    }
    resume_job(
        ticket.job.as_mut().unwrap(),
        &mut Local,
        component,
        store,
        next_id,
        cancellation,
    )
    .await?;
    ticket.job = None;
    Ok(())
}

pub(super) async fn resume_job<L: Memory, P: Progress>(
    job: &mut Job,
    progress: &mut P,
    component: &memory::extraction::Configured,
    store: &L,
    next_id: fn() -> i64,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    if job.tasks.is_none() {
        let output = cancellable(
            component.extract(job.input.clone(), cancellation),
            cancellation,
        )
        .await?;
        let [message] = output.as_slice() else {
            return Err(invalid());
        };
        if message.role != Role::Assistant || !message.content.is_empty() {
            return Err(invalid());
        }
        memory::extraction::validate(&message.values)?;
        let mut tasks = Vec::new();
        for (level, types) in &message.values {
            for (kind, content) in types.as_object().unwrap() {
                let kind = match kind.as_str() {
                    "summary" => memory::Type::Summary,
                    "fact" => memory::Type::Fact,
                    "preference" => memory::Type::Preference,
                    _ => unreachable!("validated kind"),
                };
                tasks.push(Task {
                    agent_id: (level == "agent").then_some(job.agent_id),
                    kind,
                    content: content.as_str().unwrap().to_owned(),
                    prepared: None,
                    done: false,
                });
            }
        }
        job.tasks = Some(tasks);
        // A successful extraction is never repeated on a write retry. Retain
        // the smaller extracted content, not another copy of raw history.
        job.input = Messages::default();
        progress.save(job, cancellation).await?;
    }
    let mut error = None;
    for index in 0..job.tasks.as_ref().unwrap().len() {
        if job.tasks.as_ref().unwrap()[index].done {
            continue;
        }
        let prepared = prepare(
            &mut job.tasks.as_mut().unwrap()[index],
            job.user_id,
            job.from,
            job.to,
            component,
            store,
            next_id,
            cancellation,
        )
        .await;
        match prepared {
            Ok(already_committed) => {
                if already_committed {
                    job.tasks.as_mut().unwrap()[index].done = true;
                }
                // Retain successful merge and the allocated ID before writing.
                // A failed progress save must stop before the external effect.
                progress.save(job, cancellation).await?;
                if !already_committed {
                    let result =
                        write_prepared(&job.tasks.as_ref().unwrap()[index], store, cancellation)
                            .await;
                    match result {
                        Ok(()) => {
                            job.tasks.as_mut().unwrap()[index].done = true;
                            progress.save(job, cancellation).await?;
                        }
                        Err(value) => {
                            error.get_or_insert(value);
                        }
                    }
                }
            }
            Err(value) => {
                error.get_or_insert(value);
            }
        }
        if cancellation.is_cancelled() {
            break;
        }
    }
    if let Some(error) = error {
        return Err(error);
    }
    if cancellation.is_cancelled() {
        return Err(Error::new("CANCELLED", "memory writeback cancelled"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn prepare<L: Memory>(
    task: &mut Task,
    user_id: i64,
    from: i64,
    to: i64,
    component: &memory::extraction::Configured,
    store: &L,
    next_id: fn() -> i64,
    cancellation: &Cancellation,
) -> Result<bool, Error> {
    let current = cancellable(
        store.get(
            user_id,
            task.agent_id,
            None,
            task.kind.clone(),
            cancellation,
        ),
        cancellation,
    )
    .await?;
    if let Some(item) = &current
        && (item.id <= 0
            || item.version == 0
            || item.user_id != user_id
            || item.agent_id != task.agent_id
            || item.session_id.is_some()
            || item.r#type != task.kind
            || item.from < 0
            || item.to <= item.from
            || item.content.trim().is_empty())
    {
        return Err(invalid());
    }
    if let Some(prepared) = &task.prepared {
        let version = match &prepared.previous {
            Some(previous) => previous.version.checked_add(1),
            None => Some(1),
        };
        // A lost acknowledgement must not merge the same extracted text twice.
        if current
            .as_ref()
            .is_some_and(|item| Some(item.version) == version && matches(item, &prepared.request))
        {
            return Ok(true);
        }
    }
    if task
        .prepared
        .as_ref()
        .is_none_or(|prepared| prepared.previous != current)
    {
        let (id, content, from, to, metadata) = match &current {
            Some(item) => {
                let content = cancellable(
                    component.merge(item.clone(), task.content.clone(), cancellation),
                    cancellation,
                )
                .await?;
                if content.trim().is_empty() {
                    return Err(invalid());
                }
                (
                    item.id,
                    content,
                    item.from.min(from),
                    item.to.max(to),
                    item.metadata.clone(),
                )
            }
            None => {
                // Reuse an allocated new-record ID after a failed write.
                let id = if let Some(prepared) = &task.prepared {
                    prepared.request.id
                } else {
                    let id = next_id();
                    if id <= 0 {
                        return Err(Error::new(
                            "INVALID_ARGUMENTS",
                            "memory ID function must return a positive i64",
                        ));
                    }
                    id
                };
                (id, task.content.clone(), from, to, None)
            }
        };
        task.prepared = Some(Prepared {
            previous: current,
            request: Request {
                id,
                r#type: task.kind.clone(),
                user_id,
                agent_id: task.agent_id,
                session_id: None,
                from,
                to,
                content,
                metadata,
                vector: None,
            },
        });
    }
    Ok(false)
}

async fn write_prepared<L: Memory>(
    task: &Task,
    store: &L,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    let prepared = task.prepared.as_ref().unwrap();
    let version = match &prepared.previous {
        Some(previous) => previous.version.checked_add(1),
        None => Some(1),
    }
    .ok_or_else(|| Error::new("CONFLICT", "memory version exhausted"))?;
    let result = if let Some(previous) = &prepared.previous {
        cancellable(
            store.update(prepared.request.clone(), previous.version, cancellation),
            cancellation,
        )
        .await?
    } else {
        cancellable(
            store.write(prepared.request.clone(), cancellation),
            cancellation,
        )
        .await?
    };
    if result.version != version || !matches(&result, &prepared.request) {
        return Err(invalid());
    }
    Ok(())
}

fn matches(item: &Item, request: &Request) -> bool {
    item.id == request.id
        && item.r#type == request.r#type
        && item.user_id == request.user_id
        && item.agent_id == request.agent_id
        && item.session_id == request.session_id
        && item.from == request.from
        && item.to == request.to
        && item.content == request.content
        && item.metadata == request.metadata
}
fn invalid() -> Error {
    Error::new(
        "INVALID_RESPONSE",
        "invalid long memory transformation or backend response",
    )
}

#[cfg(test)]
mod tests;
