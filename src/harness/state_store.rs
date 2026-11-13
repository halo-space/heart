//! Local persistence for the durable Harness state.
//!
//! The state model and validation remain in `harness::state`; this module
//! owns only Workspace I/O, version locking and rollback persistence.

use super::state::{Change, Goal, Item, State};
use crate::components::workspace::Workspace;
use crate::{Cancellation, Error};

const LOCK_TIMEOUT_SECONDS: u64 = 300;
const LOCK_RETRY_MILLISECONDS: u64 = 10;

/// Local Workspace persistence for Harness state.
///
/// This is intentionally a small adapter around the existing local Workspace
/// file contract. It does not expose Runtime state, create a second checkpoint
/// store, or make the Harness depend on a particular database.
pub struct Store {
    workspace: crate::core::workspace::Local,
}

struct Lock {
    path: std::path::PathBuf,
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.path);
    }
}

impl Store {
    pub fn new(workspace: crate::core::workspace::Local) -> Self {
        Self { workspace }
    }

    /// Load the current Workspace Harness state. A missing state file means
    /// that this Workspace has not persisted Harness state yet.
    pub async fn load(&self, cancellation: &Cancellation) -> Result<Option<State>, Error> {
        self.load_unlocked(cancellation).await
    }

    async fn load_unlocked(&self, cancellation: &Cancellation) -> Result<Option<State>, Error> {
        let bytes = match self
            .workspace
            .read(".heart/harness/state.json", cancellation)
            .await
        {
            Ok(bytes) => bytes,
            Err(error) if error.code == "NOT_FOUND" => return Ok(None),
            Err(error) => return Err(error),
        };
        let state: State = serde_json::from_slice(&bytes)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid Harness state JSON"))?;
        state.validate()?;
        Ok(Some(state))
    }

    /// Persist the current state and one immutable version snapshot.
    ///
    /// The state file is committed through the Local Workspace's atomic write
    /// operation. The history snapshot is written first so a failed state
    /// commit never claims that an unrecorded version was accepted.
    pub async fn save(&self, state: &State, cancellation: &Cancellation) -> Result<(), Error> {
        let _lock = self.lock(cancellation).await?;
        self.save_unlocked(state, cancellation).await
    }

    async fn save_unlocked(&self, state: &State, cancellation: &Cancellation) -> Result<(), Error> {
        state.validate()?;
        if let Some(current) = self.load_unlocked(cancellation).await? {
            if state.version < current.version {
                return Err(Error::new(
                    "CONFLICT",
                    "Harness state version is older than the current version",
                ));
            }
            if state.version == current.version {
                if serde_json::to_value(state).ok() == serde_json::to_value(current).ok() {
                    return Ok(());
                }
                return Err(Error::new(
                    "CONFLICT",
                    "Harness state version was already written with different content",
                ));
            }
        }
        let encoded = serde_json::to_vec_pretty(state)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "failed to encode Harness state"))?;
        let history_path = format!(".heart/harness/history/{}.json", state.version);
        self.workspace
            .write(&history_path, encoded.clone(), cancellation)
            .await?;
        self.workspace
            .write(".heart/harness/state.json", encoded, cancellation)
            .await
    }

    /// Apply one accepted Refinement against the latest local state and save
    /// the resulting version. The read happens immediately before the write,
    /// so callers do not update an old in-memory snapshot by accident.
    pub async fn refine(
        &self,
        item: Item,
        change: Change,
        cancellation: &Cancellation,
    ) -> Result<State, Error> {
        let _lock = self.lock(cancellation).await?;
        let mut state = self.load_unlocked(cancellation).await?.unwrap_or_default();
        state.upsert(item, change)?;
        self.save_unlocked(&state, cancellation).await?;
        Ok(state)
    }

    /// Update the current Goal as one versioned Harness change.
    pub async fn update_goal(
        &self,
        goal: Goal,
        change: Change,
        cancellation: &Cancellation,
    ) -> Result<State, Error> {
        let _lock = self.lock(cancellation).await?;
        let mut state = self.load_unlocked(cancellation).await?.unwrap_or_default();
        state.update_goal(goal, change)?;
        self.save_unlocked(&state, cancellation).await?;
        Ok(state)
    }

    /// Restore a historical snapshot as a new current version. The old
    /// snapshot remains untouched and the rollback itself is recorded as an
    /// accepted Harness change.
    pub async fn rollback(
        &self,
        version: u64,
        cancellation: &Cancellation,
    ) -> Result<State, Error> {
        if version == 0 {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "Harness rollback version must be positive",
            ));
        }
        let _lock = self.lock(cancellation).await?;
        let path = format!(".heart/harness/history/{version}.json");
        let bytes = self.workspace.read(&path, cancellation).await?;
        let mut restored: State = serde_json::from_slice(&bytes)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "invalid Harness history JSON"))?;
        restored.validate()?;
        let current = self.load_unlocked(cancellation).await?.unwrap_or_default();
        let next_version = current.version.saturating_add(1);
        restored.version = next_version;
        let now = now();
        for item in &mut restored.items {
            item.version = next_version;
            item.updated_time = now;
        }
        restored.changes.push(Change {
            id: format!("rollback-{version}-{next_version}"),
            trigger: "rollback".into(),
            version: next_version,
            changes: vec![format!("restored version {version}")],
            evidence: vec![format!("history:{version}")],
            outcome: "accepted".into(),
            created_time: now,
        });
        self.save_unlocked(&restored, cancellation).await?;
        Ok(restored)
    }

    async fn lock(&self, cancellation: &Cancellation) -> Result<Lock, Error> {
        let root = self.workspace.definition().path.as_ref().ok_or_else(|| {
            Error::new("INVALID_ARGUMENTS", "local Harness Workspace has no path")
        })?;
        let lock_path = std::path::PathBuf::from(root).join(".heart/harness/state.lock");
        let parent = lock_path
            .parent()
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "invalid Harness lock path"))?;
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|_| Error::new("IO_ERROR", "failed to prepare Harness lock"))?;
        loop {
            crate::cancellation::check(cancellation)?;
            match tokio::fs::create_dir(&lock_path).await {
                Ok(()) => return Ok(Lock { path: lock_path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let stale = tokio::fs::metadata(&lock_path)
                        .await
                        .ok()
                        .and_then(|metadata| metadata.modified().ok())
                        .and_then(|modified| modified.elapsed().ok())
                        .is_some_and(|age| age.as_secs() >= LOCK_TIMEOUT_SECONDS);
                    if stale {
                        let _ = tokio::fs::remove_dir(&lock_path).await;
                        continue;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(LOCK_RETRY_MILLISECONDS))
                        .await;
                }
                Err(_) => {
                    return Err(Error::new("IO_ERROR", "failed to acquire Harness lock"));
                }
            }
        }
    }
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_secs()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}
