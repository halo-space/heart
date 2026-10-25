//! Resource persistence, not hostile-process isolation. A local root must be
//! trusted against concurrent symlink/path replacement by other processes.

use sha2::{Digest, Sha256};
use std::{
    path::{Component, Path, PathBuf},
    sync::Arc,
    sync::atomic::{AtomicU64, Ordering},
};

use tokio::{io::AsyncWriteExt, sync::RwLock};

use crate::{
    Cancellation, Error,
    components::{
        mcp::Server,
        skill::{Definition as SkillDefinition, Skill},
    },
    workspace::{Definition, Mode},
};

use super::agent::dispatch::{Dispatch, ToolInputs};
use super::operation::{cancellable, check};

pub mod docker;
pub mod es;
pub use docker::Docker;
mod tools;

static TEMP: AtomicU64 = AtomicU64::new(0);

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        // Also cleans up if the caller drops the write Future. This is a
        // single exact path created by this operation, never recursive.
        let _ = std::fs::remove_file(&self.0);
    }
}

#[derive(Clone)]
pub struct Local {
    root: PathBuf,
    definition: Definition,
    dispatch: Dispatch,
    sandbox: Option<crate::sandbox::Local>,
    pub(super) access: Arc<RwLock<()>>,
    pub(super) skill_cache: tools::SkillCache,
}

impl Local {
    pub(super) fn skill_catalog(&self) -> Vec<SkillDefinition> {
        let mut definitions = self.dispatch.skill_catalog();
        let root = self.root.join("skills");
        let Ok(entries) = std::fs::read_dir(root) else {
            return definitions;
        };
        definitions.extend(
            entries
                .filter_map(Result::ok)
                .filter_map(|entry| {
                    let path = entry.path();
                    if !entry.file_type().ok()?.is_dir() {
                        return None;
                    }
                    let markdown = std::fs::read_to_string(path.join("SKILL.md")).ok()?;
                    let skill = crate::core::skill::Skill::from_markdown(&markdown).ok()?;
                    let name = entry.file_name().to_str()?.to_owned();
                    (skill.definition().name == name).then(|| skill.definition().clone())
                })
                .collect::<Vec<_>>(),
        );
        definitions
    }

    /// Opens an existing application-owned directory. Does not invent or
    /// initialize an application workspace layout.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, Error> {
        let root = std::fs::canonicalize(root).map_err(io_error)?;
        if !root.is_dir() {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "workspace root must be a directory",
            ));
        }
        let path = root
            .to_str()
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "workspace root must be UTF-8"))?
            .to_owned();
        Ok(Self {
            root,
            definition: Definition {
                mode: Mode::Local,
                path: Some(path),
                metadata: Default::default(),
            },
            dispatch: Dispatch::new(),
            sandbox: None,
            access: Arc::new(RwLock::new(())),
            skill_cache: Arc::new(RwLock::new(None)),
        })
    }

    /// Register the application's Tools, MCP servers and Skills in this
    /// workspace. This only stores configuration; it performs no tool calls
    /// or discovery. Standard local file tools
    /// are added when the Workspace is consumed by an Agent or Harness.
    #[allow(private_bounds)]
    pub fn tools<I>(mut self, tools: I) -> Self
    where
        I: ToolInputs,
    {
        self.dispatch = self.dispatch.tools(tools);
        self
    }

    /// Register already connected and discovered MCP servers on this Workspace.
    #[allow(private_bounds)]
    pub fn mcps<S>(mut self, servers: impl IntoIterator<Item = S>) -> Self
    where
        S: Server + 'static,
    {
        self.dispatch = self.dispatch.mcps(servers);
        self
    }

    /// Register in-process Skills on this Workspace. Local directory discovery
    /// remains owned by the Workspace backend and is cached there.
    #[allow(private_bounds)]
    pub fn skills<S>(mut self, skills: impl IntoIterator<Item = S>) -> Result<Self, Error>
    where
        S: Skill + 'static,
    {
        self.dispatch = self.dispatch.skills(skills)?;
        Ok(self)
    }

    /// Explicitly select host command execution, automatically binding its
    /// Tool to this root. This does not start a process or provide isolation.
    /// Docker execution must use its own shared-environment Workspace instead.
    pub fn sandbox(mut self, sandbox: crate::sandbox::Local) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    async fn resolve(&self, path: &str, cancellation: &Cancellation) -> Result<PathBuf, Error> {
        validate_path(path)?;
        let target = self.root.join(path);
        // Reject all existing symlink components, including a final link. A
        // not-yet-existing suffix is allowed for writes but not read success.
        let mut current = self.root.clone();
        for component in Path::new(path).components() {
            check(cancellation)?;
            current.push(component.as_os_str());
            match tokio::fs::symlink_metadata(&current).await {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err(Error::new(
                        "PERMISSION_DENIED",
                        "workspace symlinks are not allowed",
                    ));
                }
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
                Err(error) => return Err(io_error(error)),
            }
        }
        Ok(target)
    }

    /// Return a content fingerprint for the complete local Skill tree.
    ///
    /// Skill files can be replaced without changing their size or the
    /// filesystem timestamp resolution available to the process. Hashing the
    /// bytes makes cache invalidation depend on the actual Workspace content,
    /// matching the Docker-backed loader and preventing an Agent from using a
    /// stale Skill after an in-place update.
    pub(super) async fn skill_fingerprint(
        &self,
        cancellation: &Cancellation,
    ) -> Result<String, Error> {
        let mut paths = match self.list("skills", true, cancellation).await {
            Ok(paths) => paths,
            Err(error) if error.code == "NOT_FOUND" => Vec::new(),
            Err(error) => return Err(error),
        };
        paths.sort();
        let mut hasher = Sha256::new();
        for path in paths {
            let bytes = self.read(&path, cancellation).await?;
            hasher.update(path.as_bytes());
            hasher.update([0]);
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
            crate::core::operation::check(cancellation)?;
        }
        Ok(format!("sha256:{:x}", hasher.finalize()))
    }
}

impl From<Local> for Dispatch {
    fn from(mut workspace: Local) -> Self {
        let dispatch = std::mem::take(&mut workspace.dispatch);
        tools::attach(workspace.into(), dispatch)
    }
}

impl Local {
    /// Internal Tool catalog used to implement the Workspace contract.
    pub(super) fn tool_catalog(&self) -> Vec<crate::tool::ToolDefinition> {
        Dispatch::from(self.clone()).tool_catalog()
    }

    pub(super) fn mcp_catalog(&self) -> Vec<crate::mcp::Definition> {
        Dispatch::from(self.clone()).mcp_catalog()
    }

    /// Execute one configured or built-in Workspace Tool.
    pub async fn execute(
        &self,
        call: crate::tool::Call,
        cancellation: &Cancellation,
    ) -> Result<(crate::Message, crate::tool::Evaluation), Error> {
        Dispatch::from(self.clone())
            .execute(call, cancellation)
            .await
    }
}

impl crate::workspace::Workspace for Local {
    fn definition(&self) -> &Definition {
        &self.definition
    }

    fn tool(&self) -> Vec<crate::tool::ToolDefinition> {
        self.tool_catalog()
    }

    fn mcp(&self) -> Vec<crate::mcp::Definition> {
        self.mcp_catalog()
    }

    fn skill(&self) -> Vec<SkillDefinition> {
        self.skill_catalog()
    }

    fn execute<'a>(
        &'a self,
        call: crate::tool::Call,
        cancellation: &'a Cancellation,
    ) -> crate::workspace::ToolFuture<'a> {
        Box::pin(self.execute(call, cancellation))
    }
}

impl Local {
    /// List sorted workspace-relative paths. `.` is the root; shallow listing
    /// includes directories, recursive listing returns files without symlinks.
    pub async fn list(
        &self,
        path: &str,
        recursive: bool,
        cancellation: &Cancellation,
    ) -> Result<Vec<String>, Error> {
        check(cancellation)?;
        let base = if path == "." {
            self.root.clone()
        } else {
            self.resolve(path, cancellation).await?
        };
        cancellable(
            async {
                let mut pending = vec![base];
                let mut paths = Vec::new();
                while let Some(directory) = pending.pop() {
                    check(cancellation)?;
                    let mut entries = tokio::fs::read_dir(directory).await.map_err(io_error)?;
                    while let Some(entry) = entries.next_entry().await.map_err(io_error)? {
                        check(cancellation)?;
                        let kind = entry.file_type().await.map_err(io_error)?;
                        if kind.is_symlink() || (!kind.is_file() && !kind.is_dir()) {
                            continue;
                        }
                        let entry_path = entry.path();
                        if kind.is_dir() && recursive {
                            pending.push(entry_path);
                        } else {
                            let relative = entry_path.strip_prefix(&self.root).map_err(|_| {
                                Error::new("PERMISSION_DENIED", "workspace path escaped its root")
                            })?;
                            let relative = relative.to_str().ok_or_else(|| {
                                Error::new("INVALID_ARGUMENTS", "workspace path must be UTF-8")
                            })?;
                            paths.push(relative.replace(std::path::MAIN_SEPARATOR, "/"));
                        }
                    }
                }
                paths.sort();
                Ok(paths)
            },
            cancellation,
        )
        .await
    }

    pub async fn read(&self, path: &str, cancellation: &Cancellation) -> Result<Vec<u8>, Error> {
        check(cancellation)?;
        let path = self.resolve(path, cancellation).await?;
        cancellable(
            async { tokio::fs::read(path).await.map_err(io_error) },
            cancellation,
        )
        .await
    }

    pub async fn write(
        &self,
        path: &str,
        data: Vec<u8>,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        check(cancellation)?;
        let target = self.resolve(path, cancellation).await?;
        let parent = target
            .parent()
            .expect("relative workspace resource has parent");
        tokio::fs::create_dir_all(parent).await.map_err(io_error)?;
        check(cancellation)?;
        let temp = parent.join(format!(
            ".halo-write-{}-{}",
            std::process::id(),
            TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        // Open and arm cleanup in the same poll. Dropping a pending async
        // open could otherwise leave a file created later by its worker.
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(io_error)?;
        let _cleanup = Temporary(temp.clone());
        let mut file = tokio::fs::File::from_std(file);
        let written = cancellable(
            async {
                file.write_all(&data).await.map_err(io_error)?;
                file.flush().await.map_err(io_error)
            },
            cancellation,
        )
        .await;
        drop(file);
        if let Err(error) = written {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(error);
        }
        if let Err(error) = check(cancellation) {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(error);
        }
        // Rename is the commit point. Once committed, report success; do not
        // turn a persisted successful write into CANCELLED after the fact.
        let committed = tokio::fs::rename(&temp, target).await.map_err(io_error);
        if committed.is_err() {
            let _ = tokio::fs::remove_file(&temp).await;
        }
        committed
    }

    pub async fn delete(&self, path: &str, cancellation: &Cancellation) -> Result<(), Error> {
        check(cancellation)?;
        let path = self.resolve(path, cancellation).await?;
        check(cancellation)?;
        // No recursive directory deletion.
        tokio::fs::remove_file(path).await.map_err(io_error)
    }
}

fn validate_path(path: &str) -> Result<(), Error> {
    if path.is_empty()
        || path.contains('\0')
        || path.contains('\\')
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || Path::new(path)
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(Error::new(
            "INVALID_ARGUMENTS",
            "workspace resource requires a normalized relative path",
        ));
    }
    Ok(())
}

fn io_error(error: std::io::Error) -> Error {
    let code = match error.kind() {
        std::io::ErrorKind::NotFound => "NOT_FOUND",
        std::io::ErrorKind::PermissionDenied => "PERMISSION_DENIED",
        std::io::ErrorKind::AlreadyExists => "CONFLICT",
        std::io::ErrorKind::NotADirectory => "INVALID_ARGUMENTS",
        _ => "IO_ERROR",
    };
    // OS messages may reveal absolute paths; do not expose raw diagnostics.
    Error::new(code, "workspace resource operation failed")
}
