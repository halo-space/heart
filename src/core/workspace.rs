//! Resource persistence, not hostile-process isolation. A local root must be
//! trusted against concurrent symlink/path replacement by other processes.

use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
    sync::{
        RwLock,
        atomic::{AtomicU64, Ordering},
    },
};

use tokio::io::AsyncWriteExt;

use crate::{
    Cancellation, Error,
    workspace::{Definition, Mode, Workspace},
};

use super::operation::{cancellable, check};

static TEMP: AtomicU64 = AtomicU64::new(0);

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        // Also cleans up if the caller drops the write Future. This is a
        // single exact path created by this operation, never recursive.
        let _ = std::fs::remove_file(&self.0);
    }
}

pub struct Local {
    root: PathBuf,
    definition: Definition,
}

impl Local {
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
        })
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
}

impl Workspace for Local {
    fn definition(&self) -> &Definition {
        &self.definition
    }

    async fn read(&self, path: &str, cancellation: &Cancellation) -> Result<Vec<u8>, Error> {
        check(cancellation)?;
        let path = self.resolve(path, cancellation).await?;
        cancellable(
            async { tokio::fs::read(path).await.map_err(io_error) },
            cancellation,
        )
        .await
    }

    async fn write(
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

    async fn delete(&self, path: &str, cancellation: &Cancellation) -> Result<(), Error> {
        check(cancellation)?;
        let path = self.resolve(path, cancellation).await?;
        check(cancellation)?;
        // No recursive directory deletion.
        tokio::fs::remove_file(path).await.map_err(io_error)
    }
}

/// Zero-filesystem reference backend for server mode. Application database
/// implementations implement the same Workspace contract, not another trait.
pub struct InMemory {
    definition: Definition,
    resources: RwLock<BTreeMap<String, Vec<u8>>>,
}

impl InMemory {
    pub fn new() -> Self {
        Self {
            definition: Definition {
                mode: Mode::Server,
                path: None,
                metadata: Default::default(),
            },
            resources: RwLock::new(BTreeMap::new()),
        }
    }
}

impl Default for InMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl Workspace for InMemory {
    fn definition(&self) -> &Definition {
        &self.definition
    }
    async fn read(&self, path: &str, cancellation: &Cancellation) -> Result<Vec<u8>, Error> {
        check(cancellation)?;
        validate_path(path)?;
        self.resources
            .read()
            .map_err(|_| Error::new("INTERNAL", "workspace lock poisoned"))?
            .get(path)
            .cloned()
            .ok_or_else(|| Error::new("NOT_FOUND", "workspace resource not found"))
    }
    async fn write(
        &self,
        path: &str,
        data: Vec<u8>,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        check(cancellation)?;
        validate_path(path)?;
        let mut resources = self
            .resources
            .write()
            .map_err(|_| Error::new("INTERNAL", "workspace lock poisoned"))?;
        check(cancellation)?;
        resources.insert(path.to_owned(), data);
        Ok(())
    }
    async fn delete(&self, path: &str, cancellation: &Cancellation) -> Result<(), Error> {
        check(cancellation)?;
        validate_path(path)?;
        let mut resources = self
            .resources
            .write()
            .map_err(|_| Error::new("INTERNAL", "workspace lock poisoned"))?;
        check(cancellation)?;
        resources
            .remove(path)
            .map(|_| ())
            .ok_or_else(|| Error::new("NOT_FOUND", "workspace resource not found"))
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
        _ => "IO_ERROR",
    };
    // OS messages may reveal absolute paths; do not expose raw diagnostics.
    Error::new(code, "workspace resource operation failed")
}
