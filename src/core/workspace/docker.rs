//! One persistent, application-selected container for file and command tools.
//! Construction performs no Docker I/O. The first asynchronous operation starts
//! the environment; it is never silently replaced by a fresh, empty container.

use std::{collections::BTreeMap, path::PathBuf};

use serde_json::{Value, json};
use tokio::sync::Mutex;

use super::tools;
use crate::{
    Cancellation, Error,
    core::{
        agent::toolkit::Toolkit,
        operation::{cancellable, check},
        sandbox::docker as sandbox,
    },
    workspace::{Definition, Mode, Workspace},
};

mod files;

/// Image and resource limits reuse the existing Docker Sandbox configuration.
/// No host directory is mounted unless `host_workdir` is explicitly provided.
/// The prepared Linux image must provide /bin/sh, sleep, cat, mkdir, mv, rm and
/// find (including -print0, -mindepth and -maxdepth). Nothing is installed.
#[derive(Clone)]
pub struct Config {
    pub sandbox: sandbox::Config,
    pub workdir: String,
    pub host_workdir: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sandbox: Default::default(),
            workdir: "/workspace".into(),
            host_workdir: None,
            env: Default::default(),
        }
    }
}

pub struct Docker {
    backend: Box<Backend>,
    definition: Definition,
    toolkit: Toolkit,
    pub(super) discover_skills: bool,
}

impl Docker {
    /// Validate/store settings only. No CLI invocation, image pull, installation
    /// or host execution fallback. File operations lazily initialize once.
    pub fn new(config: Config) -> Result<Self, Error> {
        config.sandbox.validate()?;
        let relative = config
            .workdir
            .strip_prefix('/')
            .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "Docker workdir must be absolute"))?;
        super::validate_path(relative)?;
        if config.workdir.contains([',', ':'])
            || config.workdir.chars().any(char::is_control)
            || config
                .host_workdir
                .as_ref()
                .is_some_and(|path| path.to_str().is_none_or(|text| text.contains([',', '\0'])))
            || config.env.iter().any(|(key, value)| {
                key.is_empty()
                    || !key.bytes().enumerate().all(|(i, byte)| {
                        byte == b'_'
                            || byte.is_ascii_alphabetic()
                            || (i > 0 && byte.is_ascii_digit())
                    })
                    || value.contains('\0')
            })
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "invalid Docker workspace configuration",
            ));
        }
        Ok(Self {
            definition: Definition {
                mode: Mode::Local,
                // Container path, not a fictional host directory. No secrets
                // or host mount configuration enter the public definition.
                path: Some(config.workdir.clone()),
                metadata: serde_json::Map::from_iter([("environment".into(), json!("docker"))]),
            },
            backend: Box::new(Backend {
                config,
                state: Mutex::new(State::New),
            }),
            toolkit: Toolkit::new(),
            discover_skills: false,
        })
    }

    /// Discover `skills/<name>/SKILL.md` inside this container for each run.
    /// Opt-in because discovery starts the container before model dispatch or
    /// Tool permission checks; the default preserves no-I/O construction.
    pub fn with_skill_discovery(mut self) -> Self {
        self.discover_skills = true;
        self
    }

    pub fn toolkit(mut self, toolkit: Toolkit) -> Self {
        self.toolkit = toolkit;
        self
    }

    pub async fn list(
        &self,
        path: &str,
        recursive: bool,
        cancellation: &Cancellation,
    ) -> Result<Vec<String>, Error> {
        self.backend.list(path, recursive, cancellation).await
    }

    pub async fn read(&self, path: &str, cancellation: &Cancellation) -> Result<Vec<u8>, Error> {
        self.backend.read(path, cancellation).await
    }

    pub async fn write(
        &self,
        path: &str,
        data: Vec<u8>,
        cancellation: &Cancellation,
    ) -> Result<(), Error> {
        self.backend.write(path, data, cancellation).await
    }

    pub async fn delete(&self, path: &str, cancellation: &Cancellation) -> Result<(), Error> {
        self.backend.delete(path, cancellation).await
    }

    /// Deterministic release for direct users. Closing is terminal and does not
    /// remove host-mounted files. Tool-owned environments are released when
    /// their final owner is dropped; keep Tokio alive for best-effort cleanup.
    pub async fn close(&self) -> Result<(), Error> {
        tokio::runtime::Handle::try_current()
            .map_err(|_| Error::new("UNAVAILABLE", "Docker workspace requires a Tokio runtime"))?;
        let mut state = self.backend.state.lock().await;
        if let State::Ready(mut container) = std::mem::replace(&mut *state, State::Closed) {
            container.guard.cleanup().await?;
        }
        Ok(())
    }

    pub(super) async fn command(
        &self,
        input: Value,
        cancellation: &Cancellation,
    ) -> Result<Value, Error> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Input {
            args: Vec<String>,
        }
        let input: Input = serde_json::from_value(input)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "command input requires args"))?;
        if input.args.is_empty()
            || input.args[0].trim().is_empty()
            || input.args.iter().any(|arg| arg.contains('\0'))
            || input.args.iter().map(String::len).sum::<usize>() > 1024 * 1024
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "invalid sandbox command arguments",
            ));
        }
        let output = self
            .backend
            .execute(&input.args, None, cancellation)
            .await?;
        let stdout = String::from_utf8(output.stdout)
            .map_err(|_| Error::new("INVALID_RESPONSE", "sandbox stdout is not UTF-8"))?;
        let stderr = String::from_utf8(output.stderr)
            .map_err(|_| Error::new("INVALID_RESPONSE", "sandbox stderr is not UTF-8"))?;
        let data = json!({"stdout":stdout,"stderr":stderr,"exit_code":output.code});
        if output.code != 0 {
            return Err(Error::new("EXECUTION_FAILED", "sandbox command failed")
                .with_details(data.to_string()));
        }
        Ok(data)
    }
}

impl Workspace for Docker {
    fn definition(&self) -> &Definition {
        &self.definition
    }
}

impl From<Docker> for Toolkit {
    fn from(mut workspace: Docker) -> Self {
        let toolkit = std::mem::take(&mut workspace.toolkit);
        tools::attach(tools::Files::Docker(workspace), toolkit)
    }
}

// This is private backend lifecycle state, not Runtime or a second Workspace
// component protocol. Serializing exec also makes cancellation cleanup safe.
struct Backend {
    config: Config,
    state: Mutex<State>,
}
enum State {
    New,
    Ready(Container),
    Closed,
}
struct Container {
    id: String,
    guard: sandbox::Guard,
}

impl Backend {
    async fn execute(
        &self,
        args: &[String],
        input: Option<&[u8]>,
        cancellation: &Cancellation,
    ) -> Result<sandbox::Bytes, Error> {
        check(cancellation)?;
        tokio::runtime::Handle::try_current()
            .map_err(|_| Error::new("UNAVAILABLE", "Docker workspace requires a Tokio runtime"))?;
        let mut state = cancellable(async { Ok(self.state.lock().await) }, cancellation).await?;
        // Ownership moves into this operation. If its Future is dropped, the
        // guard cleans up, and the shared state stays Closed, never New.
        let (mut container, initialize) = match std::mem::replace(&mut *state, State::Closed) {
            State::New => (
                Container {
                    id: String::new(),
                    guard: sandbox::Guard::new("docker".into(), "heart-workspace"),
                },
                true,
            ),
            State::Ready(container) => (container, false),
            State::Closed => return Err(Error::new("UNAVAILABLE", "Docker workspace is closed")),
        };
        let result = cancellable(
            async {
                tokio::time::timeout(self.config.sandbox.timeout, async {
                    if initialize {
                        self.initialize(&mut container).await?;
                    }
                    let mut command = vec!["exec".into()];
                    if input.is_some() {
                        command.push("--interactive".into());
                    }
                    command.extend([
                        "--workdir".into(),
                        self.config.workdir.clone(),
                        "--".into(),
                        container.id.clone(),
                    ]);
                    command.extend_from_slice(args);
                    let output = sandbox::cli_bytes(
                        std::path::Path::new("docker"),
                        &command,
                        input,
                        self.config.sandbox.max_output,
                    )
                    .await?;
                    // Distinguish a failed command from a lost/stopped environment.
                    self.running(&container.id).await?;
                    Ok(output)
                })
                .await
                .map_err(|_| Error::new("TIMEOUT", "workspace operation timed out"))?
            },
            cancellation,
        )
        .await;
        match result {
            Ok(output) => {
                *state = State::Ready(container);
                Ok(output)
            }
            Err(error) => {
                // Killing the CLI alone does not kill docker exec. Close the
                // owned container so an unknown writer cannot keep running
                // behind later file operations. Never retry on a fresh image.
                if let Err(cleanup) = container.guard.cleanup().await {
                    eprintln!("workspace cleanup failed: {}", cleanup.code);
                }
                Err(error)
            }
        }
    }

    async fn initialize(&self, container: &mut Container) -> Result<(), Error> {
        let mut args = vec![
            "create".into(),
            "--name".into(),
            container.guard.name().into(),
            "--label".into(),
            format!("{}={}", sandbox::LABEL, container.guard.token()),
        ];
        args.extend(self.config.sandbox.create_args());
        if self.config.workdir != "/tmp" {
            args.extend([
                "--tmpfs".into(),
                "/tmp:rw,nosuid,nodev,size=64m,mode=1777".into(),
            ]);
        }
        if let Some(host) = &self.config.host_workdir {
            let host = tokio::fs::canonicalize(host)
                .await
                .map_err(super::io_error)?;
            if !tokio::fs::metadata(&host)
                .await
                .map_err(super::io_error)?
                .is_dir()
            {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "host_workdir must be an existing directory",
                ));
            }
            let host = host
                .to_str()
                .filter(|path| !path.contains([',', '\0']))
                .ok_or_else(|| Error::new("INVALID_ARGUMENTS", "unsupported host_workdir path"))?;
            args.extend([
                "--mount".into(),
                format!("type=bind,source={host},target={}", self.config.workdir),
            ]);
        } else {
            args.extend([
                "--tmpfs".into(),
                format!(
                    "{}:rw,nosuid,nodev,size={},mode=1777",
                    self.config.workdir, self.config.sandbox.memory
                ),
            ]);
        }
        for (name, value) in &self.config.env {
            args.extend(["--env".into(), format!("{name}={value}")]);
        }
        args.extend(["--workdir".into(), self.config.workdir.clone(), "--entrypoint".into(), "/bin/sh".into(),
            self.config.sandbox.image.clone(), "-c".into(),
            "for cmd in sleep cat mkdir mv rm find; do command -v \"$cmd\" >/dev/null || exit 127; done; trap 'exit 0' TERM INT; while :; do sleep 3600 & wait \"$!\"; done".into()]);
        let created = sandbox::cli_bytes(std::path::Path::new("docker"), &args, None, 4096).await?;
        // Do not copy Docker stderr: it can echo explicit environment secrets.
        if created.code != 0 {
            return Err(Error::new(
                "UNAVAILABLE",
                "Docker could not create workspace",
            ));
        }
        let id = std::str::from_utf8(&created.stdout)
            .unwrap_or_default()
            .trim();
        if !sandbox::container_id(id) {
            return Err(Error::new(
                "INVALID_RESPONSE",
                "invalid Docker container ID",
            ));
        }
        container.id = id.into();
        let started = sandbox::cli_bytes(
            std::path::Path::new("docker"),
            &["start".into(), id.into()],
            None,
            4096,
        )
        .await?;
        if started.code != 0 {
            return Err(Error::new(
                "UNAVAILABLE",
                "Docker could not start workspace",
            ));
        }
        self.running(id).await
    }

    async fn running(&self, id: &str) -> Result<(), Error> {
        let inspected = sandbox::cli_bytes(
            std::path::Path::new("docker"),
            &[
                "inspect".into(),
                "--format".into(),
                "{{json .State}}".into(),
                id.into(),
            ],
            None,
            4096,
        )
        .await?;
        if inspected.code != 0 {
            return Err(Error::new(
                "UNAVAILABLE",
                "Docker workspace could not be inspected",
            ));
        }
        let state: Value = serde_json::from_slice(&inspected.stdout)
            .map_err(|_| Error::new("INVALID_RESPONSE", "invalid Docker workspace state"))?;
        if state.get("OOMKilled").and_then(Value::as_bool) == Some(true) {
            return Err(Error::new(
                "RESOURCE_EXHAUSTED",
                "workspace memory limit exceeded",
            ));
        }
        if state.get("Running").and_then(Value::as_bool) != Some(true) {
            return Err(Error::new("UNAVAILABLE", "Docker workspace is not running"));
        }
        Ok(())
    }
}
