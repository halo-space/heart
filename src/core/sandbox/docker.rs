//! Optional Docker isolation. Never installs Docker, pulls an image, exposes
//! host mounts, or falls back to executing the requested command on the host.

use std::{path::PathBuf, process::Stdio, time::Duration};

use serde::Deserialize;
use serde_json::json;
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    process::Command,
};

use crate::{
    Cancellation, Error,
    sandbox::{Request, Response, Sandbox},
};

pub(crate) const LABEL: &str = "halo.agents.operation";

/// Concrete Docker settings, not Runtime configuration. Image must already be
/// available to the application's Docker daemon. No image is chosen implicitly.
#[derive(Clone, Debug)]
pub struct Config {
    pub image: String,
    pub memory: u64,
    pub cpus: f64,
    pub pids: u32,
    pub timeout: Duration,
    pub max_output: usize,
    pub network: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            image: String::new(),
            memory: 512 * 1024 * 1024,
            cpus: 1.0,
            pids: 128,
            timeout: Duration::from_secs(30),
            max_output: 1024 * 1024,
            network: false,
        }
    }
}

impl Config {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.image.is_empty()
            || self.image.starts_with('-')
            || self
                .image
                .chars()
                .any(|c| c.is_whitespace() || c.is_control())
            || self.memory < 6 * 1024 * 1024
            || !self.cpus.is_finite()
            || self.cpus <= 0.0
            || self.pids == 0
            || self.timeout.is_zero()
            || self.max_output == 0
            || self.max_output == usize::MAX
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "invalid Docker sandbox configuration",
            ));
        }
        Ok(())
    }

    /// Shared security/resource policy for one-shot Sandbox and persistent
    /// Workspace containers. Mounts and process lifetime belong to the caller.
    pub(crate) fn create_args(&self) -> Vec<String> {
        vec![
            "--pull".into(),
            "never".into(),
            "--init".into(),
            "--no-healthcheck".into(),
            "--read-only".into(),
            "--cap-drop".into(),
            "ALL".into(),
            "--security-opt".into(),
            "no-new-privileges".into(),
            "--user".into(),
            "65534:65534".into(),
            "--network".into(),
            if self.network { "bridge" } else { "none" }.into(),
            "--memory".into(),
            self.memory.to_string(),
            "--memory-swap".into(),
            self.memory.to_string(),
            "--cpus".into(),
            self.cpus.to_string(),
            "--pids-limit".into(),
            self.pids.to_string(),
        ]
    }
}

pub struct Docker {
    config: Config,
    program: PathBuf,
}

impl Docker {
    /// Construction only validates local settings; it does not start a daemon
    /// or container. Execution requires an existing Tokio I/O/time runtime.
    pub fn new(config: Config) -> Result<Self, Error> {
        config.validate()?;
        Ok(Self {
            config,
            program: PathBuf::from("docker"),
        })
    }

    async fn run(&self, args: &[String], guard: &Guard) -> Result<Response, Error> {
        let mut create = vec![
            "create".into(),
            "--name".into(),
            guard.name.clone(),
            "--label".into(),
            format!("{LABEL}={}", guard.token),
        ];
        create.extend(self.config.create_args());
        create.extend([
            "--tmpfs".into(),
            "/tmp:rw,nosuid,nodev,size=64m,mode=1777".into(),
            "--tmpfs".into(),
            "/workspace:rw,nosuid,nodev,size=64m,mode=1777".into(),
            "--workdir".into(),
            "/workspace".into(),
            "--entrypoint".into(),
            args[0].clone(),
            self.config.image.clone(),
        ]);
        create.extend_from_slice(&args[1..]);
        let created = cli(&self.program, &create, 4096).await?;
        if !created.success {
            return Err(Error::new("UNAVAILABLE", "Docker could not create sandbox")
                .with_details(created.stderr));
        }
        let id = created.stdout.trim();
        if !container_id(id) {
            return Err(Error::new(
                "INVALID_RESPONSE",
                "Docker returned an invalid container ID",
            ));
        }
        let output = cli(
            &self.program,
            &["start".into(), "--attach".into(), id.into()],
            self.config.max_output,
        )
        .await?;
        let state = cli(
            &self.program,
            &[
                "inspect".into(),
                "--format".into(),
                "{{json .State}}".into(),
                id.into(),
            ],
            4096,
        )
        .await?;
        if !state.success {
            return Err(Error::new("UNAVAILABLE", "Docker state could not be read"));
        }
        let state: State = serde_json::from_str(&state.stdout)
            .map_err(|_| Error::new("INVALID_RESPONSE", "invalid Docker execution state"))?;
        if state.running {
            return Err(Error::new(
                "UNAVAILABLE",
                "Docker attachment ended while command is still running",
            ));
        }
        if state.oom_killed {
            return Err(Error::new(
                "RESOURCE_EXHAUSTED",
                "sandbox memory limit exceeded",
            ));
        }
        if state.exit_code != 0 || !state.error.is_empty() || !output.success {
            return Err(
                Error::new("EXECUTION_FAILED", "sandbox command failed").with_details(
                    json!({"exit_code":state.exit_code, "stdout":output.stdout,
                    "stderr":output.stderr, "error":state.error})
                    .to_string(),
                ),
            );
        }
        Ok(Response {
            data: json!({"stdout":output.stdout,"stderr":output.stderr,"exit_code":0}),
            metadata: Default::default(),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    args: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct State {
    running: bool,
    exit_code: i64,
    #[serde(rename = "OOMKilled")]
    oom_killed: bool,
    error: String,
}

impl Sandbox for Docker {
    async fn execute(
        &self,
        request: Request,
        cancellation: &Cancellation,
    ) -> Result<Response, Error> {
        crate::cancellation::check(cancellation)?;
        if request.operation != "command" {
            return Err(Error::new(
                "UNSUPPORTED",
                "Docker supports the command operation",
            ));
        }
        let input: Input = serde_json::from_value(request.input)
            .map_err(|_| Error::new("INVALID_ARGUMENTS", "command input requires args"))?;
        if input.args.is_empty()
            || input.args[0].trim().is_empty()
            || input.args.iter().any(|s| s.contains('\0'))
            || input.args.iter().map(|s| s.len()).sum::<usize>() > 1024 * 1024
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "invalid sandbox command arguments",
            ));
        }
        tokio::runtime::Handle::try_current()
            .map_err(|_| Error::new("UNAVAILABLE", "Docker sandbox requires a Tokio runtime"))?;
        let token = format!("{:032x}", rand::random::<u128>());
        let mut guard = Guard {
            program: self.program.clone(),
            name: format!("halo-sandbox-{token}"),
            token,
            armed: true,
        };
        let result = crate::cancellation::cancellable(
            async {
                tokio::time::timeout(self.config.timeout, self.run(&input.args, &guard))
                    .await
                    .map_err(|_| Error::new("TIMEOUT", "sandbox operation timed out"))?
            },
            cancellation,
        )
        .await;
        // Cleanup uses its own bounded time, not the cancelled operation token.
        // On failure the guard retries best-effort cleanup when it is dropped.
        let cleaned = guard.cleanup().await;
        match (result, cleaned) {
            (Ok(mut response), Ok(())) => {
                response.metadata = request.metadata;
                Ok(response)
            }
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
        }
    }
}

struct Output {
    stdout: String,
    stderr: String,
    success: bool,
}

async fn cli(program: &std::path::Path, args: &[String], limit: usize) -> Result<Output, Error> {
    let output = cli_bytes(program, args, None, limit).await?;
    let stdout = String::from_utf8(output.stdout)
        .map_err(|_| Error::new("INVALID_RESPONSE", "sandbox stdout is not UTF-8"))?;
    let stderr = String::from_utf8(output.stderr)
        .map_err(|_| Error::new("INVALID_RESPONSE", "sandbox stderr is not UTF-8"))?;
    Ok(Output {
        stdout,
        stderr,
        success: output.code == 0,
    })
}

pub(crate) struct Bytes {
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
    pub(crate) code: i32,
}

/// Internal Docker transport shared by Sandbox and Workspace. It does not
/// execute user argv on the host, and never logs arguments/environment values.
pub(crate) async fn cli_bytes(
    program: &std::path::Path,
    args: &[String],
    input: Option<&[u8]>,
    limit: usize,
) -> Result<Bytes, Error> {
    use tokio::io::AsyncWriteExt;
    let mut child = Command::new(program)
        .args(args)
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| {
            Error::new("UNAVAILABLE", "Docker CLI could not be started").with_details(e.to_string())
        })?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let stdin = child.stdin.take();
    let (stdout, stderr, status, ()) = futures_util::try_join!(
        read(stdout, limit),
        read(stderr, limit),
        async {
            child.wait().await.map_err(|e| {
                Error::new("UNAVAILABLE", "Docker CLI wait failed").with_details(e.to_string())
            })
        },
        async {
            if let Some(mut stdin) = stdin {
                // A container-side validation failure can close stdin early;
                // the exit status/stderr remain the authoritative result.
                if let Err(error) = stdin.write_all(input.unwrap_or_default()).await
                    && error.kind() != std::io::ErrorKind::BrokenPipe
                {
                    return Err(Error::new("UNAVAILABLE", "Docker input write failed"));
                }
            }
            Ok(())
        }
    )?;
    Ok(Bytes {
        stdout,
        stderr,
        code: status.code().unwrap_or(-1),
    })
}

async fn read(reader: impl AsyncRead + Unpin, limit: usize) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    reader
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|e| {
            Error::new("UNAVAILABLE", "Docker output read failed").with_details(e.to_string())
        })?;
    if bytes.len() > limit {
        return Err(Error::new("OUTPUT_LIMIT", "sandbox output limit exceeded"));
    }
    Ok(bytes)
}

pub(crate) fn container_id(id: &str) -> bool {
    id.len() == 64 && id.bytes().all(|c| c.is_ascii_hexdigit())
}

pub(crate) struct Guard {
    program: PathBuf,
    name: String,
    token: String,
    armed: bool,
}

impl Guard {
    pub(crate) fn new(program: PathBuf, prefix: &str) -> Self {
        let token = format!("{:032x}", rand::random::<u128>());
        Self {
            program,
            name: format!("{prefix}-{token}"),
            token,
            armed: true,
        }
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }
    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    pub(crate) async fn cleanup(&mut self) -> Result<(), Error> {
        let result = cleanup(&self.program, &self.name, &self.token).await;
        if result.is_ok() {
            self.armed = false;
        }
        result
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.armed
            && let Ok(handle) = tokio::runtime::Handle::try_current()
        {
            let (program, name, token) =
                (self.program.clone(), self.name.clone(), self.token.clone());
            handle.spawn(async move {
                if let Err(error) = cleanup(&program, &name, &token).await {
                    eprintln!("sandbox cleanup failed: {}", error.code);
                }
            });
        }
    }
}

async fn cleanup(program: &std::path::Path, name: &str, token: &str) -> Result<(), Error> {
    tokio::time::timeout(Duration::from_secs(5), async {
        // Locate only this operation's labelled container. Never remove an
        // unresolved name, all containers, or a caller's pre-existing container.
        let format = format!(r#"[{{{{json .Id}}}},{{{{json (index .Config.Labels "{LABEL}")}}}}]"#);
        let inspected = cli(
            program,
            &["inspect".into(), "--format".into(), format, name.into()],
            4096,
        )
        .await?;
        if !inspected.success {
            // Missing container is idempotent; daemon/auth failures are not.
            let stderr = inspected.stderr.to_ascii_lowercase();
            if stderr.contains("no such object") || stderr.contains("no such container") {
                return Ok(());
            }
            return Err(Error::new("UNAVAILABLE", "sandbox cleanup lookup failed"));
        }
        let value: Vec<String> = serde_json::from_str(&inspected.stdout)
            .map_err(|_| Error::new("INVALID_RESPONSE", "invalid Docker cleanup inspection"))?;
        if value.len() != 2 || !container_id(&value[0]) {
            return Err(Error::new("INVALID_RESPONSE", "invalid cleanup target"));
        }
        let id = &value[0];
        if value[1] != token {
            return Err(Error::new(
                "PERMISSION_DENIED",
                "cleanup target is not owned by this operation",
            ));
        }
        let removed = cli(
            program,
            &[
                "rm".into(),
                "--force".into(),
                "--volumes".into(),
                id.clone(),
            ],
            4096,
        )
        .await?;
        if !removed.success {
            return Err(Error::new(
                "UNAVAILABLE",
                "sandbox container cleanup failed",
            ));
        }
        Ok(())
    })
    .await
    .map_err(|_| Error::new("TIMEOUT", "sandbox cleanup timed out"))?
}

#[cfg(all(test, unix))]
#[path = "../../../tests/unit/embedded/src_core_sandbox_docker_tests.rs"]
mod tests;
