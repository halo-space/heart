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

const LABEL: &str = "halo.agents.operation";

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

pub struct Docker {
    config: Config,
    program: PathBuf,
}

impl Docker {
    /// Construction only validates local settings; it does not start a daemon
    /// or container. Execution requires an existing Tokio I/O/time runtime.
    pub fn new(config: Config) -> Result<Self, Error> {
        if config.image.is_empty()
            || config.image.starts_with('-')
            || config
                .image
                .chars()
                .any(|c| c.is_whitespace() || c.is_control())
            || config.memory < 6 * 1024 * 1024
            || !config.cpus.is_finite()
            || config.cpus <= 0.0
            || config.pids == 0
            || config.timeout.is_zero()
            || config.max_output == 0
            || config.max_output == usize::MAX
        {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "invalid Docker sandbox configuration",
            ));
        }
        Ok(Self {
            config,
            program: PathBuf::from("docker"),
        })
    }

    async fn run(&self, args: &[String], guard: &Guard) -> Result<Response, Error> {
        let memory = self.config.memory.to_string();
        let mut create = vec![
            "create".into(),
            "--name".into(),
            guard.name.clone(),
            "--label".into(),
            format!("{LABEL}={}", guard.token),
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
            if self.config.network {
                "bridge".into()
            } else {
                "none".into()
            },
            "--memory".into(),
            memory.clone(),
            "--memory-swap".into(),
            memory,
            "--cpus".into(),
            self.config.cpus.to_string(),
            "--pids-limit".into(),
            self.config.pids.to_string(),
            "--tmpfs".into(),
            "/tmp:rw,nosuid,nodev,size=64m,mode=1777".into(),
            "--tmpfs".into(),
            "/workspace:rw,nosuid,nodev,size=64m,mode=1777".into(),
            "--workdir".into(),
            "/workspace".into(),
            "--entrypoint".into(),
            args[0].clone(),
            self.config.image.clone(),
        ];
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
            &[
                "start".into(),
                "--attach".into(),
                "--sig-proxy=false".into(),
                id.into(),
            ],
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
        crate::core::operation::check(cancellation)?;
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
        let result = crate::core::operation::cancellable(
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
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| {
            Error::new("UNAVAILABLE", "Docker CLI could not be started").with_details(e.to_string())
        })?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let (stdout, stderr, status) =
        futures_util::try_join!(read(stdout, limit), read(stderr, limit), async {
            child.wait().await.map_err(|e| {
                Error::new("UNAVAILABLE", "Docker CLI wait failed").with_details(e.to_string())
            })
        })?;
    let stdout = String::from_utf8(stdout)
        .map_err(|_| Error::new("INVALID_RESPONSE", "sandbox stdout is not UTF-8"))?;
    let stderr = String::from_utf8(stderr)
        .map_err(|_| Error::new("INVALID_RESPONSE", "sandbox stderr is not UTF-8"))?;
    Ok(Output {
        stdout,
        stderr,
        success: status.success(),
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

fn container_id(id: &str) -> bool {
    id.len() == 64 && id.bytes().all(|c| c.is_ascii_hexdigit())
}

struct Guard {
    program: PathBuf,
    name: String,
    token: String,
    armed: bool,
}

impl Guard {
    async fn cleanup(&mut self) -> Result<(), Error> {
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
            if inspected.stderr.contains("No such object")
                || inspected.stderr.contains("No such container")
            {
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
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    struct Fixture {
        directory: PathBuf,
        program: PathBuf,
    }
    impl Fixture {
        fn new(mode: &str) -> Self {
            let directory = std::env::temp_dir()
                .join(format!("halo-sandbox-test-{:032x}", rand::random::<u128>()));
            std::fs::create_dir(&directory).unwrap();
            let program = directory.join("docker");
            std::fs::write(&program, include_str!("../../../tests/fixtures/docker.sh")).unwrap();
            std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o700)).unwrap();
            std::fs::write(program.with_extension("mode"), mode).unwrap();
            Self { directory, program }
        }
        fn docker(&self) -> Docker {
            let mut docker = Docker::new(Config {
                image: "test-image:local".into(),
                ..Default::default()
            })
            .unwrap();
            docker.program = self.program.clone();
            docker
        }
        fn log(&self) -> String {
            std::fs::read_to_string(self.program.with_extension("log")).unwrap_or_default()
        }

        /// Keep real process startup separate from the timeout under test.
        /// The execution future is deliberately not spawned: while the clock
        /// advances it cannot poll cleanup I/O under an auto-advancing clock.
        async fn timeout_after<F: std::future::Future>(
            &self,
            execution: F,
            stage: &str,
            elapsed: Duration,
        ) -> F::Output {
            let started = async {
                while !self.log().contains(stage) {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            };
            let execution = match tokio::time::timeout(
                Duration::from_secs(15),
                futures_util::future::select(Box::pin(execution), Box::pin(started)),
            )
            .await
            .expect("Docker fixture did not reach the target stage")
            {
                futures_util::future::Either::Right(((), execution)) => execution,
                futures_util::future::Either::Left(_) => {
                    panic!("Docker operation finished before the target stage")
                }
            };

            tokio::time::pause();
            tokio::time::advance(elapsed).await;
            // Cleanup runs against real process I/O, with its original bounded
            // deadline. Resume before polling the timed-out execution again.
            tokio::time::resume();
            tokio::time::timeout(Duration::from_secs(15), execution)
                .await
                .expect("timed-out Docker operation did not finish cleanup")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }
    fn request() -> Request {
        Request {
            operation: "command".into(),
            input: json!({"args":["printf", "%s", "hello"]}),
            metadata: Default::default(),
        }
    }

    #[tokio::test]
    async fn successful_command_is_confined_and_owned_container_is_removed() {
        let fixture = Fixture::new("success");
        let mut input = request();
        input.metadata.insert("trace".into(), json!(5));
        let output = fixture
            .docker()
            .execute(input, &Cancellation::new())
            .await
            .unwrap();
        assert_eq!(output.data["stdout"], "hello");
        assert_eq!(output.metadata["trace"], 5);
        let log = fixture.log();
        for flag in [
            "--pull\nnever",
            "--network\nnone",
            "--read-only",
            "--cap-drop\nALL",
            "--security-opt\nno-new-privileges",
            "--user\n65534:65534",
            "--pids-limit\n128",
            "--memory\n536870912",
            "--memory-swap\n536870912",
            "--cpus\n1",
            "--entrypoint\nprintf",
            "rm\n--force\n--volumes",
        ] {
            assert!(log.contains(flag), "missing flag {flag}: {log}");
        }
        assert!(!log.contains("--privileged"));
        assert!(!log.contains("--volume\n"));
        assert_eq!(log.matches("create\n").count(), 1);
    }

    #[tokio::test]
    async fn command_arguments_are_not_interpreted_by_a_host_shell() {
        let fixture = Fixture::new("success");
        let marker = fixture.directory.join("must-not-exist");
        let mut input = request();
        input.input = json!({"args":["echo",format!("$(touch {})", marker.display())]});
        fixture
            .docker()
            .execute(input, &Cancellation::new())
            .await
            .unwrap();
        assert!(!marker.exists());
        assert!(fixture.log().contains("$(touch"));
    }

    #[tokio::test]
    async fn failed_command_is_direct_error_and_still_cleaned() {
        let fixture = Fixture::new("failed");
        let error = fixture
            .docker()
            .execute(request(), &Cancellation::new())
            .await
            .unwrap_err();
        assert_eq!(error.code, "EXECUTION_FAILED");
        assert!(error.details.unwrap().contains("bad command"));
        assert!(fixture.log().contains("rm\n--force"));
    }

    #[tokio::test]
    async fn invalid_create_id_does_not_prevent_labelled_cleanup() {
        let fixture = Fixture::new("bad-id");
        assert_eq!(
            fixture
                .docker()
                .execute(request(), &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "INVALID_RESPONSE"
        );
        assert!(fixture.log().contains("rm\n--force"));
    }

    #[tokio::test]
    async fn output_is_bounded_and_container_is_cleaned() {
        let fixture = Fixture::new("large");
        let mut docker = fixture.docker();
        docker.config.max_output = 3;
        assert_eq!(
            docker
                .execute(request(), &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "OUTPUT_LIMIT"
        );
        assert!(fixture.log().contains("rm\n--force"));
    }

    #[tokio::test]
    async fn timeout_stops_waiting_and_removes_actual_container() {
        let fixture = Fixture::new("wait");
        let mut docker = fixture.docker();
        docker.config.timeout = Duration::from_secs(3600);
        let cancellation = Cancellation::new();
        assert_eq!(
            fixture
                .timeout_after(
                    docker.execute(request(), &cancellation),
                    "start\n",
                    docker.config.timeout,
                )
                .await
                .unwrap_err()
                .code,
            "TIMEOUT"
        );
        assert!(fixture.log().contains("start\n"));
        assert!(fixture.log().contains("rm\n--force"));
    }

    #[tokio::test]
    async fn timeout_during_create_cleans_owned_container_without_starting_it() {
        let fixture = Fixture::new("wait-create");
        let mut docker = fixture.docker();
        docker.config.timeout = Duration::from_secs(3600);
        let cancellation = Cancellation::new();
        assert_eq!(
            fixture
                .timeout_after(
                    docker.execute(request(), &cancellation),
                    "create-ready\n",
                    docker.config.timeout,
                )
                .await
                .unwrap_err()
                .code,
            "TIMEOUT"
        );
        assert!(!fixture.log().contains("start\n"));
        assert!(fixture.log().contains("rm\n--force"));
    }

    #[tokio::test]
    async fn cancellation_during_command_performs_independent_cleanup() {
        let fixture = Fixture::new("wait");
        let docker = fixture.docker();
        let cancellation = Cancellation::new();
        let result = docker.execute(request(), &cancellation);
        let cancel = async {
            tokio::time::timeout(Duration::from_secs(5), async {
                while !fixture.log().contains("start\n") {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .unwrap();
            cancellation.cancel();
        };
        let (result, ()) = futures_util::join!(result, cancel);
        assert_eq!(result.unwrap_err().code, "CANCELLED");
        assert!(fixture.log().contains("rm\n--force"));
    }

    #[tokio::test]
    async fn dropped_execution_gets_best_effort_owned_cleanup() {
        let fixture = Fixture::new("wait");
        let docker = fixture.docker();
        let cancellation = Cancellation::new();
        let future = Box::pin(docker.execute(request(), &cancellation));
        let started = Box::pin(async {
            while !fixture.log().contains("start\n") {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });
        let pending = tokio::time::timeout(
            Duration::from_secs(5),
            futures_util::future::select(future, started),
        )
        .await
        .unwrap();
        match pending {
            futures_util::future::Either::Right(((), future)) => drop(future),
            futures_util::future::Either::Left(_) => panic!("command should remain pending"),
        }
        tokio::time::timeout(Duration::from_secs(5), async {
            while !fixture.log().contains("rm\n--force") {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn cleanup_never_removes_a_foreign_container() {
        let fixture = Fixture::new("foreign");
        assert_eq!(
            fixture
                .docker()
                .execute(request(), &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "PERMISSION_DENIED"
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!fixture.log().contains("rm\n"));
    }

    #[tokio::test]
    async fn unavailable_docker_is_not_replaced_by_host_execution() {
        let fixture = Fixture::new("offline");
        assert_eq!(
            fixture
                .docker()
                .execute(request(), &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "UNAVAILABLE"
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!fixture.log().contains("start\n"));
        assert!(!fixture.log().contains("rm\n"));
    }

    #[tokio::test]
    async fn pre_cancel_and_invalid_request_do_not_start_docker() {
        let fixture = Fixture::new("success");
        let docker = fixture.docker();
        let cancellation = Cancellation::new();
        cancellation.cancel();
        assert_eq!(
            docker
                .execute(request(), &cancellation)
                .await
                .unwrap_err()
                .code,
            "CANCELLED"
        );
        let mut input = request();
        input.operation = "unregistered".into();
        assert_eq!(
            docker
                .execute(input, &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "UNSUPPORTED"
        );
        let mut input = request();
        input.input = json!({"args":[]});
        assert_eq!(
            docker
                .execute(input, &Cancellation::new())
                .await
                .unwrap_err()
                .code,
            "INVALID_ARGUMENTS"
        );
        assert!(fixture.log().is_empty());
    }

    #[test]
    fn configuration_and_runtime_dependency_are_explicit() {
        assert!(Docker::new(Config::default()).is_err());
        let fixture = Fixture::new("success");
        assert_eq!(
            futures_executor::block_on(fixture.docker().execute(request(), &Cancellation::new()))
                .unwrap_err()
                .code,
            "UNAVAILABLE"
        );
        assert!(fixture.log().is_empty());
    }
}
