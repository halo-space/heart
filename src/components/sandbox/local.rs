//! Default host execution, NOT an isolation mechanism. Only run trusted commands.
//! The process inherits the application's environment, directory and permissions.

use std::{future::Future, process::Stdio, task::Poll, time::Duration};

use futures_util::future::poll_fn;
use serde::Deserialize;
use serde_json::json;
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    process::Command,
};

use super::{Error, Request, Response, Sandbox};
use crate::Cancellation;

/// Host-process waiting/output limits, not CPU, memory or network isolation.
#[derive(Clone, Debug)]
pub struct Config {
    /// Whole execution/read deadline; default 30 seconds, must be positive.
    pub timeout: Duration,
    /// Byte cap per stdout/stderr stream; default 1 MiB. Exceeding it errors,
    /// rather than silently truncating output. Must be positive.
    pub max_output: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_output: 1024 * 1024,
        }
    }
}

/// Ordinary local execution. Construction does not start a process or enable
/// Agent tools. Cancellation/drop kills the direct child, not its descendants;
/// it does not roll back filesystem, network or other external side effects.
#[derive(Default)]
pub struct Local {
    config: Config,
}

impl Local {
    pub fn new(config: Config) -> Result<Self, Error> {
        if config.timeout.is_zero() || config.max_output == 0 || config.max_output == usize::MAX {
            return Err(Error::new(
                "INVALID_ARGUMENTS",
                "invalid local execution configuration",
            ));
        }
        Ok(Self { config })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    args: Vec<String>,
}

impl Sandbox for Local {
    async fn execute(
        &self,
        request: Request,
        cancellation: &Cancellation,
    ) -> Result<Response, Error> {
        check(cancellation)?;
        if request.operation != "command" {
            return Err(Error::new(
                "UNSUPPORTED",
                "Local supports the command operation",
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
                "invalid local command arguments",
            ));
        }
        tokio::runtime::Handle::try_current()
            .map_err(|_| Error::new("UNAVAILABLE", "Local execution requires a Tokio runtime"))?;
        cancellable(
            async {
                tokio::time::timeout(
                    self.config.timeout,
                    run(&input.args, self.config.max_output),
                )
                .await
                .map_err(|_| Error::new("TIMEOUT", "local operation timed out"))?
            },
            cancellation,
        )
        .await
        .map(|data| Response {
            data,
            metadata: request.metadata,
        })
    }
}

async fn run(args: &[String], limit: usize) -> Result<serde_json::Value, Error> {
    // Explicit argv: no shell expansion, implicit script interpreter, retry or
    // fallback. A caller may explicitly choose a shell as the first argument.
    let mut child = Command::new(&args[0])
        .args(&args[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| {
            Error::new(
                match e.kind() {
                    std::io::ErrorKind::NotFound => "NOT_FOUND",
                    std::io::ErrorKind::PermissionDenied => "PERMISSION_DENIED",
                    _ => "UNAVAILABLE",
                },
                "local command could not be started",
            )
            .with_details(e.to_string())
        })?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let (stdout, stderr, status) =
        futures_util::try_join!(read(stdout, limit), read(stderr, limit), async {
            child.wait().await.map_err(|e| {
                Error::new("UNAVAILABLE", "local command wait failed").with_details(e.to_string())
            })
        })?;
    let data = json!({"stdout": stdout, "stderr": stderr, "exit_code": status.code()});
    if !status.success() {
        return Err(
            Error::new("EXECUTION_FAILED", "local command failed").with_details(data.to_string())
        );
    }
    Ok(data)
}

async fn read(reader: impl AsyncRead + Unpin, limit: usize) -> Result<String, Error> {
    let mut bytes = Vec::new();
    reader
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|e| {
            Error::new("UNAVAILABLE", "local output read failed").with_details(e.to_string())
        })?;
    if bytes.len() > limit {
        return Err(Error::new("OUTPUT_LIMIT", "local output limit exceeded"));
    }
    String::from_utf8(bytes)
        .map_err(|_| Error::new("INVALID_RESPONSE", "local output is not UTF-8"))
}

fn check(cancellation: &Cancellation) -> Result<(), Error> {
    if cancellation.is_cancelled() {
        Err(Error::new("CANCELLED", "operation cancelled"))
    } else {
        Ok(())
    }
}

async fn cancellable<T>(
    future: impl Future<Output = Result<T, Error>>,
    cancellation: &Cancellation,
) -> Result<T, Error> {
    let mut future = std::pin::pin!(future);
    poll_fn(|cx| {
        cancellation.register(cx.waker());
        if let Err(error) = check(cancellation) {
            return Poll::Ready(Err(error));
        }
        let result = future.as_mut().poll(cx);
        if let Err(error) = check(cancellation) {
            return Poll::Ready(Err(error));
        }
        result
    })
    .await
}
