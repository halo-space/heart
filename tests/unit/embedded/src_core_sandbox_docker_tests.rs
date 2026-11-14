use super::*;
use std::os::unix::fs::PermissionsExt;

// Keep this opt-in process fixture with the source test module. The
// repository intentionally does not publish the ignored `tests/` tree,
// so a source-only checkout must not depend on an external include_str!
// path just to compile its test target.
const DOCKER_FIXTURE: &str = r##"#!/bin/sh
# Test-only Docker protocol fixture. Never runs the provided command.
base="$0"
mode=$(cat "$base.mode")
printf '%s\n' "$@" >> "$base.log"
id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
case "$1" in
  create)
    if [ "$mode" = offline ]; then printf '%s\n' 'daemon unavailable' >&2; exit 1; fi
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --label ]; then shift; printf '%s' "${1#*=}" > "$base.label"; fi
      shift
    done
    if [ "$mode" = wait-create ]; then
      printf '%s\n' 'create-ready' >> "$base.log"
      exec sleep 30
    fi
    if [ "$mode" = bad-id ]; then printf '%s\n' 'bad-id'; else printf '%s\n' "$id"; fi
    ;;
  start)
    if [ "$#" -ne 3 ] || [ "$2" != --attach ] || [ "$3" != "$id" ]; then
      printf '%s\n' 'unsupported Docker start arguments' >&2; exit 125
    fi
    case "$mode" in
      wait) exec sleep 30 ;;
      failed) printf '%s' 'bad command' >&2; exit 2 ;;
      large) printf '%s' 'a very large result' ;;
      *) printf '%s' 'hello' ;;
    esac
    ;;
  inspect)
    if [ "$mode" = offline ]; then printf '%s\n' 'daemon unavailable' >&2; exit 1; fi
    case "$3" in
      *'.State'*)
        code=0; if [ "$mode" = failed ]; then code=2; fi
        printf '{"Running":false,"ExitCode":%s,"OOMKilled":false,"Error":""}\n' "$code"
        ;;
      *)
        token=$(cat "$base.label")
        if [ "$mode" = foreign ]; then token=foreign; fi
        printf '["%s","%s"]\n' "$id" "$token"
        ;;
    esac
    ;;
  rm) printf '%s\n' "$id" ;;
  *) printf '%s\n' 'unexpected Docker operation' >&2; exit 2 ;;
esac
"##;

struct Fixture {
    directory: PathBuf,
    program: PathBuf,
}
impl Fixture {
    fn new(mode: &str) -> Self {
        let directory =
            std::env::temp_dir().join(format!("halo-sandbox-test-{:032x}", rand::random::<u128>()));
        std::fs::create_dir(&directory).unwrap();
        let program = directory.join("docker");
        std::fs::write(&program, DOCKER_FIXTURE).unwrap();
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
async fn start_attaches_using_only_supported_docker_arguments() {
    let fixture = Fixture::new("success");
    fixture
        .docker()
        .execute(request(), &Cancellation::new())
        .await
        .unwrap();
    let log = fixture.log();
    let args: Vec<_> = log
        .lines()
        .skip_while(|line| *line != "start")
        .take(3)
        .collect();
    let id = "a".repeat(64);
    assert_eq!(args, ["start", "--attach", id.as_str()]);
    assert!(!log.contains("--sig-proxy"));
    assert!(log.contains("rm\n--force\n--volumes"));
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
