use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use futures_executor::block_on;
use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message, Messages,
    agent::{Agent, Input, Payload, toolkit::Toolkit},
    core,
    event::Event,
    harness,
    mcp::{Definition as McpDefinition, Server},
    message::Role,
    model::{
        chat::{Chat, Request},
        content::part::Part,
    },
    permission::{Behavior, Decision},
    skill::{Definition as SkillDefinition, Skill},
    tool::{Call, Tool, ToolDefinition},
    workspace::{Mode, Workspace},
};
use serde_json::{Value, json};

fn definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: "Test capability".into(),
        parameters: json!({"type":"object"}).as_object().unwrap().clone(),
    }
}

fn message(role: Role, value: &str) -> Message {
    let mut message = Message::new(role);
    message.content.push(Part {
        r#type: "text".into(),
        data: json!({"value":value}).as_object().unwrap().clone(),
    });
    message
}

fn call(name: &str, arguments: Value) -> Call {
    Call {
        call_id: format!("call_{name}"),
        name: name.into(),
        arguments: arguments.as_object().unwrap().clone(),
    }
}

fn tool_call(name: &str, arguments: Value) -> Message {
    let call = call(name, arguments);
    let mut message = Message::new(Role::Assistant);
    message.content.push(Part {
        r#type: "tool_call".into(),
        data: serde_json::to_value(call)
            .unwrap()
            .as_object()
            .unwrap()
            .clone(),
    });
    message
}

struct Echo(ToolDefinition);

impl Tool for Echo {
    fn definition(&self) -> &ToolDefinition {
        &self.0
    }

    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        Ok(message(Role::Tool, "custom result"))
    }
}

struct Remote {
    definition: McpDefinition,
    discoveries: Arc<AtomicUsize>,
}

impl Server for Remote {
    fn definition(&self) -> &McpDefinition {
        &self.definition
    }

    async fn list_tools(&self, _: &Cancellation) -> Result<Vec<ToolDefinition>, Error> {
        self.discoveries.fetch_add(1, Ordering::SeqCst);
        Ok(vec![definition("remote")])
    }

    async fn call(&self, _: Call, _: &Cancellation) -> Result<Message, Error> {
        Ok(message(Role::Tool, "remote result"))
    }
}

struct Research(SkillDefinition);

impl Skill for Research {
    fn definition(&self) -> &SkillDefinition {
        &self.0
    }

    async fn run(&self, input: Messages, _: &Cancellation) -> Result<Messages, Error> {
        Ok(input)
    }
}

struct Model {
    outputs: Mutex<VecDeque<Message>>,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl Model {
    fn new(outputs: Vec<Message>) -> (Self, Arc<Mutex<Vec<Request>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                outputs: Mutex::new(outputs.into()),
                requests: Arc::clone(&requests),
            },
            requests,
        )
    }
}

impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.requests.lock().unwrap().push(request);
        let message = self
            .outputs
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected model replay");
        Ok(stream::iter(vec![Ok(Event::Complete {
            message,
            usage: None,
            finish_reason: None,
        })]))
    }
}

fn input() -> Input {
    Input {
        metadata: json!({"chat_id":101,"agent_id":201,"session_id":301})
            .as_object()
            .unwrap()
            .clone(),
        payload: vec![Payload {
            r#type: "text".into(),
            data: json!({"value":"write and read the draft"}),
            mime_type: None,
        }],
    }
}

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "heart-workspace-assembly-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[tokio::test]
async fn workspace_preserves_tools_mcps_and_skills_without_eager_discovery() {
    let discoveries = Arc::new(AtomicUsize::new(0));
    let toolkit = Toolkit::new()
        .tool(Echo(definition("echo")))
        .mcps([Remote {
            definition: McpDefinition {
                name: "remote_server".into(),
                ..Default::default()
            },
            discoveries: Arc::clone(&discoveries),
        }])
        .skills([Research(SkillDefinition {
            name: "research".into(),
            ..Default::default()
        })])
        .unwrap();
    let directory = Directory::new();
    let workspace = core::workspace::Local::new(&directory.0)
        .unwrap()
        .toolkit(toolkit);
    assert_eq!(workspace.definition().mode, Mode::Local);
    assert!(workspace.definition().path.is_some());
    let toolkit = Toolkit::from(workspace);
    assert_eq!(discoveries.load(Ordering::SeqCst), 0);
    let names: Vec<_> = toolkit
        .definitions()
        .into_iter()
        .map(|definition| definition.name)
        .collect();
    assert_eq!(
        names,
        [
            "echo",
            "workspace_read",
            "workspace_write",
            "workspace_list",
            "workspace_edit",
            "workspace_glob",
            "workspace_grep",
            "workspace_delete"
        ]
    );
    let cancellation = Cancellation::new();
    let original = Messages(vec![message(Role::User, "skill input")]);
    assert_eq!(
        toolkit
            .run("research", original.clone(), &cancellation)
            .await
            .unwrap(),
        original
    );
    assert_eq!(discoveries.load(Ordering::SeqCst), 0);
    let (result, _) = toolkit
        .execute(call("remote", json!({})), &cancellation)
        .await
        .unwrap();
    assert_eq!(result.content[0].data["value"], "remote result");
    assert_eq!(discoveries.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn bound_file_tools_share_one_backend_and_preserve_direct_errors() {
    let directory = Directory::new();
    let toolkit = Toolkit::from(
        core::workspace::Local::new(&directory.0)
            .unwrap()
            .toolkit(Toolkit::new()),
    );
    let cancellation = Cancellation::new();
    toolkit
        .execute(
            call(
                "workspace_write",
                json!({"path":"notes/draft.txt","content":"draft"}),
            ),
            &cancellation,
        )
        .await
        .unwrap();
    let (result, _) = toolkit
        .execute(
            call("workspace_read", json!({"path":"notes/draft.txt"})),
            &cancellation,
        )
        .await
        .unwrap();
    assert_eq!(result.role, Role::Tool);
    assert_eq!(result.values["content"], "draft");
    let other_directory = Directory::new();
    let other = Toolkit::from(
        core::workspace::Local::new(&other_directory.0)
            .unwrap()
            .toolkit(Toolkit::new()),
    );
    let error = other
        .execute(
            call("workspace_read", json!({"path":"notes/draft.txt"})),
            &cancellation,
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, "NOT_FOUND");
    let cancelled = Cancellation::new();
    cancelled.cancel();
    let error = toolkit
        .execute(
            call(
                "workspace_write",
                json!({"path":"notes/draft.txt","content":"wrong"}),
            ),
            &cancelled,
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, "CANCELLED");
    let (result, _) = toolkit
        .execute(
            call("workspace_read", json!({"path":"notes/draft.txt"})),
            &cancellation,
        )
        .await
        .unwrap();
    assert_eq!(result.values["content"], "draft");
}

#[tokio::test]
async fn harness_uses_workspace_mounted_tools_in_the_existing_model_loop() {
    let directory = Directory::new();
    let (model, requests) = Model::new(vec![
        tool_call(
            "workspace_write",
            json!({"path":"draft.txt","content":"draft"}),
        ),
        tool_call("workspace_list", json!({})),
        tool_call("workspace_glob", json!({"pattern":"*.txt"})),
        tool_call("workspace_grep", json!({"pattern":"draft"})),
        tool_call(
            "workspace_edit",
            json!({"path":"draft.txt","old_string":"draft","new_string":"edited draft"}),
        ),
        tool_call("workspace_read", json!({"path":"draft.txt"})),
        tool_call("echo", json!({})),
        tool_call("workspace_delete", json!({"path":"draft.txt"})),
        message(Role::Assistant, "done"),
    ]);
    let workspace = core::workspace::Local::new(&directory.0)
        .unwrap()
        .toolkit(Toolkit::new().tool(Echo(definition("echo"))));
    let agent = harness::builder()
        .model(model)
        .workspace(workspace)
        .build()
        .unwrap();
    assert!(requests.lock().unwrap().is_empty());
    let events: Vec<_> = agent
        .run(input(), false, Cancellation::new())
        .await
        .unwrap()
        .collect()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(events.len(), 17, "nine model calls and eight tool results");
    assert!(!directory.0.join("draft.txt").exists());
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 9);
    let names: Vec<_> = requests[0]
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    for name in [
        "echo",
        "workspace_read",
        "workspace_write",
        "workspace_list",
        "workspace_edit",
        "workspace_glob",
        "workspace_grep",
        "workspace_delete",
    ] {
        assert!(names.contains(&name));
    }
    assert!(!names.iter().any(|name| name.contains("shell")));
    let read_result = requests[6]
        .messages
        .0
        .iter()
        .rev()
        .find(|message| message.role == Role::Tool)
        .unwrap();
    assert!(
        read_result.content[0].data["value"]
            .as_str()
            .unwrap()
            .contains("draft")
    );
    let custom_result = requests[7]
        .messages
        .0
        .iter()
        .rev()
        .find(|message| message.role == Role::Tool)
        .unwrap();
    assert_eq!(custom_result.content[0].data["value"], "custom result");
}

#[tokio::test]
async fn permission_denial_stops_a_workspace_write_before_side_effects() {
    let directory = Directory::new();
    let (model, requests) = Model::new(vec![
        tool_call(
            "workspace_write",
            json!({"path":"denied.txt","content":"wrong"}),
        ),
        message(Role::Assistant, "write was denied"),
    ]);
    let decisions = Arc::new(AtomicUsize::new(0));
    let checks = Arc::clone(&decisions);
    let policy = core::permission::Permission::new(
        move |request: halo_agents::permission::Request, _: Cancellation| {
            let checks = Arc::clone(&checks);
            async move {
                assert_eq!(request.resource, "workspace_write");
                checks.fetch_add(1, Ordering::SeqCst);
                Ok(Decision {
                    behavior: Behavior::Deny,
                    reason: Some("read only".into()),
                })
            }
        },
    );
    let workspace = core::workspace::Local::new(&directory.0)
        .unwrap()
        .toolkit(Toolkit::new());
    let agent = harness::builder()
        .model(model)
        .workspace(workspace)
        .permission(policy)
        .build()
        .unwrap();
    let events: Vec<_> = agent
        .run(input(), false, Cancellation::new())
        .await
        .unwrap()
        .collect()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(decisions.load(Ordering::SeqCst), 1);
    assert!(!directory.0.join("denied.txt").exists());
    assert_eq!(
        requests.lock().unwrap().len(),
        2,
        "Agent can handle the original tool error"
    );
}

#[test]
fn conflicting_workspace_tool_names_are_rejected_before_model_execution() {
    let (model, requests) = Model::new(vec![]);
    let directory = Directory::new();
    let workspace = core::workspace::Local::new(&directory.0)
        .unwrap()
        .toolkit(Toolkit::new().tool(Echo(definition("workspace_read"))));
    let result = harness::builder().model(model).workspace(workspace).build();
    let Err(error) = result else {
        panic!("duplicate tool name was accepted")
    };
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    assert!(requests.lock().unwrap().is_empty());
}

#[test]
fn harness_without_workspace_does_not_install_file_tools() {
    block_on(async {
        let (model, requests) = Model::new(vec![message(Role::Assistant, "done")]);
        let agent = harness::builder().model(model).build().unwrap();
        let events: Vec<_> = agent
            .run(input(), false, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(events.iter().all(Result::is_ok));
        assert!(
            requests.lock().unwrap()[0]
                .tools
                .iter()
                .all(|tool| !tool.name.starts_with("workspace_"))
        );
    });
}

// A contract fixture, not a database backend. It deliberately has no file
// methods, path parameter, filesystem or in-memory file store.
struct DatabaseWorkspace {
    definition: halo_agents::workspace::Definition,
    toolkit: Toolkit,
}

impl Workspace for DatabaseWorkspace {
    fn definition(&self) -> &halo_agents::workspace::Definition {
        &self.definition
    }
}

impl From<DatabaseWorkspace> for Toolkit {
    fn from(workspace: DatabaseWorkspace) -> Self {
        workspace.toolkit
    }
}

struct Query {
    definition: ToolDefinition,
    calls: Arc<AtomicUsize>,
}

impl Tool for Query {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn call(&self, call: &Call, _: &Cancellation) -> Result<Message, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(call.arguments["filter"], json!({"id":1001}));
        let mut result = message(Role::Tool, "record 1001: research");
        result
            .values
            .insert("data".into(), json!([{"id":1001,"content":"research"}]));
        Ok(result)
    }
}

fn database_workspace(calls: Arc<AtomicUsize>) -> DatabaseWorkspace {
    DatabaseWorkspace {
        definition: halo_agents::workspace::Definition {
            mode: Mode::Server,
            path: None,
            metadata: Default::default(),
        },
        toolkit: Toolkit::new().tool(Query {
            definition: ToolDefinition {
                name: "records_read".into(),
                description: "Read a bound database record using a filter, not a path".into(),
                parameters: json!({
                    "type":"object",
                    "properties":{"filter":{
                        "type":"object",
                        "properties":{"id":{"type":"integer"}},
                        "required":["id"],
                        "additionalProperties":false
                    }},
                    "required":["filter"],
                    "additionalProperties":false
                })
                .as_object()
                .unwrap()
                .clone(),
            },
            calls,
        }),
    }
}

#[tokio::test]
async fn definition_only_server_uses_its_own_schema_and_the_original_harness_loop() {
    let calls = Arc::new(AtomicUsize::new(0));
    let workspace = database_workspace(Arc::clone(&calls));
    assert_eq!(workspace.definition().mode, Mode::Server);
    assert!(workspace.definition().path.is_none());
    let (model, requests) = Model::new(vec![
        tool_call("records_read", json!({"filter":{"id":1001}})),
        message(Role::Assistant, "done"),
    ]);
    let agent = harness::builder()
        .model(model)
        .workspace(workspace)
        .build()
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let events: Vec<_> = agent
        .run(input(), false, Cancellation::new())
        .await
        .unwrap()
        .collect()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(
        events.len(),
        3,
        "retain both model and tool Complete events"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    for request in requests.iter() {
        let mut names: Vec<_> = request
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect();
        names.sort_unstable();
        assert_eq!(
            names,
            ["plan", "records_read"],
            "retain the built-in Planner, but do not install file tools for Server"
        );
        let query = request
            .tools
            .iter()
            .find(|tool| tool.name == "records_read")
            .unwrap();
        assert!(query.parameters["properties"].get("path").is_none());
    }
    assert!(requests[1].messages.as_slice().iter().any(|message| {
        message.role == Role::Tool
            && message
                .content
                .iter()
                .any(|part| part.data.get("value") == Some(&json!("record 1001: research")))
    }));
}

#[tokio::test]
async fn definition_only_server_keeps_tool_validation_cancellation_and_results() {
    let calls = Arc::new(AtomicUsize::new(0));
    let toolkit = Toolkit::from(database_workspace(Arc::clone(&calls)));
    let error = toolkit
        .execute(
            call("records_read", json!({"path":"fake/file"})),
            &Cancellation::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert_eq!(
        toolkit
            .execute(
                call("records_read", json!({"filter":{"id":1001}})),
                &cancellation
            )
            .await
            .unwrap_err()
            .code,
        "CANCELLED"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let (result, _) = toolkit
        .execute(
            call("records_read", json!({"filter":{"id":1001}})),
            &Cancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(result.role, Role::Tool);
    assert_eq!(
        result.values,
        json!({"data":[{"id":1001,"content":"research"}]})
            .as_object()
            .unwrap()
            .clone()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn definition_only_server_still_checks_existing_permission_before_call() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (model, _) = Model::new(vec![
        tool_call("records_read", json!({"filter":{"id":1001}})),
        message(Role::Assistant, "denied"),
    ]);
    let permission = core::permission::Permission::new(
        |_: halo_agents::permission::Request, _: Cancellation| async {
            Ok(Decision {
                behavior: Behavior::Deny,
                reason: Some("not allowed".into()),
            })
        },
    );
    let agent = harness::builder()
        .model(model)
        .workspace(database_workspace(Arc::clone(&calls)))
        .permission(permission)
        .build()
        .unwrap();
    let events: Vec<_> = agent
        .run(input(), false, Cancellation::new())
        .await
        .unwrap()
        .collect()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[cfg(unix)]
#[tokio::test]
async fn configured_workspace_commands_use_the_original_harness_loop_and_shared_files() {
    let directory = Directory::new();
    let (model, requests) = Model::new(vec![
        tool_call(
            "workspace_write",
            json!({"path":"draft.txt","content":"draft"}),
        ),
        tool_call(
            "workspace_exec_shell",
            json!({"args":["/bin/sh","-c","printf edited > draft.txt"]}),
        ),
        tool_call("workspace_read", json!({"path":"draft.txt"})),
        tool_call("workspace_delete", json!({"path":"draft.txt"})),
        message(Role::Assistant, "done"),
    ]);
    let agent = harness::builder()
        .model(model)
        .workspace(
            core::workspace::Local::new(&directory.0)
                .unwrap()
                .sandbox(halo_agents::sandbox::Local::default()),
        )
        .build()
        .unwrap();
    let events: Vec<_> = agent
        .run(input(), false, Cancellation::new())
        .await
        .unwrap()
        .collect()
        .await;
    assert!(events.iter().all(Result::is_ok));
    assert_eq!(events.len(), 9);
    assert!(!directory.0.join("draft.txt").exists());
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 5);
    assert_eq!(
        requests[0].tools.len(),
        9,
        "eight workspace Tools and the existing built-in plan"
    );
    let read = requests[3]
        .messages
        .as_slice()
        .iter()
        .rev()
        .find(|message| message.role == Role::Tool)
        .unwrap();
    assert!(
        read.content[0].data["value"]
            .as_str()
            .unwrap()
            .contains("edited")
    );
}

#[cfg(unix)]
#[tokio::test]
async fn permission_denial_blocks_builtin_delete_and_command_without_side_effects() {
    for name in ["workspace_delete", "workspace_exec_shell"] {
        let directory = Directory::new();
        std::fs::write(directory.0.join("keep.txt"), "keep").unwrap();
        let args = if name == "workspace_delete" {
            json!({"path":"keep.txt"})
        } else {
            json!({"args":["/bin/sh","-c","printf wrong > keep.txt; printf wrong > unexpected"]})
        };
        let (model, _) = Model::new(vec![
            tool_call(name, args),
            message(Role::Assistant, "denied"),
        ]);
        let checks = Arc::new(AtomicUsize::new(0));
        let checked = Arc::clone(&checks);
        let permission = core::permission::Permission::new(
            move |request: halo_agents::permission::Request, _: Cancellation| {
                assert_eq!(request.resource, name);
                checked.fetch_add(1, Ordering::SeqCst);
                async {
                    Ok(Decision {
                        behavior: Behavior::Deny,
                        reason: Some("read only".into()),
                    })
                }
            },
        );
        let agent = harness::builder()
            .model(model)
            .workspace(
                core::workspace::Local::new(&directory.0)
                    .unwrap()
                    .sandbox(halo_agents::sandbox::Local::default()),
            )
            .permission(permission)
            .build()
            .unwrap();
        let events: Vec<_> = agent
            .run(input(), false, Cancellation::new())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(events.iter().all(Result::is_ok));
        assert!(checks.load(Ordering::SeqCst) > 0);
        assert_eq!(
            std::fs::read_to_string(directory.0.join("keep.txt")).unwrap(),
            "keep"
        );
        assert!(!directory.0.join("unexpected").exists());
    }
}

#[test]
fn automatically_bound_operations_still_reject_custom_name_collisions() {
    for name in ["workspace_delete", "workspace_exec_shell"] {
        let directory = Directory::new();
        let (model, requests) = Model::new(vec![]);
        let workspace = core::workspace::Local::new(&directory.0)
            .unwrap()
            .sandbox(halo_agents::sandbox::Local::default())
            .toolkit(Toolkit::new().tool(Echo(definition(name))));
        let Err(error) = harness::builder().model(model).workspace(workspace).build() else {
            panic!("built-in operation collision was accepted");
        };
        assert_eq!(error.code, "INVALID_ARGUMENTS");
        assert!(requests.lock().unwrap().is_empty());
    }
}
