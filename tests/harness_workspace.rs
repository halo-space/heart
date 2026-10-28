use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::core,
    agent::mcp::{Definition as McpDefinition, Server},
    agent::message::Role,
    agent::model::content::part::Part,
    agent::skill::{Definition as SkillDefinition, Skill},
    agent::tool::{Call, Tool, ToolDefinition},
    agent::workspace::{Mode, Workspace},
    harness,
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
    tools: Vec<ToolDefinition>,
}

impl Server for Remote {
    fn definition(&self) -> &McpDefinition {
        &self.definition
    }

    fn tools(&self) -> &[ToolDefinition] {
        &self.tools
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

    async fn read(&self, _: &Cancellation) -> Result<String, Error> {
        Ok("Research instructions.".into())
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
async fn workspace_preserves_ready_tools_mcps_and_skills() {
    let directory = Directory::new();
    let workspace = core::workspace::Local::new(&directory.0)
        .unwrap()
        .tools(Echo(definition("echo")))
        .mcps([Remote {
            definition: McpDefinition {
                name: "remote_server".into(),
                ..Default::default()
            },
            tools: vec![definition("remote")],
        }])
        .skills([Research(SkillDefinition {
            name: "research".into(),
            ..Default::default()
        })])
        .unwrap();

    assert_eq!(workspace.definition().mode, Mode::Local);
    assert!(workspace.definition().path.is_some());
    let names: Vec<_> = halo::agent::workspace::Workspace::tool(&workspace)
        .into_iter()
        .map(|definition| definition.name)
        .collect();
    assert_eq!(
        names,
        [
            "echo",
            "remote",
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
    // Skill metadata stays in the tool dispatcher catalog and does not become a
    // separate executable definition before the model view is prepared.
    assert!(
        halo::agent::workspace::Workspace::tool(&workspace)
            .iter()
            .all(|definition| definition.name != "research")
    );
    let (result, _) = workspace
        .execute(call("remote", json!({})), &cancellation)
        .await
        .unwrap();
    assert_eq!(result.content[0].data["value"], "remote result");
}

#[tokio::test]
async fn bound_file_tools_share_one_backend_and_preserve_direct_errors() {
    let directory = Directory::new();
    let capabilities = core::workspace::Local::new(&directory.0).unwrap();
    let cancellation = Cancellation::new();
    capabilities
        .execute(
            call(
                "workspace_write",
                json!({"path":"notes/draft.txt","content":"draft"}),
            ),
            &cancellation,
        )
        .await
        .unwrap();
    let (result, _) = capabilities
        .execute(
            call("workspace_read", json!({"path":"notes/draft.txt"})),
            &cancellation,
        )
        .await
        .unwrap();
    assert_eq!(result.role, Role::Tool);
    assert_eq!(result.values["content"], "draft");
    let other_directory = Directory::new();
    let other = core::workspace::Local::new(&other_directory.0).unwrap();
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
    let error = capabilities
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
    let (result, _) = capabilities
        .execute(
            call("workspace_read", json!({"path":"notes/draft.txt"})),
            &cancellation,
        )
        .await
        .unwrap();
    assert_eq!(result.values["content"], "draft");
}

#[test]
fn conflicting_workspace_tool_names_are_rejected_before_model_execution() {
    let provider = halo::agent::provider::deepseek::DeepSeek::builder()
        .api_key("unused-in-construction-only")
        .build()
        .unwrap();
    let model = provider
        .chat(halo::agent::model::chat::Config::new("deepseek-flash"))
        .unwrap();
    let directory = Directory::new();
    let workspace = core::workspace::Local::new(&directory.0)
        .unwrap()
        .tools(Echo(definition("workspace_read")));
    let result = harness::builder().model(model).workspace(workspace).build();
    let Err(error) = result else {
        panic!("duplicate tool name was accepted")
    };
    assert_eq!(error.code, "INVALID_ARGUMENTS");
}

// A contract fixture, not a database backend. It deliberately has no file
// methods, path parameter, filesystem or in-memory file store.
struct DatabaseWorkspace {
    definition: halo::agent::workspace::Definition,
    query: Query,
}

impl Workspace for DatabaseWorkspace {
    fn definition(&self) -> &halo::agent::workspace::Definition {
        &self.definition
    }

    fn tool(&self) -> Vec<ToolDefinition> {
        vec![self.query.definition.clone()]
    }

    fn mcp(&self) -> Vec<McpDefinition> {
        Vec::new()
    }

    fn skill(&self) -> Vec<SkillDefinition> {
        Vec::new()
    }

    fn execute<'a>(
        &'a self,
        call: Call,
        cancellation: &'a Cancellation,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<(Message, halo::agent::tool::Evaluation), Error>,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            if cancellation.is_cancelled() {
                return Err(Error::new("CANCELLED", "operation cancelled"));
            }
            if call.name != self.query.definition.name || !call.arguments.contains_key("filter") {
                return Err(Error::new(
                    "INVALID_ARGUMENTS",
                    "invalid records_read arguments",
                ));
            }
            let message = self.query.call(&call, cancellation).await?;
            Ok((message, halo::agent::tool::Evaluation::passed()))
        })
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
        definition: halo::agent::workspace::Definition {
            mode: Mode::Server,
            path: None,
            metadata: Default::default(),
        },
        query: Query {
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
        },
    }
}

#[tokio::test]
async fn definition_only_server_keeps_tool_validation_cancellation_and_results() {
    let calls = Arc::new(AtomicUsize::new(0));
    let capabilities = database_workspace(Arc::clone(&calls));
    let error = capabilities
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
        capabilities
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
    let (result, _) = capabilities
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

#[test]
fn automatically_bound_operations_still_reject_custom_name_collisions() {
    for name in ["workspace_delete", "workspace_exec_shell"] {
        let directory = Directory::new();
        let provider = halo::agent::provider::deepseek::DeepSeek::builder()
            .api_key("unused-in-construction-only")
            .build()
            .unwrap();
        let model = provider
            .chat(halo::agent::model::chat::Config::new("deepseek-flash"))
            .unwrap();
        let workspace = core::workspace::Local::new(&directory.0)
            .unwrap()
            .sandbox(halo::agent::core::sandbox::Local::default())
            .tools(Echo(definition(name)));
        let Err(error) = harness::builder().model(model).workspace(workspace).build() else {
            panic!("built-in operation collision was accepted");
        };
        assert_eq!(error.code, "INVALID_ARGUMENTS");
    }
}
