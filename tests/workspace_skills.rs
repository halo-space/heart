use std::sync::Arc;

use elasticsearch::{Elasticsearch, http::transport::Transport};
use futures_util::{StreamExt, stream};
use halo::agent::{Cancellation, Error, Message};
use halo::{
    agent::event::Event,
    agent::mcp::{Definition as McpDefinition, Server},
    agent::model::chat::{Chat, Request},
    agent::skill::{Definition, Loader, Skill},
    agent::tool::{Call, Tool, ToolDefinition},
    agent::{self, Agent, Input},
};
use serde_json::json;
use tokio::sync::Mutex;

struct Research {
    definition: Definition,
    instructions: &'static str,
}

impl Skill for Research {
    fn definition(&self) -> &Definition {
        &self.definition
    }

    async fn read(&self, cancellation: &Cancellation) -> Result<String, Error> {
        if cancellation.is_cancelled() {
            return Err(Error::new("CANCELLED", "skill read cancelled"));
        }
        Ok(self.instructions.into())
    }
}

fn research() -> Research {
    Research {
        definition: Definition {
            name: "research".into(),
            description: Some("Find and cite reliable sources.".into()),
            ..Default::default()
        },
        instructions: "Search first, then cite the sources.",
    }
}

fn definition(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: description.into(),
        parameters: serde_json::from_value(json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }))
        .unwrap(),
    }
}

struct LocalTool(ToolDefinition);

impl Tool for LocalTool {
    fn definition(&self) -> &ToolDefinition {
        &self.0
    }

    async fn call(&self, _: &Call, _: &Cancellation) -> Result<Message, Error> {
        Ok(Message::function(json!({"result": "ok"})))
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
        Ok(Message::function(json!({"result": "remote"})))
    }
}

struct Model {
    requests: Arc<Mutex<Vec<Request>>>,
}

impl Chat for Model {
    type Stream = stream::Iter<std::vec::IntoIter<Result<Event, Error>>>;

    async fn stream(&self, request: Request) -> Result<Self::Stream, Error> {
        self.requests.lock().await.push(request);
        Ok(stream::iter(vec![Ok(Event::Complete {
            message: Message::function(json!({"answer": "done"})),
            usage: None,
            finish_reason: None,
        })]))
    }
}

#[tokio::test]
async fn mounted_capabilities_provides_skill_catalog_and_all_tool_mcp_definitions_to_model() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = agent::chat()
        .model(Model {
            requests: Arc::clone(&requests),
        })
        .system_prompt("Answer clearly.")
        .tools(LocalTool(definition("weather", "Look up current weather.")))
        .mcps([Remote {
            definition: McpDefinition {
                name: "remote_server".into(),
                ..Default::default()
            },
            tools: vec![definition("remote_search", "Search the remote index.")],
        }])
        .skills([research()])
        .unwrap()
        .build()
        .unwrap();

    let events = agent
        .run(Input::default(), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));

    let requests = requests.lock().await;
    assert_eq!(requests.len(), 1);
    let mut names: Vec<_> = requests[0]
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["remote_search", "skill", "weather"]);

    let system = &requests[0].messages.0[0];
    let prompt = system.content[0].data["value"].as_str().unwrap();
    assert!(prompt.contains("Answer clearly."));
    assert!(prompt.contains("research"));
    assert!(prompt.contains("Find and cite reliable sources."));
    assert!(prompt.contains("available Tools and MCP capabilities"));
}

#[test]
fn skill_names_do_not_replace_tool_names() {
    let directory = Directory::new();
    let workspace = halo::core::workspace::Local::new(&directory.0)
        .unwrap()
        .tools(LocalTool(definition("research", "A normal business tool.")))
        .skills([research()])
        .unwrap();
    assert!(
        halo::agent::workspace::Workspace::tool(&workspace)
            .iter()
            .any(|tool| tool.name == "research")
    );
}

struct Directory(std::path::PathBuf);

impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "heart-capabilities-skills-{}-{}",
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

fn skill_call(resource: Option<&str>) -> Call {
    let mut arguments = serde_json::Map::from_iter([("skill".into(), json!("research"))]);
    if let Some(resource) = resource {
        arguments.insert("resource".into(), json!(resource));
    }
    Call {
        call_id: "call_skill".into(),
        name: "skill".into(),
        arguments,
    }
}

#[tokio::test]
async fn local_workspace_discovers_skills_and_reads_adjacent_resources() {
    let directory = Directory::new();
    let workspace = halo::core::workspace::Local::new(&directory.0).unwrap();
    let cancellation = Cancellation::new();
    workspace
        .write(
            "skills/research/SKILL.md",
            b"---\nname: research\ndescription: Research with sources.\n---\nRead references/guide.md.\n"
                .to_vec(),
            &cancellation,
        )
        .await
        .unwrap();
    workspace
        .write(
            "skills/research/references/guide.md",
            b"Use primary sources.".to_vec(),
            &cancellation,
        )
        .await
        .unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let agent = halo::harness::builder()
        .model(Model {
            requests: Arc::clone(&requests),
        })
        .workspace(workspace)
        .build()
        .unwrap();
    let events = agent
        .run(Input::default(), true, Cancellation::new())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(events.iter().all(Result::is_ok));
    let requests = requests.lock().await;
    let prompt = requests[0].messages.0[0].content[0].data["value"]
        .as_str()
        .unwrap();
    assert!(prompt.contains("research"));
    assert!(requests[0].tools.iter().any(|tool| tool.name == "skill"));
    drop(requests);

    let workspace = halo::core::workspace::Local::new(&directory.0).unwrap();
    let (instructions, _) = workspace
        .execute(skill_call(None), &cancellation)
        .await
        .unwrap();
    assert_eq!(
        instructions.values["instructions"],
        "Read references/guide.md.\n"
    );
    let (resource, _) = workspace
        .execute(skill_call(Some("references/guide.md")), &cancellation)
        .await
        .unwrap();
    assert_eq!(resource.values["content"], "Use primary sources.");
    let error = workspace
        .execute(skill_call(Some("../other/SKILL.md")), &cancellation)
        .await
        .unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
}

#[tokio::test]
async fn local_workspace_refreshes_shared_skill_snapshot_when_files_change() {
    let directory = Directory::new();
    let workspace = halo::core::workspace::Local::new(&directory.0).unwrap();
    let cancellation = Cancellation::new();
    workspace
        .write(
            "skills/research/SKILL.md",
            b"---\nname: research\ndescription: Research with sources.\n---\nVersion one.\n"
                .to_vec(),
            &cancellation,
        )
        .await
        .unwrap();
    let (first, _) = workspace
        .execute(skill_call(None), &cancellation)
        .await
        .unwrap();
    assert_eq!(first.values["instructions"], "Version one.\n");

    // Simulate an application/editor update in the same Workspace. The
    // loader keeps the previous parsed snapshot until its directory
    // fingerprint changes, then atomically replaces it for later Agents.
    std::fs::write(
        directory.0.join("skills/research/SKILL.md"),
        b"---\nname: research\ndescription: Research with sources.\n---\nVersion two.\n",
    )
    .unwrap();
    let (second, _) = workspace
        .execute(skill_call(None), &cancellation)
        .await
        .unwrap();
    assert_eq!(second.values["instructions"], "Version two.\n");
}

struct DatabaseSkills;

impl Loader for DatabaseSkills {
    async fn list(&self, _: &Cancellation) -> Result<Vec<Definition>, Error> {
        Ok(vec![research().definition])
    }

    async fn read(&self, name: &str, _: &Cancellation) -> Result<String, Error> {
        assert_eq!(name, "research");
        Ok("Use the database skill.".into())
    }

    async fn read_resource(
        &self,
        name: &str,
        path: &str,
        _: &Cancellation,
    ) -> Result<Vec<u8>, Error> {
        assert_eq!((name, path), ("research", "guide.md"));
        Ok(b"Database reference".to_vec())
    }
}

#[tokio::test]
async fn server_workspace_uses_database_skill_loader_without_file_paths() {
    let client = Elasticsearch::new(Transport::single_node("http://127.0.0.1:9200").unwrap());
    let workspace = halo::core::workspace::es::new(client, "workspace")
        .unwrap()
        .skills(DatabaseSkills);
    let (instructions, _) = workspace
        .execute(skill_call(None), &Cancellation::new())
        .await
        .unwrap();
    assert_eq!(
        instructions.values["instructions"],
        "Use the database skill."
    );
    let (resource, _) = workspace
        .execute(skill_call(Some("guide.md")), &Cancellation::new())
        .await
        .unwrap();
    assert_eq!(resource.values["content"], "Database reference");
}
