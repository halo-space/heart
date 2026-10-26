use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use elasticsearch::{Elasticsearch, http::transport::Transport};
use halo::agent::{Cancellation, Error, Message};
use halo::agent::{
    skill::{Definition, Loader, Skill},
    tool::{Call, Tool, ToolDefinition},
};
use serde_json::json;

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
async fn skill_names_do_not_replace_tool_names() {
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

    let (instructions, _) =
        halo::agent::workspace::Workspace::execute(&workspace, skill_call(None), &cancellation)
            .await
            .unwrap();
    assert_eq!(
        instructions.values["instructions"],
        "Read references/guide.md.\n"
    );
    let (resource, _) = halo::agent::workspace::Workspace::execute(
        &workspace,
        skill_call(Some("references/guide.md")),
        &cancellation,
    )
    .await
    .unwrap();
    assert_eq!(resource.values["content"], "Use primary sources.");
    let error = halo::agent::workspace::Workspace::execute(
        &workspace,
        skill_call(Some("../other/SKILL.md")),
        &cancellation,
    )
    .await
    .unwrap_err();
    assert_eq!(error.code, "INVALID_ARGUMENTS");
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
    let (instructions, _) = halo::agent::workspace::Workspace::execute(
        &workspace,
        skill_call(None),
        &Cancellation::new(),
    )
    .await
    .unwrap();
    assert_eq!(
        instructions.values["instructions"],
        "Use the database skill."
    );
    let (resource, _) = halo::agent::workspace::Workspace::execute(
        &workspace,
        skill_call(Some("guide.md")),
        &Cancellation::new(),
    )
    .await
    .unwrap();
    assert_eq!(resource.values["content"], "Database reference");
}

struct FailingSkill {
    definition: Definition,
    reads: Arc<AtomicUsize>,
}

impl Skill for FailingSkill {
    fn definition(&self) -> &Definition {
        &self.definition
    }

    async fn read(&self, _: &Cancellation) -> Result<String, Error> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        Err(Error::new("SKILL_READ_FAILED", "skill storage unavailable"))
    }
}

#[tokio::test]
async fn skill_read_error_is_returned_directly_by_workspace_execution() {
    let directory = Directory::new();
    let reads = Arc::new(AtomicUsize::new(0));
    let workspace = halo::core::workspace::Local::new(&directory.0)
        .unwrap()
        .skills([FailingSkill {
            definition: research().definition,
            reads: Arc::clone(&reads),
        }])
        .unwrap();
    let error = halo::agent::workspace::Workspace::execute(
        &workspace,
        skill_call(None),
        &Cancellation::new(),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code, "SKILL_READ_FAILED");
    assert_eq!(reads.load(Ordering::SeqCst), 1);
}

struct CancellingSkill(Definition);

impl Skill for CancellingSkill {
    fn definition(&self) -> &Definition {
        &self.0
    }

    async fn read(&self, cancellation: &Cancellation) -> Result<String, Error> {
        cancellation.cancel();
        std::future::pending().await
    }
}

#[tokio::test]
async fn cancellation_during_skill_read_is_observed_by_workspace_execution() {
    let directory = Directory::new();
    let workspace = halo::core::workspace::Local::new(&directory.0)
        .unwrap()
        .skills([CancellingSkill(research().definition)])
        .unwrap();
    let cancellation = Cancellation::new();
    let result =
        halo::agent::workspace::Workspace::execute(&workspace, skill_call(None), &cancellation)
            .await;
    assert!(cancellation.is_cancelled());
    assert!(result.is_err());
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
    let (first, _) =
        halo::agent::workspace::Workspace::execute(&workspace, skill_call(None), &cancellation)
            .await
            .unwrap();
    assert_eq!(first.values["instructions"], "Version one.\n");

    std::fs::write(
        directory.0.join("skills/research/SKILL.md"),
        b"---\nname: research\ndescription: Research with sources.\n---\nVersion two.\n",
    )
    .unwrap();
    let (second, _) =
        halo::agent::workspace::Workspace::execute(&workspace, skill_call(None), &cancellation)
            .await
            .unwrap();
    assert_eq!(second.values["instructions"], "Version two.\n");
}
