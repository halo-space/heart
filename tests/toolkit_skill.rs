use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use futures_util::{StreamExt, stream};
use halo_agents::{
    Cancellation, Error, Message,
    agent::{self, Agent, Input, toolkit::Toolkit},
    event::Event,
    mcp::{Definition as McpDefinition, Server},
    model::chat::{Chat, Request},
    skill::{Definition, Skill},
    tool::{Call, Tool, ToolDefinition},
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
    discoveries: Arc<AtomicUsize>,
}

impl Server for Remote {
    fn definition(&self) -> &McpDefinition {
        &self.definition
    }

    async fn list_tools(&self, _: &Cancellation) -> Result<Vec<ToolDefinition>, Error> {
        self.discoveries.fetch_add(1, Ordering::SeqCst);
        Ok(vec![definition(
            "remote_search",
            "Search the remote index.",
        )])
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
async fn toolkit_provides_skill_catalog_and_all_tool_mcp_definitions_to_model() {
    let discoveries = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let toolkit = Toolkit::new()
        .tool(LocalTool(definition("weather", "Look up current weather.")))
        .mcps([Remote {
            definition: McpDefinition {
                name: "remote_server".into(),
                ..Default::default()
            },
            discoveries: Arc::clone(&discoveries),
        }])
        .skills([research()])
        .unwrap();
    let agent = agent::chat()
        .model(Model {
            requests: Arc::clone(&requests),
        })
        .system_prompt("Answer clearly.")
        .toolkit(toolkit)
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
    assert_eq!(discoveries.load(Ordering::SeqCst), 1);
}

#[test]
fn skill_names_do_not_replace_tool_names() {
    let toolkit = Toolkit::new()
        .tool(LocalTool(definition("research", "A normal business tool.")))
        .skills([research()])
        .unwrap();
    assert!(
        toolkit
            .definitions()
            .iter()
            .any(|tool| tool.name == "research")
    );
}
